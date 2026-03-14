"""
Fine-tuning SeamlessM4T for Speech-to-Text Translation (S2TT)
using a nested source/target JSON array manifest.

Expected manifest.json format:
  [
    {
      "source": {
        "id": 0,
        "lang": "eng",
        "text": "Source transcript",
        "audio_local_path": "/path/to/audio.wav",
        "sampling_rate": 16000
      },
      "target": {
        "id": 0,
        "lang": "hin",
        "text": "लक्ष्य अनुवाद पाठ",
        "audio_local_path": null,
        "sampling_rate": 16000
      }
    },
    ...
  ]

  - source.audio_local_path : path to the input speech file (wav/mp3/flac)
  - source.lang             : BCP-47 source language code  (e.g. "eng")
  - target.text             : ground-truth translation used as training label
  - target.lang             : BCP-47 target language code  (e.g. "hin")

Install dependencies:
    pip install transformers datasets torch torchaudio soundfile librosa accelerate evaluate sacrebleu
"""

import os
import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np
import soundfile as sf
from datasets import load_dataset, DatasetDict, Audio, Dataset
from transformers import (
    AutoProcessor,
    SeamlessM4TForSpeechToText,
    SeamlessM4Tv2ForSpeechToText,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Paths and language settings for your nested source/target JSON manifest."""
    train_manifest: str = "/raid/chandresh/Nivedita/STDATA/Data/Hindi/en-hi/seg_data/train/txt/manifest.json"
    dev_manifest: str   = "/raid/chandresh/Nivedita/STDATA/Data/Hindi/en-hi/seg_data/dev/txt/manifest.json"
    test_manifest: str  = "/raid/chandresh/Nivedita/STDATA/Data/Hindi/en-hi/seg_data/test/txt/manifest.json"
    audio_dir: str      = ""               # Optional base dir prepended to relative audio paths
    source_lang: str    = "eng"            # matches source.lang in your manifest
    target_lang: str    = "hin"            # matches target.lang in your manifest
    max_audio_duration: float = 1000.0       # seconds – clips longer than this are dropped
    sampling_rate: int  = 16_000


@dataclass
class ModelConfig:
    model_name_or_path: str = "facebook/seamless-m4t-v2-large"
    # Use the smaller v2-medium for lighter hardware:
    # model_name_or_path: str = "facebook/seamless-m4t-medium"


@dataclass
class TrainingConfig:
    output_dir: str             = "seamless_m4t_finetuned"
    num_train_epochs: int       = 2
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int  = 1
    gradient_accumulation_steps: int = 16       # effective batch = 4×4 = 16
    gradient_checkpointing=True
    learning_rate: float        = 1e-5
    warmup_steps: int           = 500
    weight_decay: float         = 0.01
    fp16: bool                  = True          # set False if no GPU / on MPS
    predict_with_generate: bool = True
    generation_max_length: int  = 2048
    save_strategy: str          = "epoch"
    eval_strategy: str          = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str  = "bleu"
    greater_is_better: bool     = True
    early_stopping_patience: int = 3
    logging_steps: int          = 50
    save_total_limit: int       = 3
    dataloader_num_workers: int = 4
    seed: int                   = 42


# ---------------------------------------------------------------------------
# 2. Dataset loading (nested source/target JSON array manifest)
# ---------------------------------------------------------------------------

def load_json_manifest(manifest_path: str, audio_dir: str = "") -> List[Dict]:
    """
    Load a nested source/target JSON array manifest.

    Expected format:
        [
            {
                "source": {
                    "id": 0,
                    "lang": "eng",
                    "text": "Source transcript",
                    "audio_local_path": "/path/to/audio.wav",
                    "sampling_rate": 16000,
                    ...
                },
                "target": {
                    "id": 0,
                    "lang": "hin",
                    "text": "Target translation text",
                    "audio_local_path": null,
                    ...
                }
            },
            ...
        ]

    Args:
        manifest_path: Path to the .json file.
        audio_dir:     Optional base directory prepended to relative audio paths.

    Returns:
        List of dicts with keys: audio_path, sampling_rate, transcription,
        source_lang, target_lang.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        raise ValueError(
            f"{manifest_path} must contain a JSON array at the top level, "
            f"got {type(entries).__name__}."
        )

    samples, skipped = [], 0
    for i, entry in enumerate(entries):
        # ---- validate top-level structure ----
        if "source" not in entry or "target" not in entry:
            logger.warning(f"Entry {i} missing 'source' or 'target' key — skipping.")
            skipped += 1
            continue

        src = entry["source"]
        tgt = entry["target"]

        # ---- audio path (always from source) ----
        raw_path = src.get("audio_local_path")
        if not raw_path:
            logger.warning(f"Entry {i}: source.audio_local_path is null/missing — skipping.")
            skipped += 1
            continue

        audio_path = (
            os.path.join(audio_dir, raw_path)
            if audio_dir and not os.path.isabs(raw_path)
            else raw_path
        )
        if not os.path.exists(audio_path):
            logger.warning(f"Entry {i}: audio file not found, skipping: {audio_path}")
            skipped += 1
            continue

        # ---- target translation text ----
        text = tgt.get("text", "").strip()
        if not text:
            logger.warning(f"Entry {i}: target.text is empty — skipping.")
            skipped += 1
            continue

        samples.append({
            "audio_path":    audio_path,
            "sampling_rate": src.get("sampling_rate", 16_000),
            "transcription": text,
            "source_lang":   src.get("lang", "eng"),
            "target_lang":   tgt.get("lang", "hin"),
        })

    logger.info(
        f"Loaded {len(samples)} samples from {manifest_path} "
        f"({skipped} skipped)."
    )
    return samples


def build_hf_dataset(samples: List[Dict], sampling_rate: int) -> Dataset:
    """Convert list of dicts → HuggingFace Dataset with Audio feature."""
    dataset = Dataset.from_list(
        [{"audio": s["audio_path"], "transcription": s["transcription"]} for s in samples]
    )
    # Resample all audio to a unified sampling_rate (16 kHz by default)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    return dataset


# def filter_long_audio(example, max_duration: float, sampling_rate: int) -> bool:
#     duration = len(example["audio"]["array"]) / sampling_rate
#     return duration <= max_duration


# ---------------------------------------------------------------------------
# 3. Data collator
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    target_lang: str
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # --- audio inputs ---
        audio_arrays = [f["audio"]["array"] for f in features]
        sampling_rate = features[0]["audio"]["sampling_rate"]

        input_features = self.processor(
            audios=audio_arrays,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
            src_lang=None,   # processor infers from model config
        ).input_features

        # --- text labels: tokenize directly with the tokenizer ---
        texts = [f["transcription"] for f in features]
        labels_batch = self.processor.tokenizer(
            text_target=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        # Replace padding token id with -100 so loss ignores them
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].eq(0), -100
        )
        # Remove BOS token if present (Seq2Seq trainer prepends decoder_start_token_id)
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        return {
            "input_features": input_features,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------

def build_compute_metrics(processor, target_lang: str):
    bleu_metric = evaluate.load("sacrebleu")
    wer_metric  = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids   = pred.predictions
        label_ids  = pred.label_ids

        # Replace -100 back to pad token
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        bleu = bleu_metric.compute(
            predictions=pred_str,
            references=[[r] for r in label_str],
        )
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {
            "bleu": bleu["score"],
            "wer":  wer,
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SeamlessM4T for S2TT")
    # Data
    parser.add_argument("--train_manifest", default=DataConfig.train_manifest,
                        help="Path to train manifest JSON array file")
    parser.add_argument("--dev_manifest",   default=DataConfig.dev_manifest,
                        help="Path to validation manifest JSON array file")
    parser.add_argument("--test_manifest",  default=DataConfig.test_manifest,
                        help="Path to test manifest JSON array file")
    parser.add_argument("--audio_dir",      default=DataConfig.audio_dir,
                        help="Optional base directory prepended to relative audio paths")
    parser.add_argument("--source_lang",    default=DataConfig.source_lang,
                        help="BCP-47 source language code (speech input)")
    parser.add_argument("--target_lang",    default=DataConfig.target_lang,
                        help="BCP-47 target language code (text output)")
    # parser.add_argument("--max_audio_duration", type=float,
    #                     default=DataConfig.max_audio_duration)
    # Model
    parser.add_argument("--model_name_or_path", default=ModelConfig.model_name_or_path)
    # Training
    parser.add_argument("--output_dir",          default=TrainingConfig.output_dir)
    parser.add_argument("--num_train_epochs",    type=int,   default=TrainingConfig.num_train_epochs)
    parser.add_argument("--per_device_train_batch_size", type=int,
                        default=TrainingConfig.per_device_train_batch_size)
    parser.add_argument("--learning_rate",       type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--fp16",                action="store_true", default=TrainingConfig.fp16)
    parser.add_argument("--resume_from_checkpoint", default=None,
                        help="Path to a previous checkpoint to resume training")
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                        help="Load model in 8-bit quantization to save VRAM (requires bitsandbytes)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Load processor & model ----
    logger.info(f"Loading model: {args.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(args.model_name_or_path, load_in_8bit=args.load_in_8bit)

    # Set language tokens
    processor.tokenizer.src_lang = args.source_lang
    processor.tokenizer.tgt_lang = args.target_lang

    # Forced BOS token for target language
    forced_bos_token_id = processor.tokenizer.convert_tokens_to_ids(
        f"__{args.target_lang}__"
    )
    model.config.forced_bos_token_id = forced_bos_token_id

    decoder_start_token_id = (
        model.config.decoder_start_token_id
        or processor.tokenizer.bos_token_id
    )

    # ---- Build datasets ----
    logger.info("Building datasets …")
    sr = DataConfig.sampling_rate

    def make_split(manifest):
        samples = load_json_manifest(manifest, args.audio_dir)
        ds = build_hf_dataset(samples, sr)
        # ds = ds.filter(
        #     lambda ex: filter_long_audio(ex, args.max_audio_duration, sr),
        #     desc="Filtering long audio",
        # )
        return ds

    dataset = DatasetDict({
        "train": make_split(args.train_manifest),
        "validation": make_split(args.dev_manifest),
        "test": make_split(args.test_manifest),
    })
    logger.info(f"Dataset sizes — {dict((k, len(v)) for k, v in dataset.items())}")

    # ---- Collator ----
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        target_lang=args.target_lang,
        decoder_start_token_id=decoder_start_token_id,
    )

    # ---- Training arguments ----
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=TrainingConfig.per_device_eval_batch_size,
        gradient_accumulation_steps=TrainingConfig.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=TrainingConfig.warmup_steps,
        weight_decay=TrainingConfig.weight_decay,
        fp16=args.fp16,
        predict_with_generate=True,
        generation_max_length=TrainingConfig.generation_max_length,
        save_strategy=TrainingConfig.save_strategy,
        eval_strategy=TrainingConfig.eval_strategy,
        load_best_model_at_end=TrainingConfig.load_best_model_at_end,
        metric_for_best_model=TrainingConfig.metric_for_best_model,
        greater_is_better=TrainingConfig.greater_is_better,
        logging_steps=TrainingConfig.logging_steps,
        save_total_limit=TrainingConfig.save_total_limit,
        dataloader_num_workers=TrainingConfig.dataloader_num_workers,
        seed=TrainingConfig.seed,
        report_to="tensorboard",
        push_to_hub=False,
        remove_unused_columns=False,
    )

    # ---- Trainer ----
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor.feature_extractor,   # used for collation logging
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(processor, args.target_lang),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=TrainingConfig.early_stopping_patience
        )],
    )

    # ---- Train ----
    logger.info("Starting training …")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ---- Save best model ----
    logger.info(f"Saving best model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # ---- Evaluate on test set ----
    logger.info("Evaluating on test set …")
    metrics = trainer.evaluate(
        eval_dataset=dataset["test"],
        metric_key_prefix="test",
    )
    logger.info(f"Test metrics: {metrics}")
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)


# ---------------------------------------------------------------------------
# 6. Inference helper (run after training)
# ---------------------------------------------------------------------------

def translate_audio_file(
    audio_path: str,
    model_dir: str,
    source_lang: str = "eng",
    target_lang: str = "hin",
) -> str:
    """
    Translate a single audio file using the fine-tuned model.

    Usage:
        text = translate_audio_file("sample.wav", "seamless_m4t_finetuned")
        print(text)
    """
    processor = AutoProcessor.from_pretrained(model_dir)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_dir)
    model.eval()

    audio_array, sr = sf.read(audio_path, dtype="float32")
    if sr != 16_000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16_000)

    inputs = processor(
        audios=audio_array,
        sampling_rate=16_000,
        return_tensors="pt",
    )

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            tgt_lang=target_lang,
            num_beams=5,
        )

    translation = processor.decode(output_tokens[0], skip_special_tokens=True)
    return translation


if __name__ == "__main__":
    main()