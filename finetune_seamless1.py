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
    pip install transformers==4.45.2 tokenizers==0.20.3 datasets torch torchaudio
    pip install soundfile librosa accelerate evaluate sacrebleu peft bitsandbytes
"""

import os
import json
import logging
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import numpy as np
import soundfile as sf
from datasets import DatasetDict, Audio, Dataset
from transformers import (
    AutoProcessor,
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
    train_manifest: str   = "/raid/chandresh/Nivedita/STDATA/Data/Hindi/en-hi/seg_data/train/txt/manifest.json"
    dev_manifest: str     = "/raid/chandresh/Nivedita/STDATA/Data/Hindi/en-hi/seg_data/dev/txt/manifest.json"
    test_manifest: str    = "/raid/chandresh/Nivedita/STDATA/Data/Hindi/en-hi/seg_data/test/txt/manifest.json"
    audio_dir: str        = ""       # Optional base dir prepended to relative audio paths
    source_lang: str      = "eng"   # matches source.lang in your manifest
    target_lang: str      = "hin"   # matches target.lang in your manifest
    sampling_rate: int    = 16_000


@dataclass
class ModelConfig:
    model_name_or_path: str = "facebook/seamless-m4t-v2-large"


@dataclass
class TrainingConfig:
    output_dir: str                  = "seamless_m4t_finetuned"
    num_train_epochs: int            = 10
    per_device_train_batch_size: int = 8    # keep at 1 for 40GB GPU with 8-bit + LoRA
    per_device_eval_batch_size: int  = 8
    gradient_accumulation_steps: int = 16   # effective batch = 1×16 = 16
    learning_rate: float             = 1e-4  # higher LR suits LoRA adapters
    warmup_steps: int                = 500
    weight_decay: float              = 0.01
    fp16: bool                       = True
    gradient_checkpointing: bool     = True  # trades compute for memory
    predict_with_generate: bool      = True
    generation_max_length: int       = 256
    save_strategy: str               = "epoch"
    eval_strategy: str               = "epoch"
    load_best_model_at_end: bool     = True
    metric_for_best_model: str       = "bleu"
    greater_is_better: bool          = True
    early_stopping_patience: int     = 3
    logging_steps: int               = 50
    save_total_limit: int            = 3
    dataloader_num_workers: int      = 2
    ddp_find_unused_parameters: bool = True   # SeamlessM4T has unused params in S2TT forward pass
    seed: int                        = 42


@dataclass
class LoraHyperParams:
    r: int              = 16    # LoRA rank — higher = more capacity, more memory
    lora_alpha: int     = 32    # scaling: effective multiplier = alpha/r = 2x
    lora_dropout: float = 0.05
    # Attention projection layers in both speech encoder and text decoder
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "out_proj")


# ---------------------------------------------------------------------------
# 2. Model loading + LoRA
# ---------------------------------------------------------------------------

def load_model(name_or_path: str, load_in_8bit: bool = False) -> SeamlessM4Tv2ForSpeechToText:
    """
    Load SeamlessM4Tv2ForSpeechToText with optional 8-bit quantization.

    Multi-GPU (DDP) notes:
      - Each torchrun process is assigned one GPU via LOCAL_RANK.
      - 8-bit models MUST be loaded with device_map={'': current_device} in DDP,
        NOT device_map="auto" (which spreads across all GPUs in one process).
      - Full precision: no device_map needed — DDP/accelerate handles placement.
    """
    kwargs: Dict[str, Any] = {}
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

    if load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            if is_distributed:
                # Pin to this process's GPU — local_rank is set before this call in main()
                # so current_device() is already correct, but we pass it explicitly too
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                kwargs["device_map"] = {"": local_rank}
            else:
                kwargs["device_map"] = "auto"
            logger.info(
                f"Loading model in 8-bit quantization "
                f"(device=cuda:{torch.cuda.current_device()})"
            )
        except ImportError:
            logger.warning(
                "bitsandbytes not installed — loading in full precision. "
                "Install with: pip install bitsandbytes"
            )
    else:
        if not is_distributed:
            kwargs["device_map"] = "auto"
        # DDP + full precision: no device_map — accelerate moves model to correct device

    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(name_or_path, **kwargs)
    logger.info(f"Loaded: {name_or_path}")
    return model


def apply_lora(model, load_in_8bit: bool = False):
    """
    Wrap model with LoRA adapters.
    - Required when load_in_8bit=True (quantized models cannot be trained directly).
    - Also beneficial in full-precision: trains only ~1% of params → less memory, faster.
    """
    hp = LoraHyperParams()

    if load_in_8bit:
        # Cast layer norms to fp32 and enable input gradients for quantized layers
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=TrainingConfig.gradient_checkpointing,
        )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=hp.r,
        lora_alpha=hp.lora_alpha,
        lora_dropout=hp.lora_dropout,
        target_modules=list(hp.target_modules),
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# 3. Dataset loading
# ---------------------------------------------------------------------------

def load_json_manifest(manifest_path: str, audio_dir: str = "", max_duration_secs: float = 30.0) -> List[Dict]:
    """
    Load a nested source/target JSON array manifest.

    Keys used:
      source.audio_local_path  →  path to input speech file
      source.lang              →  source language code
      target.text              →  ground-truth translation (training label)
      target.lang              →  target language code

    Args:
      max_duration_secs: clips longer than this are skipped to avoid OOM.
                         SeamlessM4T feature extractor caps at 60s, but 30s
                         is a safe limit for 8-bit + LoRA on 40GB GPUs.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        raise ValueError(
            f"{manifest_path} must be a JSON array, got {type(entries).__name__}."
        )

    samples, skipped = [], 0
    for i, entry in enumerate(entries):
        if "source" not in entry or "target" not in entry:
            logger.warning(f"Entry {i} missing 'source' or 'target' — skipping.")
            skipped += 1
            continue

        src = entry["source"]
        tgt = entry["target"]

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
            logger.warning(f"Entry {i}: audio not found — skipping: {audio_path}")
            skipped += 1
            continue

        # Check duration — long clips cause OOM even at batch_size=1
        try:
            info = sf.info(audio_path)
            duration = info.frames / info.samplerate
            if duration > max_duration_secs:
                logger.warning(
                    f"Entry {i}: audio too long ({duration:.1f}s > {max_duration_secs}s) — skipping."
                )
                skipped += 1
                continue
        except Exception:
            pass  # if we can't read info, let it through and OOM handler will catch it

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

    logger.info(f"Loaded {len(samples)} samples from {manifest_path} ({skipped} skipped).")
    return samples


def build_hf_dataset(samples: List[Dict], sampling_rate: int) -> Dataset:
    """Convert list of dicts → HuggingFace Dataset with Audio feature."""
    dataset = Dataset.from_list(
        [{"audio": s["audio_path"], "transcription": s["transcription"]} for s in samples]
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    return dataset


# ---------------------------------------------------------------------------
# 4. Data collator
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    target_lang: str
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # --- audio → input_features ---
        audio_arrays  = [f["audio"]["array"] for f in features]
        sampling_rate = features[0]["audio"]["sampling_rate"]

        input_features = self.processor(
            audios=audio_arrays,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        ).input_features

        # --- text → labels ---
        # text_target tells the tokenizer to encode in the target language direction
        texts = [f["transcription"] for f in features]
        labels_batch = self.processor.tokenizer(
            text_target=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )

        # Mask padding so loss ignores it
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].eq(0), -100
        )
        # Drop BOS — Seq2SeqTrainer prepends decoder_start_token_id automatically
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        return {"input_features": input_features, "labels": labels}


# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------

def build_compute_metrics(processor):
    bleu_metric = evaluate.load("sacrebleu")
    wer_metric  = evaluate.load("wer")

    vocab_size = processor.tokenizer.vocab_size

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids

        # --- guard against invalid token IDs (e.g. from OOM-skipped batches) ---
        # Clip to valid vocab range and replace any remaining -100 with pad
        if isinstance(pred_ids, np.ndarray):
            pred_ids = np.clip(pred_ids, 0, vocab_size - 1)
        else:
            pred_ids = [
                [max(0, min(t, vocab_size - 1)) for t in seq]
                for seq in pred_ids
            ]

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_ids = np.clip(label_ids, 0, vocab_size - 1)

        pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Filter out empty predictions (from skipped batches)
        pairs = [(p, r) for p, r in zip(pred_str, label_str) if p.strip()]
        if not pairs:
            return {"bleu": 0.0, "wer": 1.0}
        pred_str, label_str = zip(*pairs)

        bleu = bleu_metric.compute(predictions=list(pred_str), references=[[r] for r in label_str])
        wer  = wer_metric.compute(predictions=list(pred_str),  references=list(label_str))
        return {"bleu": bleu["score"], "wer": wer}

    return compute_metrics


# ---------------------------------------------------------------------------
# 6. OOM-safe Trainer
# ---------------------------------------------------------------------------

class OOMSkippingSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer that:
      1. Skips batches that cause CUDA OOM during training.
      2. Skips batches that cause CUDA OOM during evaluation/prediction.
      3. Passes tgt_lang to generation so SeamlessM4T produces the correct language.
    """

    _oom_count: int = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            if num_items_in_batch is not None:
                return super().training_step(model, inputs, num_items_in_batch)
            return super().training_step(model, inputs)
        except torch.cuda.OutOfMemoryError:
            self._oom_count += 1
            logger.warning(
                f"[OOM #{self._oom_count}] CUDA OOM during training — skipping batch."
            )
            torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)

            # In DDP, we cannot return a disconnected zero tensor — DDP will crash
            # with "undefined gradient allreduced". Instead we do a real forward pass
            # on a tiny 1-frame dummy input so the gradient graph is intact, then
            # multiply loss by 0 so no actual weight update happens.
            # We also use no_sync() to skip the all-reduce for this skipped step.
            try:
                # Build a minimal 1-frame dummy input on the correct device
                device = self.args.device
                seq_len = 4  # minimum valid length for the speech encoder
                dummy_inputs = {
                    "input_features": torch.zeros(
                        1, seq_len, model.config.feature_size if hasattr(model.config, "feature_size") else 160,
                        device=device, dtype=torch.float32
                    ),
                    "labels": torch.zeros(1, 2, device=device, dtype=torch.long),
                }
                # no_sync skips DDP gradient all-reduce for this step
                ctx = model.no_sync() if hasattr(model, "no_sync") else torch.no_grad()
                with ctx:
                    out = model(**dummy_inputs)
                    loss = out.loss * 0.0  # real graph, zero contribution
                    self.accelerator.backward(loss)
                model.zero_grad(set_to_none=True)
                return loss.detach()
            except Exception:
                # If even the dummy fails, zero out grads and return scalar 0
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                return torch.tensor(0.0, device=self.args.device)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        try:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
        except torch.cuda.OutOfMemoryError:
            self._oom_count += 1
            logger.warning(
                f"[OOM #{self._oom_count}] CUDA OOM during evaluation — skipping batch."
            )
            torch.cuda.empty_cache()
            # Return None tensors — the Trainer will skip this batch in metric computation
            return (None, None, None)

    def _inject_tgt_lang(self):
        """Returns patched gen_kwargs with tgt_lang, and the original to restore later."""
        orig = dict(getattr(self, "_gen_kwargs", {}) or {})
        tgt_lang = getattr(self.data_collator, "target_lang", None)
        if tgt_lang:
            self._gen_kwargs = {**orig, "tgt_lang": tgt_lang}
        return orig

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        orig = self._inject_tgt_lang()
        result = super().evaluate(eval_dataset, ignore_keys=ignore_keys,
                                  metric_key_prefix=metric_key_prefix)
        self._gen_kwargs = orig
        return result

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        orig = self._inject_tgt_lang()
        result = super().predict(test_dataset, ignore_keys=ignore_keys,
                                 metric_key_prefix=metric_key_prefix)
        self._gen_kwargs = orig
        return result



def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SeamlessM4T for S2TT")

    # Data
    parser.add_argument("--train_manifest", default=DataConfig.train_manifest)
    parser.add_argument("--dev_manifest",   default=DataConfig.dev_manifest)
    parser.add_argument("--test_manifest",  default=DataConfig.test_manifest)
    parser.add_argument("--audio_dir",      default=DataConfig.audio_dir)
    parser.add_argument("--source_lang",    default=DataConfig.source_lang)
    parser.add_argument("--target_lang",    default=DataConfig.target_lang)

    # Model
    parser.add_argument("--model_name_or_path", default=ModelConfig.model_name_or_path)
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                        help="Load in 8-bit quantization (requires bitsandbytes). "
                             "LoRA adapters are added automatically.")

    # Training
    parser.add_argument("--output_dir",                  default=TrainingConfig.output_dir)
    parser.add_argument("--num_train_epochs",    type=int,   default=TrainingConfig.num_train_epochs)
    parser.add_argument("--per_device_train_batch_size", type=int,
                        default=TrainingConfig.per_device_train_batch_size)
    parser.add_argument("--learning_rate",       type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--fp16",                action="store_true", default=TrainingConfig.fp16)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--train_fraction", type=float, default=1.0,
                        help="Fraction of training set to use, e.g. 0.01 for 1%%.")
    parser.add_argument("--val_fraction", type=float, default=1.0,
                        help="Fraction of validation set to use, e.g. 0.01 for 1%%.")
    parser.add_argument("--test_fraction", type=float, default=1.0,
                        help="Fraction of test set to use, e.g. 0.01 for 1%%.")
    parser.add_argument("--max_duration", type=float, default=30.0,
                        help="Skip audio clips longer than this many seconds (default: 30). "
                             "Longer clips cause OOM even at batch_size=1.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Multi-GPU info ----
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        # CRITICAL: must set device BEFORE any CUDA operations including model loading
        torch.cuda.set_device(local_rank)
        logger.info(f"Multi-GPU training: {world_size} GPUs | local_rank={local_rank} | device=cuda:{local_rank}")
    else:
        logger.info("Single-GPU training")

    # Reduce CUDA memory fragmentation (recommended when near GPU memory limit)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # ---- Processor ----
    logger.info(f"Loading processor: {args.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    processor.tokenizer.src_lang = args.source_lang
    processor.tokenizer.tgt_lang = args.target_lang

    # ---- Model ----
    model = load_model(args.model_name_or_path, load_in_8bit=args.load_in_8bit)

    # Force decoder to produce target language tokens
    forced_bos_token_id = processor.tokenizer.convert_tokens_to_ids(
        f"__{args.target_lang}__"
    )
    model.config.forced_bos_token_id = forced_bos_token_id
    decoder_start_token_id = (
        model.config.decoder_start_token_id or processor.tokenizer.bos_token_id
    )

    # ---- LoRA adapters ----
    # Always applied: required for 8-bit, also saves memory in full precision.
    model = apply_lora(model, load_in_8bit=args.load_in_8bit)

    # Gradient checkpointing for full-precision (8-bit handled inside apply_lora)
    if TrainingConfig.gradient_checkpointing and not args.load_in_8bit:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("Gradient checkpointing enabled")

    # ---- Datasets ----
    logger.info("Building datasets …")
    sr = DataConfig.sampling_rate

    def make_split(manifest, fraction: float = 1.0):
        samples = load_json_manifest(manifest, args.audio_dir, max_duration_secs=args.max_duration)
        if fraction < 1.0:
            n = max(1, int(len(samples) * fraction))
            samples = samples[:n]
            logger.info(f"Using {n}/{len(samples) + (len(samples) - n)} samples "
                        f"({fraction*100:.1f}% of {manifest})")
        return build_hf_dataset(samples, sr)

    dataset = DatasetDict({
        "train":      make_split(args.train_manifest, fraction=args.train_fraction),
        "validation": make_split(args.dev_manifest,   fraction=args.val_fraction),
        "test":       make_split(args.test_manifest,  fraction=args.test_fraction),
    })
    logger.info(f"Dataset sizes — {dict((k, len(v)) for k, v in dataset.items())}")

    # ---- Collator ----
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        target_lang=args.target_lang,
        decoder_start_token_id=decoder_start_token_id,
    )

    # ---- Training arguments ----
    # transformers 4.46+ renamed 'tokenizer' → 'processing_class' in Trainer
    import transformers as _tf
    _tf_ver = tuple(int(x) for x in _tf.__version__.split(".")[:2])
    _proc_kwarg = (
        {"processing_class": processor.feature_extractor}
        if _tf_ver >= (4, 46)
        else {"tokenizer": processor.feature_extractor}
    )

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
        gradient_checkpointing=TrainingConfig.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
        remove_unused_columns=False,  # collator maps audio/transcription → input_features/labels
        eval_accumulation_steps=8,    # offload eval predictions to CPU every 8 steps to save GPU memory
        ddp_find_unused_parameters=TrainingConfig.ddp_find_unused_parameters,
    )

    # ---- Trainer ----
    trainer = OOMSkippingSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        **_proc_kwarg,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(processor),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=TrainingConfig.early_stopping_patience
        )],
    )

    # ---- Train ----
    logger.info("Starting training …")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if trainer._oom_count > 0:
        logger.warning(f"Training completed with {trainer._oom_count} OOM-skipped batch(es).")

    # ---- Save ----
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # ---- Test evaluation ----
    logger.info("Evaluating on test set …")
    test_output = trainer.predict(dataset["test"], metric_key_prefix="test")
    metrics = test_output.metrics
    logger.info(f"Test metrics: {metrics}")
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)


# ---------------------------------------------------------------------------
# 8. Inference helper
# ---------------------------------------------------------------------------

def translate_audio_file(
    audio_path: str,
    model_dir: str,
    source_lang: str = "eng",
    target_lang: str = "hin",
    device: str = "cuda",
) -> str:
    """
    Translate a single audio file using the fine-tuned LoRA model.

    Usage:
        text = translate_audio_file("sample.wav", "seamless_m4t_finetuned")
        print(text)
    """
    from peft import PeftModel

    processor  = AutoProcessor.from_pretrained(model_dir)
    base_model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_dir, device_map=device)
    model      = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()

    audio_array, sr = sf.read(audio_path, dtype="float32")
    if sr != 16_000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16_000)

    inputs = processor(
        audios=audio_array,
        sampling_rate=16_000,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_tokens = model.generate(**inputs, tgt_lang=target_lang, num_beams=5)

    return processor.decode(output_tokens[0], skip_special_tokens=True)


if __name__ == "__main__":
    main()