"""
Fine-tuning Qwen2-Audio for Speech-to-Text Translation
=======================================================
Supports:
  • Single-GPU, multi-GPU (DDP via torchrun), and multi-node training
  • Fraction-based data sampling  (--train_frac 0.1 uses 10% of train set)
  • QLoRA / LoRA parameter-efficient fine-tuning
  • Inference on a single audio file after training

Custom JSON dataset format expected:
{
    "source": {
        "id": 0,
        "lang": "eng",
        "text": "...",
        "audio_local_path": "/path/to/audio.wav",
        "waveform": null,
        "sampling_rate": 16000,
        "units": null
    },
    "target": {
        "id": 0,
        "lang": "hin",
        "text": "...",          ← training label
        "audio_local_path": null,
        ...
    }
}

──────────────────────────────────────────────
LAUNCH COMMANDS
──────────────────────────────────────────────

# 1. Single GPU
python finetune_qwen2_audio.py --mode train

# 2. Multi-GPU on 1 node  (e.g. 4 GPUs)
torchrun --nproc_per_node=4 finetune_qwen2_audio.py --mode train

# 3. Multi-node  (2 nodes, 4 GPUs each)
torchrun --nnodes=2 --nproc_per_node=4 \
         --rdzv_backend=c10d --rdzv_endpoint=<MASTER_IP>:29500 \
         finetune_qwen2_audio.py --mode train

# 4. With accelerate (auto-detects GPUs)
accelerate launch finetune_qwen2_audio.py --mode train

# 5. Train on 10% of data, evaluate on 5%
python finetune_qwen2_audio.py --mode train --train_frac 0.1 --eval_frac 0.05

# 6. Test / score the model on held-out test split
python finetune_qwen2_audio.py --mode test --test_json data/test.json

──────────────────────────────────────────────

Requirements:
    pip install transformers torch torchaudio datasets accelerate \
                peft bitsandbytes librosa soundfile evaluate sacrebleu
"""

# ─────────────────────────────────────────────
# 0.  Imports
# ─────────────────────────────────────────────
import os
import sys
import types
import importlib
import importlib.util
import importlib.abc
import importlib.machinery

# ── Permanently disable flash_attn BEFORE importing transformers ──────────
#
# Problem: flash_attn was compiled against libcudart.so.12 (CUDA 12).
# That library is missing here, so any attempt to load flash_attn crashes.
# transformers checks availability via importlib.util.find_spec() AND then
# does unconditional top-level imports inside modeling_flash_attention_utils.
#
# Three-layer fix:
#   1. Clear any cached flash_attn from sys.modules (stale stubs / partial loads)
#   2. Patch importlib.util.find_spec → return None for flash_attn so that
#      is_flash_attn_2_available() returns False and transformers skips the
#      entire flash-attention code path.
#   3. Install a MetaPathFinder at the front of sys.meta_path so that any
#      remaining `from flash_attn import X` statement (e.g. unconditional
#      imports inside modeling_flash_attention_utils) gets a harmless stub
#      instead of a CUDA crash.
# ──────────────────────────────────────────────────────────────────────────
def _disable_flash_attn() -> None:

    # ── Layer 1: purge any existing flash_attn entries ────────────────────
    for _key in list(sys.modules.keys()):
        if _key == "flash_attn" or _key.startswith("flash_attn."):
            del sys.modules[_key]

    # ── Layer 2: patch importlib.util.find_spec ───────────────────────────
    # is_flash_attn_2_available() calls importlib.util.find_spec("flash_attn").
    # Returning None makes it evaluate to False immediately.
    _orig_find_spec = importlib.util.find_spec

    def _patched_find_spec(name, package=None):
        if isinstance(name, str) and (
            name == "flash_attn" or name.startswith("flash_attn.")
        ):
            return None   # ← tells transformers: flash_attn not installed
        return _orig_find_spec(name, package)

    importlib.util.find_spec = _patched_find_spec

    # ── Layer 3: MetaPathFinder stub for direct `from flash_attn import X` ─
    # Even if is_flash_attn_2_available() returns False, some transformers
    # versions do unconditional top-level imports of flash_attn symbols.
    # This finder intercepts those and returns a module with None-valued stubs.
    _FA_ATTRS = {
        "flash_attn": [
            "flash_attn_func", "flash_attn_varlen_func",
            "flash_attn_with_kvcache", "flash_attn_kvpacked_func",
        ],
        "flash_attn.bert_padding": [
            "index_first_axis", "pad_input", "unpad_input",
            "index_put_first_axis",
        ],
        "flash_attn.flash_attn_interface": [
            "flash_attn_func", "flash_attn_varlen_func",
            "flash_attn_with_kvcache",
        ],
        "flash_attn.flash_attn_utils": [],
        "flash_attn.flash_attn_triton": [],
    }

    class _FlashAttnStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path, target=None):
            if fullname == "flash_attn" or fullname.startswith("flash_attn."):
                return importlib.machinery.ModuleSpec(fullname, self,
                                                      is_package=True)
            return None

        def create_module(self, spec):
            mod = types.ModuleType(spec.name)
            mod.__path__   = []
            mod.__spec__   = spec
            mod.__loader__ = self
            mod.__package__ = spec.name
            for attr in _FA_ATTRS.get(spec.name, []):
                setattr(mod, attr, None)
            return mod

        def exec_module(self, module):
            pass   # nothing to execute; attributes set in create_module

    sys.meta_path.insert(0, _FlashAttnStubFinder())


_disable_flash_attn()
# ──────────────────────────────────────────────────────────────────────────

import json
import math
import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import torch
import soundfile as sf
from datasets import Dataset, DatasetDict
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────
def is_main_process() -> bool:
    """True only on rank-0 (or when not running distributed)."""
    rank = int(os.environ.get("RANK", "0"))
    return rank == 0


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


# ─────────────────────────────────────────────
# 1.  Configuration
# ─────────────────────────────────────────────
@dataclass
class ScriptConfig:
    # ── Model ──────────────────────────────────
    model_id: str = "Qwen/Qwen2-Audio-7B-Instruct"
    output_dir: str = "./qwen2-audio-translation-finetuned"

    # ── Dataset paths ──────────────────────────
    train_json: str = "/raid/chandresh/Nivedita/STDATA/Data/Hindi/en-hi/seg_data/train/txt/manifest.json"
    eval_json:  str = "/raid/chandresh/Nivedita/STDATA/Data/Hindi/en-hi/seg_data/dev/txt/manifest.json"
    test_json:  str = "/raid/chandresh/Nivedita/STDATA/Data/Hindi/en-hi/seg_data/test/txt/manifest.json"
    # Optional base directory to rebase audio_local_path values.
    # Leave "" to use paths exactly as written in the JSON.
    audio_root: str = ""

    # ── Fraction-based sampling ─────────────────
    # Range: (0.0, 1.0].  1.0 = use all data (default).
    train_frac: float = 1.0
    eval_frac:  float = 1.0
    test_frac:  float = 1.0

    # ── Language / prompt ──────────────────────
    target_language_name: str = "Hindi"

    # ── LoRA ───────────────────────────────────
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # ── Training ───────────────────────────────
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    seed: int = 42
    fp16: bool = True
    bf16: bool = False

    # ── Distributed ────────────────────────────
    # Passed automatically by torchrun / accelerate; no need to set manually.
    local_rank: int = -1   # set by torchrun via env or CLI

    # ── Inference ──────────────────────────────
    audio_file: str = ""


# ─────────────────────────────────────────────
# 2.  Audio / path helpers
# ─────────────────────────────────────────────
def build_system_prompt(target_lang_name: str) -> str:
    return (
        f"You are a speech translation system. "
        f"Listen to the audio and output only the {target_lang_name} translation. "
        f"Do not add any explanation."
    )


def resolve_audio_path(raw_path: str, audio_root: str) -> str:
    if not audio_root:
        return raw_path
    p = Path(raw_path)
    return str(Path(audio_root) / (p.name if p.is_absolute() else p))


def load_audio_array(path: str, target_sr: int = 16_000) -> np.ndarray:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


# ─────────────────────────────────────────────
# 3.  Custom JSON dataset loading
# ─────────────────────────────────────────────
def load_json_file(
    json_path: str,
    audio_root: str,
    frac: float = 1.0,
    split_name: str = "",
) -> List[Dict[str, Any]]:
    """
    Read JSON, resolve audio paths, drop bad records, apply fraction sampling.
    Returns flat list of dicts: {sample_id, audio_path, source_text, target_text}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    samples, skipped = [], 0
    for rec in records:
        src = rec["source"]
        tgt = rec["target"]

        raw_path = src.get("audio_local_path") or ""
        if not raw_path:
            skipped += 1
            continue

        audio_path = resolve_audio_path(raw_path, audio_root)
        if not os.path.exists(audio_path):
            if is_main_process():
                logger.warning(f"Missing audio, skipping id={src['id']}: {audio_path}")
            skipped += 1
            continue

        target_text = tgt.get("text", "").strip()
        if not target_text:
            skipped += 1
            continue

        samples.append({
            "sample_id":   src["id"],
            "audio_path":  audio_path,
            "source_text": src.get("text", ""),
            "target_text": target_text,
        })

    # ── Fraction sampling ──────────────────────
    if frac < 1.0:
        n = max(1, math.ceil(len(samples) * frac))
        # Deterministic: take first n (shuffle before calling if you prefer random)
        samples = samples[:n]

    if is_main_process():
        tag = f"[{split_name}]" if split_name else ""
        logger.info(
            f"{tag} {len(samples)} samples loaded from {json_path}  "
            f"(frac={frac:.2f}, skipped={skipped})"
        )
    return samples


def load_custom_dataset(cfg: ScriptConfig) -> DatasetDict:
    train_samples = load_json_file(cfg.train_json, cfg.audio_root, cfg.train_frac, "train")
    eval_samples  = load_json_file(cfg.eval_json,  cfg.audio_root, cfg.eval_frac,  "eval")
    return DatasetDict({
        "train":      Dataset.from_list(train_samples),
        "validation": Dataset.from_list(eval_samples),
    })


def load_test_dataset(cfg: ScriptConfig) -> Dataset:
    test_samples = load_json_file(cfg.test_json, cfg.audio_root, cfg.test_frac, "test")
    return Dataset.from_list(test_samples)


# ─────────────────────────────────────────────
# 4.  Preprocessing
# ─────────────────────────────────────────────
def make_preprocess_fn(processor: AutoProcessor, system_prompt: str):
    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        audio_array = load_audio_array(example["audio_path"])

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "audio", "audio_url": "placeholder"}]},
        ]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        model_inputs = processor(
            text=text,
            audios=[audio_array],
            sampling_rate=16_000,
            return_tensors="pt",
            padding=False,
        )

        label_ids = processor.tokenizer(
            example["target_text"] + processor.tokenizer.eos_token,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids

        return {
            "input_ids":               model_inputs["input_ids"][0],
            "attention_mask":          model_inputs["attention_mask"][0],
            "input_features":          model_inputs["input_features"][0],
            "feature_attention_mask":  model_inputs["feature_attention_mask"][0],
            "labels":                  label_ids[0],
        }

    return preprocess


# ─────────────────────────────────────────────
# 5.  Data collator
# ─────────────────────────────────────────────
@dataclass
class SpeechTranslationCollator:
    processor: AutoProcessor
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            [f["input_ids"]      for f in features], batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        attn_mask_padded = torch.nn.utils.rnn.pad_sequence(
            [f["attention_mask"] for f in features], batch_first=True, padding_value=0,
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            [f["labels"]         for f in features], batch_first=True,
            padding_value=self.label_pad_token_id,
        )
        input_features         = torch.stack([f["input_features"]          for f in features])
        feature_attention_mask = torch.stack([f["feature_attention_mask"]  for f in features])

        return {
            "input_ids":               input_ids_padded,
            "attention_mask":          attn_mask_padded,
            "input_features":          input_features,
            "feature_attention_mask":  feature_attention_mask,
            "labels":                  labels_padded,
        }


# ─────────────────────────────────────────────
# 6.  Model loading
# ─────────────────────────────────────────────
def load_model_and_processor(cfg: ScriptConfig):
    if is_main_process():
        logger.info(f"Loading processor : {cfg.model_id}")
    processor = AutoProcessor.from_pretrained(cfg.model_id)

    quant_kwargs: Dict[str, Any] = {}
    if cfg.use_qlora:
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    if is_main_process():
        logger.info(f"Loading model     : {cfg.model_id}")

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",   # avoids flash_attn / libcudart.so.12 issues
        **quant_kwargs,
    )

    if cfg.use_qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    if is_main_process():
        model.print_trainable_parameters()

    return model, processor


# ─────────────────────────────────────────────
# 7.  Evaluation metric  (BLEU)
# ─────────────────────────────────────────────
def build_compute_metrics(processor: AutoProcessor):
    bleu_metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_pred):
        predictions, label_ids = eval_pred
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        decoded_preds  = processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = processor.tokenizer.batch_decode(label_ids,   skip_special_tokens=True)

        result = bleu_metric.compute(
            predictions=decoded_preds,
            references=[[lbl] for lbl in decoded_labels],
        )
        return {"bleu": round(result["score"], 4)}

    return compute_metrics


# ─────────────────────────────────────────────
# 8.  Build TrainingArguments (distributed-aware)
# ─────────────────────────────────────────────
def build_training_args(cfg: ScriptConfig) -> TrainingArguments:
    world_size = get_world_size()
    effective_batch = (
        cfg.per_device_train_batch_size
        * cfg.gradient_accumulation_steps
        * world_size
    )
    if is_main_process():
        logger.info(
            f"Distributed training: world_size={world_size}  "
            f"effective_batch_size={effective_batch}"
        )

    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_grad_norm=cfg.max_grad_norm,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        # ── Distributed ──────────────────────────
        local_rank=cfg.local_rank,           # honours torchrun env var
        ddp_find_unused_parameters=False,    # LoRA sets unused params; False = faster
        # ── Logging / saving ─────────────────────
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=256,
        report_to="tensorboard",
        seed=cfg.seed,
        dataloader_num_workers=0,    # safest for custom audio I/O
        remove_unused_columns=False,
    )


# ─────────────────────────────────────────────
# 9.  Train
# ─────────────────────────────────────────────
def train(cfg: ScriptConfig):
    set_seed(cfg.seed)

    raw_ds        = load_custom_dataset(cfg)
    system_prompt = build_system_prompt(cfg.target_language_name)
    model, processor = load_model_and_processor(cfg)
    preprocess_fn = make_preprocess_fn(processor, system_prompt)

    raw_cols = ["sample_id", "audio_path", "source_text", "target_text"]

    if is_main_process():
        logger.info("Preprocessing train split …")
    train_ds = raw_ds["train"].map(preprocess_fn, remove_columns=raw_cols, num_proc=1)

    if is_main_process():
        logger.info("Preprocessing validation split …")
    eval_ds = raw_ds["validation"].map(preprocess_fn, remove_columns=raw_cols, num_proc=1)

    collator      = SpeechTranslationCollator(processor=processor)
    training_args = build_training_args(cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=build_compute_metrics(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    if is_main_process():
        logger.info("Starting training …")
    trainer.train()

    # Only rank-0 saves
    if is_main_process():
        model.save_pretrained(cfg.output_dir)
        processor.save_pretrained(cfg.output_dir)
        logger.info(f"Model saved → {cfg.output_dir}")


# ─────────────────────────────────────────────
# 10. Test  (evaluate on held-out test split)
# ─────────────────────────────────────────────
def test(cfg: ScriptConfig):
    if is_main_process():
        logger.info(f"Loading model for testing from {cfg.output_dir} …")

    processor  = AutoProcessor.from_pretrained(cfg.output_dir)
    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",   # avoids flash_attn / libcudart.so.12 issues
    )
    model = PeftModel.from_pretrained(base_model, cfg.output_dir)
    model.eval()

    system_prompt = build_system_prompt(cfg.target_language_name)
    preprocess_fn = make_preprocess_fn(processor, system_prompt)

    test_raw = load_test_dataset(cfg)
    raw_cols = ["sample_id", "audio_path", "source_text", "target_text"]

    if is_main_process():
        logger.info("Preprocessing test split …")
    test_ds = test_raw.map(preprocess_fn, remove_columns=raw_cols, num_proc=1)

    collator = SpeechTranslationCollator(processor=processor)
    test_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=256,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        report_to="none",
        remove_unused_columns=False,
        local_rank=cfg.local_rank,
    )

    trainer = Trainer(
        model=model,
        args=test_args,
        data_collator=collator,
        compute_metrics=build_compute_metrics(processor),
    )

    if is_main_process():
        logger.info("Running evaluation on test split …")
    results = trainer.evaluate(eval_dataset=test_ds)

    if is_main_process():
        logger.info("──────────────── Test Results ────────────────")
        for k, v in results.items():
            logger.info(f"  {k}: {v}")
        logger.info("──────────────────────────────────────────────")

    return results


# ─────────────────────────────────────────────
# 11. Infer  (single audio file)
# ─────────────────────────────────────────────
def infer(cfg: ScriptConfig):
    if not cfg.audio_file or not os.path.exists(cfg.audio_file):
        raise FileNotFoundError(f"Audio file not found: {cfg.audio_file}")

    logger.info(f"Loading model from {cfg.output_dir} …")
    processor  = AutoProcessor.from_pretrained(cfg.output_dir)
    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",   # avoids flash_attn / libcudart.so.12 issues
    )
    model = PeftModel.from_pretrained(base_model, cfg.output_dir)
    model.eval()

    audio         = load_audio_array(cfg.audio_file)
    system_prompt = build_system_prompt(cfg.target_language_name)

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "audio", "audio_url": "placeholder"}]},
    ]
    text   = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audios=[audio], sampling_rate=16_000, return_tensors="pt").to(model.device)

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=256, num_beams=4, early_stopping=True)

    translation = processor.decode(gen_ids[0], skip_special_tokens=True)
    print("\n" + "=" * 60)
    print(f"Audio        : {cfg.audio_file}")
    print(f"Translation  ({cfg.target_language_name}): {translation}")
    print("=" * 60 + "\n")
    return translation


# ─────────────────────────────────────────────
# 12. CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2-Audio for speech-to-text translation "
                    "with distributed training support."
    )
    parser.add_argument(
        "--mode", choices=["train", "test", "infer"], default="train",
        help="train | test (evaluate on test split) | infer (single audio file)",
    )

    # ── Model ──────────────────────────────────
    parser.add_argument("--model_id",   default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--output_dir", default="./qwen2-audio-translation-finetuned")

    # ── Dataset ────────────────────────────────
    parser.add_argument("--train_json", default="/raid/chandresh/Nivedita/STDATA/Data/Hindi/en-hi/seg_data/train/txt/manifest.json")
    parser.add_argument("--eval_json",  default="/raid/chandresh/Nivedita/STDATA/Data/Hindi/en-hi/seg_data/dev/txt/manifest.json")
    parser.add_argument("--test_json",  default="/raid/chandresh/Nivedita/STDATA/Data/Hindi/en-hi/seg_data/test/txt/manifest.json")
    parser.add_argument("--audio_root", default="",
                        help="Rebase audio_local_path onto this directory")
    parser.add_argument("--target_language_name", default="Hindi")

    # ── Fraction sampling ──────────────────────
    parser.add_argument("--train_frac", type=float, default=1.0,
                        help="Fraction of train data to use, e.g. 0.1 = 10%%")
    parser.add_argument("--eval_frac",  type=float, default=1.0,
                        help="Fraction of validation data to use")
    parser.add_argument("--test_frac",  type=float, default=1.0,
                        help="Fraction of test data to use")

    # ── LoRA ───────────────────────────────────
    parser.add_argument("--use_qlora",  action="store_true", default=True)
    parser.add_argument("--lora_r",     type=int,   default=16)
    parser.add_argument("--lora_alpha", type=int,   default=32)

    # ── Training ───────────────────────────────
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=2)
    parser.add_argument("--grad_accum", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--fp16",       action="store_true", default=True)
    parser.add_argument("--bf16",       action="store_true", default=False)
    parser.add_argument("--seed",       type=int,   default=42)

    # ── Distributed (set automatically by torchrun, but can override) ──
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Set automatically by torchrun; do not set manually")

    # ── Inference ──────────────────────────────
    parser.add_argument("--audio_file", default="",
                        help="(infer mode) path to .wav/.flac to translate")

    args = parser.parse_args()

    # Also honour the LOCAL_RANK env var set by torchrun
    if args.local_rank == -1:
        args.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

    cfg = ScriptConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        train_json=args.train_json,
        eval_json=args.eval_json,
        test_json=args.test_json,
        audio_root=args.audio_root,
        target_language_name=args.target_language_name,
        train_frac=args.train_frac,
        eval_frac=args.eval_frac,
        test_frac=args.test_frac,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        local_rank=args.local_rank,
        audio_file=args.audio_file,
    )
    return cfg, args.mode


# ─────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    cfg, mode = parse_args()

    if mode == "train":
        train(cfg)
    elif mode == "test":
        test(cfg)
    elif mode == "infer":
        infer(cfg)