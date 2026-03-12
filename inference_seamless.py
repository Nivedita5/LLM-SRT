"""
Multi-GPU Inference script for fine-tuned SeamlessM4T S2TT model.

Strategy: each GPU process gets a shard of the test set, runs independently,
then rank 0 gathers all results, merges them, and computes BLEU/WER.
No gradient sync needed — pure data parallelism.

Usage
-----
# Single GPU (plain python)
python inference_seamless_m4t.py \
    --model_dir seamless_m4t_finetuned \
    --test_manifest /path/to/test/manifest.json \
    --source_lang eng --target_lang hin \
    --batch_size 4 --output_file results.tsv

# Multi-GPU (torchrun) — automatically shards test set across GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    inference_seamless_m4t.py \
    --model_dir seamless_m4t_finetuned \
    --test_manifest /path/to/test/manifest.json \
    --source_lang eng --target_lang hin \
    --batch_size 4 --output_file results.tsv

# Single audio file (always single GPU)
python inference_seamless_m4t.py \
    --model_dir seamless_m4t_finetuned \
    --audio_file /path/to/audio.wav \
    --source_lang eng --target_lang hin
"""

import os
import json
import logging
import argparse
import pickle
from typing import List, Dict, Optional

import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
from peft import PeftModel
import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Distributed helpers
# ---------------------------------------------------------------------------

def init_distributed():
    """Initialize torch.distributed if launched via torchrun, else no-op."""
    if "RANK" not in os.environ:
        return False
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if torch.distributed.get_rank() != 0:
        logging.getLogger().setLevel(logging.WARNING)
    return True


def get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def shard_samples(samples: List[Dict], rank: int, world_size: int) -> List[Dict]:
    """Interleaved sharding: rank 0 gets [0, ws, 2ws, ...], rank 1 gets [1, ws+1, ...]"""
    return samples[rank::world_size]


def gather_results(local_results: List[Dict]) -> List[Dict]:
    """Gather results from all ranks onto rank 0."""
    if get_world_size() == 1:
        return local_results

    data = pickle.dumps(local_results)
    data_tensor = torch.ByteTensor(list(data)).cuda()
    size_tensor = torch.LongTensor([data_tensor.numel()]).cuda()

    # Share sizes
    size_list = [torch.LongTensor([0]).cuda() for _ in range(get_world_size())]
    torch.distributed.all_gather(size_list, size_tensor)
    max_size = max(s.item() for s in size_list)

    # Pad and gather
    padded = torch.zeros(max_size, dtype=torch.uint8).cuda()
    padded[:data_tensor.numel()] = data_tensor

    if is_main_process():
        tensor_list = [torch.zeros(max_size, dtype=torch.uint8).cuda()
                       for _ in range(get_world_size())]
        torch.distributed.gather(padded, gather_list=tensor_list, dst=0)
        all_results = []
        for tensor, size in zip(tensor_list, size_list):
            all_results.extend(pickle.loads(bytes(tensor[:size.item()].cpu().tolist())))
        return all_results
    else:
        torch.distributed.gather(padded, dst=0)
        return []


# ---------------------------------------------------------------------------
# 2. Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-GPU inference for fine-tuned SeamlessM4T S2TT model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Model ----
    parser.add_argument("--model_dir", required=True,
                        help="Path to finetuned model directory (contains adapter_config.json).")
    parser.add_argument("--base_model", default=None,
                        help="Base HuggingFace model name or path. "
                             "Auto-detected from adapter_config.json if not set.")
    parser.add_argument("--no_lora", action="store_true", default=False,
                        help="Skip LoRA loading (use for fully merged or base models).")
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                        help="Load model in 8-bit quantization (saves ~50%% GPU memory).")

    # ---- Languages ----
    parser.add_argument("--source_lang", default="eng",
                        help="Source language BCP-47 code, e.g. 'eng'.")
    parser.add_argument("--target_lang", default="hin",
                        help="Target language BCP-47 code, e.g. 'hin'.")

    # ---- Input ----
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio_file",
                       help="Single audio file — always runs on one GPU.")
    group.add_argument("--test_manifest",
                       help="Manifest JSON for batch inference across multiple GPUs.")

    # ---- Data filtering ----
    parser.add_argument("--audio_dir", default="",
                        help="Base directory prepended to relative audio paths.")
    parser.add_argument("--max_duration", type=float, default=30.0,
                        help="Skip audio clips longer than this many seconds.")
    parser.add_argument("--min_duration", type=float, default=0.1,
                        help="Skip audio clips shorter than this many seconds.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap total samples (applied before sharding).")

    # ---- Generation ----
    parser.add_argument("--num_beams", type=int, default=5,
                        help="Beam search width.")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum output tokens per sample.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Samples per batch per GPU.")

    # ---- Output ----
    parser.add_argument("--output_file", default="predictions.tsv",
                        help="TSV output (written by rank 0 only). "
                             "Columns: audio_path, reference, prediction.")
    parser.add_argument("--no_metrics", action="store_true", default=False,
                        help="Skip BLEU/WER computation.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 3. Model loading
# ---------------------------------------------------------------------------

def load_model_and_processor(args, device: str):
    if is_main_process():
        logger.info(f"Loading processor from: {args.model_dir}")
    processor = AutoProcessor.from_pretrained(args.model_dir)
    processor.tokenizer.src_lang = args.source_lang
    processor.tokenizer.tgt_lang = args.target_lang

    base_model_name = args.base_model
    if base_model_name is None and not args.no_lora:
        adapter_cfg_path = os.path.join(args.model_dir, "adapter_config.json")
        if os.path.exists(adapter_cfg_path):
            with open(adapter_cfg_path) as f:
                base_model_name = json.load(f).get("base_model_name_or_path")
            if is_main_process():
                logger.info(f"Base model from adapter_config.json: {base_model_name}")
        else:
            base_model_name = args.model_dir
    if base_model_name is None:
        base_model_name = args.model_dir

    load_kwargs: Dict = {"device_map": {"": device}}
    if args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        if is_main_process():
            logger.info(f"Loading in 8-bit on {device}")

    if is_main_process():
        logger.info(f"Loading base model: {base_model_name} → {device}")

    base_model = SeamlessM4Tv2ForSpeechToText.from_pretrained(base_model_name, **load_kwargs)

    if args.no_lora:
        model = base_model
    else:
        if is_main_process():
            logger.info(f"Loading LoRA adapters from: {args.model_dir}")
        model = PeftModel.from_pretrained(base_model, args.model_dir)
        model = model.merge_and_unload()
        if is_main_process():
            logger.info("LoRA adapters merged for faster inference.")

    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# 4. Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(
    manifest_path: str,
    audio_dir: str = "",
    max_duration: float = 30.0,
    min_duration: float = 0.1,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        raise ValueError(f"{manifest_path} must be a JSON array.")

    samples, skipped = [], 0
    for i, entry in enumerate(entries):
        if max_samples and len(samples) >= max_samples:
            break

        src = entry.get("source", {})
        tgt = entry.get("target", {})
        raw_path = src.get("audio_local_path")
        if not raw_path:
            skipped += 1
            continue

        audio_path = (
            os.path.join(audio_dir, raw_path)
            if audio_dir and not os.path.isabs(raw_path)
            else raw_path
        )
        if not os.path.exists(audio_path):
            skipped += 1
            continue

        try:
            info = sf.info(audio_path)
            duration = info.frames / info.samplerate
            if duration < min_duration or duration > max_duration:
                skipped += 1
                continue
        except Exception:
            pass

        samples.append({
            "audio_path": audio_path,
            "reference":  tgt.get("text", "").strip(),
            "index":      i,
        })

    if is_main_process():
        logger.info(f"Loaded {len(samples)} samples ({skipped} skipped) from {manifest_path}")
    return samples


# ---------------------------------------------------------------------------
# 5. Audio + inference helpers
# ---------------------------------------------------------------------------

def load_audio(audio_path: str, target_sr: int = 16_000) -> np.ndarray:
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def translate_batch(audio_arrays, model, processor, target_lang, num_beams,
                    max_new_tokens, device) -> List[str]:
    inputs = processor(
        audios=audio_arrays,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True,
    ).to(device)
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            tgt_lang=target_lang,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
    return processor.batch_decode(output_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# 6. Single file mode
# ---------------------------------------------------------------------------

def run_single_file(args, model, processor, device: str):
    logger.info(f"Translating: {args.audio_file}")
    audio = load_audio(args.audio_file)
    preds = translate_batch([audio], model, processor, args.target_lang,
                            args.num_beams, args.max_new_tokens, device)
    print("\n" + "=" * 60)
    print(f"  Audio      : {args.audio_file}")
    print(f"  Translation: {preds[0]}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# 7. Manifest / multi-GPU batch mode
# ---------------------------------------------------------------------------

def run_manifest(args, model, processor, device: str):
    rank       = get_rank()
    world_size = get_world_size()

    all_samples = load_manifest(
        manifest_path=args.test_manifest,
        audio_dir=args.audio_dir,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        max_samples=args.max_samples,
    )
    if not all_samples:
        if is_main_process():
            logger.error("No valid samples found. Exiting.")
        return

    # Shard samples across GPUs
    my_samples = shard_samples(all_samples, rank, world_size)

    if is_main_process():
        logger.info(
            f"Sharding {len(all_samples)} samples across {world_size} GPU(s) "
            f"→ ~{len(my_samples)} per GPU"
        )

    # Run inference on this rank's shard
    local_results: List[Dict] = []
    for batch_start in tqdm(
        range(0, len(my_samples), args.batch_size),
        desc=f"GPU:{rank}",
        disable=not is_main_process(),
    ):
        batch = my_samples[batch_start : batch_start + args.batch_size]
        audio_arrays, valid_batch = [], []
        for s in batch:
            try:
                audio_arrays.append(load_audio(s["audio_path"]))
                valid_batch.append(s)
            except Exception as e:
                logger.warning(f"[GPU {rank}] Load failed {s['audio_path']}: {e}")

        if not audio_arrays:
            continue

        try:
            preds = translate_batch(audio_arrays, model, processor, args.target_lang,
                                    args.num_beams, args.max_new_tokens, device)
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"[GPU {rank}] OOM at batch {batch_start} — skipping. "
                           "Try reducing --batch_size.")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            logger.warning(f"[GPU {rank}] Error at batch {batch_start}: {e} — skipping.")
            continue

        for s, pred in zip(valid_batch, preds):
            local_results.append({
                "index":      s["index"],
                "audio_path": s["audio_path"],
                "reference":  s["reference"],
                "prediction": pred,
            })

    # Sync all ranks, then gather
    if world_size > 1:
        torch.distributed.barrier()

    all_results = gather_results(local_results)

    if not is_main_process():
        return

    # Restore original manifest order
    all_results.sort(key=lambda x: x["index"])
    logger.info(f"Total predictions: {len(all_results)}")

    # ---- Save TSV ----
    logger.info(f"Saving to: {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("audio_path\treference\tprediction\n")
        for r in all_results:
            f.write(
                f"{r['audio_path']}\t"
                f"{r['reference'].replace(chr(9), ' ')}\t"
                f"{r['prediction'].replace(chr(9), ' ')}\n"
            )

    # ---- Metrics ----
    if not args.no_metrics:
        pairs = [(r["prediction"], r["reference"])
                 for r in all_results if r["reference"].strip()]
        if pairs:
            preds_m, refs_m = zip(*pairs)
            bleu = evaluate.load("sacrebleu").compute(
                predictions=list(preds_m), references=[[r] for r in refs_m]
            )
            wer = evaluate.load("wer").compute(
                predictions=list(preds_m), references=list(refs_m)
            )
            print("\n" + "=" * 60)
            print(f"  GPUs used         : {world_size}")
            print(f"  Samples evaluated : {len(preds_m)}")
            print(f"  BLEU              : {bleu['score']:.2f}")
            print(f"  WER               : {wer:.4f}")
            print("=" * 60 + "\n")

            metrics_path = args.output_file.replace(".tsv", "_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump({"bleu": bleu["score"], "wer": wer,
                           "num_samples": len(preds_m), "num_gpus": world_size}, f, indent=2)
            logger.info(f"Metrics saved to: {metrics_path}")
        else:
            logger.warning("No references found — skipping metrics.")


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main():
    is_dist = init_distributed()
    args    = parse_args()
    rank    = get_rank()
    device  = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    if is_main_process():
        logger.info(
            f"Inference | GPUs={get_world_size()} | distributed={is_dist} | device={device}"
        )

    model, processor = load_model_and_processor(args, device)

    if args.audio_file:
        if is_main_process():
            run_single_file(args, model, processor, device)
    else:
        run_manifest(args, model, processor, device)

    if is_dist:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()