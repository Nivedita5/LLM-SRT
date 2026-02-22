import torch
import yaml
import argparse
import torchaudio
import os
import gc
import math
from transformers import AutoProcessor, SeamlessM4TModel

# Global Memory Optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision('high')

def main():
    parser = argparse.ArgumentParser(description="Seamless-M4T Medium with Skip Logic")
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tgt", default="hin")
    parser.add_argument("--fraction", type=float, default=1.0)
    # Set the max duration you want to allow (in seconds)
    parser.add_argument("--max_dur", type=float, default=100.0, help="Skip files longer than this")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "facebook/hf-seamless-m4t-medium"

    print(f"Loading Model: {model_id}")
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = SeamlessM4TModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            use_safetensors=True,
            low_cpu_mem_usage=True
        ).to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model. Ensure you have disk space and internet. Error: {e}")
        return

    with open(args.yaml, 'r') as f:
        meta_entries = yaml.safe_load(f)
    
    subset_size = math.ceil(len(meta_entries) * args.fraction)
    meta_subset = meta_entries[:subset_size]

    print(f"Processing {subset_size} samples. Skipping files > {args.max_dur}s.")

    with open(args.out, 'w', encoding='utf-8') as f_out:
        for i, meta in enumerate(meta_subset):
            wav_id = meta.get('wav') or meta.get('id')
            wav_filename = f"{wav_id}.wav" if not str(wav_id).endswith(".wav") else wav_id
            wav_path = os.path.join(args.wav_dir, wav_filename)

            if not os.path.exists(wav_path):
                f_out.write(f"{wav_id}|SKIPPED_NOT_FOUND\n")
                continue

            try:
                # 1. Check metadata duration BEFORE loading to GPU
                actual_dur = meta.get('duration', 0)
                if actual_dur > args.max_dur:
                    print(f"⏩ Skipping {wav_id}: Duration {actual_dur:.2f}s is too long.")
                    f_out.write(f"{wav_id}|SKIPPED_TOO_LONG\n")
                    continue

                # 2. Load and Resample
                audio, orig_freq = torchaudio.load(wav_path)
                if orig_freq != 16000:
                    audio = torchaudio.functional.resample(audio, orig_freq, 16000)
                
                start_f = int(meta.get('offset', 0) * 16000)
                audio = audio[:, start_f : start_f + int(actual_dur * 16000)] if actual_dur > 0 else audio[:, start_f:]

                # 3. Final safety check on tensor size
                if audio.shape[1] > (args.max_dur * 16000):
                    print(f"Skipping {wav_id}: Final audio tensor exceeds limits.")
                    f_out.write(f"{wav_id}|SKIPPED_LIMIT_EXCEEDED\n")
                    continue

                # 4. Inference
                inputs = processor(audio=audio, sampling_rate=16000, return_tensors="pt").to(device)
                if "input_features" in inputs:
                    inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)

                with torch.inference_mode():
                    output_tokens = model.generate(
                        **inputs, 
                        tgt_lang=args.tgt, 
                        generate_speech=False,
                        num_beams=1
                    )
                
                prediction = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
                f_out.write(f"{wav_id}|{prediction}\n")

            except torch.cuda.OutOfMemoryError:
                print(f"💥 OOM on {wav_id} despite checks. Clearing VRAM.")
                f_out.write(f"{wav_id}|OOM_SKIPPED\n")
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"⚠️ Error processing {wav_id}: {e}")
                f_out.write(f"{wav_id}|ERROR\n")

            # Clean up every iteration
            if (i + 1) % 5 == 0:
                f_out.flush()
                torch.cuda.empty_cache()
                print(f"Progress: {i+1}/{subset_size}")

    print(f"🏁 Done! Results saved to {args.out}")

if __name__ == "__main__":
    main()