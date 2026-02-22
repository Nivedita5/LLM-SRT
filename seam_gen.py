import torch
import yaml
import argparse
import torchaudio
import os
import gc
import math
from transformers import AutoProcessor, SeamlessM4TModel

# Memory optimizations for A100
torch.set_float32_matmul_precision('high')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    parser = argparse.ArgumentParser(description="Seamless-M4T Medium Generation")
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tgt", default="hin")
    parser.add_argument("--fraction", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "facebook/hf-seamless-m4t-medium"

    print(f"Attempting to load {model_id} using Safetensors...")

    processor = AutoProcessor.from_pretrained(model_id)
    
    # FIX: use_safetensors=True avoids the security vulnerability block
    # FIX: attn_implementation="eager" is required for the Medium model
    model = SeamlessM4TModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        use_safetensors=True,
        low_cpu_mem_usage=True
    ).to(device)

    # ... [Rest of your YAML loading and subset logic] ...
    with open(args.yaml, 'r') as f:
        meta_entries = yaml.safe_load(f)
    subset_size = math.ceil(len(meta_entries) * args.fraction)
    meta_subset = meta_entries[:subset_size]

    model.eval()
    
    with open(args.out, 'w', encoding='utf-8') as f_out:
        for i, meta in enumerate(meta_subset):
            wav_id = meta.get('wav') or meta.get('id')
            wav_path = os.path.join(args.wav_dir, f"{wav_id}.wav" if not str(wav_id).endswith(".wav") else wav_id)

            if not os.path.exists(wav_path):
                continue

            audio, orig_freq = torchaudio.load(wav_path)
            if orig_freq != 16000:
                audio = torchaudio.functional.resample(audio, orig_freq, 16000)
            
            start_f = int(meta.get('offset', 0) * 16000)
            dur = meta.get('duration')
            audio = audio[:, start_f : start_f + int(dur * 16000)] if dur else audio[:, start_f:]

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

            del inputs, output_tokens
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
                f_out.flush()
                print(f"Progress: {i+1}/{subset_size}")

    print(f"✅ Completed. Results in {args.out}")

if __name__ == "__main__":
    main()