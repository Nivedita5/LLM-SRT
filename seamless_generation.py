import torch
import yaml
import argparse
import torchaudio
import os
import gc
import math
from transformers import AutoProcessor, SeamlessM4Tv2Model

# Optimize for A100 Tensor Cores
torch.set_float32_matmul_precision('high')

def main():
    parser = argparse.ArgumentParser(description="Seamless-M4T v2 Translation Generation")
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tgt", default="hin")
    # Added argument for data fraction
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of data to process (0.0 to 1.0)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading SeamlessM4T v2 onto {device}...")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to(device)

    with open(args.yaml, 'r') as f:
        meta_entries = yaml.safe_load(f)

    # Calculate the slice index
    total_count = len(meta_entries)
    subset_size = math.ceil(total_count * args.fraction)
    meta_subset = meta_entries[:subset_size] # Slicing the first N entries

    print(f"Total entries: {total_count}")
    print(f"Processing first {args.fraction*100}%: {subset_size} samples")

    model.eval()
    
    with open(args.out, 'w', encoding='utf-8') as f_out:
        for i, meta in enumerate(meta_subset):
            wav_id = meta.get('wav') or meta.get('wav_name') or meta.get('id')
            if not wav_id: continue

            wav_filename = f"{wav_id}.wav" if not str(wav_id).endswith(".wav") else wav_id
            wav_path = os.path.join(args.wav_dir, wav_filename)

            if not os.path.exists(wav_path):
                print(f"❌ Missing: {wav_path}")
                continue

            # Load and slice audio
            audio, orig_freq = torchaudio.load(wav_path)
            if orig_freq != 16000:
                audio = torchaudio.functional.resample(audio, orig_freq, 16000)
            
            start_frame = int(meta.get('offset', 0) * 16000)
            dur = meta.get('duration')
            audio = audio[:, start_frame : start_frame + int(dur * 16000)] if dur else audio[:, start_frame:]

            # Generation
            inputs = processor(audio=audio, sampling_rate=16000, return_tensors="pt").to(device)
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    output_tokens = model.generate(**inputs, tgt_lang=args.tgt, generate_speech=False)
        # output_tokens = model.generate(
        #     **inputs,
        #     tgt_lang=args.tgt,
        #     generate_speech=False,
        #     num_beams=1
        # )
        #     with torch.no_grad():
                # output_tokens = model.generate(**inputs, tgt_lang=args.tgt, generate_speech=False)
            
            prediction = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

            # Write to file
            f_out.write(f"{wav_filename}|{prediction}\n")

            del inputs
            del output_tokens
            torch.cuda.empty_cache()
            gc.collect()

            # if (i + 1) % 5 == 0:
            #     print(f"Processed {i+1}/{len(meta_subset)}...")
            
            if (i + 1) % 5 == 0:
                print(f"Progress: {i+1}/{subset_size}")
                f_out.flush()

    print(f"✅ Finished generating {subset_size} samples to {args.out}")

if __name__ == "__main__":
    main()