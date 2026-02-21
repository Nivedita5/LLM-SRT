import torch
import yaml
import argparse
import torchaudio
import os
from transformers import AutoProcessor, SeamlessM4Tv2Model
from bert_score import score as bert_score
from sacrebleu.metrics import BLEU, CHRF
from comet import download_model, load_from_checkpoint

# Optimize for A100 Tensor Cores
torch.set_float32_matmul_precision('high')

def calculate_metrics(pred, ref, lang, device, comet_model):
    """Computes metrics with safety handling for COMET version bugs."""
    # BLEU & ChrF++
    bleu = BLEU().sentence_score(pred, [ref]).score
    chrf = CHRF(word_order=2).sentence_score(pred, [ref]).score

    # BERTScore
    _, _, F1 = bert_score([pred], [ref], lang=lang, device=device, verbose=False)
    
    # COMET with safety wrapper
    comet_score = 0.0
    try:
        data = [{"src": ref, "mt": pred, "ref": ref}]
        # Using gpus=0 (CPU) for COMET can sometimes avoid Lightning/Multi-GPU conflicts
        # during inference loops, but set to 1 if you've updated COMET.
        comet_output = comet_model.predict(data, batch_size=1, gpus=0, progress_bar=False)
        comet_score = comet_output.scores[0]
    except ValueError as e:
        if "unpack" in str(e):
            print("⚠️ COMET unpacking error: library mismatch. Skipping COMET for this line.")
        else:
            print(f"⚠️ COMET ValueError: {e}")
    except Exception as e:
        print(f"⚠️ COMET failed: {e}")
    
    return {
        "BLEU": bleu, "ChrF++": chrf, 
        "BERTScore": F1.item(), "COMET": comet_score
    }

def main():
    parser = argparse.ArgumentParser(description="Seamless-M4T v2 Evaluation")
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--ref_file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tgt", default="hin")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading models onto {device}...")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to(device)
    
    # COMET setup
    comet_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_path)

    with open(args.yaml, 'r') as f:
        meta_entries = yaml.safe_load(f)
    with open(args.ref_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f.readlines()]

    with open(args.out, 'w', encoding='utf-8') as report:
        for i, (meta, ref_text) in enumerate(zip(meta_entries, references)):
            # Robust key checking for wav filename
            wav_id = meta.get('wav') or meta.get('wav_name') or meta.get('id')
            if not wav_id:
                continue

            wav_filename = f"{wav_id}.wav" if not str(wav_id).endswith(".wav") else wav_id
            wav_path = os.path.join(args.wav_dir, wav_filename)

            if not os.path.exists(wav_path):
                print(f"File not found: {wav_path}")
                continue

            # Audio processing
            audio, orig_freq = torchaudio.load(wav_path)
            if orig_freq != 16000:
                audio = torchaudio.functional.resample(audio, orig_freq, 16000)
            
            start_f = int(meta.get('offset', 0) * 16000)
            dur = meta.get('duration')
            audio = audio[:, start_f : start_f + int(dur * 16000)] if dur else audio[:, start_f:]

            # Inference
            inputs = processor(audio=audio, sampling_rate=16000, return_tensors="pt").to(device)
            output_tokens = model.generate(**inputs, tgt_lang=args.tgt, generate_speech=False)
            prediction = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

            # Metrics
            scores = calculate_metrics(prediction, ref_text, args.tgt, device, comet_model)

            # Log results
            print(f"[{i+1}/{len(meta_entries)}] {wav_filename} -> BLEU: {scores['BLEU']:.2f}")
            report.write(f"{wav_filename}|{prediction}|{scores['BLEU']:.2f}|{scores['COMET']:.4f}\n")
            report.flush() # Ensure it writes to disk immediately

    print(f"✅ Completed. Results in {args.out}")

if __name__ == "__main__":
    main()