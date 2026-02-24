import os
import yaml
import torchaudio
import argparse
from tqdm import tqdm

def segment_dataset(root, split, src_lang, tgt_lang, output_dir):
    """
    Segments long audio files into individual clips and updates YAML/Text metadata.
    """
    # 1. Setup Input Paths
    wav_dir = os.path.join(root, split, "wav")
    txt_dir = os.path.join(root, split, "txt")
    
    src_input = os.path.join(txt_dir, f"{split}.{src_lang}")
    tgt_input = os.path.join(txt_dir, f"{split}.{tgt_lang}")
    yaml_input = os.path.join(txt_dir, f"{split}.yaml")

    # 2. Setup Output Paths
    out_wav_dir = os.path.join(output_dir, split, "wav")
    out_txt_dir = os.path.join(output_dir, split, "txt")
    os.makedirs(out_wav_dir, exist_ok=True)
    os.makedirs(out_txt_dir, exist_ok=True)

    # 3. Load All Data
    print(f"--- Loading {split} split ---")
    with open(yaml_input, 'r') as f:
        original_meta = yaml.safe_load(f)
    
    with open(src_input, 'r', encoding='utf-8') as f:
        src_lines = f.readlines()
    
    with open(tgt_input, 'r', encoding='utf-8') as f:
        tgt_lines = f.readlines()

    if not (len(original_meta) == len(src_lines) == len(tgt_lines)):
        raise ValueError("Mismatch: YAML entries and Text lines do not match in length!")

    new_meta = []
    
    print(f"Processing {len(original_meta)} segments...")

    # 4. Processing Loop
    for i, entry in enumerate(tqdm(original_meta)):
        original_wav_name = entry["wav"]
        input_audio_path = os.path.join(wav_dir, original_wav_name)
        
        # Load audio (Force 16kHz for SeamlessM4T compatibility)
        waveform, sr = torchaudio.load(input_audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000
        
        # Calculate frame boundaries
        start_frame = int(entry["offset"] * sr)
        end_frame = int((entry["offset"] + entry["duration"]) * sr)
        
        # Slice segment
        segment = waveform[:, start_frame:end_frame]
        
        # Create unique filename: recording_ID_seg.wav
        base_name = os.path.splitext(original_wav_name)[0]
        new_wav_name = f"{base_name}_seg{i:05d}.wav"
        output_audio_path = os.path.join(out_wav_dir, new_wav_name)
        
        # Save segment
        torchaudio.save(output_audio_path, segment, sr)
        
        # Create updated YAML entry
        new_meta.append({
            "wav": new_wav_name,
            "offset": 0.0,
            "duration": float(entry["duration"])
        })

    # 5. Save Updated Metadata and Text Files
    print("Saving updated text and YAML files...")
    
    with open(os.path.join(out_txt_dir, f"{split}.yaml"), 'w') as f:
        yaml.dump(new_meta, f, default_flow_style=False)
        
    with open(os.path.join(out_txt_dir, f"{split}.{src_lang}"), 'w', encoding='utf-8') as f:
        f.writelines(src_lines)
        
    with open(os.path.join(out_txt_dir, f"{split}.{tgt_lang}"), 'w', encoding='utf-8') as f:
        f.writelines(tgt_lines)

    print(f"\nSuccess! Segmented dataset created at: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment SpeechTranslation Dataset into FLEURS-style clips.")
    parser.add_argument("--root", type=str, required=True, help="Path to original dataset root")
    parser.add_argument("--output", type=str, required=True, help="Path to save segmented dataset")
    parser.add_argument("--split", type=str, default="train", help="Split to process (train/dev/test)")
    parser.add_argument("--src_lang", type=str, required=True, help="Source language code (e.g., en)")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language code (e.g., fr)")

    args = parser.parse_args()

    segment_dataset(
        root=args.root,
        output_dir=args.output,
        split=args.split,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang
    )