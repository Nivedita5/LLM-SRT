import os
import yaml
import argparse
from tqdm import tqdm

def restructure_to_long_form(root_path, split, src_lang, tgt_lang, target_duration=60.0):
    txt_dir = os.path.join(root_path, split, "txt")
    
    # Define paths
    src_path = os.path.join(txt_dir, f"{split}.{src_lang}")
    tgt_path = os.path.join(txt_dir, f"{split}.{tgt_lang}")
    yaml_path = os.path.join(txt_dir, f"{split}.yaml")
    
    # Output paths
    out_src_path = os.path.join(txt_dir, f"{split}_long_30min.{src_lang}")
    out_tgt_path = os.path.join(txt_dir, f"{split}_long_30min.{tgt_lang}")
    out_yaml_path = os.path.join(txt_dir, f"{split}_long_0min.yaml")

    # Load data
    with open(src_path, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f]
    with open(tgt_path, 'r', encoding='utf-8') as f:
        tgt_lines = [line.strip() for line in f]
    with open(yaml_path, 'r', encoding='utf-8') as f:
        meta = yaml.safe_load(f)

    long_src, long_tgt, long_meta = [], [], []
    
    curr_src, curr_tgt = [], []
    curr_start_offset = None
    curr_duration = 0.0
    curr_wav = None
    curr_spk = None

    for i in tqdm(range(len(meta)), desc="Merging"):
        m = meta[i]
        
        # Flush if WAV changes OR adding next segment exceeds target_duration
        if (curr_wav is not None and m['wav'] != curr_wav) or (curr_duration >= target_duration):
            long_src.append(" ".join(curr_src))
            long_tgt.append(" ".join(curr_tgt))
            long_meta.append({
                "duration": curr_duration,
                "offset": curr_start_offset,
                "speaker_id": curr_spk,
                "wav": curr_wav
            })
            curr_src, curr_tgt, curr_duration = [], [], 0.0

        if not curr_src:
            curr_start_offset = m['offset']
            curr_wav = m['wav']
            curr_spk = m.get('speaker_id', 'unknown')
            
        curr_src.append(src_lines[i])
        curr_tgt.append(tgt_lines[i])
        curr_duration += m['duration']

    # Final flush
    if curr_src:
        long_src.append(" ".join(curr_src))
        long_tgt.append(" ".join(curr_tgt))
        long_meta.append({
            "duration": curr_duration,
            "offset": curr_start_offset,
            "speaker_id": curr_spk,
            "wav": curr_wav
        })

    # Save text files
    with open(out_src_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(long_src) + "\n")
    with open(out_tgt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(long_tgt) + "\n")
        
    # Save YAML with the custom "Mixed" style: List is block, Items are flow
    with open(out_yaml_path, 'w', encoding='utf-8') as f:
        for item in long_meta:
            # item_str will look like "{duration: ..., offset: ...}"
            item_str = yaml.dump(item, default_flow_style=True, sort_keys=True).strip()
            f.write(f"- {item_str}\n")

    print(f"Done! Restructured to {len(long_src)} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--src_lang", type=str, default="en")
    parser.add_argument("--tgt_lang", type=str, default="hi")
    parser.add_argument("--duration", type=float, default=1800.0)
    
    args = parser.parse_args()
    restructure_to_long_form(args.root, args.split, args.src_lang, args.tgt_lang, args.duration)