import json
import yaml
import argparse

def generate_speech_json(yaml_path, src_txt_path, tgt_txt_path, src_lang, tgt_lang, output_file, common_path):
    # Load audio data (This is now a LIST based on your error)
    with open(yaml_path, 'r') as f:
        audio_entries = yaml.safe_load(f)
    
    # Load text lines
    with open(src_txt_path, 'r', encoding='utf-8') as f:
        src_texts = [line.strip() for line in f]
        
    with open(tgt_txt_path, 'r', encoding='utf-8') as f:
        tgt_texts = [line.strip() for line in f]

    json_output = []

    # Iterate using zip to align the YAML list with the text files
    # This assumes the YAML list order matches the text file line order
    for i, (audio_item, s_txt, t_txt) in enumerate(zip(audio_entries, src_texts, tgt_texts)):
        
        # Adjust 'wav' or 'audio' key based on your actual YAML structure
        # If your YAML has separate source/target paths per item:
        s_aud = audio_item.get('source_wav') or audio_item.get('wav') 
        t_aud = audio_item.get('target_wav') or audio_item.get('wav')

        s_aud = common_path + s_aud
        entry = {
            "source": {
                "id": i,
                "lang": src_lang,
                "text": s_txt,
                "audio_local_path": s_aud,
                "waveform": None,
                "sampling_rate": 16000,
                "units": None
            },
            "target": {
                "id": i,
                "lang": tgt_lang,
                "text": t_txt,
                "audio_local_path": None,
                "waveform": None,
                "sampling_rate": 16000,
                "units": None
            }
        }
        json_output.append(entry)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=4, ensure_ascii=False)

    print(f"Successfully generated {output_file} with {len(json_output)} entries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--src_txt", required=True)
    parser.add_argument("--tgt_txt", required=True)
    parser.add_argument("--src_lang", default="eng")
    parser.add_argument("--tgt_lang", default="urd")
    parser.add_argument("--common_path", required=True)
    parser.add_argument("--out", default="manifest.json")

    args = parser.parse_args()
    generate_speech_json(args.yaml, args.src_txt, args.tgt_txt, args.src_lang, args.tgt_lang, args.out, args.common_path)