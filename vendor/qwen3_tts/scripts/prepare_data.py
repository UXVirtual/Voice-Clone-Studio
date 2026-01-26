# coding=utf-8
import argparse
import json
import torch
from qwen_tts import Qwen3TTSTokenizer

BATCH_INFER_NUM = 32

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    args = parser.parse_args()

    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = [json.loads(line.strip()) for line in lines if line.strip()]

    final_lines = []
    batch_lines = []
    batch_audios = []
    
    for line in total_lines:
        batch_lines.append(line)
        batch_audios.append(line['audio'])

        if len(batch_lines) >= BATCH_INFER_NUM:
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, bline in zip(enc_res.audio_codes, batch_lines):
                bline['audio_codes'] = code.cpu().tolist()
                final_lines.append(bline)
            batch_lines.clear()
            batch_audios.clear()

    if len(batch_audios) > 0:
        enc_res = tokenizer_12hz.encode(batch_audios)
        for code, bline in zip(enc_res.audio_codes, batch_lines):
            bline['audio_codes'] = code.cpu().tolist()
            final_lines.append(bline)
        batch_lines.clear()
        batch_audios.clear()

    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for line in final_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
