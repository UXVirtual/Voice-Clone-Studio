import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from pathlib import Path

SAMPLES_DIR = Path(__file__).parent / "samples"

def get_available_samples():
    """Find all .wav files in samples folder that have matching .txt files."""
    if not SAMPLES_DIR.exists():
        print(f"Error: samples folder not found at {SAMPLES_DIR}")
        print("Please create a 'samples' folder with .wav files and matching .txt files.")
        return []

    samples = []
    for wav_file in SAMPLES_DIR.glob("*.wav"):
        txt_file = wav_file.with_suffix(".txt")
        if txt_file.exists():
            samples.append((wav_file, txt_file))
    return samples

def select_sample(samples):
    """Display available samples and let user select one."""
    print("\nAvailable voice samples:")
    print("-" * 40)
    for i, (wav_file, txt_file) in enumerate(samples, 1):
        ref_text = txt_file.read_text(encoding="utf-8").strip()
        preview = ref_text[:50] + "..." if len(ref_text) > 50 else ref_text
        print(f"  {i}. {wav_file.stem}")
        print(f"     Text: {preview}")
    print("-" * 40)

    while True:
        try:
            choice = int(input("\nSelect a sample (number): "))
            if 1 <= choice <= len(samples):
                return samples[choice - 1]
            print(f"Please enter a number between 1 and {len(samples)}")
        except ValueError:
            print("Please enter a valid number")

def main():
    # Find available samples
    samples = get_available_samples()
    if not samples:
        print("\nNo valid samples found. Each sample needs:")
        print("  - A .wav file (the voice reference)")
        print("  - A .txt file with the same name (transcript of the audio)")
        return

    # Let user select a sample
    wav_file, txt_file = select_sample(samples)
    ref_audio = str(wav_file)
    ref_text = txt_file.read_text(encoding="utf-8").strip()

    print(f"\nSelected: {wav_file.stem}")
    print(f"Reference text: {ref_text}")

    # Get text to generate
    print("\nEnter the text you want to generate (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    text_to_generate = " ".join(lines)
    if not text_to_generate.strip():
        print("No text entered. Exiting.")
        return

    print("\nLoading model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    print("Generating audio...")
    wavs, sr = model.generate_voice_clone(
        text=text_to_generate,
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
    )

    output_file = "output_voice_clone.wav"
    sf.write(output_file, wavs[0], sr)
    print(f"\nSaved to: {output_file}")

if __name__ == "__main__":
    main()