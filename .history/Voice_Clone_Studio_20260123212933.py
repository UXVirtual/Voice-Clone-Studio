import torch
import soundfile as sf
import gradio as gr
from qwen_tts import Qwen3TTSModel
from pathlib import Path
from datetime import datetime
import numpy as np
import pickle
import hashlib

# Directories
SAMPLES_DIR = Path(__file__).parent / "samples"

OUTPUT_DIR = Path(__file__).parent / "output"
DESIGNS_DIR = Path(__file__).parent / "designs"
TEMP_DIR = Path(__file__).parent / "temp"
OUTPUT_DIR.mkdir(exist_ok=True)
DESIGNS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Clear temp folder on launch
import shutil
for f in TEMP_DIR.iterdir():
    if f.is_file():
        f.unlink()
    elif f.is_dir():
        shutil.rmtree(f)

# Global model cache - now stores (model, size) tuples
_tts_model = None
_tts_model_size = None
_voice_design_model = None
_custom_voice_model = None
_custom_voice_model_size = None
_whisper_model = None
_voice_prompt_cache = {}  # In-memory cache for voice prompts

# Model size options
MODEL_SIZES = ["1.7B", "0.6B"]
MODEL_SIZES_BASE = ["1.7B", "0.6B"]  # Base model has both sizes
MODEL_SIZES_CUSTOM = ["1.7B", "0.6B"]  # CustomVoice has both sizes
MODEL_SIZES_DESIGN = ["1.7B"]  # VoiceDesign only has 1.7B

# Supported languages for TTS
LANGUAGES = [
    "Auto", "English", "Chinese", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian"
]

# Custom Voice speakers with descriptions
CUSTOM_VOICE_SPEAKERS = {
    "Vivian": "Bright, slightly edgy young female voice (Chinese)",
    "Serena": "Warm, gentle young female voice (Chinese)",
    "Uncle_Fu": "Seasoned male voice with low, mellow timbre (Chinese)",
    "Dylan": "Youthful Beijing male voice, clear and natural (Chinese/Beijing)",
    "Eric": "Lively Chengdu male voice, slightly husky brightness (Chinese/Sichuan)",
    "Ryan": "Dynamic male voice with strong rhythmic drive (English)",
    "Aiden": "Sunny American male voice with clear midrange (English)",
    "Ono_Anna": "Playful Japanese female voice, light and nimble (Japanese)",
    "Sohee": "Warm Korean female voice with rich emotion (Korean)"
}

def get_tts_model(size="1.7B"):
    """Lazy-load the TTS Base model for voice cloning."""
    global _tts_model, _tts_model_size

    # If we need a different size, unload current model
    if _tts_model is not None and _tts_model_size != size:
        print(f"Switching Base model from {_tts_model_size} to {size}...")
        del _tts_model
        _tts_model = None
        torch.cuda.empty_cache()

    if _tts_model is None:
        model_name = f"Qwen/Qwen3-TTS-12Hz-{size}-Base"
        print(f"Loading {model_name}...")

        # Try flash_attention_2 first, fall back to sdpa if not available
        try:
            _tts_model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            print(f"TTS Base model ({size}) loaded with Flash Attention 2!")
        except Exception as e:
            if "flash" in str(e).lower():
                print("Flash Attention 2 not available, using SDPA instead...")
                _tts_model = Qwen3TTSModel.from_pretrained(
                    model_name,
                    device_map="cuda:0",
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                )
                print(f"TTS Base model ({size}) loaded with SDPA!")
            else:
                raise e
        _tts_model_size = size
    return _tts_model


def get_voice_design_model():
    """Lazy-load the VoiceDesign model (only 1.7B available)."""
    global _voice_design_model
    if _voice_design_model is None:
        print("Loading Qwen3-TTS VoiceDesign model (1.7B)...")

        # Try flash_attention_2 first, fall back to sdpa if not available
        try:
            _voice_design_model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            print("VoiceDesign model loaded with Flash Attention 2!")
        except Exception as e:
            if "flash" in str(e).lower():
                print("Flash Attention 2 not available, using SDPA instead...")
                _voice_design_model = Qwen3TTSModel.from_pretrained(
                    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                    device_map="cuda:0",
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                )
                print("VoiceDesign model loaded with SDPA!")
            else:
                raise e
    return _voice_design_model


def get_custom_voice_model(size="1.7B"):
    """Lazy-load the CustomVoice model."""
    global _custom_voice_model, _custom_voice_model_size

    # If we need a different size, unload current model
    if _custom_voice_model is not None and _custom_voice_model_size != size:
        print(f"Switching CustomVoice model from {_custom_voice_model_size} to {size}...")
        del _custom_voice_model
        _custom_voice_model = None
        torch.cuda.empty_cache()

    if _custom_voice_model is None:
        model_name = f"Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice"
        print(f"Loading {model_name}...")

        # Try flash_attention_2 first, fall back to sdpa if not available
        try:
            _custom_voice_model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            print(f"CustomVoice model ({size}) loaded with Flash Attention 2!")
        except Exception as e:
            if "flash" in str(e).lower():
                print("Flash Attention 2 not available, using SDPA instead...")
                _custom_voice_model = Qwen3TTSModel.from_pretrained(
                    model_name,
                    device_map="cuda:0",
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                )
                print(f"CustomVoice model ({size}) loaded with SDPA!")
            else:
                raise e
        _custom_voice_model_size = size
    return _custom_voice_model


def get_whisper_model():
    """Lazy-load the Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model...")
        import whisper
        _whisper_model = whisper.load_model("medium")
        print("Whisper model loaded!")
    return _whisper_model


def get_prompt_cache_path(sample_name):
    """Get the path to the cached voice prompt file."""
    return SAMPLES_DIR / f"{sample_name}.prompt"


def compute_sample_hash(wav_path, ref_text):
    """Compute a hash of the sample to detect changes."""
    hasher = hashlib.md5()
    # Hash the audio file
    with open(wav_path, 'rb') as f:
        hasher.update(f.read())
    # Hash the reference text
    hasher.update(ref_text.encode('utf-8'))
    return hasher.hexdigest()


def save_voice_prompt(sample_name, prompt_items, sample_hash):
    """Save the voice clone prompt to disk."""
    cache_path = get_prompt_cache_path(sample_name)
    try:
        # Move tensors to CPU before saving
        # Handle both dict and list formats
        if isinstance(prompt_items, dict):
            cpu_prompt = {}
            for key, value in prompt_items.items():
                if isinstance(value, torch.Tensor):
                    cpu_prompt[key] = value.cpu()
                else:
                    cpu_prompt[key] = value
        elif isinstance(prompt_items, (list, tuple)):
            cpu_prompt = []
            for item in prompt_items:
                if isinstance(item, torch.Tensor):
                    cpu_prompt.append(item.cpu())
                else:
                    cpu_prompt.append(item)
        else:
            # Single tensor or other type
            if isinstance(prompt_items, torch.Tensor):
                cpu_prompt = prompt_items.cpu()
            else:
                cpu_prompt = prompt_items

        cache_data = {
            'prompt': cpu_prompt,
            'hash': sample_hash,
            'version': '1.0'
        }
        torch.save(cache_data, cache_path)
        print(f"Saved voice prompt cache: {cache_path}")
        return True
    except Exception as e:
        print(f"Failed to save voice prompt: {e}")
        return False


def load_voice_prompt(sample_name, expected_hash, device='cuda:0'):
    """Load the voice clone prompt from disk if valid."""
    global _voice_prompt_cache

    # Check in-memory cache first
    if sample_name in _voice_prompt_cache:
        cached = _voice_prompt_cache[sample_name]
        if cached['hash'] == expected_hash:
            print(f"Using in-memory cached prompt for: {sample_name}")
            return cached['prompt']

    # Check disk cache
    cache_path = get_prompt_cache_path(sample_name)
    if not cache_path.exists():
        return None

    try:
        cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)

        # Verify hash matches (sample hasn't changed)
        if cache_data.get('hash') != expected_hash:
            print(f"Sample changed, invalidating cache for: {sample_name}")
            return None

        # Move tensors back to device
        # Handle both dict and list formats
        cached_prompt = cache_data['prompt']
        if isinstance(cached_prompt, dict):
            prompt_items = {}
            for key, value in cached_prompt.items():
                if isinstance(value, torch.Tensor):
                    prompt_items[key] = value.to(device)
                else:
                    prompt_items[key] = value
        elif isinstance(cached_prompt, (list, tuple)):
            prompt_items = []
            for item in cached_prompt:
                if isinstance(item, torch.Tensor):
                    prompt_items.append(item.to(device))
                else:
                    prompt_items.append(item)
        else:
            # Single tensor or other type
            if isinstance(cached_prompt, torch.Tensor):
                prompt_items = cached_prompt.to(device)
            else:
                prompt_items = cached_prompt

        # Store in memory cache
        _voice_prompt_cache[sample_name] = {
            'prompt': prompt_items,
            'hash': expected_hash
        }

        print(f"Loaded voice prompt from cache: {cache_path}")
        return prompt_items

    except Exception as e:
        print(f"Failed to load voice prompt cache: {e}")
        return None


def get_or_create_voice_prompt(model, sample_name, wav_path, ref_text, progress_callback=None):
    """Get cached voice prompt or create new one."""
    # Compute hash to check if sample has changed
    sample_hash = compute_sample_hash(wav_path, ref_text)

    # Try to load from cache
    prompt_items = load_voice_prompt(sample_name, sample_hash)

    if prompt_items is not None:
        if progress_callback:
            progress_callback(0.35, desc="Using cached voice prompt...")
        return prompt_items, True  # True = was cached

    # Create new prompt
    if progress_callback:
        progress_callback(0.2, desc="Processing voice sample (first time)...")

    prompt_items = model.create_voice_clone_prompt(
        ref_audio=wav_path,
        ref_text=ref_text,
        x_vector_only_mode=False,
    )

    # Save to cache
    if progress_callback:
        progress_callback(0.35, desc="Caching voice prompt...")

    save_voice_prompt(sample_name, prompt_items, sample_hash)

    # Store in memory cache too
    _voice_prompt_cache[sample_name] = {
        'prompt': prompt_items,
        'hash': sample_hash
    }

    return prompt_items, False  # False = newly created


def get_available_samples():
    """Find all .wav files in samples folder that have matching .txt files."""
    if not SAMPLES_DIR.exists():
        return []

    samples = []
    import json
    for wav_file in sorted(SAMPLES_DIR.glob("*.wav")):
        json_file = wav_file.with_suffix(".json")
        if json_file.exists():
            try:
                meta = json.loads(json_file.read_text(encoding="utf-8"))
                ref_text = meta.get("Text", "")
            except Exception:
                meta = {}
                ref_text = ""
            samples.append({
                "name": wav_file.stem,
                "wav_path": str(wav_file),
                "json_path": str(json_file),
                "ref_text": ref_text,
                "meta": meta
            })
    return samples


def get_sample_choices():
    """Get sample names for dropdown."""
    samples = get_available_samples()
    return [s["name"] for s in samples]


def get_output_files():
    """Get list of generated output files."""
    if not OUTPUT_DIR.exists():
        return []
    files = sorted(OUTPUT_DIR.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True)
    return [str(f) for f in files]


def get_audio_duration(audio_path):
    """Get duration of audio file in seconds."""
    try:
        info = sf.info(audio_path)
        return info.duration
    except:
        return 0.0


def format_time(seconds):
    """Format seconds as MM:SS.ms"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def on_sample_select(sample_name):
    """When a sample is selected, show its reference text, audio, and cache status."""
    samples = get_available_samples()
    for s in samples:
        if s["name"] == sample_name:
            cache_path = get_prompt_cache_path(sample_name)
            cache_indicator = " ⚡" if cache_path.exists() else ""
            # Show all info if available
            meta = s.get("meta", {})
            if meta:
                info = "\n".join(f"{k}: {v}" for k, v in meta.items())
                return s["wav_path"], info + cache_indicator
            else:
                return s["wav_path"], s["ref_text"] + cache_indicator
    return None, ""


def generate_audio(sample_name, text_to_generate, language, seed, model_size="1.7B", progress=gr.Progress()):
    """Generate audio using voice cloning with cached prompts."""
    if not sample_name:
        return None, "❌ Please select a voice sample first."

    if not text_to_generate or not text_to_generate.strip():
        return None, "❌ Please enter text to generate."

    # Find the selected sample
    samples = get_available_samples()
    sample = None
    for s in samples:
        if s["name"] == sample_name:
            sample = s
            break

    if not sample:
        return None, f"❌ Sample '{sample_name}' not found."

    try:
        # Set the seed for reproducibility
        seed = int(seed) if seed is not None else -1
        if seed < 0:
            # Generate a random seed and use it
            import random
            seed = random.randint(0, 2147483647)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seed_msg = f"🎲 Seed: {seed}"

        progress(0.1, desc=f"Loading Base model ({model_size})...")
        model = get_tts_model(model_size)

        # Get or create the voice prompt (with caching)
        prompt_items, was_cached = get_or_create_voice_prompt(
            model=model,
            sample_name=sample_name,
            wav_path=sample["wav_path"],
            ref_text=sample["ref_text"],
            progress_callback=progress
        )

        cache_status = "cached" if was_cached else "newly processed"
        progress(0.4, desc=f"Generating audio ({cache_status} prompt)...")

        # Generate using the cached prompt
        wavs, sr = model.generate_voice_clone(
            text=text_to_generate.strip(),
            language=language if language != "Auto" else "Auto",
            voice_clone_prompt=prompt_items,
        )

        progress(0.8, desc="Saving audio...")
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() else "_" for c in sample_name)
        output_file = OUTPUT_DIR / f"{safe_name}_{timestamp}.wav"

        sf.write(str(output_file), wavs[0], sr)

        # Save metadata file
        metadata_file = output_file.with_suffix(".txt")
        metadata = f"""Generated: {timestamp}
Sample: {sample_name}
Model: Base {model_size}
Language: {language}
Seed: {seed}
Text: {text_to_generate.strip()}
"""
        metadata_file.write_text(metadata, encoding="utf-8")

        progress(1.0, desc="Done!")
        cache_msg = "⚡ Used cached prompt" if was_cached else "💾 Created & cached prompt"
        return str(output_file), f"✅ Audio saved to: {output_file.name}\n{cache_msg} | {seed_msg} | 🤖 {model_size}"

    except Exception as e:
        return None, f"❌ Error generating audio: {str(e)}"


def generate_voice_design(text_to_generate, language, instruct, seed, progress=gr.Progress()):
    """Generate audio using voice design with natural language instructions."""
    if not text_to_generate or not text_to_generate.strip():
        return None, "❌ Please enter text to generate."

    if not instruct or not instruct.strip():
        return None, "❌ Please enter voice design instructions."

    try:
        # Set the seed for reproducibility
        seed = int(seed) if seed is not None else -1
        if seed < 0:
            import random
            seed = random.randint(0, 2147483647)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seed_msg = f"🎲 Seed: {seed}"

        progress(0.1, desc="Loading VoiceDesign model...")
        model = get_voice_design_model()

        progress(0.3, desc="Generating designed voice...")
        wavs, sr = model.generate_voice_design(
            text=text_to_generate.strip(),
            language=language if language != "Auto" else "Auto",
            instruct=instruct.strip(),
        )

        progress(0.8, desc="Saving audio (temp)...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file = TEMP_DIR / f"voice_design_{timestamp}.wav"
        sf.write(str(temp_file), wavs[0], sr)

        # User must save to samples explicitly; return temp file path
        progress(1.0, desc="Done!")
        return str(temp_file), f"✅ Voice design generated. Save to samples to keep.\n{seed_msg}"

    except Exception as e:
        return None, f"❌ Error generating audio: {str(e)}"


def generate_custom_voice(text_to_generate, language, speaker, instruct, seed, model_size="1.7B", progress=gr.Progress()):
    """Generate audio using the CustomVoice model with premium speakers."""
    if not text_to_generate or not text_to_generate.strip():
        return None, "❌ Please enter text to generate."

    if not speaker:
        return None, "❌ Please select a speaker."

    try:
        # Set the seed for reproducibility
        seed = int(seed) if seed is not None else -1
        if seed < 0:
            import random
            seed = random.randint(0, 2147483647)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seed_msg = f"🎲 Seed: {seed}"

        progress(0.1, desc=f"Loading CustomVoice model ({model_size})...")
        model = get_custom_voice_model(model_size)

        progress(0.3, desc="Generating with custom voice...")

        # Call with or without instruct
        kwargs = {
            "text": text_to_generate.strip(),
            "language": language if language != "Auto" else "Auto",
            "speaker": speaker,
        }
        if instruct and instruct.strip():
            kwargs["instruct"] = instruct.strip()

        wavs, sr = model.generate_custom_voice(**kwargs)

        progress(0.8, desc="Saving audio...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"custom_{speaker}_{timestamp}.wav"

        sf.write(str(output_file), wavs[0], sr)

        # Save metadata file
        metadata_file = output_file.with_suffix(".txt")
        metadata = f"""Generated: {timestamp}
Type: Custom Voice
Model: CustomVoice {model_size}
Speaker: {speaker}
Language: {language}
Seed: {seed}
Instruct: {instruct.strip() if instruct else ''}
Text: {text_to_generate.strip()}
"""
        metadata_file.write_text(metadata, encoding="utf-8")

        progress(1.0, desc="Done!")
        instruct_msg = f" with style: {instruct.strip()[:30]}..." if instruct and instruct.strip() else ""
        return str(output_file), f"✅ Audio saved to: {output_file.name}\n🎭 Speaker: {speaker}{instruct_msg}\n{seed_msg} | 🤖 {model_size}"

    except Exception as e:
        return None, f"❌ Error generating audio: {str(e)}"


def generate_conversation(conversation_data, pause_duration, language, seed, model_size="1.7B", progress=gr.Progress()):
    """Generate a multi-speaker conversation from structured data.

    conversation_data is a string with format:
    Speaker1: Line of dialogue
    Speaker2: Another line
    Speaker1: Response
    ...
    """
    if not conversation_data or not conversation_data.strip():
        return None, "❌ Please enter conversation lines."

    try:
        # Parse conversation lines
        lines = []
        for line in conversation_data.strip().split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue
            speaker, text = line.split(':', 1)
            speaker = speaker.strip()
            text = text.strip()
            if speaker and text:
                lines.append((speaker, text))

        if not lines:
            return None, "❌ No valid conversation lines found. Use format: Speaker: Text"

        # Validate speakers
        valid_speakers = list(CUSTOM_VOICE_SPEAKERS.keys())
        for speaker, _ in lines:
            if speaker not in valid_speakers:
                return None, f"❌ Unknown speaker: '{speaker}'\n\nValid speakers: {', '.join(valid_speakers)}"

        # Set seed
        seed = int(seed) if seed is not None else -1
        if seed < 0:
            import random
            seed = random.randint(0, 2147483647)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        progress(0.1, desc=f"Loading CustomVoice model ({model_size})...")
        model = get_custom_voice_model(model_size)

        # Generate all lines
        all_wavs = []
        sr = None

        for i, (speaker, text) in enumerate(lines):
            progress_val = 0.1 + (0.8 * i / len(lines))
            progress(progress_val, desc=f"Generating line {i + 1}/{len(lines)} ({speaker})...")

            wavs, sr = model.generate_custom_voice(
                text=text,
                language=language if language != "Auto" else "Auto",
                speaker=speaker,
            )
            all_wavs.append(wavs[0])

        # Concatenate with pauses
        progress(0.9, desc="Stitching conversation...")
        pause_samples = int(sr * pause_duration)
        pause = np.zeros(pause_samples)

        conversation_audio = []
        for i, wav in enumerate(all_wavs):
            conversation_audio.append(wav)
            if i < len(all_wavs) - 1:  # Don't add pause after last line
                conversation_audio.append(pause)

        final_audio = np.concatenate(conversation_audio)

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"conversation_{timestamp}.wav"
        sf.write(str(output_file), final_audio, sr)

        # Save metadata
        metadata_file = output_file.with_suffix(".txt")
        speakers_used = list(set(s for s, _ in lines))
        metadata = f"""Generated: {timestamp}
Type: Conversation
Model: CustomVoice {model_size}
Language: {language}
Seed: {seed}
Pause Duration: {pause_duration}s
Speakers: {', '.join(speakers_used)}
Lines: {len(lines)}

--- Script ---
{conversation_data.strip()}
"""
        metadata_file.write_text(metadata, encoding="utf-8")

        progress(1.0, desc="Done!")
        duration = len(final_audio) / sr
        return str(output_file), f"✅ Conversation saved: {output_file.name}\n📝 {len(lines)} lines | ⏱️ {duration:.1f}s | 🎲 Seed: {seed} | 🤖 {model_size}"

    except Exception as e:
        return None, f"❌ Error generating conversation: {str(e)}"


def generate_design_then_clone(design_text, design_instruct, clone_text, language, seed, progress=gr.Progress()):
    """Generate a voice design, then clone it for new text."""
    if not design_text or not design_text.strip():
        return None, None, "❌ Please enter reference text for voice design."

    if not design_instruct or not design_instruct.strip():
        return None, None, "❌ Please enter voice design instructions."

    if not clone_text or not clone_text.strip():
        return None, None, "❌ Please enter text to generate with the cloned voice."

    try:
        # Set the seed for reproducibility
        seed = int(seed) if seed is not None else -1
        if seed < 0:
            import random
            seed = random.randint(0, 2147483647)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seed_msg = f"🎲 Seed: {seed}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Step 1: Generate the designed voice
        progress(0.1, desc="Loading VoiceDesign model...")
        design_model = get_voice_design_model()

        progress(0.2, desc="Creating designed voice reference...")
        ref_wavs, sr = design_model.generate_voice_design(
            text=design_text.strip(),
            language=language if language != "Auto" else "Auto",
            instruct=design_instruct.strip(),
        )

        # Save the reference
        ref_file = OUTPUT_DIR / f"design_ref_{timestamp}.wav"
        sf.write(str(ref_file), ref_wavs[0], sr)

        # Step 2: Clone the designed voice
        progress(0.5, desc="Loading Base model for cloning...")
        clone_model = get_tts_model()

        progress(0.6, desc="Creating voice clone prompt...")
        voice_clone_prompt = clone_model.create_voice_clone_prompt(
            ref_audio=(ref_wavs[0], sr),
            ref_text=design_text.strip(),
        )

        progress(0.7, desc="Generating cloned audio...")
        wavs, sr = clone_model.generate_voice_clone(
            text=clone_text.strip(),
            language=language if language != "Auto" else "Auto",
            voice_clone_prompt=voice_clone_prompt,
        )

        progress(0.9, desc="Saving audio...")
        output_file = OUTPUT_DIR / f"design_clone_{timestamp}.wav"
        sf.write(str(output_file), wavs[0], sr)

        # Save metadata file
        metadata_file = output_file.with_suffix(".txt")
        metadata = f"""Generated: {timestamp}
Type: Design → Clone
Language: {language}
Seed: {seed}
Design Instruct: {design_instruct.strip()}
Design Text: {design_text.strip()}
Clone Text: {clone_text.strip()}
"""
        metadata_file.write_text(metadata, encoding="utf-8")

        progress(1.0, desc="Done!")
        return str(ref_file), str(output_file), f"✅ Generated!\n📎 Reference: {ref_file.name}\n🎵 Output: {output_file.name}\n{seed_msg}"

    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"


def save_designed_voice(audio_file, name, instruct, language, seed, ref_text):
    """Save a designed voice as a sample (wav+txt in samples)."""
    if not audio_file:
        return "❌ No audio to save. Generate a voice first.", gr.update()

    if not name or not name.strip():
        return "❌ Please enter a name for this design.", gr.update()

    name = name.strip()
    safe_name = "".join(c if c.isalnum() or c in "_ -" else "_" for c in name)

    # Check if already exists
    target_wav = SAMPLES_DIR / f"{safe_name}.wav"
    if target_wav.exists():
        return f"❌ Sample '{safe_name}' already exists. Choose a different name.", gr.update()

    try:
        import shutil, json
        shutil.copy(audio_file, target_wav)

        # Save .json metadata
        meta = {
            "Type": "Voice Design",
            "Language": language,
            "Seed": int(seed) if seed else -1,
            "Instruct": instruct.strip() if instruct else "",
            "Text": ref_text.strip() if ref_text else ""
        }
        json_file = target_wav.with_suffix(".json")
        json_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return f"✅ Saved as sample: {safe_name}", gr.update()

    except Exception as e:
        return f"❌ Error saving: {str(e)}", gr.update()


# def load_design_info(design_name):
#     """Load info for a selected design."""
#     if not design_name:
#         return None, ""

#     designs = get_available_designs()
#     for d in designs:
#         if d["name"] == design_name:
#             meta = d.get("meta", {})
#             info = f"""**{design_name}**

# **Instruct:** {meta.get('instruct', 'N/A')}
# **Language:** {meta.get('language', 'N/A')}
# **Seed:** {meta.get('seed', 'N/A')}
# **Reference Text:** {meta.get('ref_text', 'N/A')}
# **Created:** {meta.get('created', 'N/A')}"""
#             return d["wav_path"], info
#     return None, ""


# def refresh_designs():
#     """Refresh the designs dropdown."""
#     choices = get_design_choices()
#     return gr.update(choices=choices, value=choices[0] if choices else None)


# def delete_design(design_name):
#     """Delete a saved design."""
#     if not design_name:
#         return "❌ No design selected.", gr.update()

#     wav_file = DESIGNS_DIR / f"{design_name}.wav"
#     json_file = DESIGNS_DIR / f"{design_name}.json"

#     try:
#         if wav_file.exists():
#             wav_file.unlink()
#         if json_file.exists():
#             json_file.unlink()

#         choices = get_design_choices()
#         return f"✅ Deleted: {design_name}", gr.update(choices=choices, value=choices[0] if choices else None)
#     except Exception as e:
#         return f"❌ Error: {str(e)}", gr.update()


def refresh_samples():
    """Refresh the sample dropdown."""
    choices = get_sample_choices()
    return gr.update(choices=choices, value=choices[0] if choices else None)


def refresh_outputs():
    """Refresh the output file list."""
    files = get_output_files()
    return gr.update(choices=files, value=files[0] if files else None)


def load_output_audio(file_path):
    """Load a selected output file for playback and show metadata."""
    if file_path and Path(file_path).exists():
        # Check for metadata file
        metadata_file = Path(file_path).with_suffix(".txt")
        if metadata_file.exists():
            try:
                metadata = metadata_file.read_text(encoding="utf-8")
                return file_path, metadata
            except:
                pass
        return file_path, "No metadata available"
    return None, ""


# ============== Prep Samples Functions ==============

def on_prep_audio_load(audio_file):
    """When audio is loaded in prep tab, get its info."""
    if audio_file is None:
        return "No audio loaded"

    try:
        duration = get_audio_duration(audio_file)
        info_text = f"Duration: {format_time(duration)} ({duration:.2f}s)"
        return info_text
    except Exception as e:
        return f"Error: {str(e)}"


def normalize_audio(audio_file):
    """Normalize audio levels."""
    if audio_file is None:
        return None

    try:
        data, sr = sf.read(audio_file)

        # Normalize to -1 to 1 range
        max_val = np.max(np.abs(data))
        if max_val > 0:
            normalized = data / max_val * 0.95  # Leave some headroom
        else:
            normalized = data

        temp_path = TEMP_DIR / f"normalized_{datetime.now().strftime('%H%M%S')}.wav"
        sf.write(str(temp_path), normalized, sr)

        return str(temp_path)

    except Exception as e:
        return None


def convert_to_mono(audio_file):
    """Convert stereo audio to mono."""
    if audio_file is None:
        return None

    try:
        data, sr = sf.read(audio_file)

        if len(data.shape) > 1 and data.shape[1] > 1:
            mono = np.mean(data, axis=1)
            temp_path = TEMP_DIR / f"mono_{datetime.now().strftime('%H%M%S')}.wav"
            sf.write(str(temp_path), mono, sr)
            return str(temp_path)
        else:
            return audio_file

    except Exception as e:
        return None

def transcribe_audio(audio_file, whisper_language, progress=gr.Progress()):
    """Transcribe audio using Whisper."""
    if audio_file is None:
        return "❌ Please load an audio file first."

    try:
        progress(0.2, desc="Loading Whisper model...")
        model = get_whisper_model()

        progress(0.4, desc="Transcribing...")

        audio_path = audio_file

        # Transcribe
        options = {}
        if whisper_language and whisper_language != "Auto-detect":
            lang_code = {
                "English": "en", "Chinese": "zh", "Japanese": "ja",
                "Korean": "ko", "German": "de", "French": "fr",
                "Russian": "ru", "Portuguese": "pt", "Spanish": "es",
                "Italian": "it"
            }.get(whisper_language, None)
            if lang_code:
                options["language"] = lang_code

        result = model.transcribe(audio_path, **options)
        progress(1.0, desc="Done!")

        transcription = result["text"].strip()

        return transcription

    except Exception as e:
        return f"❌ Error transcribing: {str(e)}"


def save_as_sample(audio_file, transcription, sample_name):
    """Save audio and transcription as a new sample."""
    if not audio_file:
        return "❌ No audio file to save.", gr.update(), gr.update()

    if not transcription or transcription.startswith("❌"):
        return "❌ Please provide a transcription first.", gr.update(), gr.update()

    if not sample_name or not sample_name.strip():
        return "❌ Please enter a sample name.", gr.update(), gr.update()

    # Clean sample name
    clean_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in sample_name).strip()
    clean_name = clean_name.replace(" ", "_")

    if not clean_name:
        return "❌ Invalid sample name.", gr.update(), gr.update()

    try:
        import json
        # Read audio file
        audio_data, sr = sf.read(audio_file)

        # Save wav file
        wav_path = SAMPLES_DIR / f"{clean_name}.wav"
        sf.write(str(wav_path), audio_data, sr)

        # Save .json metadata
        meta = {
            "Type": "Sample",
            "Text": transcription.strip() if transcription else ""
        }
        json_path = SAMPLES_DIR / f"{clean_name}.json"
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Refresh samples dropdown
        choices = get_sample_choices()

        return (
            f"✅ Sample saved as '{clean_name}'",
            gr.update(choices=choices),
            gr.update(choices=choices)
        )

    except Exception as e:
        return f"❌ Error saving sample: {str(e)}", gr.update(), gr.update()


def load_existing_sample(sample_name):
    """Load an existing sample for editing."""
    if not sample_name:
        return None, "", "No sample selected"

    samples = get_available_samples()
    for s in samples:
        if s["name"] == sample_name:
            duration = get_audio_duration(s["wav_path"])
            cache_path = get_prompt_cache_path(sample_name)
            cache_status = "⚡ Cached" if cache_path.exists() else "📝 Not cached"
            info = f"Duration: {format_time(duration)} ({duration:.2f}s)\nPrompt: {cache_status}"
            return s["wav_path"], s["ref_text"], info

    return None, "", "Sample not found"


def delete_sample(sample_name):
    """Delete a sample (wav, txt, and prompt cache files)."""
    if not sample_name:
        return "❌ No sample selected", gr.update(), gr.update()

    try:
        wav_path = SAMPLES_DIR / f"{sample_name}.wav"
        json_path = SAMPLES_DIR / f"{sample_name}.json"
        prompt_path = get_prompt_cache_path(sample_name)

        deleted = []
        if wav_path.exists():
            wav_path.unlink()
            deleted.append("wav")
        if json_path.exists():
            json_path.unlink()
            deleted.append("json")
        if prompt_path.exists():
            prompt_path.unlink()
            deleted.append("prompt cache")

        # Also remove from memory cache
        if sample_name in _voice_prompt_cache:
            del _voice_prompt_cache[sample_name]

        if deleted:
            choices = get_sample_choices()
            return (
                f"✅ Deleted {sample_name} ({', '.join(deleted)} files)",
                gr.update(choices=choices, value=choices[0] if choices else None),
                gr.update(choices=choices, value=choices[0] if choices else None)
            )
        else:
            return "❌ Files not found", gr.update(), gr.update()

    except Exception as e:
        return f"❌ Error deleting: {str(e)}", gr.update(), gr.update()


def clear_sample_cache(sample_name):
    """Clear the voice prompt cache for a sample."""
    if not sample_name:
        return "❌ No sample selected", "No sample selected"

    try:
        prompt_path = get_prompt_cache_path(sample_name)

        # Remove from disk
        if prompt_path.exists():
            prompt_path.unlink()

        # Remove from memory cache
        if sample_name in _voice_prompt_cache:
            del _voice_prompt_cache[sample_name]

        # Update info
        samples = get_available_samples()
        for s in samples:
            if s["name"] == sample_name:
                duration = get_audio_duration(s["wav_path"])
                info = f"Duration: {format_time(duration)} ({duration:.2f}s)\nPrompt: 📝 Not cached"
                return f"✅ Cache cleared for '{sample_name}'", info

        return f"✅ Cache cleared for '{sample_name}'", "Cache cleared"

    except Exception as e:
        return f"❌ Error clearing cache: {str(e)}", str(e)


def create_ui():
    """Create the Gradio interface."""

    with gr.Blocks(title="Qwen3-TTS Voice Clone Studio") as app:
        gr.Markdown("""
        # 🎙️ Qwen3-TTS Voice Clone Studio

        Clone voices using Qwen3-TTS. Select a voice sample, enter your text, and generate speech!
        """)

        with gr.Tabs():
            # ============== TAB 1: Voice Clone ==============
            with gr.TabItem("🎤 Voice Clone"):
                with gr.Row():
                    # Left column - Sample selection (1/3 width)
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎯 Voice Sample")

                        sample_dropdown = gr.Dropdown(
                            choices=get_sample_choices(),
                            label="Select Sample",
                            info="Manage samples in Prep Samples tab"
                        )

                        with gr.Row():
                            load_sample_btn = gr.Button("▶️ Load", size="sm")
                            refresh_samples_btn = gr.Button("🔄 Refresh", size="sm")

                        sample_audio = gr.Audio(
                            label="Sample Preview",
                            type="filepath",
                            interactive=False,
                            visible=True
                        )

                        sample_text = gr.Textbox(
                            label="Sample Text",
                            interactive=False,
                            lines=3
                        )

                        sample_info = gr.Markdown("")

                    # Right column - Generation (2/3 width)
                    with gr.Column(scale=2):
                        gr.Markdown("### ✍️ Generate Speech")

                        text_input = gr.Textbox(
                            label="Text to Generate",
                            placeholder="Enter the text you want to speak in the cloned voice...",
                            lines=4
                        )

                        with gr.Row():
                            language_dropdown = gr.Dropdown(
                                choices=LANGUAGES,
                                value="English",
                                label="Language",
                                info="Language of the text to generate",
                                scale=2
                            )
                            seed_input = gr.Number(
                                label="Seed",
                                value=-1,
                                precision=0,
                                info="-1 for random",
                                scale=1
                            )
                            clone_model_size = gr.Dropdown(
                                choices=MODEL_SIZES_BASE,
                                value="1.7B",
                                label="Model",
                                info="1.7B = better quality, 0.6B = faster",
                                scale=1
                            )

                        generate_btn = gr.Button("🚀 Generate Audio", variant="primary", size="lg")

                        status_text = gr.Textbox(label="Status", interactive=False)

                        output_audio = gr.Audio(
                            label="Generated Audio",
                            type="filepath"
                        )

                gr.Markdown("---")
                gr.Markdown("### 📂 Output History")

                output_dropdown = gr.Dropdown(
                    choices=get_output_files(),
                    label="Previous Outputs",
                    info="Select a previously generated file to play"
                )

                with gr.Row():
                    load_output_btn = gr.Button("▶️ Play", size="sm")
                    refresh_outputs_btn = gr.Button("🔄 Refresh", size="sm")

                history_audio = gr.Audio(
                    label="Playback",
                    type="filepath"
                )

                history_metadata = gr.Textbox(
                    label="Generation Info",
                    interactive=False,
                    lines=5
                )

                # Event handlers for Voice Clone tab
                def load_selected_sample(sample_name):
                    """Load audio, text, and info for the selected sample."""
                    if not sample_name:
                        return None, "", ""
                    samples = get_available_samples()
                    for s in samples:
                        if s["name"] == sample_name:
                            # Get cache status and duration info
                            cache_path = get_prompt_cache_path(sample_name)
                            cache_status = "⚡ Cached" if cache_path.exists() else "📦 Not cached"
                            try:
                                audio_data, sr = sf.read(s["wav_path"])
                                duration = len(audio_data) / sr
                                info = f"**Info**\n\nDuration: {duration:.2f}s | Prompt: {cache_status}"
                            except:
                                info = f"**Info**\n\nPrompt: {cache_status}"
                            return s["wav_path"], s["ref_text"], info
                    return None, "", ""

                sample_dropdown.change(
                    load_selected_sample,
                    inputs=[sample_dropdown],
                    outputs=[sample_audio, sample_text, sample_info]
                )

                load_sample_btn.click(
                    load_selected_sample,
                    inputs=[sample_dropdown],
                    outputs=[sample_audio, sample_text, sample_info]
                )

                refresh_samples_btn.click(
                    refresh_samples,
                    outputs=[sample_dropdown]
                )

                generate_btn.click(
                    generate_audio,
                    inputs=[sample_dropdown, text_input, language_dropdown, seed_input, clone_model_size],
                    outputs=[output_audio, status_text]
                ).then(
                    refresh_outputs,
                    outputs=[output_dropdown]
                )

                refresh_outputs_btn.click(
                    refresh_outputs,
                    outputs=[output_dropdown]
                )

                output_dropdown.change(
                    load_output_audio,
                    inputs=[output_dropdown],
                    outputs=[history_audio, history_metadata]
                )

                load_output_btn.click(
                    load_output_audio,
                    inputs=[output_dropdown],
                    outputs=[history_audio, history_metadata]
                )

            # ============== TAB 2: Voice Design ==============
            with gr.TabItem("🎨 Voice Design"):
                gr.Markdown("""
                ### Design a Voice with Natural Language

                Describe the voice characteristics you want (age, gender, emotion, tone, accent)
                and the model will generate speech matching that description. Save designs you like for reuse!
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### ✍️ Create Design")

                        design_text_input = gr.Textbox(
                            label="Reference Text",
                            placeholder="Enter the text for the voice design (this will be spoken in the designed voice)...",
                            lines=3,
                            value="Thank you for listening to this voice design sample. This sentence is intentionally a bit longer so you can hear the full range and quality of the generated voice."
                        )

                        design_instruct_input = gr.Textbox(
                            label="Voice Design Instructions",
                            placeholder="Describe the voice: e.g., 'Young female voice, bright and cheerful, slightly breathy' or 'Deep male voice with a warm, comforting tone, speak slowly'",
                            lines=3
                        )

                        with gr.Row():
                            design_language = gr.Dropdown(
                                choices=LANGUAGES,
                                value="English",
                                label="Language",
                                info="Language of the text to generate",
                                scale=2
                            )
                            design_seed = gr.Number(
                                label="Seed",
                                value=-1,
                                precision=0,
                                info="-1 for random",
                                scale=1
                            )

                        design_generate_btn = gr.Button("🎨 Generate Voice", variant="primary", size="lg")
                        design_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column(scale=1):
                        gr.Markdown("### 🔊 Preview & Save")
                        design_output_audio = gr.Audio(
                            label="Generated Audio",
                            type="filepath"
                        )

                        gr.Markdown("---")
                        gr.Markdown("**Save this design for reuse:**")

                        design_save_name = gr.Textbox(
                            label="Design Name",
                            placeholder="Enter a name for this voice design...",
                            lines=1
                        )

                        design_save_btn = gr.Button("💾 Save Design", variant="secondary")
                        design_save_status = gr.Textbox(label="Save Status", interactive=False)

                # Voice Design event handlers
                design_generate_btn.click(
                    generate_voice_design,
                    inputs=[design_text_input, design_language, design_instruct_input, design_seed],
                    outputs=[design_output_audio, design_status]
                )

                # Note: save_designed_voice returns (status, dropdown_update) but we only capture status here
                # The Clone Design tab has its own refresh button to update the dropdown
                design_save_btn.click(
                    lambda *args: save_designed_voice(*args)[0],  # Only return status, ignore dropdown update
                    inputs=[design_output_audio, design_save_name, design_instruct_input, design_language, design_seed, design_text_input],
                    outputs=[design_save_status]
                )

            # (Clone Design tab removed)

            # ============== TAB 4: Custom Voice ==============
            with gr.TabItem("🎭 Custom Voice"):
                gr.Markdown("""
                ### Generate with Premium Voices

                Use pre-built premium voices with optional style instructions. These voices are high-quality
                and support instruction-based style control (emotion, tone, speed, etc.).
                """)

                with gr.Row():
                    # Left - Speaker selection
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎤 Select Speaker")

                        # Create speaker choices with descriptions
                        speaker_choices = [f"{name} - {desc}" for name, desc in CUSTOM_VOICE_SPEAKERS.items()]
                        custom_speaker_dropdown = gr.Dropdown(
                            choices=speaker_choices,
                            label="Speaker",
                            info="Choose a premium voice"
                        )

                        gr.Markdown("""
                        **Speaker Guide:**
                        - 🇨🇳 **Chinese**: Vivian, Serena, Uncle_Fu, Dylan (Beijing), Eric (Sichuan)
                        - 🇺🇸 **English**: Ryan, Aiden
                        - 🇯🇵 **Japanese**: Ono_Anna
                        - 🇰🇷 **Korean**: Sohee

                        *Tip: Each speaker works best in their native language but can speak any supported language.*
                        """)

                    # Right - Generation
                    with gr.Column(scale=2):
                        gr.Markdown("### ✍️ Generate Speech")

                        custom_text_input = gr.Textbox(
                            label="Text to Generate",
                            placeholder="Enter the text you want spoken...",
                            lines=4
                        )

                        custom_instruct_input = gr.Textbox(
                            label="Style Instructions (Optional)",
                            placeholder="e.g., 'Speak with excitement' or 'Very sad and slow' or '用愤怒的语气说'",
                            lines=2,
                            info="Control emotion, tone, speed, etc."
                        )

                        with gr.Row():
                            custom_language = gr.Dropdown(
                                choices=LANGUAGES,
                                value="Auto",
                                label="Language",
                                info="Auto-detect or specify",
                                scale=2
                            )
                            custom_seed = gr.Number(
                                label="Seed",
                                value=-1,
                                precision=0,
                                info="-1 for random",
                                scale=1
                            )
                            custom_model_size = gr.Dropdown(
                                choices=MODEL_SIZES_CUSTOM,
                                value="1.7B",
                                label="Model",
                                info="1.7B = better, 0.6B = faster",
                                scale=1
                            )

                        custom_generate_btn = gr.Button("🚀 Generate Audio", variant="primary", size="lg")
                        custom_status = gr.Textbox(label="Status", interactive=False)

                        custom_output_audio = gr.Audio(
                            label="Generated Audio",
                            type="filepath"
                        )

                # Custom Voice event handlers
                def extract_speaker_name(selection):
                    """Extract speaker name from dropdown selection."""
                    if not selection:
                        return None
                    return selection.split(" - ")[0]

                custom_generate_btn.click(
                    lambda text, lang, speaker_sel, instruct, seed, model_size, progress=gr.Progress(): generate_custom_voice(text, lang, extract_speaker_name(speaker_sel), instruct, seed, model_size, progress),
                    inputs=[custom_text_input, custom_language, custom_speaker_dropdown, custom_instruct_input, custom_seed, custom_model_size],
                    outputs=[custom_output_audio, custom_status]
                )

            # ============== TAB 5: Conversation ==============
            with gr.TabItem("💬 Conversation"):
                gr.Markdown("""
                ### Create Multi-Speaker Conversations

                Generate dialogues between different speakers. Write your script with speaker names,
                and the system will generate each line and stitch them together automatically.
                """)

                with gr.Row():
                    # Left - Script input
                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 Conversation Script")

                        conversation_input = gr.Textbox(
                            label="Script",
                            placeholder="""Enter conversation lines in format:
Speaker: Dialogue text

Example:
Ryan: Hey, how's it going?
Vivian: I'm doing great, thanks for asking!
Ryan: That's wonderful to hear.
Aiden: Mind if I join this conversation?""",
                            lines=12,
                            info="One line per speaker turn. Format: SpeakerName: Text"
                        )

                        gr.Markdown("""
                        **Available Speakers:**

                        | Speaker | Voice | Language |
                        |---------|-------|----------|
                        | Vivian | Bright young female | Chinese |
                        | Serena | Warm gentle female | Chinese |
                        | Uncle_Fu | Seasoned mellow male | Chinese |
                        | Dylan | Youthful Beijing male | Chinese |
                        | Eric | Lively Chengdu male | Chinese |
                        | Ryan | Dynamic male | English |
                        | Aiden | Sunny American male | English |
                        | Ono_Anna | Playful female | Japanese |
                        | Sohee | Warm female | Korean |
                        """)

                    # Right - Settings and output
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Settings")

                        conv_language = gr.Dropdown(
                            choices=LANGUAGES,
                            value="Auto",
                            label="Language",
                            info="Language for all lines (Auto recommended)"
                        )

                        conv_pause = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.5,
                            step=0.1,
                            label="Pause Between Lines (seconds)",
                            info="Silence between each speaker turn"
                        )

                        conv_seed = gr.Number(
                            label="Seed",
                            value=-1,
                            precision=0,
                            info="-1 for random"
                        )

                        conv_model_size = gr.Dropdown(
                            choices=MODEL_SIZES_CUSTOM,
                            value="1.7B",
                            label="Model",
                            info="CustomVoice model size"
                        )

                        conv_generate_btn = gr.Button("🎬 Generate Conversation", variant="primary", size="lg")
                        conv_status = gr.Textbox(label="Status", interactive=False)

                        gr.Markdown("### 🔊 Output")
                        conv_output_audio = gr.Audio(
                            label="Generated Conversation",
                            type="filepath"
                        )

                # Conversation event handlers
                conv_generate_btn.click(
                    generate_conversation,
                    inputs=[conversation_input, conv_pause, conv_language, conv_seed, conv_model_size],
                    outputs=[conv_output_audio, conv_status]
                )

            # ============== TAB 6: Prep Samples ==============
            with gr.TabItem("🎛️ Prep Samples"):
                gr.Markdown("""
                ### Prepare Voice Samples

                Load, trim, edit, transcribe, and manage your voice samples. This is your workspace for preparing
                high-quality reference audio for voice cloning.
                """)

                with gr.Row():
                    # Left column - Existing samples browser
                    with gr.Column(scale=1):
                        gr.Markdown("### 📚 Existing Samples")

                        existing_sample_dropdown = gr.Dropdown(
                            choices=get_sample_choices(),
                            label="Browse Samples",
                            info="Select a sample to preview or edit"
                        )

                        with gr.Row():
                            load_sample_btn = gr.Button("📂 Load to Editor", size="sm")
                            clear_cache_btn = gr.Button("🔄 Clear Cache", size="sm")
                            delete_sample_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")

                        existing_sample_audio = gr.Audio(
                            label="Sample Preview",
                            type="filepath",
                            interactive=False
                        )

                        existing_sample_text = gr.Textbox(
                            label="Sample Text",
                            lines=3,
                            interactive=False
                        )

                        existing_sample_info = gr.Textbox(
                            label="Info",
                            interactive=False
                        )

                    # Right column - Audio editing
                    with gr.Column(scale=2):
                        gr.Markdown("### ✂️ Edit Audio")

                        prep_audio_input = gr.Audio(
                            label="Working Audio (Use Trim icon to edit)",
                            type="filepath",
                            sources=["upload", "microphone"],
                            interactive=True
                        )

                        # gr.Markdown("#### Quick Actions")
                        with gr.Row():
                            normalize_btn = gr.Button("Normalize Volume")
                            mono_btn = gr.Button("Convert to Mono")

                        prep_audio_info = gr.Textbox(
                            label="Audio Info",
                            interactive=False
                        )

                gr.Markdown("---")

                with gr.Row():
                    # Transcription section
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 Transcribe")

                        whisper_language = gr.Dropdown(
                            choices=["Auto-detect"] + LANGUAGES[1:],
                            value="Auto-detect",
                            label="Language Hint",
                            info="Optional: specify language for better accuracy"
                        )

                        transcribe_btn = gr.Button("📝 Transcribe Audio", variant="primary")

                    with gr.Column(scale=2):
                        gr.Markdown("### 💬 Transcription / Reference Text")

                        transcription_output = gr.Textbox(
                            label="Text",
                            lines=4,
                            interactive=True,
                            placeholder="Transcription will appear here, or enter/edit text manually...",
                            info="This will be saved as the reference text for the sample"
                        )

                gr.Markdown("---")
                gr.Markdown("### 💾 Save as New Sample")

                with gr.Row():
                    new_sample_name = gr.Textbox(
                        label="Sample Name",
                        placeholder="Enter a name for this voice sample...",
                        scale=2
                    )
                    save_sample_btn = gr.Button("💾 Save Sample", variant="primary", scale=1)

                save_status = gr.Textbox(label="Save Status", interactive=False)

                # ---- Event handlers for Prep Samples tab ----

                # Load existing sample to editor
                def load_sample_to_editor(sample_name):
                    """Load sample into the working audio editor."""
                    if not sample_name:
                        return None, "", "No sample selected"
                    samples = get_available_samples()
                    for s in samples:
                        if s["name"] == sample_name:
                            duration = get_audio_duration(s["wav_path"])
                            info = f"Duration: {format_time(duration)} ({duration:.2f}s)"
                            return s["wav_path"], s["ref_text"], info
                    return None, "", "Sample not found"

                load_sample_btn.click(
                    load_sample_to_editor,
                    inputs=[existing_sample_dropdown],
                    outputs=[prep_audio_input, transcription_output, prep_audio_info]
                )

                # Preview on dropdown change
                existing_sample_dropdown.change(
                    load_existing_sample,
                    inputs=[existing_sample_dropdown],
                    outputs=[existing_sample_audio, existing_sample_text, existing_sample_info]
                )

                # Delete sample
                delete_sample_btn.click(
                    delete_sample,
                    inputs=[existing_sample_dropdown],
                    outputs=[save_status, existing_sample_dropdown, sample_dropdown]
                )

                # Clear cache
                clear_cache_btn.click(
                    clear_sample_cache,
                    inputs=[existing_sample_dropdown],
                    outputs=[save_status, existing_sample_info]
                )

                # When audio is loaded/changed in editor
                prep_audio_input.change(
                    on_prep_audio_load,
                    inputs=[prep_audio_input],
                    outputs=[prep_audio_info]
                )

                # Normalize
                normalize_btn.click(
                    normalize_audio,
                    inputs=[prep_audio_input],
                    outputs=[prep_audio_input]
                )

                # Convert to mono
                mono_btn.click(
                    convert_to_mono,
                    inputs=[prep_audio_input],
                    outputs=[prep_audio_input]
                )

                # Transcribe
                transcribe_btn.click(
                    transcribe_audio,
                    inputs=[prep_audio_input, whisper_language],
                    outputs=[transcription_output]
                )

                # Save as sample
                save_sample_btn.click(
                    save_as_sample,
                    inputs=[prep_audio_input, transcription_output, new_sample_name],
                    outputs=[save_status, existing_sample_dropdown, sample_dropdown]
                )

        gr.Markdown("""
        ---
        **Tips:**
        - **Voice Clone**: Clone from your own audio samples (3-10 seconds of clear audio)
        - **Voice Design**: Create voices from text descriptions, save designs you like!
        - **Clone Design**: Use saved designs to generate new content consistently
        - **Custom Voice**: Use premium pre-built voices with style control (emotion, tone, speed)
        - **Conversation**: Create multi-speaker dialogues - just write a script!
        - Use the **Prep Samples** tab to trim, clean, and transcribe audio before saving
        - ⚡ **Voice prompts are cached!** First generation processes the sample, subsequent ones are faster
        """)

    return app


if __name__ == "__main__":
    print(f"Samples directory: {SAMPLES_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Found {len(get_sample_choices())} samples")

    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft()
    )
