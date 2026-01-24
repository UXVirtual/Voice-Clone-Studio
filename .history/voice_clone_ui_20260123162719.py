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
OUTPUT_DIR.mkdir(exist_ok=True)

# Global model cache
_tts_model = None
_whisper_model = None
_voice_prompt_cache = {}  # In-memory cache for voice prompts

# Supported languages for TTS
LANGUAGES = [
    "Auto", "English", "Chinese", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian"
]


def get_tts_model():
    """Lazy-load the TTS model."""
    global _tts_model
    if _tts_model is None:
        print("Loading Qwen3-TTS model...")
        _tts_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        print("TTS model loaded!")
    return _tts_model


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
    for wav_file in sorted(SAMPLES_DIR.glob("*.wav")):
        txt_file = wav_file.with_suffix(".txt")
        if txt_file.exists():
            ref_text = txt_file.read_text(encoding="utf-8").strip()
            samples.append({
                "name": wav_file.stem,
                "wav_path": str(wav_file),
                "txt_path": str(txt_file),
                "ref_text": ref_text
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
            return s["wav_path"], s["ref_text"] + cache_indicator
    return None, ""


def generate_audio(sample_name, text_to_generate, language, seed, progress=gr.Progress()):
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
        seed = int(seed) if seed is not None else None
        if seed is not None and seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            seed_msg = f"🎲 Seed: {seed}"
        else:
            seed_msg = "🎲 Seed: random"
        
        progress(0.1, desc="Loading model...")
        model = get_tts_model()

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

        progress(1.0, desc="Done!")
        cache_msg = "⚡ Used cached prompt" if was_cached else "💾 Created & cached prompt"
        return str(output_file), f"✅ Audio saved to: {output_file.name}\n{cache_msg} | {seed_msg}"

    except Exception as e:
        return None, f"❌ Error generating audio: {str(e)}"


def refresh_samples():
    """Refresh the sample dropdown."""
    choices = get_sample_choices()
    return gr.update(choices=choices, value=choices[0] if choices else None)


def refresh_outputs():
    """Refresh the output file list."""
    files = get_output_files()
    return gr.update(choices=files, value=files[0] if files else None)


def load_output_audio(file_path):
    """Load a selected output file for playback."""
    if file_path and Path(file_path).exists():
        return file_path
    return None


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


def trim_audio(audio_file):
    """Apply trim from the audio component's waveform selection."""
    if audio_file is None:
        return None, "❌ No audio file loaded"

    try:
        # The audio component returns the trimmed file when user uses waveform trim
        # We just need to save it to our output folder with a proper name
        data, sr = sf.read(audio_file)
        duration = len(data) / sr
        
        # Save to output folder
        temp_path = OUTPUT_DIR / f"trimmed_{datetime.now().strftime('%H%M%S')}.wav"
        sf.write(str(temp_path), data, sr)

        return str(temp_path), f"✅ Saved trimmed audio: {format_time(duration)} ({duration:.2f}s)"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def normalize_audio(audio_file):
    """Normalize audio levels."""
    if audio_file is None:
        return None, "❌ No audio file loaded"

    try:
        data, sr = sf.read(audio_file)

        # Normalize to -1 to 1 range
        max_val = np.max(np.abs(data))
        if max_val > 0:
            normalized = data / max_val * 0.95  # Leave some headroom
        else:
            normalized = data

        temp_path = OUTPUT_DIR / f"normalized_{datetime.now().strftime('%H%M%S')}.wav"
        sf.write(str(temp_path), normalized, sr)

        return str(temp_path), "✅ Audio normalized"

    except Exception as e:
        return None, f"❌ Error normalizing: {str(e)}"


def convert_to_mono(audio_file):
    """Convert stereo audio to mono."""
    if audio_file is None:
        return None, "❌ No audio file loaded"

    try:
        data, sr = sf.read(audio_file)

        if len(data.shape) > 1 and data.shape[1] > 1:
            mono = np.mean(data, axis=1)
            temp_path = OUTPUT_DIR / f"mono_{datetime.now().strftime('%H%M%S')}.wav"
            sf.write(str(temp_path), mono, sr)
            return str(temp_path), "✅ Converted to mono"
        else:
            return audio_file, "ℹ️ Audio is already mono"

    except Exception as e:
        return None, f"❌ Error converting: {str(e)}"


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

        detected_lang = result.get("language", "unknown")
        transcription = result["text"].strip()

        return f"[Detected language: {detected_lang}]\n\n{transcription}"

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
        # Read audio file
        audio_data, sr = sf.read(audio_file)

        # Save wav file
        wav_path = SAMPLES_DIR / f"{clean_name}.wav"
        sf.write(str(wav_path), audio_data, sr)

        # Save text file (remove the detected language line if present)
        text_content = transcription
        if text_content.startswith("[Detected language:"):
            lines = text_content.split("\n")
            text_content = "\n".join(lines[2:]).strip()

        txt_path = SAMPLES_DIR / f"{clean_name}.txt"
        txt_path.write_text(text_content, encoding="utf-8")

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
        txt_path = SAMPLES_DIR / f"{sample_name}.txt"
        prompt_path = get_prompt_cache_path(sample_name)

        deleted = []
        if wav_path.exists():
            wav_path.unlink()
            deleted.append("wav")
        if txt_path.exists():
            txt_path.unlink()
            deleted.append("txt")
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

    custom_css = """
    .seed-random-btn {
        background: var(--input-background-fill) !important;
        border: 1px solid var(--border-color-primary) !important;
        min-width: 36px !important;
        max-width: 36px !important;
        height: 42px !important;
        margin-top: 24px !important;
        align-self: flex-end !important;
    }
    .seed-random-btn-wrap {
        background: var(--block-background-fill) !important;
        padding: 8px !important;
        border-radius: 8px !important;
        margin-top: auto !important;
    }
    .lang-seed-group {
        background: var(--block-background-fill) !important;
        padding: 12px !important;
        border-radius: 8px !important;
    }
    """

    with gr.Blocks(title="Qwen3-TTS Voice Clone Studio", theme=gr.themes.Soft(), css=custom_css) as app:
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

                        with gr.Group():
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
                                with gr.Column(scale=0, min_width=50, elem_classes=["seed-random-btn-wrap"]):
                                    randomize_seed_btn = gr.Button("🎲", min_width=36, elem_classes=["seed-random-btn"])

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

                def get_random_seed():
                    """Generate a random seed."""
                    import random
                    return random.randint(0, 2147483647)

                randomize_seed_btn.click(
                    get_random_seed,
                    outputs=[seed_input]
                )

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
                    inputs=[sample_dropdown, text_input, language_dropdown, seed_input],
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
                    outputs=[history_audio]
                )

                load_output_btn.click(
                    load_output_audio,
                    inputs=[output_dropdown],
                    outputs=[history_audio]
                )

            # ============== TAB 2: Prep Samples ==============
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

                        with gr.Row():
                            prep_audio_input = gr.Audio(
                                label="Working Audio (upload, record, or load from samples)",
                                type="filepath",
                                sources=["upload", "microphone"],
                                interactive=True,
                                scale=2
                            )
                            prep_audio_info = gr.Textbox(
                                label="Audio Info",
                                interactive=False,
                                scale=1
                            )

                        gr.Markdown("*Use the waveform to select a region, then click Trim in the player or use the button below*")
                        
                        gr.Markdown("#### Quick Actions")
                        with gr.Row():
                            trim_btn = gr.Button("✂️ Apply Trim / Save", variant="primary")
                            normalize_btn = gr.Button("📊 Normalize Volume")
                            mono_btn = gr.Button("🔊 Convert to Mono")

                        edit_status = gr.Textbox(label="Edit Status", interactive=False)

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

                # Trim audio
                trim_btn.click(
                    trim_audio,
                    inputs=[prep_audio_input],
                    outputs=[prep_audio_input, edit_status]
                )

                # Normalize
                normalize_btn.click(
                    normalize_audio,
                    inputs=[prep_audio_input],
                    outputs=[prep_audio_input, edit_status]
                )

                # Convert to mono
                mono_btn.click(
                    convert_to_mono,
                    inputs=[prep_audio_input],
                    outputs=[prep_audio_input, edit_status]
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
        - For best results, use clear reference audio (3-10 seconds)
        - The reference text should exactly match what's spoken in the reference audio
        - Use the **Prep Samples** tab to trim, clean, and transcribe audio before saving
        - Normalize audio to ensure consistent volume levels
        - ⚡ **Voice prompts are cached!** First generation processes the sample, subsequent ones are faster
        - If you modify a sample's audio or text, the cache auto-invalidates and rebuilds
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
        inbrowser=True
    )
