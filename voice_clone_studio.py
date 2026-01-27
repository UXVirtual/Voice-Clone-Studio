import os
import sys
from pathlib import Path

# Add modules directory to Python path
MODULES_DIR = Path(__file__).parent / "modules"
if str(MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(MODULES_DIR))

import torch
import soundfile as sf
import gradio as gr
from qwen_tts import Qwen3TTSModel
from datetime import datetime
import numpy as np
import hashlib
import random
import json
import shutil
import re
from textwrap import dedent
from modules.ui_components import ui_help
import markdown
from modules.ui_components.confirmation_modal import (
    CONFIRMATION_MODAL_CSS,
    CONFIRMATION_MODAL_HEAD,
    CONFIRMATION_MODAL_HTML,
    show_confirmation_modal_js
)

# Directories
SAMPLES_DIR = Path(__file__).parent / "samples"
OUTPUT_DIR = Path(__file__).parent / "output"
TEMP_DIR = Path(__file__).parent / "temp"
DATASETS_DIR = Path(__file__).parent / "datasets"
CONFIG_FILE = Path(__file__).parent / "config.json"
SAMPLES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Clear temp folder on launch
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
_vibe_voice_model = None
_vibevoice_tts_model = None  # VibeVoice TTS for long-form multi-speaker
_voice_prompt_cache = {}  # In-memory cache for voice prompts

# Model size options
MODEL_SIZES = ["Small", "Large"]  # Small=0.6B, Large=1.7B
MODEL_SIZES_BASE = ["Small", "Large"]  # Base model: Small=0.6B, Large=1.7B
MODEL_SIZES_CUSTOM = ["Small", "Large"]  # CustomVoice: Small=0.6B, Large=1.7B
MODEL_SIZES_DESIGN = ["1.7B"]  # VoiceDesign only has 1.7B
MODEL_SIZES_VIBEVOICE = ["Small", "Large"]  # VibeVoice: Small=1.5B, Large=Large

# Voice Clone engine and model options
VOICE_CLONE_OPTIONS = [
    "Qwen3 - Small",
    "Qwen3 - Large",
    "VibeVoice - Small",
    "VibeVoice - Large"
]

# Default to Large models for better quality
DEFAULT_VOICE_CLONE_MODEL = "Qwen3 - Large"

# Supported languages for TTS
LANGUAGES = [
    "Auto", "English", "Chinese", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian"
]

# Custom Voice speakers
CUSTOM_VOICE_SPEAKERS = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee"
]

# ============== Configuration Management ==============

def load_config():
    """Load user preferences from config file."""
    default_config = {
        "transcribe_model": "Whisper",
        "tts_base_size": "Large",
        "custom_voice_size": "Large",
        "language": "Auto",
        "conv_pause_duration": 0.5,
        "whisper_language": "Auto-detect"
    }

    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                saved_config = json.load(f)
                # Merge with defaults to handle new settings
                default_config.update(saved_config)
    except Exception as e:
        print(f"Warning: Could not load config: {e}")

    return default_config


def save_config(config):
    """Save user preferences to config file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save config: {e}")


# Load config on startup
_user_config = load_config()

# Check Whisper availability
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠ Whisper not available - only VibeVoice ASR will be offered for transcription")

# ============== Model Management ==============

def unload_tts_models():
    """Unload all TTS models to free VRAM."""
    global _tts_model, _tts_model_size, _voice_design_model, _custom_voice_model, _custom_voice_model_size, _vibevoice_tts_model

    freed = []
    if _tts_model is not None:
        del _tts_model
        _tts_model = None
        _tts_model_size = None
        freed.append("Base TTS")

    if _voice_design_model is not None:
        del _voice_design_model
        _voice_design_model = None
        freed.append("VoiceDesign")

    if _custom_voice_model is not None:
        del _custom_voice_model
        _custom_voice_model = None
        _custom_voice_model_size = None
        freed.append("CustomVoice")

    if _vibevoice_tts_model is not None:
        del _vibevoice_tts_model
        _vibevoice_tts_model = None
        freed.append("VibeVoice TTS")

    if freed:
        torch.cuda.empty_cache()
        print(f"🗑️ Unloaded TTS models: {', '.join(freed)}")
        return True
    return False


def unload_asr_models():
    """Unload all ASR models to free VRAM."""
    global _whisper_model, _vibe_voice_model

    freed = []
    if _whisper_model is not None:
        del _whisper_model
        _whisper_model = None
        freed.append("Whisper")

    if _vibe_voice_model is not None:
        del _vibe_voice_model
        _vibe_voice_model = None
        freed.append("VibeVoice ASR")

    if freed:
        torch.cuda.empty_cache()
        print(f"🗑️ Unloaded ASR models: {', '.join(freed)}")
        return True
    return False


def get_tts_model(size="1.7B"):
    """Lazy-load the TTS Base model for voice cloning."""
    global _tts_model, _tts_model_size

    # Unload ASR models before loading TTS
    unload_asr_models()

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

    # Unload ASR models before loading TTS
    unload_asr_models()

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

    # Unload ASR models before loading TTS
    unload_asr_models()

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

    if not WHISPER_AVAILABLE:
        raise ImportError("Whisper is not available on this system.")

    # Unload TTS models before loading ASR
    unload_tts_models()

    if _whisper_model is None:
        print("Loading Whisper model...")
        _whisper_model = whisper.load_model("medium")
        print("Whisper model loaded!")
    return _whisper_model


def get_vibe_voice_model():
    """Lazy-load the VibeVoice ASR model."""
    global _vibe_voice_model

    # Unload TTS models before loading ASR
    unload_tts_models()

    if _vibe_voice_model is None:
        print("Loading VibeVoice ASR model...")

        try:
            # Import from renamed vibevoice_asr package (no conflict with TTS)
            from modules.vibevoice_asr.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
            from modules.vibevoice_asr.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

            model_path = "microsoft/VibeVoice-ASR"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            # Suppress expected warnings (missing preprocessor_config.json and tokenizer class mismatch)
            import logging
            import warnings
            prev_level = logging.getLogger("transformers.tokenization_utils_base").level
            logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                processor = VibeVoiceASRProcessor.from_pretrained(model_path)

            logging.getLogger("transformers.tokenization_utils_base").setLevel(prev_level)

            # Load model with flash attention if available
            try:
                model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                    model_path,
                    dtype=dtype,
                    device_map=device if device == "auto" else None,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=True
                )
                print("VibeVoice ASR loaded with Flash Attention 2!")
            except Exception as e:
                if "flash" in str(e).lower():
                    print("Flash Attention 2 not available, using SDPA...")
                    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                        model_path,
                        dtype=dtype,
                        device_map=device if device == "auto" else None,
                        attn_implementation="sdpa",
                        trust_remote_code=True
                    )
                    print("VibeVoice ASR loaded with SDPA!")
                else:
                    raise e

            if device != "auto":
                model = model.to(device)

            model.eval()

            # Create simple inference wrapper
            class VibeVoiceWrapper:
                def __init__(self, model, processor, device):
                    self.model = model
                    self.processor = processor
                    self.device = device

                def transcribe(self, audio_path):
                    """Simple transcribe method compatible with Whisper API."""
                    # Process audio
                    inputs = self.processor(
                        audio=audio_path,
                        return_tensors="pt",
                        add_generation_prompt=True
                    )

                    # Move to device
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                              for k, v in inputs.items()}

                    # Generate with conservative settings
                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=None,  # Greedy
                            do_sample=False,
                            num_beams=1,
                            pad_token_id=self.processor.pad_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id,
                        )

                    # Decode output
                    generated_ids = output_ids[0, inputs['input_ids'].shape[1]:]
                    generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)

                    # Use processor's post-processing to parse structured output
                    try:
                        segments = self.processor.post_process_transcription(generated_text)

                        # Format as [Speaker X]: text (with brackets for compatibility)
                        formatted_lines = []
                        for segment in segments:
                            speaker = segment.get("Speaker", segment.get("speaker_id", 0))
                            content = segment.get("Content", segment.get("text", "")).strip()
                            if content:
                                formatted_lines.append(f"[Speaker {speaker}]: {content}")

                        formatted_text = "\n".join(formatted_lines)
                    except Exception as e:
                        # Fallback: try to parse raw JSON if processor fails
                        try:
                            # Remove "assistant" prefix and other non-JSON text
                            json_start = generated_text.find("[")
                            if json_start != -1:
                                json_text = generated_text[json_start:]
                                segments = json.loads(json_text)

                                formatted_lines = []
                                for segment in segments:
                                    speaker = segment.get("Speaker", 0)
                                    content = segment.get("Content", "").strip()
                                    if content:
                                        formatted_lines.append(f"[Speaker {speaker}]: {content}")

                                formatted_text = "\n".join(formatted_lines)
                            else:
                                # No JSON found, use raw text
                                formatted_text = generated_text
                        except:
                            # Last resort: return raw text
                            formatted_text = generated_text

                    # Return in Whisper-compatible format
                    return {"text": formatted_text}

            _vibe_voice_model = VibeVoiceWrapper(model, processor, device)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"VibeVoice ASR model loaded! ({total_params / 1e9:.2f}B parameters)")

        except ImportError as e:
            print(f"❌ VibeVoice ASR not available: {e}")
            print("Make sure modules/vibevoice_asr directory exists and contains the vibevoice_asr module.")
            raise e
        except Exception as e:
            print(f"❌ Error loading VibeVoice ASR: {e}")
            raise e

    return _vibe_voice_model


def get_vibevoice_tts_model(model_size="1.5B"):
    """Lazy-load the VibeVoice TTS model for long-form multi-speaker generation."""
    global _vibevoice_tts_model

    # Unload ASR models before loading TTS
    unload_asr_models()

    if _vibevoice_tts_model is None:
        print(f"Loading VibeVoice TTS model ({model_size})...")
        try:
            from modules.vibevoice_tts.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
            import warnings

            # Map size to HuggingFace model path
            model_path = f"FranckyB/VibeVoice-{model_size}"

            # Suppress tokenizer mismatch warning (Qwen2Tokenizer wrapped in VibeVoice is intentional)
            import logging
            logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)

                # Model automatically downloads from HF if not cached
                _vibevoice_tts_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    model_path,
                    dtype=torch.bfloat16,
                    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                    attn_implementation="sdpa"  # Use scaled dot-product attention
                )

            print(f"VibeVoice TTS ({model_size}) loaded!")

        except ImportError as e:
            print(f"❌ VibeVoice TTS not available: {e}")
            print("Make sure modules/vibevoice_tts directory exists and contains the vibevoice_tts module.")
            raise e
        except Exception as e:
            print(f"❌ Error loading VibeVoice TTS: {e}")
            raise e

    return _vibevoice_tts_model


def get_prompt_cache_path(sample_name, model_size="1.7B"):
    """Get the path to the cached voice prompt file."""
    return SAMPLES_DIR / f"{sample_name}_{model_size}.prompt"


def compute_sample_hash(wav_path, ref_text):
    """Compute a hash of the sample to detect changes."""
    hasher = hashlib.md5()
    # Hash the audio file
    with open(wav_path, 'rb') as f:
        hasher.update(f.read())
    # Hash the reference text
    hasher.update(ref_text.encode('utf-8'))
    return hasher.hexdigest()


def save_voice_prompt(sample_name, prompt_items, sample_hash, model_size="1.7B"):
    """Save the voice clone prompt to disk."""
    cache_path = get_prompt_cache_path(sample_name, model_size)
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


def load_voice_prompt(sample_name, expected_hash, model_size="1.7B", device='cuda:0'):
    """Load the voice clone prompt from disk if valid."""
    global _voice_prompt_cache

    cache_key = f"{sample_name}_{model_size}"

    # Check in-memory cache first
    if cache_key in _voice_prompt_cache:
        cached = _voice_prompt_cache[cache_key]
        if cached['hash'] == expected_hash:
            print(f"Using in-memory cached prompt for: {sample_name} ({model_size})")
            return cached['prompt']

    # Check disk cache
    cache_path = get_prompt_cache_path(sample_name, model_size)
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
        _voice_prompt_cache[cache_key] = {
            'prompt': prompt_items,
            'hash': expected_hash
        }

        print(f"Loaded voice prompt from cache: {cache_path}")
        return prompt_items

    except Exception as e:
        print(f"Failed to load voice prompt cache: {e}")
        return None


def get_or_create_voice_prompt(model, sample_name, wav_path, ref_text, model_size="1.7B", progress_callback=None):
    """Get cached voice prompt or create new one."""
    # Compute hash to check if sample has changed
    sample_hash = compute_sample_hash(wav_path, ref_text)

    # Try to load from cache
    prompt_items = load_voice_prompt(sample_name, sample_hash, model_size)

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

    save_voice_prompt(sample_name, prompt_items, sample_hash, model_size)

    # Store in memory cache too
    cache_key = f"{sample_name}_{model_size}"
    _voice_prompt_cache[cache_key] = {
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
    """Get list of generated output files with None as first option."""
    if not OUTPUT_DIR.exists():
        return []
    files = sorted(OUTPUT_DIR.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True)
    # Return just filenames instead of full paths
    return [f.name for f in files]


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


def generate_audio(sample_name, text_to_generate, language, seed, model_selection="Qwen3 - Small", progress=gr.Progress()):
    """Generate audio using voice cloning - supports both Qwen and VibeVoice engines."""
    if not sample_name:
        return None, "❌ Please select a voice sample first."

    if not text_to_generate or not text_to_generate.strip():
        return None, "❌ Please enter text to generate."

    # Parse model selection to determine engine and size
    if "VibeVoice" in model_selection:
        engine = "vibevoice"
        if "Small" in model_selection:
            model_size = "1.5B"
        else:  # Large
            model_size = "Large"
    else:  # Qwen3
        engine = "qwen"
        if "Small" in model_selection:
            model_size = "0.6B"
        else:  # Large
            model_size = "1.7B"

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
            seed = random.randint(0, 2147483647)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seed_msg = f"🎲 Seed: {seed}"

        if engine == "qwen":
            # Qwen engine - uses cached prompts
            progress(0.1, desc=f"Loading Qwen3 model ({model_size})...")
            model = get_tts_model(model_size)

            # Get or create the voice prompt (with caching)
            prompt_items, was_cached = get_or_create_voice_prompt(
                model=model,
                sample_name=sample_name,
                wav_path=sample["wav_path"],
                ref_text=sample["ref_text"],
                model_size=model_size,
                progress_callback=progress
            )

            cache_status = "cached" if was_cached else "newly processed"
            progress(0.6, desc=f"Generating audio ({cache_status} prompt)...")

            # Generate using the cached prompt
            wavs, sr = model.generate_voice_clone(
                text=text_to_generate.strip(),
                language=language if language != "Auto" else "Auto",
                voice_clone_prompt=prompt_items,
            )

            engine_display = f"Qwen3-{model_size}"

        else:  # vibevoice engine
            progress(0.1, desc=f"Loading VibeVoice model ({model_size})...")
            model = get_vibevoice_tts_model(model_size)

            from modules.vibevoice_tts.processor.vibevoice_processor import VibeVoiceProcessor
            import warnings
            import logging

            model_path = f"FranckyB/VibeVoice-{model_size}"

            # Suppress tokenizer mismatch warning
            prev_level = logging.getLogger("transformers.tokenization_utils_base").level
            logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                processor = VibeVoiceProcessor.from_pretrained(model_path)

            logging.getLogger("transformers.tokenization_utils_base").setLevel(prev_level)

            progress(0.5, desc="Processing voice sample...")

            # Format script for VibeVoice (single speaker)
            formatted_script = f"Speaker 1: {text_to_generate.strip()}"

            # Process inputs
            inputs = processor(
                text=[formatted_script],
                voice_samples=[[sample["wav_path"]]],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Move to device
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(device)

            progress(0.6, desc="Generating audio...")

            # Set inference steps
            model.set_ddpm_inference_steps(num_steps=10)

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=3.0,
                tokenizer=processor.tokenizer,
                generation_config={'do_sample': False},
                verbose=False,
            )

            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                # Convert bfloat16 to float32 for soundfile compatibility
                # Squeeze to remove batch dimension if present
                audio_tensor = outputs.speech_outputs[0].cpu().to(torch.float32)
                wavs = [audio_tensor.squeeze().numpy()]
                sr = 24000  # VibeVoice uses 24kHz
            else:
                return None, "❌ VibeVoice failed to generate audio."

            engine_display = f"VibeVoice-{model_size}"
            cache_status = "no caching (VibeVoice)"

        progress(0.8, desc="Saving audio...")
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() else "_" for c in sample_name)
        output_file = OUTPUT_DIR / f"{safe_name}_{timestamp}.wav"

        sf.write(str(output_file), wavs[0], sr)

        # Save metadata file
        metadata_file = output_file.with_suffix(".txt")
        metadata = dedent(f"""\
            Generated: {timestamp}
            Sample: {sample_name}
            Engine: {engine_display}
            Language: {language}
            Seed: {seed}
            Text: {text_to_generate.strip()}
            """)
        metadata_file.write_text(metadata, encoding="utf-8")

        progress(1.0, desc="Done!")
        if engine == "qwen":
            cache_msg = "⚡ Used cached prompt" if was_cached else "💾 Created & cached prompt"
            return str(output_file), f"✅ Audio saved to: {output_file.name}\n{cache_msg} | {seed_msg} | 🤖 {engine_display}"
        else:
            return str(output_file), f"✅ Audio saved to: {output_file.name}\n{seed_msg} | 🤖 {engine_display}"

    except Exception as e:
        return None, f"❌ Error generating audio: {str(e)}"

    except Exception as e:
        return None, f"❌ Error generating audio: {str(e)}"


def generate_voice_design(text_to_generate, language, instruct, seed, progress=gr.Progress(), save_to_output=False):
    """Generate audio using voice design with natural language instructions."""
    if not text_to_generate or not text_to_generate.strip():
        return None, "❌ Please enter text to generate."

    if not instruct or not instruct.strip():
        return None, "❌ Please enter voice design instructions."

    try:
        # Set the seed for reproducibility
        seed = int(seed) if seed is not None else -1
        if seed < 0:
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

        progress(0.8, desc=f"Saving audio ({'output' if save_to_output else 'temp'})...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_to_output:
            out_file = OUTPUT_DIR / f"voice_design_{timestamp}.wav"
        else:
            out_file = TEMP_DIR / f"voice_design_{timestamp}.wav"
        sf.write(str(out_file), wavs[0], sr)

        # User must save to samples explicitly; return file path
        progress(1.0, desc="Done!")
        return str(out_file), f"✅ Voice design generated. Save to samples to keep.\n{seed_msg}"

    except Exception as e:
        return None, f"❌ Error generating audio: {str(e)}"


def extract_style_instructions(text):
    """Extract style instructions from parentheses and return clean text + instructions.

    Example: "(nervous) Hello there (excited)" -> ("Hello there", "nervous, excited")
    """
    import re

    # Find all text within parentheses
    instructions = re.findall(r'\(([^)]+)\)', text)

    # Remove all parentheses and their content from the text
    clean_text = re.sub(r'\s*\([^)]+\)\s*', ' ', text)

    # Clean up extra spaces
    clean_text = ' '.join(clean_text.split())

    # Combine all instructions
    combined_instruct = ', '.join(instructions) if instructions else ''

    return clean_text, combined_instruct


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
        metadata = dedent(f"""\
            Generated: {timestamp}
            Type: Custom Voice
            Model: CustomVoice {model_size}
            Speaker: {speaker}
            Language: {language}
            Seed: {seed}
            Instruct: {instruct.strip() if instruct else ''}
            Text: {text_to_generate.strip()}
            """)
        metadata_file.write_text(metadata, encoding="utf-8")

        progress(1.0, desc="Done!")
        instruct_msg = f" with style: {instruct.strip()[:30]}..." if instruct and instruct.strip() else ""
        return str(output_file), f"✅ Audio saved to: {output_file.name}\n🎭 Speaker: {speaker}{instruct_msg}\n{seed_msg} | 🤖 {model_size}"

    except Exception as e:
        return None, f"❌ Error generating audio: {str(e)}"


def generate_with_trained_model(text_to_generate, language, speaker_name, checkpoint_path, instruct, seed, progress=gr.Progress()):
    """Generate audio using a trained custom voice model checkpoint."""
    if not text_to_generate or not text_to_generate.strip():
        return None, "❌ Please enter text to generate."

    try:
        # Set the seed for reproducibility
        seed = int(seed) if seed is not None else -1
        if seed < 0:
            seed = random.randint(0, 2147483647)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seed_msg = f"🎲 Seed: {seed}"

        progress(0.1, desc=f"Loading trained model from {checkpoint_path}...")

        # Load the trained model checkpoint
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(
            checkpoint_path,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        progress(0.3, desc="Generating with trained voice...")

        # Call with or without instruct
        kwargs = {
            "text": text_to_generate.strip(),
            "language": language if language != "Auto" else "Auto",
            "speaker": speaker_name,  # Use the speaker name the model was trained with
        }
        if instruct and instruct.strip():
            kwargs["instruct"] = instruct.strip()

        wavs, sr = model.generate_custom_voice(**kwargs)

        progress(0.8, desc="Saving audio...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"trained_{speaker_name}_{timestamp}.wav"

        sf.write(str(output_file), wavs[0], sr)

        # Save metadata file
        metadata_file = output_file.with_suffix(".txt")
        metadata = dedent(f"""\
            Generated: {timestamp}
            Type: Trained Model
            Model: {checkpoint_path}
            Speaker: {speaker_name}
            Language: {language}
            Seed: {seed}
            Instruct: {instruct.strip() if instruct else ''}
            Text: {text_to_generate.strip()}
            """)
        metadata_file.write_text(metadata, encoding="utf-8")

        progress(1.0, desc="Done!")
        instruct_msg = f" with style: {instruct.strip()[:30]}..." if instruct and instruct.strip() else ""
        return str(output_file), f"✅ Audio saved to: {output_file.name}\n🎭 Speaker: {speaker_name}{instruct_msg}\n{seed_msg} | 🤖 Trained Model"

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
        # Speaker number to name mapping from CUSTOM_VOICE_SPEAKERS
        speaker_list = CUSTOM_VOICE_SPEAKERS

        # Parse conversation lines - support [Speaker N]:, [N]:, and SpeakerName: formats
        lines = []
        for line in conversation_data.strip().split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue

            # Check if format is [Speaker N]: or [N]:
            if line.startswith('[') and ']' in line:
                bracket_end = line.index(']')
                bracket_content = line[1:bracket_end].strip()
                text = line[bracket_end + 1:].lstrip(':').strip()

                # Try [Speaker N]: format (from transcription, 0-based)
                if bracket_content.lower().startswith('speaker'):
                    num_str = bracket_content[7:].strip()  # After "speaker"
                    if num_str.isdigit():
                        speaker_num = int(num_str)
                        if 0 <= speaker_num < len(speaker_list):
                            speaker = speaker_list[speaker_num]
                            if text:
                                lines.append((speaker, text))
                            continue
                # Try [N]: format (user input, 1-based)
                elif bracket_content.isdigit():
                    speaker_num = int(bracket_content)
                    if 1 <= speaker_num <= len(speaker_list):
                        speaker = speaker_list[speaker_num - 1]
                        if text:
                            lines.append((speaker, text))
                        continue

            # Fallback to SpeakerName: format
            speaker, text = line.split(':', 1)
            speaker = speaker.strip()
            text = text.strip()
            if speaker in speaker_list and text:
                lines.append((speaker, text))

        if not lines:
            return None, "❌ No valid conversation lines found. Use format: [N]: Text or SpeakerName: Text"

        # All speakers validated during parsing

        # Set seed
        seed = int(seed) if seed is not None else -1
        if seed < 0:
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

            # Extract style instructions from parentheses
            clean_text, style_instruct = extract_style_instructions(text)

            if style_instruct:
                progress(progress_val, desc=f"Line {i + 1}/{len(lines)} [{style_instruct[:15]}...]")
            else:
                progress(progress_val, desc=f"Line {i + 1}/{len(lines)} ({speaker})")

            # Generate with optional style instructions
            kwargs = {
                "text": clean_text,
                "language": language if language != "Auto" else "Auto",
                "speaker": speaker,
            }
            if style_instruct:
                kwargs["instruct"] = style_instruct

            wavs, sr = model.generate_custom_voice(**kwargs)
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
        output_file = OUTPUT_DIR / f"conversation_qwen3_{timestamp}.wav"
        sf.write(str(output_file), final_audio, sr)

        # Save metadata
        metadata_file = output_file.with_suffix(".txt")
        speakers_used = list(set(s for s, _ in lines))
        metadata = (
            f"Generated: {timestamp}\n"
            f"Type: Qwen3-TTS Conversation\n"
            f"Model: CustomVoice {model_size}\n"
            f"Language: {language}\n"
            f"Seed: {seed}\n"
            f"Pause Duration: {pause_duration}s\n"
            f"Speakers: {', '.join(speakers_used)}\n"
            f"Lines: {len(lines)}\n"
            f"\n"
            f"--- Script ---\n"
            f"{conversation_data.strip()}\n"
        )
        metadata_file.write_text(metadata, encoding="utf-8")

        progress(1.0, desc="Done!")
        duration = len(final_audio) / sr
        return str(output_file), f"✅ Conversation saved: {output_file.name}\n📝 {len(lines)} lines | ⏱️ {duration:.1f}s | 🎲 Seed: {seed} | 🤖 {model_size}"

    except Exception as e:
        return None, f"❌ Error generating conversation: {str(e)}"


def generate_vibevoice_longform(script_text, voice_samples_dict, model_size="1.5B", cfg_scale=3.0, seed=-1, progress=gr.Progress()):
    """Generate long-form multi-speaker audio using VibeVoice TTS (up to 90 minutes)."""
    if not script_text or not script_text.strip():
        return None, "❌ Please enter a script."

    try:
        # Set seed
        seed = int(seed) if seed is not None else -1
        if seed < 0:
            seed = random.randint(0, 2147483647)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        progress(0.1, desc=f"Loading VibeVoice TTS ({model_size})...")
        model = get_vibevoice_tts_model(model_size)

        # Import processor
        from modules.vibevoice_tts.processor.vibevoice_processor import VibeVoiceProcessor
        import warnings
        import logging

        model_path = f"FranckyB/VibeVoice-{model_size}"

        # Suppress tokenizer mismatch warning
        prev_level = logging.getLogger("transformers.tokenization_utils_base").level
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            processor = VibeVoiceProcessor.from_pretrained(model_path)

        logging.getLogger("transformers.tokenization_utils_base").setLevel(prev_level)

        # Parse script to extract speaker labels and map to voice samples
        progress(0.3, desc="Processing script...")

        # Parse lines - support [Speaker N]:, [N]:, and SpeakerX: formats
        lines = []
        for line in script_text.strip().split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue

            # Check if format is [Speaker N]: or [N]:
            if line.startswith('[') and ']' in line:
                bracket_end = line.index(']')
                bracket_content = line[1:bracket_end].strip()
                text = line[bracket_end + 1:].lstrip(':').strip()

                # Try [Speaker N]: format (from transcription, 0-based)
                if bracket_content.lower().startswith('speaker'):
                    num_str = bracket_content[7:].strip()  # After "speaker"
                    if num_str.isdigit():
                        speaker_num = int(num_str)
                        if text:
                            # Map Speaker 0,1,2,3... to Speaker1,2,3,4 (wrapping)
                            wrapped_num = (speaker_num % 4) + 1
                            lines.append((f"Speaker{wrapped_num}", text, speaker_num))
                        continue
                # Try [N]: format (user input, 1-based)
                elif bracket_content.isdigit():
                    speaker_num = int(bracket_content)
                    if text:
                        # Use 1-based numbering for user-facing, wrap beyond 4
                        wrapped_num = ((speaker_num - 1) % 4) + 1
                        lines.append((f"Speaker{wrapped_num}", text, speaker_num))
                    continue

            # Fallback to SpeakerX: or Speaker X: format
            parts = line.split(':', 1)
            if len(parts) == 2:
                speaker, text = parts
                speaker = speaker.strip()
                text = text.strip()
                if speaker and text:
                    # Extract number from "SpeakerN" or "Speaker N" format
                    if speaker.lower().startswith("speaker"):
                        num_str = speaker[7:].strip()
                        if num_str.isdigit():
                            lines.append((f"Speaker{num_str}", text, int(num_str)))
                        else:
                            lines.append((speaker, text, None))

        # Build available samples list from provided voice samples
        available_samples = []
        for i in range(1, 5):  # Speaker1 through Speaker4
            speaker_key = f"Speaker{i}"
            if speaker_key in voice_samples_dict and voice_samples_dict[speaker_key]:
                available_samples.append((speaker_key, voice_samples_dict[speaker_key]))

        if not available_samples:
            return None, "❌ Please provide at least one voice sample (Speaker1)."

        # Build voice samples list and mapping
        voice_samples = [sample for _, sample in available_samples]
        speaker_to_sample = {speaker: idx for idx, (speaker, _) in enumerate(available_samples)}

        if not voice_samples:
            return None, "❌ Please provide at least one voice sample."

        # Reconstruct script with proper formatting for VibeVoice
        # VibeVoice expects: "Speaker 0: text\nSpeaker 1: text" (0-based indexing)
        # Strip style instructions (parentheses) as VibeVoice would read them aloud
        formatted_lines = []
        for speaker, text, original_num in lines:
            # Map to 0-based index for VibeVoice
            if speaker in speaker_to_sample:
                vv_speaker_num = speaker_to_sample[speaker]
                # Remove style instructions from text for VibeVoice
                clean_text, _ = extract_style_instructions(text)
                formatted_lines.append(f"Speaker {vv_speaker_num}: {clean_text}")

        formatted_script = '\n'.join(formatted_lines)

        # Process inputs with script and voice samples
        # Note: processor expects lists for text and voice_samples
        inputs = processor(
            text=[formatted_script],
            voice_samples=[voice_samples],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move to device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)

        progress(0.6, desc="Generating audio...")

        # Set inference steps
        model.set_ddpm_inference_steps(num_steps=10)

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=processor.tokenizer,
            generation_config={'do_sample': False},
            verbose=False,
        )

        progress(0.8, desc="Saving audio...")

        # Get generated audio
        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            # Convert bfloat16 to float32 for soundfile compatibility
            # Squeeze to remove batch dimension if present
            audio_tensor = outputs.speech_outputs[0].cpu().to(torch.float32)
            generated_audio = audio_tensor.squeeze().numpy()
            sr = 24000  # VibeVoice uses 24kHz

            # Save output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = OUTPUT_DIR / f"Conversation_vibevoice_{timestamp}.wav"
            sf.write(str(output_file), generated_audio, sr)

            # Save metadata
            metadata_file = output_file.with_suffix(".txt")
            duration = len(generated_audio) / sr
            metadata = (
                f"Generated: {timestamp}\n"
                f"Type: VibeVoice Conversation\n"
                f"Model: VibeVoice-{model_size}\n"
                f"CFG Scale: {cfg_scale}\n"
                f"Seed: {seed}\n"
                f"Duration: {duration:.1f}s ({duration / 60:.1f} min)\n"
                f"Speakers: {len(voice_samples)}\n"
                f"\n"
                f"--- Script ---\n"
                f"{script_text.strip()}\n"
            )
            metadata_file.write_text(metadata, encoding="utf-8")

            progress(1.0, desc="Done!")
            return str(output_file), f"✅ Generated: {output_file.name}\n⏱️ {duration:.1f}s ({duration / 60:.1f} min) | 🎲 Seed: {seed} | 🤖 {model_size}"
        else:
            return None, "❌ No audio generated."

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Full error traceback:\n{error_details}")
        return None, f"❌ Error: {str(e)}\n\nSee console for full traceback."


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
        metadata = dedent(f"""\
            Generated: {timestamp}
            Type: Design → Clone
            Language: {language}
            Seed: {seed}
            Design Instruct: {design_instruct.strip()}
            Design Text: {design_text.strip()}
            Clone Text: {clone_text.strip()}
            """)
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


def refresh_samples():
    """Refresh the sample dropdown."""
    choices = get_sample_choices()
    return gr.update(choices=choices, value=choices[0] if choices else None)


def refresh_outputs():
    """Refresh the output file list."""
    files = get_output_files()
    return gr.update(choices=files, value=None)


def load_output_audio(file_path):
    """Load a selected output file for playback and show metadata."""
    if not file_path:
        return None, ""

    if not Path(file_path).is_absolute():
        file_path = OUTPUT_DIR / file_path
    else:
        file_path = Path(file_path)

    if file_path.exists():
        # Check for metadata file
        metadata_file = file_path.with_suffix(".txt")
        if metadata_file.exists():
            try:
                metadata = metadata_file.read_text(encoding="utf-8")
                return str(file_path), metadata
            except:
                pass
        return str(file_path), "No metadata available"
    return None, ""


# ============== Prep Samples Functions ==============

def is_video_file(filepath):
    """Check if file is a video based on extension."""
    if not filepath:
        return False
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpeg', '.mpg'}
    return Path(filepath).suffix.lower() in video_extensions


def is_audio_file(filepath):
    """Check if file is an audio file based on extension."""
    if not filepath:
        return False
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus'}
    return Path(filepath).suffix.lower() in audio_extensions


def extract_audio_from_video(video_path):
    """Extract audio from video file using ffmpeg."""
    try:
        import subprocess

        # Create temp output path
        timestamp = datetime.now().strftime('%H%M%S')
        audio_output = TEMP_DIR / f"extracted_audio_{timestamp}.wav"

        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '24000',  # 24kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output
            str(audio_output)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and audio_output.exists():
            return str(audio_output)
        else:
            print(f"ffmpeg error: {result.stderr}")
            return None

    except FileNotFoundError:
        print("❌ ffmpeg not found. Please install ffmpeg to extract audio from video.")
        return None
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None


def on_prep_audio_load(audio_file):
    """When audio/video is loaded in prep tab, get its info and extract audio if needed."""
    if audio_file is None:
        return None, "No file loaded"

    try:
        # Check if it's a video file
        if is_video_file(audio_file):
            print(f"Video file detected: {Path(audio_file).name}")
            print("Extracting audio from video...")
            audio_path = extract_audio_from_video(audio_file)

            if audio_path:
                duration = get_audio_duration(audio_path)
                info_text = f"🎬 Video → Audio extracted\nDuration: {format_time(duration)} ({duration:.2f}s)"
                return audio_path, info_text
            else:
                return None, "❌ Failed to extract audio from video. Make sure file has audio track."

        # It's an audio file
        elif is_audio_file(audio_file):
            duration = get_audio_duration(audio_file)
            info_text = f"Duration: {format_time(duration)} ({duration:.2f}s)"
            return audio_file, info_text

        else:
            return None, "❌ Unsupported file type. Please upload audio (.wav, .mp3, etc.) or video (.mp4, .mov, etc.)"

    except Exception as e:
        return None, f"Error: {str(e)}"


def normalize_audio(audio_file):
    """Normalize audio levels."""
    if audio_file is None:
        return None

    try:
        data, sr = sf.read(audio_file)

        # Normalize to -1 to 1 range with conservative headroom
        max_val = np.max(np.abs(data))
        if max_val > 0:
            normalized = data / max_val * 0.85  # Leave 15% headroom to prevent clipping in TTS
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

def transcribe_audio(audio_file, whisper_language, transcribe_model, progress=gr.Progress()):
    """Transcribe audio using Whisper or VibeVoice ASR."""
    if audio_file is None:
        return "❌ Please load an audio file first."

    try:
        if transcribe_model == "VibeVoice ASR":
            progress(0.2, desc="Loading VibeVoice ASR model...")
            try:
                model = get_vibe_voice_model()
            except Exception as e:
                return f"❌ VibeVoice ASR not available: {str(e)}\n\nInstall with: pip install vibevoice"

            progress(0.4, desc="Transcribing with VibeVoice ASR...")
            result = model.transcribe(audio_file)

        else:  # Default to Whisper
            if not WHISPER_AVAILABLE:
                return "❌ Whisper not available. Please use VibeVoice ASR instead."

            progress(0.2, desc="Loading Whisper model...")
            try:
                model = get_whisper_model()
            except ImportError as e:
                return f"❌ {str(e)}"

            progress(0.4, desc="Transcribing with Whisper...")

            # Transcribe with language options
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

            result = model.transcribe(audio_file, **options)

        progress(1.0, desc="Done!")
        transcription = result["text"].strip()

        # Keep [Speaker N]: format for compatibility with Conversation and VibeVoice tabs
        return transcription

    except Exception as e:
        return f"❌ Error transcribing: {str(e)}"


def batch_transcribe_folder(folder, replace_existing, whisper_language, transcribe_model, progress=gr.Progress()):
    """Batch transcribe all audio files in a dataset folder."""
    if not folder or folder == "(No folders)":
        return "❌ Please select a dataset folder first."

    try:
        base_dir = DATASETS_DIR / folder
        if not base_dir.exists():
            return f"❌ Folder not found: {folder}"

        # Get all audio files
        audio_files = sorted(list(base_dir.glob("*.wav")) + list(base_dir.glob("*.mp3")))

        if not audio_files:
            return f"❌ No audio files found in {folder}"

        # Check if there's anything to do BEFORE loading model
        files_to_process = []
        for audio_file in audio_files:
            txt_file = audio_file.with_suffix(".txt")
            if not txt_file.exists() or replace_existing:
                files_to_process.append(audio_file)

        if not files_to_process:
            return f"✅ All {len(audio_files)} files already have transcripts. Check 'Replace existing transcripts' to re-transcribe."

        # Load model once
        status_log = []
        status_log.append(f"📁 Batch transcribing folder: {folder}")
        status_log.append(f"Found {len(audio_files)} audio files ({len(files_to_process)} to process)")
        status_log.append("")

        if transcribe_model == "VibeVoice ASR":
            progress(0.05, desc="Loading VibeVoice ASR model...")
            try:
                model = get_vibe_voice_model()
                status_log.append("✅ Loaded VibeVoice ASR model")
            except Exception as e:
                return f"❌ VibeVoice ASR not available: {str(e)}"
        else:
            if not WHISPER_AVAILABLE:
                return "❌ Whisper not available. Please use VibeVoice ASR instead."

            progress(0.05, desc="Loading Whisper model...")
            try:
                model = get_whisper_model()
                status_log.append("✅ Loaded Whisper model")
            except ImportError as e:
                return f"❌ {str(e)}"

            # Prepare language options
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

        status_log.append("")
        status_log.append("=" * 60)

        transcribed_count = 0
        skipped_count = 0
        error_count = 0

        for i, audio_file in enumerate(audio_files):
            txt_file = audio_file.with_suffix(".txt")

            # Check if transcript already exists
            if txt_file.exists() and not replace_existing:
                status_log.append(f"⏭️  Skipped: {audio_file.name} (transcript exists)")
                skipped_count += 1
                continue

            # Update progress
            progress_val = 0.1 + (0.9 * i / len(audio_files))
            progress(progress_val, desc=f"Transcribing {i + 1}/{len(audio_files)}: {audio_file.name[:30]}...")

            try:
                # Transcribe
                if transcribe_model == "VibeVoice ASR":
                    result = model.transcribe(str(audio_file))
                else:
                    result = model.transcribe(str(audio_file), **options)

                transcription = result["text"].strip()

                # For VibeVoice ASR, remove text in brackets [...] and surrounding colons
                if transcribe_model == "VibeVoice ASR":
                    # Remove [text] and any colons that immediately follow
                    transcription = re.sub(r'\[.*?\]\s*:', '', transcription)  # Remove [ ... ]:
                    transcription = re.sub(r'\[.*?\]', '', transcription)      # Remove remaining [ ... ]
                    transcription = ' '.join(transcription.split())  # Clean up extra whitespace

                # Save transcript
                txt_file.write_text(transcription, encoding="utf-8")

                status_log.append(f"✅ {audio_file.name} → {len(transcription)} chars")
                transcribed_count += 1

            except Exception as e:
                status_log.append(f"❌ Error: {audio_file.name} - {str(e)}")
                error_count += 1

        status_log.append("=" * 60)
        status_log.append("")
        status_log.append("📊 Summary:")
        status_log.append(f"   ✅ Transcribed: {transcribed_count}")
        status_log.append(f"   ⏭️  Skipped: {skipped_count}")
        status_log.append(f"   ❌ Errors: {error_count}")
        status_log.append(f"   📝 Total: {len(audio_files)}")

        progress(1.0, desc="Batch transcription complete!")

        return "\n".join(status_log)

    except Exception as e:
        return f"❌ Error during batch transcription: {str(e)}"


def save_as_sample(audio_file, transcription, sample_name):
    """Save audio and transcription as a new sample."""
    if not audio_file:
        return "❌ No audio file to save.", gr.update(), gr.update(), gr.update()

    if not transcription or transcription.startswith("❌"):
        return "❌ Please provide a transcription first.", gr.update(), gr.update(), gr.update()

    if not sample_name or not sample_name.strip():
        return "❌ Please enter a sample name.", gr.update(), gr.update(), gr.update()

    # Clean sample name
    clean_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in sample_name).strip()
    clean_name = clean_name.replace(" ", "_")

    if not clean_name:
        return "❌ Invalid sample name.", gr.update(), gr.update(), gr.update()

    try:
        # Read audio file
        audio_data, sr = sf.read(audio_file)

        # Clean transcription: remove ALL text in square brackets [...]
        # This removes [Speaker X], [human sounds], [lyrics], etc.
        cleaned_transcription = re.sub(r'\[.*?\]\s*', '', transcription)
        cleaned_transcription = cleaned_transcription.strip()

        # Save wav file
        wav_path = SAMPLES_DIR / f"{clean_name}.wav"
        sf.write(str(wav_path), audio_data, sr)

        # Save .json metadata
        meta = {
            "Type": "Sample",
            "Text": cleaned_transcription if cleaned_transcription else ""
        }
        json_path = SAMPLES_DIR / f"{clean_name}.json"
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Refresh samples dropdown
        choices = get_sample_choices()

        return (
            f"✅ Sample saved as '{clean_name}'",
            gr.update(choices=choices),
            gr.update(choices=choices),
            ""  # Clear the sample name field
        )

    except Exception as e:
        return f"❌ Error saving sample: {str(e)}", gr.update(), gr.update(), gr.update()


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

            # Add design instructions if this was a Voice Design sample
            meta = s.get("meta", {})
            if meta.get("Type") == "Voice Design" and meta.get("Instruct"):
                info += f"\n\nVoice Design:\n{meta['Instruct']}"

            return s["wav_path"], s["ref_text"], info

    return None, "", "Sample not found"


def delete_sample(action, sample_name):
    """Delete a sample (wav, txt, and prompt cache files)."""
    # Ignore empty calls or actions not for this callback
    if not action or not action.strip() or not action.startswith("sample_"):
        return gr.update(), gr.update(), gr.update(), gr.update()

    # If cancelled, return without doing anything
    if "cancel" in action:
        return "❌ Deletion cancelled", gr.update(), gr.update(), action

    # Only process confirm actions
    if "confirm" not in action:
        return gr.update(), gr.update(), gr.update(), action

    if not sample_name:
        return "❌ No sample selected", gr.update(), gr.update(), action

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
                gr.update(choices=choices, value=choices[0] if choices else None),
                action
            )
        else:
            return "❌ Files not found", gr.update(), gr.update(), action

    except Exception as e:
        return f"❌ Error deleting: {str(e)}", gr.update(), gr.update(), action


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


# ============== Training Dataset Functions ==============

def get_trained_models():
    """Get list of trained custom voice models from models directory.

    Returns a flat list of model entries with format:
    - "ModelName" for direct models (folder with model.safetensors)
    - "ModelName - Epoch N" for checkpoint-based models

    Each entry includes the full path for loading.
    """
    trained_models_dir = Path("models")
    if not trained_models_dir.exists():
        return []

    models = []

    for folder in trained_models_dir.iterdir():
        if not folder.is_dir():
            continue

        # Check if this folder directly contains model.safetensors
        if (folder / "model.safetensors").exists():
            models.append({
                "display_name": folder.name,
                "path": str(folder),
                "speaker_name": folder.name
            })
        else:
            # Check for checkpoint subfolders
            checkpoints = []
            for subfolder in folder.iterdir():
                if subfolder.is_dir() and (subfolder / "model.safetensors").exists():
                    # Extract epoch number from "checkpoint-epoch-N" format
                    if subfolder.name.startswith("checkpoint-epoch-"):
                        try:
                            epoch_num = int(subfolder.name.split("-")[-1])
                            checkpoints.append({
                                "epoch": epoch_num,
                                "path": str(subfolder),
                                "display_name": f"{folder.name} - Epoch {epoch_num}"
                            })
                        except ValueError:
                            continue

            # Add all checkpoints to the models list
            for cp in sorted(checkpoints, key=lambda x: x["epoch"]):
                models.append({
                    "display_name": cp["display_name"],
                    "path": cp["path"],
                    "speaker_name": folder.name
                })

    return sorted(models, key=lambda x: x["display_name"])


def get_dataset_folders():
    """Get list of subfolders in datasets directory."""
    if not DATASETS_DIR.exists():
        return ["(No folders)"]
    folders = sorted([d.name for d in DATASETS_DIR.iterdir() if d.is_dir()])
    return folders if folders else ["(No folders)"]


def get_dataset_files(folder=None):
    """Get list of audio files in datasets directory or subfolder."""
    if not DATASETS_DIR.exists():
        return []

    # Determine the directory to scan
    if folder and folder != "(No folders)":
        scan_dir = DATASETS_DIR / folder
    else:
        scan_dir = DATASETS_DIR

    if not scan_dir.exists():
        return []

    audio_files = sorted(
        list(scan_dir.glob("*.wav")) + list(scan_dir.glob("*.mp3")),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    return [f.name for f in audio_files]


def load_dataset_item(folder, filename):
    """Load audio file and its transcript (auto-transcribe if missing)."""
    if not filename:
        return None, ""

    # Determine the directory
    if folder and folder != "(No folders)":
        base_dir = DATASETS_DIR / folder
    else:
        base_dir = DATASETS_DIR

    audio_path = base_dir / filename
    txt_path = audio_path.with_suffix(".txt")

    # Load or create transcript
    if txt_path.exists():
        try:
            transcript = txt_path.read_text(encoding="utf-8")
        except:
            transcript = ""
    else:
        transcript = ""

    return str(audio_path), transcript


def save_dataset_transcript(folder, filename, transcript):
    """Save transcript for a training dataset audio file."""
    if not filename or not transcript:
        return "❌ Filename and transcript required"

    try:
        # Determine the directory
        if folder and folder != "(No folders)":
            base_dir = DATASETS_DIR / folder
        else:
            base_dir = DATASETS_DIR

        audio_path = base_dir / filename
        txt_path = audio_path.with_suffix(".txt")
        txt_path.write_text(transcript.strip(), encoding="utf-8")
        return f"✅ Saved transcript for {filename}"
    except Exception as e:
        return f"❌ Error saving: {str(e)}"


def delete_dataset_item(action, folder, filename):
    """Delete both audio and transcript files."""
    # Ignore empty calls or actions not for this callback
    if not action or not action.strip() or not action.startswith("finetune_"):
        return gr.update(), gr.update(), gr.update()

    # If cancelled, return without doing anything
    if "cancel" in action:
        return "❌ Deletion cancelled", gr.update(), action

    # Only process confirm actions
    if "confirm" not in action:
        return gr.update(), gr.update(), action

    if not filename:
        return "❌ No file selected", gr.update(), action

    try:
        # Determine the directory
        if folder and folder != "(No folders)":
            base_dir = DATASETS_DIR / folder
        else:
            base_dir = DATASETS_DIR

        audio_path = base_dir / filename
        txt_path = audio_path.with_suffix(".txt")

        deleted = []
        if audio_path.exists():
            audio_path.unlink()
            deleted.append("audio")
        if txt_path.exists():
            txt_path.unlink()
            deleted.append("transcript")

        files = get_dataset_files(folder)
        msg = f"✅ Deleted {filename} ({', '.join(deleted)})" if deleted else "❌ File not found"
        return msg, gr.update(choices=files, value=None), action
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(choices=get_dataset_files(folder), value=None), action


def auto_transcribe_finetune(folder, filename, transcribe_model="Whisper", language="Auto-detect", progress=gr.Progress()):
    """Auto-transcribe a finetune audio file."""
    if not filename:
        return "", "❌ No file selected"

    try:
        # Determine the directory
        if folder and folder != "(No folders)":
            base_dir = DATASETS_DIR / folder
        else:
            base_dir = DATASETS_DIR

        audio_path = base_dir / filename
        if not audio_path.exists():
            return "", "❌ File not found"

        # Use existing transcription logic
        transcript = transcribe_audio(str(audio_path), language, transcribe_model, progress)

        # Check if transcription failed (starts with ❌)
        if transcript.startswith("❌"):
            return "", transcript

        # For VibeVoice ASR, remove text in brackets [...] and surrounding colons
        if transcribe_model == "VibeVoice ASR":
            # Remove [text] and any colons that immediately follow
            transcript = re.sub(r'\[.*?\]\s*:', '', transcript)  # Remove [ ... ]:
            transcript = re.sub(r'\[.*?\]', '', transcript)      # Remove remaining [ ... ]
            transcript = ' '.join(transcript.split())  # Clean up extra whitespace

        # Save transcript
        txt_path = audio_path.with_suffix(".txt")
        txt_path.write_text(transcript.strip(), encoding="utf-8")

        return transcript, f"✅ Transcribed and saved for {filename}"
    except Exception as e:
        return "", f"❌ Transcription failed: {str(e)}"


def convert_audio_to_finetune_format(audio_path, progress=gr.Progress()):
    """Convert audio to 24kHz, 16-bit, mono format required for finetuning."""
    try:
        import subprocess

        if not audio_path or not Path(audio_path).exists():
            return None, "❌ No audio file"

        progress(0.3, desc="Converting with ffmpeg...")

        # Use ffmpeg to convert (already installed)
        output_path = Path(audio_path)
        temp_output = output_path.parent / f"temp_{output_path.name}"

        # ffmpeg command: convert to 24kHz, mono, 16-bit PCM
        cmd = [
            'ffmpeg', '-y', '-i', str(audio_path),
            '-ar', '24000',  # 24kHz sample rate
            '-ac', '1',       # mono
            '-sample_fmt', 's16',  # 16-bit
            '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
            str(temp_output)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return None, f"❌ ffmpeg error: {result.stderr}"

        # Replace original with converted
        if temp_output.exists():
            output_path.unlink()
            temp_output.rename(output_path)

        progress(1.0, desc="Done!")
        return str(output_path), "✅ Converted to 24kHz 16-bit mono"
    except FileNotFoundError:
        return None, "❌ ffmpeg not found. Please install ffmpeg"
    except Exception as e:
        return None, f"❌ Conversion failed: {str(e)}"


def save_trimmed_audio(audio_path, trimmed_audio):
    """Save trimmed audio, overwriting the original file."""
    try:
        if not trimmed_audio:
            return None, "❌ No audio to save"

        if not audio_path or not Path(audio_path).exists():
            return None, "❌ Invalid audio path"

        # trimmed_audio can be either:
        # - filepath string (when type="filepath")
        # - tuple (sample_rate, audio_data) (when type="numpy")
        if isinstance(trimmed_audio, str):
            # It's a filepath - copy the trimmed file to original location
            import shutil
            shutil.copy(trimmed_audio, audio_path)
            output_path = Path(audio_path)
            return str(output_path), f"✅ Saved trimmed audio to {output_path.name}"
        else:
            # It's numpy format - convert and save
            sr, audio_data = trimmed_audio

            # Save over the original file
            output_path = Path(audio_path)
            sf.write(str(output_path), audio_data, sr, subtype='PCM_16')

            # Return the saved audio data so it updates in the UI
            return (sr, audio_data), f"✅ Saved trimmed audio to {output_path.name}"
    except Exception as e:
        return None, f"❌ Error saving: {str(e)}"


def check_audio_format(audio_path):
    """Check if audio is 24kHz, 16-bit, mono."""
    try:
        info = sf.info(audio_path)
        is_correct = (info.samplerate == 24000 and
                      info.channels == 1 and
                      info.subtype == 'PCM_16')
        return is_correct, info
    except:
        return False, None


def convert_all_finetune_audio(folder, progress=gr.Progress()):
    """Convert all audio files that aren't 24kHz 16-bit mono."""
    try:
        import subprocess

        files = get_dataset_files(folder)
        if not files:
            return "❌ No audio files found in datasets/"

        # Determine the directory
        if folder and folder != "(No folders)":
            base_dir = DATASETS_DIR / folder
        else:
            base_dir = DATASETS_DIR

        total = len(files)
        converted = 0
        skipped = 0
        errors = []

        for i, filename in enumerate(files):
            progress((i + 1) / total, desc=f"Checking {filename}...")

            audio_path = base_dir / filename

            # Check if already correct format
            is_correct, info = check_audio_format(str(audio_path))

            if is_correct:
                skipped += 1
                continue

            # Convert
            progress((i + 1) / total, desc=f"Converting {filename}...")
            temp_output = audio_path.parent / f"temp_{filename}"

            cmd = [
                'ffmpeg', '-y', '-i', str(audio_path),
                '-ar', '24000',
                '-ac', '1',
                '-sample_fmt', 's16',
                '-acodec', 'pcm_s16le',
                str(temp_output)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0 and temp_output.exists():
                audio_path.unlink()
                temp_output.rename(audio_path)
                converted += 1
            else:
                errors.append(f"{filename}: {result.stderr[:100]}")

        progress(1.0, desc="Done!")

        msg = f"✅ Converted: {converted} | Skipped (already correct): {skipped}"
        if errors:
            msg += f"\n❌ Errors: {len(errors)}\n" + "\n".join(errors[:3])

        return msg
    except FileNotFoundError:
        return "❌ ffmpeg not found. Please install ffmpeg"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============== Training Functions ==============

def train_model(folder, speaker_name, ref_audio_filename, model_size, batch_size, learning_rate, num_epochs, save_interval, progress=gr.Progress()):
    """Complete training workflow: validate, prepare data, and train model."""
    import subprocess
    import json
    import sys

    # ============== STEP 1: Validation ==============
    progress(0.0, desc="Step 1/3: Validating dataset...")

    if not folder or folder == "(No folders)" or folder == "(Select Dataset)":
        return "❌ Please select a dataset folder"

    if not speaker_name or not speaker_name.strip():
        return "❌ Please enter a speaker name"

    if not ref_audio_filename:
        return "❌ Please select a reference audio file"

    # Validate save_interval
    if save_interval is None:
        save_interval = 5  # Default value

    # Create output directory - use absolute path
    project_root = Path(__file__).parent
    output_dir = project_root / "models" / speaker_name.strip()
    if output_dir.exists():
        return f"❌ Output folder already exists: {output_dir}\n\nPlease choose a different speaker name or delete the existing folder."

    output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = DATASETS_DIR / folder
    if not base_dir.exists():
        return f"❌ Folder not found: {folder}"

    # Reference audio path
    ref_audio_path = base_dir / ref_audio_filename
    if not ref_audio_path.exists():
        return f"❌ Reference audio not found: {ref_audio_filename}"

    # Only get audio files, ignore .txt, .jsonl, etc.
    audio_files = [f for f in (list(base_dir.glob("*.wav")) + list(base_dir.glob("*.mp3")))
                   if not f.name.endswith('.txt') and not f.name.endswith('.jsonl')]
    if not audio_files:
        return "❌ No audio files found in folder"

    issues = []
    valid_files = []
    converted_count = 0
    total = len(audio_files)

    status_log = []
    status_log.append("=" * 60)
    status_log.append("STEP 1/3: DATASET VALIDATION")
    status_log.append("=" * 60)

    for i, audio_path in enumerate(audio_files):
        progress(0.0 + (0.2 * (i + 1) / total), desc=f"Validating {audio_path.name}...")

        txt_path = audio_path.with_suffix(".txt")

        # Check if transcript exists
        if not txt_path.exists():
            issues.append(f"❌ {audio_path.name}: Missing transcript")
            continue

        # Check if transcript is not empty
        try:
            transcript = txt_path.read_text(encoding="utf-8").strip()
            if not transcript:
                issues.append(f"❌ {audio_path.name}: Empty transcript")
                continue
        except:
            issues.append(f"❌ {audio_path.name}: Cannot read transcript")
            continue

        # Check audio format and convert if needed
        is_correct, info = check_audio_format(str(audio_path))
        if not is_correct:
            if not info:
                issues.append(f"❌ {audio_path.name}: Cannot read audio file")
                continue

            # Auto-convert to 24kHz 16-bit mono
            progress(0.0 + (0.2 * (i + 1) / total), desc=f"Converting {audio_path.name}...")
            temp_output = audio_path.parent / f"temp_{audio_path.name}"
            cmd = [
                'ffmpeg', '-y', '-i', str(audio_path),
                '-ar', '24000', '-ac', '1', '-sample_fmt', 's16',
                '-acodec', 'pcm_s16le', str(temp_output)
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0 and temp_output.exists():
                    audio_path.unlink()
                    temp_output.rename(audio_path)
                    converted_count += 1
                else:
                    issues.append(f"❌ {audio_path.name}: Conversion failed - {result.stderr[:100]}")
                    continue
            except FileNotFoundError:
                issues.append(f"❌ {audio_path.name}: ffmpeg not found")
                continue
            except Exception as e:
                issues.append(f"❌ {audio_path.name}: Conversion error - {str(e)[:100]}")
                continue

        valid_files.append(audio_path.name)

    if not valid_files:
        return "❌ No valid training samples found\n" + "\n".join(issues[:10])

    status_log.append(f"✅ Found {len(valid_files)} valid training samples")
    if converted_count > 0:
        status_log.append(f"✅ Auto-converted {converted_count} files to 24kHz 16-bit mono")
    if issues:
        status_log.append(f"⚠️  {len(issues)} files skipped:")
        for issue in issues[:5]:
            status_log.append(f"   {issue}")
        if len(issues) > 5:
            status_log.append(f"   ... and {len(issues) - 5} more")

    # Ensure reference audio is in correct format
    progress(0.2, desc="Preparing reference audio...")
    is_correct, info = check_audio_format(str(ref_audio_path))
    if not is_correct:
        temp_output = ref_audio_path.parent / f"temp_{ref_audio_path.name}"
        cmd = [
            'ffmpeg', '-y', '-i', str(ref_audio_path),
            '-ar', '24000', '-ac', '1', '-sample_fmt', 's16',
            '-acodec', 'pcm_s16le', str(temp_output)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and temp_output.exists():
            ref_audio_path.unlink()
            temp_output.rename(ref_audio_path)
        else:
            return f"❌ Failed to convert reference audio: {result.stderr[:200]}"

    # Generate train_raw.jsonl
    progress(0.25, desc="Generating train_raw.jsonl...")
    train_raw_path = base_dir / "train_raw.jsonl"
    jsonl_entries = []

    for filename in valid_files:
        audio_path = base_dir / filename
        txt_path = audio_path.with_suffix(".txt")
        transcript = txt_path.read_text(encoding="utf-8").strip()

        entry = {
            "audio": str(audio_path.absolute()),
            "text": transcript,
            "ref_audio": str(ref_audio_path.absolute())
        }
        jsonl_entries.append(entry)

    try:
        with open(train_raw_path, 'w', encoding='utf-8') as f:
            for entry in jsonl_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        status_log.append(f"✅ Generated train_raw.jsonl with {len(jsonl_entries)} entries")
    except Exception as e:
        return f"❌ Failed to write train_raw.jsonl: {str(e)}"

    # ============== STEP 2: Prepare Data (extract audio codes) ==============
    status_log.append("")
    status_log.append("=" * 60)
    status_log.append("STEP 2/3: EXTRACTING AUDIO CODES")
    status_log.append("=" * 60)
    progress(0.3, desc="Step 2/3: Extracting audio codes...")

    train_with_codes_path = base_dir / "train_with_codes.jsonl"
    modules_dir = Path(__file__).parent / "modules"
    prepare_script = modules_dir / "qwen_finetune" / "prepare_data.py"

    if not prepare_script.exists():
        status_log.append("❌ Qwen3-TTS finetuning scripts not found!")
        status_log.append("   Please ensure Qwen3-TTS repository is cloned.")
        return "\n".join(status_log)

    # Get venv Python executable
    venv_python = Path(__file__).parent / "venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        status_log.append("❌ Virtual environment not found!")
        status_log.append(f"   Expected at: {venv_python}")
        return "\n".join(status_log)

    prepare_cmd = [
        str(venv_python),
        str(prepare_script.absolute()),
        "--device", "cuda:0",
        "--tokenizer_model_path", "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "--input_jsonl", str(train_raw_path),
        "--output_jsonl", str(train_with_codes_path)
    ]

    status_log.append(f"Running: {' '.join(prepare_cmd)}")
    status_log.append("")

    try:
        result = subprocess.Popen(
            prepare_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=str(base_dir)
        )

        for line in result.stdout:
            line = line.strip()
            if line:
                status_log.append(f"  {line}")

        result.wait()

        if result.returncode != 0:
            status_log.append(f"❌ prepare_data.py failed with exit code {result.returncode}")
            return "\n".join(status_log)

        if not train_with_codes_path.exists():
            status_log.append("❌ train_with_codes.jsonl was not generated")
            return "\n".join(status_log)

        status_log.append("")
        status_log.append("✅ Audio codes extracted successfully")

    except Exception as e:
        status_log.append(f"❌ Error running prepare_data.py: {str(e)}")
        return "\n".join(status_log)

    # ============== STEP 3: Fine-tune ==============
    status_log.append("")
    status_log.append("=" * 60)
    status_log.append("STEP 3/3: TRAINING MODEL")
    status_log.append("=" * 60)
    progress(0.5, desc="Step 3/3: Training model (this may take a while)...")

    modules_dir = Path(__file__).parent / "modules"
    sft_script = modules_dir / "qwen_finetune" / "sft_12hz.py"

    if not sft_script.exists():
        status_log.append("❌ sft_12hz.py not found!")
        return "\n".join(status_log)

    # Determine base model path and ensure it's cached
    if "Large" in model_size or "1.7B" in model_size:
        base_model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    else:
        base_model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    # Get the cached model path (downloads if not cached)
    status_log.append(f"Locating base model: {base_model_id}")
    try:
        from huggingface_hub import snapshot_download
        # This checks cache first, only downloads if missing, then returns cache path
        base_model_path = snapshot_download(
            repo_id=base_model_id,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.npz"],
            local_files_only=False  # Will download if not in cache
        )
        status_log.append(f"✅ Using cached model at: {base_model_path}")
    except Exception as e:
        status_log.append(f"❌ Failed to locate/download base model: {str(e)}")
        return "\n".join(status_log)

    sft_cmd = [
        str(venv_python),
        str(sft_script.absolute()),
        "--init_model_path", base_model_path,  # Use local path instead of model ID
        "--output_model_path", str(output_dir),
        "--train_jsonl", str(train_with_codes_path),
        "--batch_size", str(int(batch_size)),
        "--lr", str(learning_rate),
        "--num_epochs", str(int(num_epochs)),
        "--save_interval", str(int(save_interval)),
        "--speaker_name", speaker_name.strip()
    ]

    status_log.append("Training configuration:")
    status_log.append(f"  Base model: {base_model_id}")
    status_log.append(f"  Batch size: {int(batch_size)}")
    status_log.append(f"  Learning rate: {learning_rate}")
    status_log.append(f"  Epochs: {int(num_epochs)}")
    status_log.append(f"  Save interval: Every {int(save_interval)} epoch(s)" if save_interval > 0 else "  Save interval: Every epoch")
    status_log.append(f"  Speaker name: {speaker_name.strip()}")
    status_log.append(f"  Output: {output_dir}")
    status_log.append("")
    status_log.append("🚀 Starting training...")
    status_log.append(f"Running: {' '.join([str(arg) for arg in sft_cmd])}")
    status_log.append("")

    try:
        # Set environment variables to suppress warnings
        env = os.environ.copy()
        env['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        env['TOKENIZERS_PARALLELISM'] = 'false'

        # Capture output in real-time
        result = subprocess.Popen(
            sft_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )

        epoch_count = 0
        for line in result.stdout:
            line = line.strip()
            if line:
                status_log.append(f"  {line}")

                # Update progress based on epoch indicators
                if "Epoch" in line and "Step" in line:
                    try:
                        # Parse "Epoch 0 | Step 0 | Loss: 19.8072"
                        epoch_num = int(line.split("Epoch")[1].split("|")[0].strip())
                        # Progress from 50% to 100% based on epoch
                        progress_val = 0.5 + (0.5 * (epoch_num + 1) / int(num_epochs))
                        progress(progress_val, desc=f"Training: {line[:60]}...")
                    except:
                        pass

        result.wait()

        if result.returncode != 0:
            status_log.append("")
            status_log.append(f"❌ Training failed with exit code {result.returncode}")
            return "\n".join(status_log)

        status_log.append("")
        status_log.append("=" * 60)
        status_log.append("✅ TRAINING COMPLETED SUCCESSFULLY!")
        status_log.append("=" * 60)
        status_log.append(f"📁 Model will be saved to: {output_dir}")
        status_log.append(f"🎤 Speaker name: {speaker_name.strip()}")
        status_log.append("")
        status_log.append("Monitor the terminal window for progress.")
        status_log.append("When training completes, you'll see:")
        status_log.append(f"  - Checkpoints in: {output_dir}/checkpoint-epoch-N/")
        status_log.append("")
        status_log.append("To use your trained model after completion:")
        status_log.append("  1. Go to Voice Presets tab")
        status_log.append("  2. Select 'Trained Models' radio button")
        status_log.append(f"  3. Click refresh and select '{speaker_name.strip()}'")

        progress(1.0, desc="Training launched in terminal!")

    except Exception as e:
        status_log.append(f"❌ Error during training: {str(e)}")
        return "\n".join(status_log)

    return "\n".join(status_log)


def create_ui():
    """Create the Gradio interface."""

    # Load custom theme from local theme.json (colors pre-configured with orange)
    theme = gr.themes.Base.load('modules/ui_components/theme.json')

    # Custom CSS for vertical file list
    custom_css = """
    #confirm-trigger {
        display: none !important;
    }
    #finetune-files-group > div {
        display: grid !important;
    }
    #finetune-files-container {
        max-height: 400px;
        overflow-y: auto;
    }
    #finetune-files-group label {
        background: none !important;
        border: none !important;
        padding: 4px 8px !important;
        margin: 2px 0 !important;
        box-shadow: none !important;
    }
    #finetune-files-group label:hover {
        background: rgba(255, 255, 255, 0.05) !important;
    }
    #output-files-group > div {
        display: grid !important;
    }
    #output-files-container {
        max-height: 600px;
        overflow-y: auto;
    }
    #output-files-group label {
        background: none !important;
        border: none !important;
        padding: 4px 8px !important;
        margin: 2px 0 !important;
        box-shadow: none !important;
    }
    #output-files-group label:hover {
        background: rgba(255, 255, 255, 0.05) !important;
    }
    """

    # Helper function to format help content with markdown
    def format_help_html(markdown_text, height="70vh"):
        """Convert markdown to HTML with scrollable container styling that matches Gradio components.

        Args:
            markdown_text: Markdown content to convert
            height: CSS height value (default: "70vh")
        """
        html_content = markdown.markdown(
            markdown_text,
            extensions=['fenced_code', 'tables', 'nl2br']
        )
        return f"""
        <div style="
            width: 100%;
            max-height: {height};
            overflow-y: auto;
            box-sizing: border-box;
            color: var(--block-label-text-color);
            font-size: var(--block-text-size);
            font-family: var(--font);
            line-height: 1.6;
        ">
            {html_content}
        </div>
        """

    with gr.Blocks(title="Voice Clone Studio") as app:
        # Add confirmation modal HTML
        gr.HTML(CONFIRMATION_MODAL_HTML)

        # Hidden trigger for confirmation modal - visible but hidden via CSS
        confirm_trigger = gr.Textbox(label="Confirm Trigger", value="", elem_id="confirm-trigger")

        gr.Markdown("""
        # 🎙️ Voice Clone Studio
        <p style="font-size: 0.9em; color: #ffffff; margin-top: -10px;">  Powered by Qwen3-TTS, VibeVoice and Whisper</p>
        """)

        with gr.Tabs():
            # ============== TAB 1: Voice Clone ==============
            with gr.TabItem("Voice Clone"):
                gr.Markdown("Clone Voices from Your Samples, using Qwen3-TTS or VibeVoice")
                with gr.Row():
                    # Left column - Sample selection (1/3 width)
                    with gr.Column(scale=1):
                        gr.Markdown("### Voice Sample")

                        sample_choices = get_sample_choices()
                        sample_dropdown = gr.Dropdown(
                            choices=sample_choices,
                            value=sample_choices[0] if sample_choices else None,
                            label="Select Sample",
                            info="Manage samples in Prep Samples tab"
                        )

                        with gr.Row():
                            load_sample_btn = gr.Button("Load", size="sm")
                            refresh_samples_btn = gr.Button("Refresh", size="sm")

                        sample_audio = gr.Audio(
                            label="Sample Preview",
                            type="filepath",
                            interactive=False,
                            visible=True
                        )

                        sample_text = gr.Textbox(
                            label="Sample Text",
                            interactive=False,
                            max_lines=10
                        )

                        sample_info = gr.Textbox(
                            label="Info",
                            interactive=False,
                            max_lines=3
                        )

                    # Right column - Generation (2/3 width)
                    with gr.Column(scale=3):
                        gr.Markdown("### Generate Speech")

                        text_input = gr.Textbox(
                            label="Text to Generate",
                            placeholder="Enter the text you want to speak in the cloned voice...",
                            lines=6
                        )

                        with gr.Row():
                            language_dropdown = gr.Dropdown(
                                choices=LANGUAGES,
                                value=_user_config.get("language", "Auto"),
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

                        clone_model_dropdown = gr.Dropdown(
                            choices=VOICE_CLONE_OPTIONS,
                            value=_user_config.get("voice_clone_model", DEFAULT_VOICE_CLONE_MODEL),
                            label="Engine & Model",
                            info="Choose between Qwen3 (fast, cached prompts) or VibeVoice (high-quality, long-form capable)"
                        )

                        generate_btn = gr.Button("Generate Audio", variant="primary", size="lg")

                        output_audio = gr.Audio(
                            label="Generated Audio",
                            type="filepath"
                        )

                        status_text = gr.Textbox(label="Status", interactive=False, max_lines=5)

                # Event handlers for Voice Clone tab
                def load_selected_sample(sample_name):
                    """Load audio, text, and info for the selected sample."""
                    if not sample_name:
                        return None, "", ""
                    samples = get_available_samples()
                    for s in samples:
                        if s["name"] == sample_name:
                            # Check cache status for both model sizes
                            cache_small = get_prompt_cache_path(sample_name, "0.6B").exists()
                            cache_large = get_prompt_cache_path(sample_name, "1.7B").exists()

                            if cache_small and cache_large:
                                cache_status = "Qwen Cache: ⚡ Small, Large"
                            elif cache_small:
                                cache_status = "Qwen Cache: ⚡ Small"
                            elif cache_large:
                                cache_status = "Qwen Cache: ⚡ Large"
                            else:
                                cache_status = "Qwen Cache: 📦 Not cached"

                            try:
                                audio_data, sr = sf.read(s["wav_path"])
                                duration = len(audio_data) / sr
                                info = f"**Info**\n\nDuration: {duration:.2f}s | {cache_status}"
                            except:
                                info = f"**Info**\n\n{cache_status}"

                            # Add design instructions if this was a Voice Design sample
                            meta = s.get("meta", {})
                            if meta.get("Type") == "Voice Design" and meta.get("Instruct"):
                                info += f"\n\n**Voice Design:**\n{meta['Instruct']}"

                            return s["wav_path"], s["ref_text"], info
                    return None, "", ""

                # Connect event handlers for Voice Clone tab
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
                    inputs=[sample_dropdown, text_input, language_dropdown, seed_input, clone_model_dropdown],
                    outputs=[output_audio, status_text]
                )

            # ============== TAB 2: Custom Voice ==============
            with gr.TabItem("Voice Presets"):
                gr.Markdown("Use Qwen3-TTS pre-trained models or Custom Trained models with style control")

                with gr.Row():
                    # Left - Speaker selection
                    with gr.Column(scale=1):
                        gr.Markdown("### Select Voice Type")

                        voice_type_radio = gr.Radio(
                            choices=["Premium Speakers", "Trained Models"],
                            value="Premium Speakers",
                            label="Voice Source"
                        )

                        # Premium speakers dropdown
                        with gr.Column(visible=True) as premium_section:
                            speaker_choices = CUSTOM_VOICE_SPEAKERS
                            custom_speaker_dropdown = gr.Dropdown(
                                choices=speaker_choices,
                                label="Speaker",
                                info="Choose a premium voice"
                            )

                            custom_model_size = gr.Dropdown(
                                choices=MODEL_SIZES_CUSTOM,
                                value=_user_config.get("custom_voice_size", "Large"),
                                label="Model",
                                info="Small = faster, Large = better quality",
                                scale=1
                            )

                            premium_speaker_guide = dedent("""\
                                **Premium Speakers:**

                                | Speaker | Voice | Language |
                                |---------|-------|----------|
                                | Vivian | Bright young female    | 🇨🇳 Chinese |
                                | Serena | Warm gentle female    | 🇨🇳 Chinese |
                                | Uncle_Fu | Seasoned mellow male    | 🇨🇳 Chinese |
                                | Dylan | Youthful Beijing male    | 🇨🇳 Chinese |
                                | Eric | Lively Chengdu male    | 🇨🇳 Chinese |
                                | Ryan | Dynamic male | 🇺🇸 English    |
                                | Aiden | Sunny American male    | 🇺🇸 English |
                                | Ono_Anna | Playful female    | 🇯🇵 Japanese |
                                | Sohee | Warm female    | 🇰🇷 Korean |

                                *Each speaker works best in native language.*
                                """)

                            gr.HTML(
                                value=format_help_html(premium_speaker_guide),
                                container=True,   # give it the normal block/card container
                                padding=True      # match block padding
                            )

                        # Trained models dropdown
                        with gr.Column(visible=False) as trained_section:
                            def get_initial_model_list():
                                """Get initial list of trained models for dropdown initialization."""
                                models = get_trained_models()
                                if not models:
                                    return ["(No trained models found)"]
                                return ["(Select Model)"] + [m['display_name'] for m in models]

                            def refresh_trained_models():
                                """Refresh model list."""
                                models = get_trained_models()
                                if not models:
                                    return gr.update(choices=["(No trained models found)"], value="(No trained models found)")
                                choices = ["(Select Model)"] + [m['display_name'] for m in models]
                                return gr.update(choices=choices, value="(Select Model)")

                            trained_model_dropdown = gr.Dropdown(
                                choices=get_initial_model_list(),
                                value="(Select Model)",
                                label="Trained Model",
                                info="Select your custom trained voice"
                            )

                            refresh_trained_btn = gr.Button("Refresh", size="sm")

                            trained_models_tip = dedent("""\
                            **Trained Models:**

                            Custom voices you've trained in the Train Model tab.
                            Models are listed as:
                            - "ModelName" for standalone models
                            - "ModelName - Epoch N" for checkpoint-based models

                            *Tip: Later epochs are usually better trained*
                            """)
                            gr.HTML(
                                value=format_help_html(trained_models_tip),
                                container=True,   # give it the normal block/card container
                                padding=True,      # match block padding
                            )

                    # Right - Generation
                    with gr.Column(scale=3):
                        gr.Markdown("### Generate Speech")

                        custom_text_input = gr.Textbox(
                            label="Text to Generate",
                            placeholder="Enter the text you want spoken...",
                            lines=6
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
                                value=_user_config.get("language", "Auto"),
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

                        custom_generate_btn = gr.Button("Generate Audio", variant="primary", size="lg")

                        custom_output_audio = gr.Audio(
                            label="Generated Audio",
                            type="filepath"
                        )
                        custom_status = gr.Textbox(label="Status", max_lines=3, interactive=False)

                # Custom Voice event handlers
                def extract_speaker_name(selection):
                    """Extract speaker name from dropdown selection."""
                    if not selection:
                        return None
                    return selection.split(" - ")[0].split(" (")[0]  # Handle both formats

                def toggle_voice_type(voice_type):
                    """Toggle between premium and trained model sections."""
                    is_premium = voice_type == "Premium Speakers"
                    return {
                        premium_section: gr.update(visible=is_premium),
                        trained_section: gr.update(visible=not is_premium)
                    }

                def generate_with_voice_type(text, lang, speaker_sel, instruct, seed, model_size, voice_type, premium_speaker, trained_model, progress=gr.Progress()):
                    """Generate audio with either premium or trained voice."""

                    if voice_type == "Premium Speakers":
                        # Use premium speaker with CustomVoice model
                        speaker = extract_speaker_name(premium_speaker)
                        if not speaker:
                            return None, "❌ Please select a premium speaker"

                        return generate_custom_voice(
                            text, lang, speaker, instruct, seed,
                            "1.7B" if model_size == "Large" else "0.6B",
                            progress
                        )
                    else:
                        # Use trained model
                        if not trained_model or trained_model in ["(No trained models found)", "(Select Model)"]:
                            return None, "❌ Please select a trained model or train one first"

                        # Find the model path from the model list
                        models = get_trained_models()
                        model_path = None
                        speaker_name = None
                        for model in models:
                            if model['display_name'] == trained_model:
                                model_path = model['path']
                                speaker_name = model['speaker_name']
                                break

                        if not model_path:
                            return None, f"❌ Model not found: {trained_model}"

                        # Generate with trained model
                        return generate_with_trained_model(
                            text, lang, speaker_name, model_path, instruct, seed, progress
                        )

                voice_type_radio.change(
                    toggle_voice_type,
                    inputs=[voice_type_radio],
                    outputs=[premium_section, trained_section]
                )

                refresh_trained_btn.click(
                    refresh_trained_models,
                    outputs=[trained_model_dropdown]
                )

                custom_generate_btn.click(
                    generate_with_voice_type,
                    inputs=[
                        custom_text_input, custom_language, custom_speaker_dropdown,
                        custom_instruct_input, custom_seed, custom_model_size,
                        voice_type_radio, custom_speaker_dropdown, trained_model_dropdown
                    ],
                    outputs=[custom_output_audio, custom_status]
                )

            # ============== TAB 3: Unified Conversation ==============
            with gr.TabItem("Conversation"):
                gr.Markdown("Create Conversation, using Qwen3-TTS or VibeVoice")

                # Model selector at top
                initial_conv_model = _user_config.get("conv_model_type", "Qwen")
                is_qwen_initial = initial_conv_model == "Qwen"
                with gr.Row():
                    conv_model_type = gr.Radio(
                        choices=["Qwen", "VibeVoice"],
                        value=initial_conv_model,
                        label="Conversation Engine",
                    )

                with gr.Row():
                    # Left - Script input and model-specific controls
                    with gr.Column(scale=2):
                        gr.Markdown("### Conversation Script")

                        conversation_script = gr.Textbox(
                            label="Script:",
                            placeholder=dedent("""\
                                Use [N]: format for speaker labels. Add (style) for emotions:

                                [1]: (cheerful) Hey, how's it going?
                                [2]: (excited) I'm doing great, thanks for asking!
                                [1]: That's wonderful to hear.
                                [3]: (curious) Mind if I join this conversation?

                                Style instructions work with Qwen only (VibeVoice ignores them)."""),
                            lines=18
                        )

                        # Qwen speaker mapping (visible when Qwen selected)
                        speaker_guide = dedent("""\
                            **Qwen Speaker Numbers → Preset Voices:**

                            | # | Speaker | Voice | Language |   | # | Speaker | Voice | Language |
                            |---|---------|-------|----------|---|---|---------|-------|----------|
                            | 1 | Vivian | Bright young female | 🇨🇳 Chinese |   | 6 | Ryan | Dynamic male | 🇺🇸 English |
                            | 2 | Serena | Warm gentle female | 🇨🇳 Chinese |   | 7 | Aiden | Sunny American male | 🇺🇸 English |
                            | 3 | Uncle_Fu | Seasoned mellow male | 🇨🇳 Chinese |   | 8 | Ono_Anna | Playful female | 🇯🇵 Japanese |
                            | 4 | Dylan | Youthful Beijing male | 🇨🇳 Chinese |   | 9 | Sohee | Warm female | 🇰🇷 Korean |
                            | 5 | Eric | Lively Chengdu male | 🇨🇳 Chinese |  |  |  |  |  |

                            *Each speaker works best in their native language.*
                            """)

                        qwen_speaker_table = gr.HTML(
                            value=format_help_html(speaker_guide),
                            container=True,   # give it the normal block/card container
                            padding=True,      # match block padding
                            visible=is_qwen_initial
                        )

                        # VibeVoice voice sample selectors (visible when VibeVoice selected)
                        with gr.Column(visible=not is_qwen_initial) as vibevoice_voices_section:
                            gr.Markdown("### Voice Samples (Up to 4 Speakers)")

                            with gr.Row():
                                with gr.Column():
                                    voice_sample_1 = gr.Dropdown(
                                        choices=get_sample_choices(),
                                        label="[1] Voice Sample (Required)",
                                        info="Select from your prepared samples"
                                    )
                                with gr.Column():
                                    voice_sample_2 = gr.Dropdown(
                                        choices=get_sample_choices(),
                                        label="[2] Voice Sample (Optional)",
                                        info="Select from your prepared samples"
                                    )

                            with gr.Row():
                                with gr.Column():
                                    voice_sample_3 = gr.Dropdown(
                                        choices=get_sample_choices(),
                                        label="[3] Voice Sample (Optional)",
                                        info="Select from your prepared samples"
                                    )
                                with gr.Column():
                                    voice_sample_4 = gr.Dropdown(
                                        choices=get_sample_choices(),
                                        label="[4] Voice Sample (Optional)",
                                        info="Select from your prepared samples"
                                    )

                    # Right - Settings and output
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Settings")

                        # Qwen-specific settings
                        with gr.Column(visible=is_qwen_initial) as qwen_settings:
                            conv_model_size = gr.Dropdown(
                                choices=MODEL_SIZES_CUSTOM,
                                value=_user_config.get("conv_model_size", "Large"),
                                label="Model Size",
                                info="Small = Faster, Large = Better Quality"
                            )

                            conv_pause = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=_user_config.get("conv_pause_duration", 0.5),
                                step=0.1,
                                label="Pause Between Lines (seconds)",
                                info="Silence between each speaker turn"
                            )

                            conv_language = gr.Dropdown(
                                choices=LANGUAGES,
                                value=_user_config.get("language", "Auto"),
                                label="Language",
                                info="Language for all lines (Auto recommended)"
                            )

                        # VibeVoice-specific settings
                        with gr.Column(visible=not is_qwen_initial) as vibevoice_settings:
                            longform_model_size = gr.Dropdown(
                                choices=MODEL_SIZES_VIBEVOICE,
                                value=_user_config.get("vibevoice_model_size", "Large"),
                                label="Model Size",
                                info="Small = Faster, Large = Better Quality"
                            )

                            longform_cfg_scale = gr.Slider(
                                minimum=1.0,
                                maximum=5.0,
                                value=3.0,
                                step=0.5,
                                label="CFG Scale",
                                info="Higher = more adherence to prompt (3.0 recommended)"
                            )

                        # Shared settings
                        conv_seed = gr.Number(
                            label="Seed",
                            value=-1,
                            precision=0,
                            info="-1 for random"
                        )

                        conv_generate_btn = gr.Button("Generate Conversation", variant="primary", size="lg")

                        gr.Markdown("### Output")
                        conv_output_audio = gr.Audio(
                            label="Generated Conversation",
                            type="filepath"
                        )
                        conv_status = gr.Textbox(label="Status", interactive=False, max_lines=3)

                        # Model-specific tips
                        qwen_tips_text = dedent("""\
                        **Qwen Tips:**
                        - Fast generation with preset voices
                        - Up to 9 different speakers
                        - Each voice optimized for their native language
                        """)

                        vibevoice_tips_text = dedent("""\
                        **VibeVoice Tips:**
                        - Up to 90 minutes continuous generation
                        - Up to 4 speakers with custom voices
                        - May spontaneously add background music/sounds
                        - Longer scripts work best with Large model
                        """)

                        qwen_tips = gr.HTML(
                            value=format_help_html(qwen_tips_text),
                            container=True,   # give it the normal block/card container
                            padding=True,      # match block padding
                            visible=is_qwen_initial
                        )

                        vibevoice_tips = gr.HTML(
                            value=format_help_html(vibevoice_tips_text),
                            container=True,   # give it the normal block/card container
                            padding=True,      # match block padding
                            visible=not is_qwen_initial
                        )

                # Helper function for voice samples
                def prepare_voice_samples_dict(v1, v2, v3, v4):
                    """Prepare voice samples dictionary for generation."""
                    samples = {}
                    available_samples = get_available_samples()

                    # Convert sample names to file paths
                    for speaker_num, sample_name in [("Speaker1", v1), ("Speaker2", v2), ("Speaker3", v3), ("Speaker4", v4)]:
                        if sample_name:
                            for s in available_samples:
                                if s["name"] == sample_name:
                                    samples[speaker_num] = s["wav_path"]
                                    break
                    return samples

                # Unified generate handler
                def unified_conversation_generate(
                    model_type, script,
                    # Qwen params
                    qwen_lang, qwen_pause, qwen_model_size,
                    # VibeVoice params
                    vv_v1, vv_v2, vv_v3, vv_v4, vv_model_size, vv_cfg,
                    # Shared
                    seed, progress=gr.Progress()
                ):
                    """Route to appropriate generation function based on model type."""
                    if model_type == "Qwen":
                        # Map UI labels to actual model sizes
                        qwen_size = "1.7B" if qwen_model_size == "Large" else "0.6B"
                        return generate_conversation(script, qwen_pause, qwen_lang, seed, qwen_size)
                    else:  # VibeVoice
                        # Map UI labels to actual model sizes
                        vv_size = "1.5B" if vv_model_size == "Small" else "Large"
                        voice_samples = prepare_voice_samples_dict(vv_v1, vv_v2, vv_v3, vv_v4)
                        return generate_vibevoice_longform(script, voice_samples, vv_size, vv_cfg, seed, progress)

                # Event handlers
                conv_generate_btn.click(
                    unified_conversation_generate,
                    inputs=[
                        conv_model_type, conversation_script,
                        # Qwen
                        conv_language, conv_pause, conv_model_size,
                        # VibeVoice
                        voice_sample_1, voice_sample_2, voice_sample_3, voice_sample_4,
                        longform_model_size, longform_cfg_scale,
                        # Shared
                        conv_seed
                    ],
                    outputs=[conv_output_audio, conv_status]
                )

                # Toggle UI based on model selection
                def toggle_conv_ui(model_type):
                    is_qwen = model_type == "Qwen"
                    return {
                        qwen_speaker_table: gr.update(visible=is_qwen),
                        vibevoice_voices_section: gr.update(visible=not is_qwen),
                        qwen_settings: gr.update(visible=is_qwen),
                        vibevoice_settings: gr.update(visible=not is_qwen),
                        qwen_tips: gr.update(visible=is_qwen),
                        vibevoice_tips: gr.update(visible=not is_qwen)
                    }

                conv_model_type.change(
                    toggle_conv_ui,
                    inputs=[conv_model_type],
                    outputs=[qwen_speaker_table, vibevoice_voices_section, qwen_settings, vibevoice_settings, qwen_tips, vibevoice_tips]
                )

            # ============== TAB 4: Voice Design ==============
            with gr.TabItem("Voice Design"):
                gr.Markdown("Create new voices from natural language descriptions")

                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### Create Design")

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
                                value=_user_config.get("language", "Auto"),
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

                        save_to_output_checkbox = gr.Checkbox(
                            label="Save to Output folder instead of Temp",
                            value=False
                        )

                        design_generate_btn = gr.Button("Generate Voice", variant="primary", size="lg")
                        design_status = gr.Textbox(label="Status", interactive=False, max_lines=3)

                    with gr.Column(scale=1):
                        gr.Markdown("### Preview & Save")
                        design_output_audio = gr.Audio(
                            label="Generated Audio",
                            type="filepath"
                        )

                        design_save_name = gr.Textbox(
                            label="Sample Name",
                            placeholder="Enter a name for this voice Sample...",
                            info="Name under which to save the designed voice",
                            lines=1
                        )

                        design_save_btn = gr.Button("Save Sample", variant="primary")

                # Voice Design event handlers
                def generate_voice_design_with_checkbox(text, language, instruct, seed, save_to_output, progress=gr.Progress()):
                    return generate_voice_design(text, language, instruct, seed, progress=progress, save_to_output=save_to_output)

                design_generate_btn.click(
                    generate_voice_design_with_checkbox,
                    inputs=[design_text_input, design_language, design_instruct_input, design_seed, save_to_output_checkbox],
                    outputs=[design_output_audio, design_status]
                )

                # Note: save_designed_voice returns (status, dropdown_update) but we only capture status here
                # The Clone Design tab has its own refresh button to update the dropdown
                design_save_btn.click(
                    lambda *args: save_designed_voice(*args)[0],  # Only return status, ignore dropdown update
                    inputs=[design_output_audio, design_save_name, design_instruct_input, design_language, design_seed, design_text_input],
                    outputs=[design_status]
                )

            # ============== TAB 5: Prep Samples ==============
            with gr.TabItem("Prep Samples"):
                gr.Markdown("Prepare audio samples for voice cloning")
                with gr.Row():
                    # Left column - Existing samples browser
                    with gr.Column(scale=1):
                        gr.Markdown("### Existing Samples")

                        existing_sample_choices = get_sample_choices()
                        existing_sample_dropdown = gr.Dropdown(
                            choices=existing_sample_choices,
                            value=existing_sample_choices[0] if existing_sample_choices else None,
                            label="Browse Samples",
                            info="Select a sample to preview or edit"
                        )

                        with gr.Row():
                            preview_sample_btn = gr.Button("Preview Sample", size="sm")
                            refresh_preview_btn = gr.Button("Refresh Preview", size="sm")
                            load_sample_btn = gr.Button("Load to Editor", size="sm")
                            clear_cache_btn = gr.Button("Clear Cache", size="sm")
                            delete_sample_btn = gr.Button("Delete", size="sm", variant="primary")

                        existing_sample_audio = gr.Audio(
                            label="Sample Preview",
                            type="filepath",
                            interactive=False
                        )

                        existing_sample_text = gr.Textbox(
                            label="Sample Text",
                            max_lines=10,
                            interactive=False
                        )

                        existing_sample_info = gr.Textbox(
                            label="Info",
                            interactive=False
                        )

                        save_status = gr.Textbox(label="Status", interactive=False, scale=1)

                    # Right column - Audio/Video editing
                    with gr.Column(scale=2):
                        gr.Markdown("### Edit Audio/Video")

                        prep_file_input = gr.File(
                            label="Audio or Video File",
                            type="filepath",
                            file_types=["audio", "video"],
                            interactive=True
                        )

                        prep_audio_editor = gr.Audio(
                            label="Audio Editor (Use Trim icon ✂️ to edit)",
                            type="filepath",
                            interactive=True,
                            visible=False
                        )

                        # gr.Markdown("#### Quick Actions")
                        with gr.Row():
                            clear_btn = gr.Button("Clear", size="sm")
                            normalize_btn = gr.Button("Normalize Volume", size="sm")
                            mono_btn = gr.Button("Convert to Mono", size="sm")

                        prep_audio_info = gr.Textbox(
                            label="Audio Info",
                            interactive=False
                        )
                        with gr.Column(scale=2):
                            gr.Markdown("### Transcription / Reference Text")
                            transcription_output = gr.Textbox(
                                label="Text",
                                lines=4,
                                max_lines=10,
                                interactive=True,
                                placeholder="Transcription will appear here, or enter/edit text manually..."
                            )
                            with gr.Row():
                                whisper_language = gr.Dropdown(
                                    choices=["Auto-detect"] + LANGUAGES[1:],
                                    value=_user_config.get("whisper_language", "Auto-detect"),
                                    label="Language",
                                )

                                # Offer available transcription models
                                available_models = ['VibeVoice ASR']
                                if WHISPER_AVAILABLE:
                                    available_models.insert(0, 'Whisper')

                                default_model = _user_config.get("transcribe_model", "Whisper")
                                if default_model not in available_models:
                                    default_model = available_models[0]

                                transcribe_model = gr.Dropdown(
                                    choices=available_models,
                                    value=default_model,
                                    label="Model",
                                )

                            transcribe_btn = gr.Button("Transcribe Audio", variant="primary")

                            # Save as new sample
                            # gr.Markdown("### Save as New Sample")
                            new_sample_name = gr.Textbox(
                                label="Sample Name",
                                placeholder="Enter a name for this voice sample...",
                                scale=2
                            )
                            save_sample_btn = gr.Button("Save Sample", variant="primary")

                # Load existing sample to editor
                def load_sample_to_editor(sample_name):
                    """Load sample into the working audio editor."""
                    if not sample_name:
                        return None, None, "", "No sample selected", gr.update(visible=False)
                    samples = get_available_samples()
                    for s in samples:
                        if s["name"] == sample_name:
                            duration = get_audio_duration(s["wav_path"])
                            info = f"Duration: {format_time(duration)} ({duration:.2f}s)"
                            return s["wav_path"], s["wav_path"], s["ref_text"], info, gr.update(visible=True)
                    return None, None, "", "Sample not found", gr.update(visible=False)

                load_sample_btn.click(
                    load_sample_to_editor,
                    inputs=[existing_sample_dropdown],
                    outputs=[prep_file_input, prep_audio_editor, transcription_output, prep_audio_info, prep_audio_editor]
                )

                # Preview on dropdown change
                existing_sample_dropdown.change(
                    load_existing_sample,
                    inputs=[existing_sample_dropdown],
                    outputs=[existing_sample_audio, existing_sample_text, existing_sample_info]
                )

                # Preview button
                preview_sample_btn.click(
                    load_existing_sample,
                    inputs=[existing_sample_dropdown],
                    outputs=[existing_sample_audio, existing_sample_text, existing_sample_info]
                )

                # Refresh preview button - refreshes the dropdown list
                refresh_preview_btn.click(
                    refresh_samples,
                    outputs=[existing_sample_dropdown]
                )

                # Delete sample
                # Show modal on delete button click
                delete_sample_btn.click(
                    fn=None,
                    inputs=None,
                    outputs=None,
                    js=show_confirmation_modal_js(
                        title="Delete Sample?",
                        message="This will permanently delete the sample audio, text, and cached files. This action cannot be undone.",
                        confirm_button_text="Delete",
                        context="sample_"
                    )
                )

                # Process confirmation
                confirm_trigger.change(
                    delete_sample,
                    inputs=[confirm_trigger, existing_sample_dropdown],
                    outputs=[save_status, existing_sample_dropdown, sample_dropdown, confirm_trigger]
                )

                # Clear cache
                clear_cache_btn.click(
                    clear_sample_cache,
                    inputs=[existing_sample_dropdown],
                    outputs=[save_status, existing_sample_info]
                )

                # When file is loaded/changed
                prep_file_input.change(
                    on_prep_audio_load,
                    inputs=[prep_file_input],
                    outputs=[prep_audio_editor, prep_audio_info]
                ).then(
                    lambda audio: (
                        gr.update(visible=audio is not None),
                        gr.update(visible=audio is None)
                    ),
                    inputs=[prep_audio_editor],
                    outputs=[prep_audio_editor, prep_file_input]
                )

                # Clear file input and reset
                clear_btn.click(
                    lambda: (None, None, ""),
                    outputs=[prep_file_input, prep_audio_editor, prep_audio_info]
                )

                # Normalize
                normalize_btn.click(
                    normalize_audio,
                    inputs=[prep_audio_editor],
                    outputs=[prep_audio_editor]
                )

                # Convert to mono
                mono_btn.click(
                    convert_to_mono,
                    inputs=[prep_audio_editor],
                    outputs=[prep_audio_editor]
                )

                # Transcribe
                transcribe_btn.click(
                    transcribe_audio,
                    inputs=[prep_audio_editor, whisper_language, transcribe_model],
                    outputs=[transcription_output]
                )

                # Save as sample
                save_sample_btn.click(
                    save_as_sample,
                    inputs=[prep_audio_editor, transcription_output, new_sample_name],
                    outputs=[save_status, existing_sample_dropdown, sample_dropdown, new_sample_name]
                )

            # ============== TAB 6: Output History ==============
            with gr.TabItem("Output History"):
                gr.Markdown("Browse and manage previously generated audio files")
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Column(scale=1, elem_id="output-files-container"):
                            output_dropdown = gr.Radio(
                                choices=get_output_files(),
                                show_label=False,
                                interactive=True,
                                elem_id="output-files-group"
                            )
                        refresh_outputs_btn = gr.Button("Refresh", size="sm")

                    with gr.Column(scale=1):
                        history_audio = gr.Audio(
                            label="Playback",
                            type="filepath"
                        )

                        history_metadata = gr.Textbox(
                            label="Generation Info",
                            interactive=False,
                            max_lines=10
                        )
                        delete_output_btn   = gr.Button("Delete", size="sm", variant="primary")

                def delete_output_file(action, selected_file):
                    # Ignore empty calls or actions not for this callback
                    if not action or not action.strip() or not action.startswith("output_"):
                        return gr.update(), gr.update(), gr.update(), gr.update()

                    # If cancelled, return without doing anything
                    if "cancel" in action:
                        return gr.update(), gr.update(), gr.update(value="❌ Deletion cancelled"), action

                    # Only process confirm actions
                    if "confirm" not in action:
                        return gr.update(), gr.update(), gr.update(), action

                    try:
                        # Convert filename to full path if needed
                        if not Path(selected_file).is_absolute():
                            audio_path = OUTPUT_DIR / selected_file
                        else:
                            audio_path = Path(selected_file)

                        txt_path = audio_path.with_suffix(".txt")
                        deleted = []
                        if audio_path.exists():
                            audio_path.unlink()
                            deleted.append("audio")
                        if txt_path.exists():
                            txt_path.unlink()
                            deleted.append("text")
                        # Refresh dropdown
                        choices = get_output_files()
                        msg = f"✅ Deleted: {audio_path.name} ({', '.join(deleted)})" if deleted else "❌ Files not found"

                        # Refresh list and clear selection
                        return gr.update(choices=choices, value=None), gr.update(value=None), gr.update(value=msg), action
                    except Exception as e:
                        return gr.update(), gr.update(value=None), gr.update(value=f"❌ Error: {str(e)}"), action

                # Show modal on delete button click
                delete_output_btn.click(
                    fn=None,
                    inputs=None,
                    outputs=None,
                    js=show_confirmation_modal_js(
                        title="Delete Output File?",
                        message="This will permanently delete the generated audio and its metadata. This action cannot be undone.",
                        confirm_button_text="Delete",
                        context="output_"
                    )
                )

                # Process confirmation
                confirm_trigger.change(
                    delete_output_file,
                    inputs=[confirm_trigger, output_dropdown],
                    outputs=[output_dropdown, history_audio, history_metadata, confirm_trigger]
                )

                refresh_outputs_btn.click(
                    refresh_outputs,
                    outputs=[output_dropdown]
                )

                # Load on dropdown change
                output_dropdown.change(
                    load_output_audio,
                    inputs=[output_dropdown],
                    outputs=[history_audio, history_metadata]
                )

            # ============== TAB 7: Finetune Dataset ==============
            with gr.TabItem("Finetune Dataset"):
                gr.Markdown("Manage and prepare your finetuning dataset")
                with gr.Row():
                    # Left - File list and management
                    with gr.Column(scale=1):
                        gr.Markdown("### Dataset Files")

                        finetune_folder_dropdown = gr.Dropdown(
                            choices=get_dataset_folders(),
                            label="Dataset Folder",
                            info="Subfolders in datasets",
                            interactive=True,
                            value=None
                        )

                        refresh_folder_btn = gr.Button("Refresh Folders", size="sm")

                        with gr.Column(elem_id="finetune-files-container"):
                            finetune_dropdown = gr.Radio(
                                choices=[],
                                show_label=False,
                                interactive=True,
                                elem_id="finetune-files-group"
                            )

                        with gr.Row():
                            refresh_finetune_btn = gr.Button("Refresh", size="sm", scale=1)
                            delete_finetune_btn = gr.Button("Delete", size="sm", scale=1, variant="primary")

                        finetune_audio_preview = gr.Audio(
                            label="Audio Preview & Trim",
                            type="filepath",
                            interactive=True
                        )

                        save_trimmed_btn = gr.Button("Save Trimmed Audio")

                        gr.Markdown("### Transcription Settings")

                        finetune_transcribe_model = gr.Radio(
                            choices=["Whisper", "VibeVoice ASR"],
                            value=_user_config.get("transcribe_model", "Whisper"),
                            label="Transcription Model",
                            info="Choose transcription engine"
                        )

                        finetune_transcribe_lang = gr.Dropdown(
                            choices=["Auto-detect", "English", "Chinese", "Japanese", "Korean",
                                     "French", "German", "Spanish", "Russian"],
                            value=_user_config.get("whisper_language", "Auto-detect"),
                            label="Language (Whisper only)",
                            visible=(_user_config.get("transcribe_model", "Whisper") == "Whisper")
                        )

                    # Right - Transcript editor
                    with gr.Column(scale=2):
                        gr.Markdown("### Edit Transcript")

                        finetune_transcript = gr.Textbox(
                            label="Transcript",
                            placeholder="Load an audio file or auto-transcribe to edit the transcript...",
                            lines=10,
                            info="Edit the transcript to match the audio exactly"
                        )

                        with gr.Row():
                            auto_transcribe_btn = gr.Button("Auto-Transcribe", variant="primary", scale=1)
                            save_transcript_btn = gr.Button("Save Transcript", variant="primary", scale=1)

                        with gr.Column(scale=1):
                            gr.Markdown("**Batch Transcribe Folder**")

                            batch_replace_existing = gr.Checkbox(
                                label="Replace existing transcripts",
                                value=False
                            )

                            batch_transcribe_btn = gr.Button("Batch Transcribe", variant="primary", size="lg")

                            finetune_status = gr.Textbox(
                                label="Status",
                                interactive=False,
                                max_lines=15
                            )

                            finetune_quick_guide = dedent("""\
                            **Quick Guide:**
                            - Create subfolders in /datasets to organize training sets
                            - Use **Batch Transcribe** to Transcribe all files at once
                            - Or edit individual files, trimming track and adjusting transcripts as needed.

                            *See Help Guide tab → Finetune Dataset for detailed instructions*
                            """)
                            gr.HTML(
                                value=format_help_html(finetune_quick_guide),
                                container=True,   # give it the normal block/card container
                                padding=True      # match block padding
                            )

                # Event handlers
                def refresh_folder_list():
                    """Refresh folder list."""
                    folders = get_dataset_folders()
                    return gr.update(choices=folders, value=None)

                def refresh_finetune_list(folder):
                    """Refresh file list for the current folder."""
                    files = get_dataset_files(folder)
                    return gr.update(choices=files, value=None)

                def update_file_list(folder):
                    """Update file list when folder changes."""
                    files = get_dataset_files(folder)
                    return gr.update(choices=files, value=None)

                # When folder changes, update file list
                finetune_folder_dropdown.change(
                    update_file_list,
                    inputs=[finetune_folder_dropdown],
                    outputs=[finetune_dropdown]
                )

                refresh_folder_btn.click(
                    refresh_folder_list,
                    outputs=[finetune_folder_dropdown]
                )

                refresh_finetune_btn.click(
                    refresh_finetune_list,
                    inputs=[finetune_folder_dropdown],
                    outputs=[finetune_dropdown]
                )

                finetune_dropdown.change(
                    load_dataset_item,
                    inputs=[finetune_folder_dropdown, finetune_dropdown],
                    outputs=[finetune_audio_preview, finetune_transcript]
                )

                save_transcript_btn.click(
                    save_dataset_transcript,
                    inputs=[finetune_folder_dropdown, finetune_dropdown, finetune_transcript],
                    outputs=[finetune_status]
                )

                # Show modal on delete button click
                delete_finetune_btn.click(
                    fn=None,
                    inputs=None,
                    outputs=None,
                    js=show_confirmation_modal_js(
                        title="Delete Dataset Item?",
                        message="This will permanently delete the audio file and its transcript. This action cannot be undone.",
                        confirm_button_text="Delete",
                        context="finetune_"
                    )
                )

                # Process confirmation
                confirm_trigger.change(
                    delete_dataset_item,
                    inputs=[confirm_trigger, finetune_folder_dropdown, finetune_dropdown],
                    outputs=[finetune_status, finetune_dropdown, confirm_trigger]
                )

                auto_transcribe_btn.click(
                    auto_transcribe_finetune,
                    inputs=[finetune_folder_dropdown, finetune_dropdown, finetune_transcribe_model, finetune_transcribe_lang],
                    outputs=[finetune_transcript, finetune_status]
                )

                batch_transcribe_btn.click(
                    batch_transcribe_folder,
                    inputs=[finetune_folder_dropdown, batch_replace_existing, finetune_transcribe_lang, finetune_transcribe_model],
                    outputs=[finetune_status]
                )

                def save_and_reload(folder, filename, audio):
                    """Save trimmed audio, then return values to refresh and reload."""
                    # Determine the directory
                    if folder and folder != "(No folders)":
                        base_dir = DATASETS_DIR / folder
                    else:
                        base_dir = DATASETS_DIR

                    # Save the audio
                    saved_audio, status = save_trimmed_audio(str(base_dir / filename) if filename else None, audio)

                    # Return: clear audio, status, and filename to preserve for reload
                    return None, status, filename

                save_trimmed_event = save_trimmed_btn.click(
                    save_and_reload,
                    inputs=[finetune_folder_dropdown, finetune_dropdown, finetune_audio_preview],
                    outputs=[finetune_audio_preview, finetune_status, finetune_dropdown]
                )

                # After saving, reload the same file
                save_trimmed_event.then(
                    load_dataset_item,
                    inputs=[finetune_folder_dropdown, finetune_dropdown],
                    outputs=[finetune_audio_preview, finetune_transcript]
                )

                # Toggle language dropdown based on transcribe model
                def toggle_language_dropdown(model):
                    return gr.update(visible=(model == "Whisper"))

                finetune_transcribe_model.change(
                    toggle_language_dropdown,
                    inputs=[finetune_transcribe_model],
                    outputs=[finetune_transcribe_lang]
                )

                # Save finetune transcription preferences
                finetune_transcribe_model.change(
                    lambda x: save_preference("transcribe_model", x),
                    inputs=[finetune_transcribe_model],
                    outputs=[]
                )

                finetune_transcribe_lang.change(
                    lambda x: save_preference("whisper_language", x),
                    inputs=[finetune_transcribe_lang],
                    outputs=[]
                )

            # ============== TAB 8: Train Model ==============
            with gr.TabItem("Train Model"):
                gr.Markdown("Train a custom voice model using your finetuning dataset")
                with gr.Row():
                    # Left column - Dataset selection and validation
                    with gr.Column(scale=1):
                        gr.Markdown("### Dataset Selection")

                        train_folder_dropdown = gr.Dropdown(
                            choices=["(Select Dataset)"] + get_dataset_folders(),
                            value="(Select Dataset)",
                            label="Training Dataset",
                            info="Select prepared subfolder",
                            interactive=True
                        )

                        refresh_train_folder_btn = gr.Button("Refresh Datasets", size="sm")

                        gr.Markdown("### Model Configuration")

                        speaker_name_input = gr.Textbox(
                            label="Speaker Name",
                            value="speaker",
                            info="Name for the trained voice model",
                            interactive=True
                        )

                        gr.Markdown("### Select Reference")

                        ref_audio_dropdown = gr.Dropdown(
                            choices=[],
                            label="Reference Audio",
                            info="Select one sample from your dataset as reference",
                            interactive=True
                        )

                        ref_audio_preview = gr.Audio(
                            label="Preview",
                            type="filepath",
                            interactive=False
                        )

                        start_training_btn = gr.Button("Start Training", variant="primary", size="lg")

                        train_quick_guide = dedent("""\
                            **Quick Guide:**
                            1. Select dataset folder
                            2. Enter speaker name
                            3. Choose reference audio from dataset
                            4. Configure parameters & start training (defaults work well for most cases)

                            *See Help Guide tab → Train Model for detailed instructions*
                        """)
                        gr.HTML(
                            value=format_help_html(train_quick_guide),
                            container=True,   # give it the normal block/card container
                            padding=True      # match block padding)
                        )

                    # Right column - Training configuration
                    with gr.Column(scale=1):
                        gr.Markdown("### Training Parameters")

                        model_size_radio = gr.Radio(
                            choices=["1.7B (Large)", "0.6B (Small)"],
                            value="1.7B (Large)",
                            label="Base Model Size",
                            info="Larger model = better quality, slower training"
                        )

                        batch_size_slider = gr.Slider(
                            minimum=1,
                            maximum=8,
                            value=2,
                            step=1,
                            label="Batch Size",
                            info="Reduce if you get out of memory errors"
                        )

                        learning_rate_slider = gr.Slider(
                            minimum=1e-6,
                            maximum=1e-4,
                            value=2e-6,
                            label="Learning Rate",
                            info="Default: 2e-6"
                        )

                        num_epochs_slider = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=5,
                            step=1,
                            label="Number of Epochs",
                            info="How many times to train on the full dataset"
                        )

                        save_interval_slider = gr.Slider(
                            minimum=0,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Save Interval (Epochs)",
                            info="Save checkpoint every N epochs (0 = save every epoch)"
                        )

                        training_status = gr.Textbox(
                            label="Status",
                            lines=20,
                            interactive=False
                        )

                # Event handlers for training tab
                def update_ref_audio_dropdown(folder):
                    """Update reference audio dropdown when folder changes."""
                    files = get_dataset_files(folder)
                    return gr.update(choices=files, value=None), None

                def load_ref_audio_preview(folder, filename):
                    """Load reference audio preview."""
                    if not folder or not filename or folder == "(No folders)" or folder == "(Select Dataset)":
                        return None
                    audio_path = DATASETS_DIR / folder / filename
                    if audio_path.exists():
                        return str(audio_path)
                    return None

                train_folder_dropdown.change(
                    update_ref_audio_dropdown,
                    inputs=[train_folder_dropdown],
                    outputs=[ref_audio_dropdown, ref_audio_preview]
                )

                refresh_train_folder_btn.click(
                    lambda: gr.update(choices=["(Select Dataset)"] + get_dataset_folders(), value="(Select Dataset)"),
                    outputs=[train_folder_dropdown]
                )

                ref_audio_dropdown.change(
                    load_ref_audio_preview,
                    inputs=[train_folder_dropdown, ref_audio_dropdown],
                    outputs=[ref_audio_preview]
                )

                start_training_btn.click(
                    train_model,
                    inputs=[
                        train_folder_dropdown,
                        speaker_name_input,
                        ref_audio_dropdown,
                        model_size_radio,
                        batch_size_slider,
                        learning_rate_slider,
                        num_epochs_slider,
                        save_interval_slider
                    ],
                    outputs=[training_status]
                )

            # ============== TAB 9: Help & Guide ==============
            with gr.TabItem("Help Guide"):
                gr.Markdown("# Voice Clone Studio - Help & Guide")

                help_topic = gr.Radio(
                    choices=[
                        "Voice Clone",
                        "Voice Presets",
                        "Conversation",
                        "Voice Design",
                        "Prep Samples",
                        "Finetune Dataset",
                        "Train Model",
                        "Tips & Tricks"
                    ],
                    value="Voice Clone",
                    show_label=False,
                    interactive=True,
                    container=False
                )

                help_content = gr.HTML(
                    value=format_help_html(ui_help.show_voice_clone_help()),
                    container=True,   # give it the normal block/card container
                    padding=True      # match block padding
                )

                # Map radio selection to help function
                def show_help(topic):
                    help_map = {
                        "Voice Clone": ui_help.show_voice_clone_help,
                        "Conversation": ui_help.show_conversation_help,
                        "Voice Presets": ui_help.show_voice_presets_help,
                        "Voice Design": ui_help.show_voice_design_help,
                        "Prep Samples": ui_help.show_prep_samples_help,
                        "Finetune Dataset": ui_help.show_finetune_help,
                        "Train Model": ui_help.show_train_help,
                        "Tips & Tricks": ui_help.show_tips_help
                    }
                    return format_help_html(help_map[topic]())

                # Event handler for radio selection
                help_topic.change(fn=show_help, inputs=help_topic, outputs=help_content)

        # ============== Config Auto-Save ==============
        # Save preferences when users change settings
        def save_preference(key, value):
            _user_config[key] = value
            save_config(_user_config)

        # Register change handlers for preferences
        transcribe_model.change(
            lambda x: save_preference("transcribe_model", x),
            inputs=[transcribe_model],
            outputs=[]
        )

        whisper_language.change(
            lambda x: save_preference("whisper_language", x),
            inputs=[whisper_language],
            outputs=[]
        )

        clone_model_dropdown.change(
            lambda x: save_preference("voice_clone_model", x),
            inputs=[clone_model_dropdown],
            outputs=[]
        )

        custom_model_size.change(
            lambda x: save_preference("custom_voice_size", x),
            inputs=[custom_model_size],
            outputs=[]
        )

        language_dropdown.change(
            lambda x: save_preference("language", x),
            inputs=[language_dropdown],
            outputs=[]
        )

        custom_language.change(
            lambda x: save_preference("language", x),
            inputs=[custom_language],
            outputs=[]
        )

        conv_pause.change(
            lambda x: save_preference("conv_pause_duration", x),
            inputs=[conv_pause],
            outputs=[]
        )

        conv_model_type.change(
            lambda x: save_preference("conv_model_type", x),
            inputs=[conv_model_type],
            outputs=[]
        )

        conv_model_size.change(
            lambda x: save_preference("conv_model_size", x),
            inputs=[conv_model_size],
            outputs=[]
        )

        longform_model_size.change(
            lambda x: save_preference("vibevoice_model_size", x),
            inputs=[longform_model_size],
            outputs=[]
        )

        design_language.change(
            lambda x: save_preference("language", x),
            inputs=[design_language],
            outputs=[]
        )

        conv_language.change(
            lambda x: save_preference("language", x),
            inputs=[conv_language],
            outputs=[]
        )

    return app, theme, custom_css, CONFIRMATION_MODAL_CSS, CONFIRMATION_MODAL_HEAD
if __name__ == "__main__":

    app, theme, custom_css, modal_css, modal_head = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=theme,
        css=custom_css + modal_css,
        head=modal_head
    )
