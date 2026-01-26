import sys
import os
from pathlib import Path

# Add vendor directories to Python path
VENDOR_DIR = Path(__file__).parent / "vendor"

# Add vendor to path (contains both 'vibevoice_tts' and 'vibevoice_asr' packages)
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

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
import subprocess
from textwrap import dedent
import logging
import traceback
import warnings
import sys
import types

# --- DeepFilterNet / Torchaudio Compatibility Shim ---
try:
    from patches import deepfilternet_torchaudio_patch
    deepfilternet_torchaudio_patch.apply_patches()
except ImportError:
    print("Warning: compatibility_patches module not found. DeepFilterNet may fail to load.")

# Try importing DeepFilterNet
try:
    from df.enhance import enhance, init_df, load_audio, save_audio
    from df.io import load_audio as df_load_audio
    DEEPFILTER_AVAILABLE = True
except ImportError as e:
    # If it still fails with the specific backend error, print guidance
    if "torchaudio.backend" in str(e):
        print(f"⚠ DeepFilterNet failed to load due to torchaudio incompatibility: {e}")
    else:
        print(f"⚠ DeepFilterNet not available: {e}")
    DEEPFILTER_AVAILABLE = False
# -----------------------------------------------------

# Directories
SAMPLES_DIR = Path(__file__).parent / "samples"
OUTPUT_DIR = Path(__file__).parent / "output"
TEMP_DIR = Path(__file__).parent / "temp"
FINETUNED_MODELS_DIR = Path(__file__).parent / "finetuned_models"
CONFIG_FILE = Path(__file__).parent / "config.json"
SAMPLES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
FINETUNED_MODELS_DIR.mkdir(exist_ok=True)

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
_deepfilter_model = None     # DeepFilterNet model for audio cleaning
_deepfilter_state = None     # DeepFilterNet state
_deepfilter_params = None    # DeepFilterNet parameters
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

def load_finetuned_speakers():
    """Load fine-tuned speakers from local directory."""
    if not FINETUNED_MODELS_DIR.exists():
        return
        
    for model_dir in FINETUNED_MODELS_DIR.iterdir():
        if model_dir.is_dir():
            # Check if it has config.json and model.safetensors
            if (model_dir / "config.json").exists() and (model_dir / "model.safetensors").exists():
                 name = model_dir.name
                 if name not in CUSTOM_VOICE_SPEAKERS:
                     CUSTOM_VOICE_SPEAKERS[name] = f"Fine-tuned Qwen3-TTS model: {name}"

# Load fine-tuned speakers on startup
load_finetuned_speakers()

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

    if _deepfilter_model is not None:
        # DeepFilterNet models are small, but good practice to clean up if needed
        # However, they don't have a standard delete/unload method, just relying on GC
        # del _deepfilter_model
        # _deepfilter_model = None
        pass

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


def get_deepfilter_model():
    """Lazy-load the DeepFilterNet model."""
    global _deepfilter_model, _deepfilter_state, _deepfilter_params

    if not DEEPFILTER_AVAILABLE:
        raise ImportError("DeepFilterNet is not available on this system.")

    if _deepfilter_model is None:
        print("Loading DeepFilterNet model...")
        try:
            # Initialize with default settings (DeepFilterNet3)
            # init_df returns (model, df_state, params) in newer versions
            res = init_df()
            if isinstance(res, tuple):
                _deepfilter_model = res[0]
                _deepfilter_state = res[1]
                _deepfilter_params = res[2]
            else:
                _deepfilter_model = res
                _deepfilter_state = None
                _deepfilter_params = None
                
            print("DeepFilterNet model loaded!")
        except Exception as e:
            print(f"❌ Error loading DeepFilterNet: {e}")
            raise e
    
    return _deepfilter_model, _deepfilter_state, _deepfilter_params


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



def get_custom_voice_model(size_or_path="1.7B"):
    """Lazy-load the CustomVoice model or a Fine-Tuned model."""
    global _custom_voice_model, _custom_voice_model_size

    # Unload ASR models before loading TTS
    unload_asr_models()

    target_id = size_or_path
    is_path = "/" in str(target_id) or "\\" in str(target_id)

    # If we need a different size/path, unload current model
    if _custom_voice_model is not None and _custom_voice_model_size != target_id:
        print(f"Switching CustomVoice model from {_custom_voice_model_size} to {target_id}...")
        del _custom_voice_model
        _custom_voice_model = None
        torch.cuda.empty_cache()

    if _custom_voice_model is None:
        if is_path:
             model_name = str(target_id)
             print(f"Loading Fine-Tuned Model from {model_name}...")
        else:
             model_name = f"Qwen/Qwen3-TTS-12Hz-{target_id}-CustomVoice"
             print(f"Loading {model_name}...")

        # Try flash_attention_2 first, fall back to sdpa if not available
        try:
            _custom_voice_model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            print(f"CustomVoice model ({target_id}) loaded with Flash Attention 2!")
        except Exception as e:
            if "flash" in str(e).lower():
                print("Flash Attention 2 not available, using SDPA instead...")
                _custom_voice_model = Qwen3TTSModel.from_pretrained(
                    model_name,
                    device_map="cuda:0",
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                )
                print(f"CustomVoice model ({target_id}) loaded with SDPA!")
            else:
                raise e
        _custom_voice_model_size = target_id
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
            from vibevoice_asr.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
            from vibevoice_asr.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

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
            print("Make sure vendor/vibevoice_asr directory exists and contains the vibevoice_asr module.")
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
            from vibevoice_tts.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
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
            print("Make sure vendor/vibevoice_tts directory exists and contains the vibevoice_tts module.")
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
        return ["(Select a file)"]
    files = sorted(OUTPUT_DIR.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True)
    # Return just filenames instead of full paths
    return ["(Select a file)"] + [f.name for f in files]


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

            from vibevoice_tts.processor.vibevoice_processor import VibeVoiceProcessor
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


def resolve_speaker_from_config(model_path, requested_speaker):
    """Resolve the correct internal speaker name from a model config."""
    p = Path(model_path)
    if not p.exists():
        return requested_speaker
    
    config_path = p / "config.json"
    if not config_path.exists():
        # Sometimes structure is deeper or parallel, but standard finetune has config.json
        return requested_speaker
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        talker_config = config.get("talker_config", {})
        if not talker_config:
            return requested_speaker
            
        valid_speakers = list(talker_config.keys())
        if not valid_speakers:
            return requested_speaker
            
        # 1. Exact match
        if requested_speaker in valid_speakers:
            return requested_speaker
            
        # 2. Case-insensitive match & cleanup
        req_clean = requested_speaker.lower().replace("_tuned", "")
        
        for s in valid_speakers:
            s_lower = s.lower()
            if s_lower == requested_speaker.lower():
                return s
            if s_lower == req_clean:
                return s
                
        # 3. If single speaker model, default to it
        if len(valid_speakers) == 1:
            return valid_speakers[0]
            
        return requested_speaker
        
    except Exception as e:
        print(f"Warning: Failed to resolve speaker from config: {e}")
        return requested_speaker


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
        # Speaker number to name mapping (in order of CUSTOM_VOICE_SPEAKERS)
        speaker_list = list(CUSTOM_VOICE_SPEAKERS.keys())

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

        # Generate all lines
        all_wavs = []
        sr = 24000
        
        PRIMITIVE_SPEAKERS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]

        for i, (speaker, text) in enumerate(lines):
            progress_val = 0.1 + (0.8 * i / len(lines))
            
            # Determine which model to use for this speaker
            target_model_id = model_size
            if speaker not in PRIMITIVE_SPEAKERS:
                 # Check for fine-tuned model path
                 possible_paths = [
                     FINETUNED_MODELS_DIR / f"{speaker}_tuned",
                     FINETUNED_MODELS_DIR / speaker
                 ]
                 for d in possible_paths:
                      if d.exists():
                           checkpoints = sorted([sd for sd in d.iterdir() if sd.is_dir() and "checkpoint" in sd.name], key=lambda x: x.stat().st_mtime)
                           if checkpoints:
                                target_model_id = str(checkpoints[-1])
                           else:
                                if (d / "model.safetensors").exists():
                                     target_model_id = str(d)
                           break

            # Switch Model if needed
            model = get_custom_voice_model(target_model_id)

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
        from vibevoice_tts.processor.vibevoice_processor import VibeVoiceProcessor
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


def run_finetuning(sample_name, num_epochs, batch_size, lr, model_size, progress=gr.Progress()):
    """Run fine-tuning process for a selected sample."""
    if not sample_name:
        return "❌ Please select a voice sample first."

    # Look for the sample
    samples = get_available_samples()
    sample = next((s for s in samples if s["name"] == sample_name), None)
    
    if not sample:
        return f"❌ Sample '{sample_name}' not found."
    
    # Paths
    wav_path = sample["wav_path"]
    ref_text = sample.get("meta", {}).get("Text") or sample.get("ref_text", "")
    
    # Try to load if missing in dict but present in file
    if not ref_text:
        try:
             with open(sample["json_path"], 'r', encoding='utf-8') as f:
                 meta = json.load(f)
                 ref_text = meta.get("Text")
        except:
             pass
    
    if not ref_text:
        return "❌ Sample has no reference text."
        
    finetune_folder = VENDOR_DIR / "qwen3_tts"
    scripts_py = sys.executable

    # Prepare temp jsonl
    raw_jsonl_path = TEMP_DIR / "train_raw.jsonl"
    train_jsonl_path = TEMP_DIR / "train_with_codes.jsonl"
    output_model_dir = FINETUNED_MODELS_DIR / f"{sample_name}_tuned"
    
    # 1. Create Raw JSONL
    try:
        data = {
            "audio": wav_path,
            "text": ref_text,
            "ref_audio": wav_path 
        }
        with open(raw_jsonl_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    except Exception as e:
        return f"❌ Failed to create training data: {e}"

    # 2. Extract Codes (prepare_data.py)
    progress(0.1, desc="Extracting audio codes...")
    
    cmd_prep = [
        scripts_py, 
        str(finetune_folder / "scripts/prepare_data.py"),
        "--input_jsonl", str(raw_jsonl_path),
        "--output_jsonl", str(train_jsonl_path),
        "--device", "cuda:0" if torch.cuda.is_available() else "cpu"
    ]
    
    try:
        result = subprocess.run(cmd_prep, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr)
            return f"❌ Data preparation failed:\n{result.stderr}"
    except Exception as e:
        return f"❌ Error running prepare_data: {e}"

    # 3. Training (sft_12hz.py)
    progress(0.3, desc=f"Fine-tuning Qwen3-TTS ({num_epochs} epochs)...")
    
    # Determine base model correctly
    init_model = "Qwen/Qwen3-TTS-12Hz-1.7B-Base" if "1.7B" in model_size or "Large" in model_size else "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    
    cmd_train = [
        scripts_py, "-u", # Unbuffered output
        str(finetune_folder / "scripts/sft_12hz.py"),
        "--init_model_path", init_model,
        "--output_model_path", str(output_model_dir),
        "--train_jsonl", str(train_jsonl_path),
        "--batch_size", str(int(batch_size)),
        "--lr", str(lr),
        "--num_epochs", str(int(num_epochs)),
        "--speaker_name", sample_name
    ]
    
    try:
        # Use Popen to capture text output in real-time
        process = subprocess.Popen(
            cmd_train,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8' # Force UTF-8 for reading
        )
        
        steps_per_epoch = 1  # default fallback
        current_epoch_idx = 0
        total_epochs_count = int(num_epochs)

        print("\n=== Fine-Tuning Log Start ===")
        for line in process.stdout:
            # Echo to console
            print(line, end="")
            
            line_str = line.strip()
            # Parse custom info
            if "TRAIN_INFO: steps_per_epoch=" in line_str:
                try:
                    steps_per_epoch = int(line_str.split("=")[1])
                except: pass
            
            # Parse progress: "Epoch 0 | Step 10 | Loss: 3.2541"
            # Since sft_12hz.py might print every 10 steps, we update progress accordingly
            if "Epoch" in line_str and "Step" in line_str and "|" in line_str:
                try:
                    parts = line_str.split("|")
                    ep_str = parts[0].strip()   # Epoch X
                    step_str = parts[1].strip() # Step Y
                    
                    e_val = int(ep_str.split()[1])
                    s_val = int(step_str.split()[1])
                    
                    current_epoch_idx = e_val
                    
                    # Calculate overall completion (0.0 to 1.0 within training phase)
                    # Global step = e_val * steps_per_epoch + s_val
                    total_global_steps = total_epochs_count * steps_per_epoch
                    current_global_step = e_val * steps_per_epoch + s_val
                    
                    completion = current_global_step / max(total_global_steps, 1)
                    if completion > 1.0: completion = 1.0
                    
                    # Map 0.0-1.0 to 0.3-1.0 in UI
                    ui_val = 0.3 + (completion * 0.7)
                    
                    progress(ui_val, desc=f"Training: Epoch {e_val+1}/{total_epochs_count} (Step {s_val}/{steps_per_epoch})")
                    
                except Exception:
                    pass

        # Wait for finish
        process.wait()
        print("\n=== Fine-Tuning Log End ===")
        
        if process.returncode != 0:
            return f"❌ Training failed with exit code {process.returncode}. Please check the console log for details."
            
    except Exception as e:
        return f"❌ Error running training: {e}"

    progress(1.0, desc="Done!")
    
    # Reload speakers
    load_finetuned_speakers()
    
    # Look for last checkpoint to confirm
    checkpoints = sorted([d for d in output_model_dir.iterdir() if d.is_dir() and "checkpoint" in d.name], key=lambda x: x.stat().st_mtime)
    
    msg = f"✅ Fine-tuning complete! Created {len(checkpoints)} checkpoints."
    if checkpoints:
         msg += "\nUse the 'Checkpoint Management' section below to preview and select the best epoch."
    
    return msg


def get_training_checkpoints(sample_name):
    """Get list of available checkpoints for a sample."""
    if not sample_name:
        return []
    
    tuned_dir = FINETUNED_MODELS_DIR / f"{sample_name}_tuned"
    if not tuned_dir.exists():
        return []
        
    checkpoints = [d.name for d in tuned_dir.iterdir() if d.is_dir() and "checkpoint" in d.name]
    
    # Sort by epoch number
    def get_epoch_num(name):
        try:
            parts = name.split('-') # checkpoint-epoch-10
            if parts[-1].isdigit():
                return int(parts[-1])
            if parts[-2] == "epoch" and parts[-1].isdigit():
                return int(parts[-1])
            return 9999
        except:
            return 9999

    return sorted(checkpoints, key=get_epoch_num)

def _repair_checkpoint_config(checkpoint_path):
    """Ensure speaker keys in config.json are lowercase (required by Qwen3-TTS)."""
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        return

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        modified = False
        if "talker_config" in config and isinstance(config["talker_config"], dict):
            spk_id = config["talker_config"].get("spk_id", {})
            new_spk_id = {}
            for k, v in spk_id.items():
                if k != k.lower():
                    new_spk_id[k.lower()] = v
                    modified = True
                else:
                    new_spk_id[k] = v
            
            if modified:
                config["talker_config"]["spk_id"] = new_spk_id
                
                # Also fix dialect map
                spk_is_dialect = config["talker_config"].get("spk_is_dialect", {})
                new_dialect = {}
                for k, v in spk_is_dialect.items():
                    if k != k.lower():
                        new_dialect[k.lower()] = v
                    else:
                        new_dialect[k] = v
                config["talker_config"]["spk_is_dialect"] = new_dialect
                
                print(f"Repairing config at {config_path}: Lowercasing speaker keys")
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                    
    except Exception as e:
        print(f"Warning: Failed to repair config at {config_path}: {e}")

def preview_checkpoint(sample_name, checkpoint_name, text, seed):
    """Generate audio using a specific fine-tuning checkpoint."""
    if not sample_name or not checkpoint_name:
        return None, "❌ Select a checkpoint first."
        
    checkpoint_path = FINETUNED_MODELS_DIR / f"{sample_name}_tuned" / checkpoint_name
    
    if not checkpoint_path.exists():
        return None, f"❌ Checkpoint not found: {checkpoint_path}"
        
    # Attempt to repair config if needed (fix speaker name casing)
    _repair_checkpoint_config(checkpoint_path)

    try:
        # Load model directly from checkpoint path
        model = get_custom_voice_model(str(checkpoint_path)) 
        
        torch.manual_seed(int(seed) if seed else 42)
        
        # Determine valid speaker ID - for single-speaker fine-tunes it's often the sample name or mapped in config
        # But get_custom_voice_model with path loads it as a generic model
        # The finetuned model has speaker config inside.
        # We can pass "Reference" or "Speaker 0" or let the model handle defaults.
        # Qwen3TTSModel usually needs a speaker name that exists in the config.
        # Our training script sets speaker_name = sample_name.
        speaker_name = sample_name
        
        wavs, sr = model.generate_custom_voice(
            text=text,
            language="Auto",
            speaker=speaker_name 
        )
        
        # Save temp
        timestamp = datetime.now().strftime("%H%M%S")
        out_file = TEMP_DIR / f"preview_{sample_name}_chk_{timestamp}.wav"
        sf.write(str(out_file), wavs[0], sr)
        
        return str(out_file), f"✅ Preview generated from {checkpoint_name}"
    except Exception as e:
        return None, f"❌ Preview failed: {e}\n(Tip: Ensure speaker name matches training)"

def finalize_checkpoint(sample_name, checkpoint_name, cleanup):
    """Promote a checkpoint to be the active fine-tuned model."""
    if not sample_name or not checkpoint_name:
        return "❌ Select a checkpoint first."
        
    tuned_dir = FINETUNED_MODELS_DIR / f"{sample_name}_tuned"
    source_dir = tuned_dir / checkpoint_name
    
    if not source_dir.exists():
        return f"❌ Checkpoint source not found: {source_dir}"
        
    # Attempt to repair config if needed
    _repair_checkpoint_config(source_dir)

    try:
        # Determine strict compatible folder name from config
        target_folder_name = f"{sample_name}_tuned"
        config_path = source_dir / "config.json"
        if config_path.exists():
             try:
                 with open(config_path, 'r', encoding='utf-8') as f:
                     config = json.load(f)
                 talker_config = config.get("talker_config", {})
                 # If only one speaker, use that name as folder name
                 if len(talker_config) == 1:
                     target_folder_name = list(talker_config.keys())[0]
             except Exception as e:
                 print(f"Warning: Could not read speaker name from config: {e}")

        # Ensure target directory exists
        # We might be renaming the project essentially
        target_dir = FINETUNED_MODELS_DIR / target_folder_name
        
        # If the target directory allows, ensure it's clean or just overwrite root files
        target_dir.mkdir(exist_ok=True)

        files_moved = []
        for item in source_dir.iterdir():
             dest = target_dir / item.name
             if item.is_file():
                 shutil.copy2(item, dest)
                 files_moved.append(item.name)
             elif item.is_dir():
                 if dest.exists():
                     shutil.rmtree(dest)
                 shutil.copytree(item, dest)
                 files_moved.append(item.name + "/")

        msg = f"✅ Checkpoint {checkpoint_name} promoted to main model."
        
        if cleanup:
            try:
                # Remove the original tuned folder if we moved to a new name
                original_tuned_dir = FINETUNED_MODELS_DIR / f"{sample_name}_tuned"
                
                # Check if we moved to a different folder
                if original_tuned_dir.exists() and original_tuned_dir.resolve() != target_dir.resolve():
                    shutil.rmtree(original_tuned_dir)
                else:
                    # If we stayed in same folder, just remove checkpoints
                    count = 0
                    for item in target_dir.iterdir():
                        if item.is_dir() and "checkpoint" in item.name:
                            shutil.rmtree(item)
                            count += 1
                    if count:
                        msg += f"\n🗑️ Cleanup: Removed {count} checkpoint folders."

            except Exception as cleanup_error:
                print(f"Cleanup warning: {cleanup_error}")
            
        # Force unload to refresh
        global _custom_voice_model
        _custom_voice_model = None
        torch.cuda.empty_cache()
        
        # Reload speakers registry
        load_finetuned_speakers()
        
        return msg
        
    except Exception as e:
        return f"❌ Finalize failed: {e}"


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
    return gr.update(choices=files, value=files[0] if files else None)


def load_output_audio(file_path):
    """Load a selected output file for playback and show metadata."""
    if file_path and file_path != "(Select a file)":
        # Convert filename to full path if needed
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


def clean_audio(audio_file, progress=gr.Progress()):
    """Clean audio using DeepFilterNet."""
    if audio_file is None:
        return None

    if not DEEPFILTER_AVAILABLE:
        print("DeepFilterNet not installed. Skipping cleaning.")
        return audio_file

    try:
        progress(0.1, desc="Loading Audio Cleaner...")
        df_model, df_state, df_params = get_deepfilter_model()
        
        # Get sample rate from params or use default
        target_sr = df_params.sr if df_params is not None and hasattr(df_params, 'sr') else 48000

        progress(0.3, desc="Processing audio...")
        
        # Load audio using DeepFilterNet's loader
        # This returns audio tensor and sample rate
        audio, _ = df_load_audio(audio_file, sr=target_sr)
        
        # Run enhancement
        # enhance method expects audio tensor and model
        enhanced_audio = enhance(df_model, df_state=df_state, audio=audio)
        
        # Save output
        timestamp = datetime.now().strftime("%H%M%S")
        output_path = TEMP_DIR / f"cleaned_{timestamp}.wav"
        
        # Save using DeepFilterNet's save function
        save_audio(str(output_path), enhanced_audio, target_sr)
        
        progress(1.0, desc="Done!")
        return str(output_path)

    except Exception as e:
        print(f"Error cleaning audio: {e}")
        # Return original if cleaning fails
        return audio_file


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


def get_speaker_table_markdown():
    """Generate markdown table for available speakers including fine-tuned ones."""
    base_md = """
    **Qwen Speaker Numbers → Preset Voices:**

    | # | Speaker | Voice | Language |
    |---|---------|-------|----------|
    | 1 | Vivian | Bright young female | 🇨🇳 Chinese |
    | 2 | Serena | Warm gentle female | 🇨🇳 Chinese |
    | 3 | Uncle_Fu | Seasoned mellow male | 🇨🇳 Chinese |
    | 4 | Dylan | Youthful Beijing male | 🇨🇳 Chinese |
    | 5 | Eric | Lively Chengdu male | 🇨🇳 Chinese |
    | 6 | Ryan | Dynamic male | 🇺🇸 English |
    | 7 | Aiden | Sunny American male | 🇺🇸 English |
    | 8 | Ono_Anna | Playful female | 🇯🇵 Japanese |
    | 9 | Sohee | Warm female | 🇰🇷 Korean |
    """
    
    # Add fine-tuned
    finetuned = []
    primitive_keys = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]
    
    idx = 10
    for name in CUSTOM_VOICE_SPEAKERS:
        if name not in primitive_keys:
            finetuned.append(f"| {idx} | {name} | Fine-tuned Model | Custom |")
            idx += 1
            
    if finetuned:
        base_md += "\n" + "\n".join(finetuned)
        
    base_md += "\n\n*Each speaker works best in their native language.*"
    return base_md

def get_custom_speaker_choices():
    return [f"{name} - {desc}" for name, desc in CUSTOM_VOICE_SPEAKERS.items()]

def refresh_speakers_ui():
    """Reload speakers and returns updated UI components."""
    load_finetuned_speakers()
    return get_speaker_table_markdown(), gr.update(choices=get_custom_speaker_choices())

def create_ui():
    """Create the Gradio interface."""

    with gr.Blocks(title="Voice Clone Studio") as app:
        gr.Markdown("""
        # 🎙️ Voice Clone Studio
        <p style="font-size: 0.9em; color: #ffffff; margin-top: -10px;">  Powered by Qwen3-TTS, VibeVoice and Whisper</p>
        """)

        with gr.Tabs():
            # ============== TAB 1: Voice Clone ==============
            with gr.TabItem("Voice Clone"):
                gr.Markdown("""
                ### Clone Voices from Your Samples

                Select a prepared voice sample and generate speech in that voice. Use the Prep Samples tab to add or edit your samples.
                """)
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
                            lines=3
                        )

                        sample_info = gr.Textbox(
                            label="Info",
                            interactive=False
                        )

                    # Right column - Generation (2/3 width)
                    with gr.Column(scale=2):
                        gr.Markdown("### Generate Speech")

                        text_input = gr.Textbox(
                            label="Text to Generate",
                            placeholder="Enter the text you want to speak in the cloned voice...",
                            lines=4
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

                        status_text = gr.Textbox(label="Status", interactive=False)

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

            # ============== TAB 2: Unified Conversation ==============
            with gr.TabItem("Conversation"):
                gr.Markdown("""
                ### Create Multi-Speaker Conversations

                Choose between **Qwen** (preset voices with support for Style Instructions) or **VibeVoice** (high-quality, custom voices, up to 90 minutes).
                """)

                # Model selector at top
                initial_conv_model = _user_config.get("conv_model_type", "Qwen")
                is_qwen_initial = initial_conv_model == "Qwen"

                with gr.Row():
                    conv_model_type = gr.Radio(
                        choices=["Qwen", "VibeVoice"],
                        value=initial_conv_model,
                        label="TTS Engine"
                    )

                with gr.Row():
                    # Left - Script input and model-specific controls
                    with gr.Column(scale=2):
                        gr.Markdown("### Conversation Script")

                        conversation_script = gr.Textbox(
                            label="Script",
                            placeholder=dedent("""\
                                Use [N]: format for speaker labels. Add (style) for emotions:

                                [1]: (cheerful) Hey, how's it going?
                                [2]: (excited) I'm doing great, thanks for asking!
                                [1]: That's wonderful to hear.
                                [3]: (curious) Mind if I join this conversation?

                                Style instructions work with Qwen only (VibeVoice ignores them)."""),
                            lines=12,
                            info="One line per speaker turn. Format: [N]: Text with (optional style) for Qwen."
                        )

                        # Qwen speaker mapping (visible when Qwen selected)
                        with gr.Column(visible=is_qwen_initial) as qwen_speaker_col:
                            with gr.Row(): 
                                gr.Markdown("### Available Speakers")
                                refresh_spk_btn = gr.Button("🔄 Refresh", size="sm")
                            
                            qwen_speaker_table = gr.Markdown(get_speaker_table_markdown())

                        # VibeVoice voice sample selectors (visible when VibeVoice selected)
                        with gr.Column(visible=not is_qwen_initial) as vibevoice_voices_section:
                            gr.Markdown("### Voice Samples (Up to 4 Speakers)")
                            gr.Markdown("**[1]** = Voice Sample 1, **[2]** = Sample 2, etc. Numbers beyond 4 wrap around (5→1, 6→2, etc.)")

                            with gr.Row():
                                voice_sample_1 = gr.Dropdown(
                                    choices=get_sample_choices(),
                                    label="[1] Voice Sample (Required)",
                                    info="Select from your prepared samples"
                                )
                                voice_sample_2 = gr.Dropdown(
                                    choices=get_sample_choices(),
                                    label="[2] Voice Sample (Optional)",
                                    info="Select from your prepared samples"
                                )

                            with gr.Row():
                                voice_sample_3 = gr.Dropdown(
                                    choices=get_sample_choices(),
                                    label="[3] Voice Sample (Optional)",
                                    info="Select from your prepared samples"
                                )
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
                                info="Small = 0.6B (faster), Large = 1.7B (better quality)"
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
                                info="Small = 1.5B (faster), Large = more stable"
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
                        conv_status = gr.Textbox(label="Status", interactive=False, lines=3)

                        # Model-specific tips
                        qwen_tips = gr.Markdown("""
                        **Qwen Tips:**
                        - Fast generation with preset voices
                        - Up to 9 preset speakers + fine-tuned ones
                        - Each voice optimized for their native language
                        """, visible=is_qwen_initial)

                        vibevoice_tips = gr.Markdown("""
                        **VibeVoice Tips:**
                        - Up to 90 minutes continuous generation
                        - Up to 4 speakers with custom voices
                        - May spontaneously add background music/sounds
                        - Longer scripts work best with Large model
                        """, visible=not is_qwen_initial)

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
                    outputs=[qwen_speaker_col, vibevoice_voices_section, qwen_settings, vibevoice_settings, qwen_tips, vibevoice_tips]
                )
                
            # ============== TAB 3: Custom Voice ==============
            with gr.TabItem("Voice Presets"):
                gr.Markdown("""
                ### Generate with Qwen3-TTS Preset Voices

                Use pre-built premium voices with optional style instructions. These voices models support instruction-based style control (emotion, tone, speed, etc.).
                """)

                with gr.Row():
                    # Left - Speaker selection
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎤 Select Speaker")

                        # Create speaker choices with descriptions
                        with gr.Row():
                             custom_speaker_dropdown = gr.Dropdown(
                                choices=get_custom_speaker_choices(),
                                label="Speaker",
                                info="Choose a premium voice",
                                scale=3
                             )
                             refresh_presets_btn = gr.Button("🔄", size="sm", scale=0)

                        refresh_presets_btn.click(
                            refresh_speakers_ui,
                            outputs=[qwen_speaker_table, custom_speaker_dropdown]
                        )

                        gr.Markdown("""
                        **Available Speakers:**

                        | Speaker | Voice | Language |
                        |---------|-------|----------|
                        | Vivian | Bright young female | 🇨🇳 Chinese |
                        | Serena | Warm gentle female | 🇨🇳 Chinese |
                        | Uncle_Fu | Seasoned mellow male | 🇨🇳 Chinese |
                        | Dylan | Youthful Beijing male | 🇨🇳 Chinese |
                        | Eric | Lively Chengdu male | 🇨🇳 Chinese |
                        | Ryan | Dynamic male | 🇺🇸 English |
                        | Aiden | Sunny American male | 🇺🇸 English |
                        | Ono_Anna | Playful female | 🇯🇵 Japanese |
                        | Sohee | Warm female | 🇰🇷 Korean |

                        *Tip: Each speaker works best in their native language but can speak any supported language.*
                        """)

                    # Right - Generation
                    with gr.Column(scale=2):
                        gr.Markdown("### Generate Speech")

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
                            custom_model_size = gr.Dropdown(
                                choices=MODEL_SIZES_CUSTOM,
                                value=_user_config.get("custom_voice_size", "Large"),
                                label="Model",
                                info="Small = faster, Large = better quality",
                                scale=1
                            )

                        custom_generate_btn = gr.Button("Generate Audio", variant="primary", size="lg")

                        custom_output_audio = gr.Audio(
                            label="Generated Audio",
                            type="filepath"
                        )
                        custom_status = gr.Textbox(label="Status", lines=3, interactive=False)

                # Custom Voice event handlers
                def extract_speaker_name(selection):
                    """Extract speaker name from dropdown selection."""
                    if not selection:
                        return None
                    return selection.split(" - ")[0]

                custom_generate_btn.click(
                    lambda text, lang, speaker_sel, instruct, seed, model_size, progress=gr.Progress(): generate_custom_voice(
                        text, lang, extract_speaker_name(speaker_sel), instruct, seed,
                        "1.7B" if model_size == "Large" else "0.6B",  # Map UI labels to actual model sizes
                        progress
                    ),
                    inputs=[custom_text_input, custom_language, custom_speaker_dropdown, custom_instruct_input, custom_seed, custom_model_size],
                    outputs=[custom_output_audio, custom_status]
                )

            # ============== TAB 4: Voice Design ==============
            with gr.TabItem("Voice Design"):
                gr.Markdown("""
                ### Design a Voice with Natural Language

                Describe the voice characteristics you want (age, gender, emotion, tone, accent)
                and the model will generate speech matching that description. Save designs you like for reuse!
                """)

                with gr.Row():
                    with gr.Column(scale=2):
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
                        design_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column(scale=1):
                        gr.Markdown("### Preview & Save")
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

                        design_save_btn = gr.Button("Save Design", variant="secondary")
                        design_save_status = gr.Textbox(label="Save Status", interactive=False)

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
                    outputs=[design_save_status]
                )

            # ============== TAB 5: Fine-tune ==============
            with gr.TabItem("Fine-tune"):
                gr.Markdown("""
                ### Fine-tune Qwen3-TTS
                
                Create a permanent fine-tuned model from a sample. This is recommended to improve quality and stability for a specific voice.
                """)
                
                # Logic to pre-calculate values for better UX on reload
                ft_choices = get_sample_choices()
                ft_val = ft_choices[0] if ft_choices else None
                
                ft_cps = get_training_checkpoints(ft_val) if ft_val else []
                ft_cp_val = ft_cps[-1] if ft_cps else None
                
                with gr.Row():
                    with gr.Column(scale=1):
                        ft_sample_dropdown = gr.Dropdown(
                            choices=ft_choices,
                            value=ft_val,
                            label="Select Voice Sample",
                            info="Choose a sample to fine-tune on."
                        )
                        refresh_ft_btn = gr.Button("Refresh Samples", size="sm")

                        ft_model_size_radio = gr.Radio(
                            choices=["Small (0.6B)", "Large (1.7B)"],
                            value="Large (1.7B)",
                            label="Base Model Size"
                        )
                        
                        with gr.Accordion("Advanced Settings", open=False):
                            ft_epochs = gr.Number(value=10, label="Epochs", precision=0)
                            ft_batch_size = gr.Number(value=2, label="Batch Size", precision=0)
                            ft_lr = gr.Number(value=2e-5, label="Learning Rate")

                        ft_start_btn = gr.Button("Start Fine-tuning", variant="primary")

                        gr.Markdown("---")
                        gr.Markdown("### 🏁 Checkpoint Management")
                        
                        ft_checkpoints_dropdown = gr.Dropdown(
                            choices=ft_cps,
                            value=ft_cp_val,
                            label="Available Checkpoints",
                            info="Select an epoch to test.",
                            interactive=True
                        )
                        refresh_cp_btn = gr.Button("Refresh Checkpoints", size="sm")

                        ft_preview_text = gr.Textbox(
                            label="Preview Text",
                            value="This is a test of the fine-tuned voice.",
                            lines=1
                        )
                        ft_preview_btn = gr.Button("Preview Audio", size="sm")
                        ft_preview_audio = gr.Audio(label="Checkpoint Preview", interactive=False)
                        ft_preview_status = gr.Textbox(label="Preview Status", lines=1)

                        gr.Markdown("#### Finalize")
                        ft_cleanup_chk = gr.Checkbox(label="Delete other checkpoints after finalizing", value=True)
                        ft_finalize_btn = gr.Button("Use This Checkpoint", variant="secondary")
                        ft_final_status = gr.Textbox(label="Finalization Status", lines=2)

                    with gr.Column(scale=1):
                         ft_status_box = gr.Textbox(label="Status / Output", lines=20, interactive=False)
                
                refresh_ft_btn.click(
                    lambda: gr.update(choices=get_sample_choices()),
                    outputs=[ft_sample_dropdown]
                )

                def update_cp_dropdown(sample):
                     chex = get_training_checkpoints(sample)
                     return gr.update(choices=chex, value=chex[-1] if chex else None)
                
                ft_sample_dropdown.change(
                    update_cp_dropdown,
                    inputs=[ft_sample_dropdown],
                    outputs=[ft_checkpoints_dropdown]
                )

                refresh_cp_btn.click(
                    update_cp_dropdown,
                    inputs=[ft_sample_dropdown],
                    outputs=[ft_checkpoints_dropdown]
                )
                
                ft_start_btn.click(
                    run_finetuning,
                    inputs=[ft_sample_dropdown, ft_epochs, ft_batch_size, ft_lr, ft_model_size_radio],
                    outputs=[ft_status_box]
                ).success(
                    update_cp_dropdown,
                    inputs=[ft_sample_dropdown],
                    outputs=[ft_checkpoints_dropdown]
                )

                ft_preview_btn.click(
                    lambda s, c, t: preview_checkpoint(s, c, t, -1),
                    inputs=[ft_sample_dropdown, ft_checkpoints_dropdown, ft_preview_text],
                    outputs=[ft_preview_audio, ft_preview_status]
                )

                ft_finalize_btn.click(
                    finalize_checkpoint,
                    inputs=[ft_sample_dropdown, ft_checkpoints_dropdown, ft_cleanup_chk],
                    outputs=[ft_final_status]
                )

            # ============== TAB 6: Prep Samples ==============
            with gr.TabItem("Prep Samples"):
                gr.Markdown("""
                ### Prepare Voice Samples

                Load, trim, edit, transcribe, and manage your voice samples. This is your workspace for preparing
                reference audio for voice cloning.
                """)

                with gr.Row():
                    # Left column - Existing samples browser
                    with gr.Column(scale=1):
                        gr.Markdown("### 📚 Existing Samples")

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
                            delete_sample_btn = gr.Button("Delete", size="sm", variant="stop")

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
                            clean_btn = gr.Button("AI Denoise", size="sm", variant="secondary")
                            if not DEEPFILTER_AVAILABLE:
                                clean_btn.interactive = False
                                clean_btn.value = "AI Denoise (Not Installed)"

                        prep_audio_info = gr.Textbox(
                            label="Audio Info",
                            interactive=False
                        )
                        with gr.Column(scale=2):
                            gr.Markdown("### 💬 Transcription / Reference Text")
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

                gr.Markdown("---")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Info")
                        save_status = gr.Textbox(label="Save Status", interactive=False, scale=1)
                    with gr.Column(scale=2):
                        # Save as new sample
                        gr.Markdown("### Save as New Sample")
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

                # Clean audio
                clean_btn.click(
                    clean_audio,
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
                gr.Markdown("""
                ### Browse Previous Outputs

                View, play back, and manage your previously generated audio files.
                """)
                gr.Markdown("### 📂 Output History")

                with gr.Row():
                    output_dropdown = gr.Dropdown(
                        choices=get_output_files(),
                        label="Previous Outputs",
                        info="Select a previously generated file to play",
                        scale=20
                    )
                    with gr.Column(scale=0):
                        load_output_btn = gr.Button("Load", size="sm")
                        refresh_outputs_btn = gr.Button("Refresh", size="sm")
                        delete_output_btn   = gr.Button("Delete", size="sm")

                history_audio = gr.Audio(
                    label="Playback",
                    type="filepath"
                )

                history_metadata = gr.Textbox(
                    label="Generation Info",
                    interactive=False,
                    lines=5
                )

                def delete_output_file(selected_file):
                    if not selected_file or selected_file == "(Select a file)":
                        return gr.update(), gr.update(value=None), gr.update(value="❌ No file selected.")
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

                        # Set to None placeholder to force reset
                        return gr.update(choices=choices, value="(Select a file)"), gr.update(value=None), gr.update(value=msg)
                    except Exception as e:
                        return gr.update(), gr.update(value=None), gr.update(value=f"❌ Error: {str(e)}")
                delete_output_btn.click(
                    delete_output_file,
                    inputs=[output_dropdown],
                    outputs=[output_dropdown, history_audio, history_metadata]
                )

                refresh_outputs_btn.click(
                    refresh_outputs,
                    outputs=[output_dropdown]
                )

                load_output_btn.click(
                    load_output_audio,
                    inputs=[output_dropdown],
                    outputs=[history_audio, history_metadata]
                )

                # Simple load on dropdown change or button click
                output_dropdown.change(
                    load_output_audio,
                    inputs=[output_dropdown],
                    outputs=[history_audio, history_metadata]
                )

        # ============== Config Auto-Save ==============
        # Cross-Tab Event Handlers
        # We define this here to ensure all components (like custom_speaker_dropdown) are defined
        refresh_spk_btn.click(
            refresh_speakers_ui,
            outputs=[qwen_speaker_table, custom_speaker_dropdown]
        )

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

        gr.Markdown("""
        ---
        **Tips:**
        - **Voice Clone**    : Clone from your own audio samples.
        - **Conversation**   : Create multi-speaker dialogues with Qwen3-TTS or VibeVoice.
        - **Voice Presets**  : Use Qwen premium pre-built voices with style control (emotion, tone, speed)
        - **Voice Design**   : Create voices from text descriptions, save designs you like.
        - **Prep Samples**   : Trim, clean, and transcribe audio and save as voice samples.
        - **Output History** : Browse, play, and manage your generated audio files.
        - ⚡ **Voice prompts are cached!** First generation processes the sample, subsequent ones are faster
        - 💾 **Your preferences are auto-saved!** Model choices persist across sessions
        """)

    return app


if __name__ == "__main__":
    print(f"Samples directory: {SAMPLES_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Found {len(get_sample_choices())} samples")

    app = create_ui()
    app.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft()
    )
