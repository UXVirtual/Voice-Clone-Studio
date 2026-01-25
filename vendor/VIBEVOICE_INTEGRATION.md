# VibeVoice Integration

This document explains the VibeVoice ASR and TTS integration in Voice Clone Studio.

## Overview

Voice Clone Studio now includes two VibeVoice models:

### 1. VibeVoice ASR (7B) - Speech Recognition
- **Source:** https://github.com/microsoft/VibeVoice
- **Installation:** From GitHub via requirements.txt
- **Features:**
  - 60-minute single-pass transcription
  - Automatic speaker diarization ([Speaker 0], [Speaker 1], etc.)
  - Multilingual (100+ languages)
  - Timestamping and speaker tracking

### 2. VibeVoice TTS (1.5B/Large) - Long-Form Text-to-Speech
- **Source:** Vendored from https://github.com/rsxdalv/VibeVoice
- **Models:** `FranckyB/VibeVoice-1.5B` and `FranckyB/VibeVoice-Large` on HuggingFace
- **Features:**
  - Up to 90 minutes continuous generation
  - Up to 4 distinct speakers
  - Spontaneous background music/sounds
  - Cross-lingual support

## Installation

All dependencies are handled automatically:

```bash
# Install from requirements.txt
pip install -r requirements.txt
```

This installs:
- `qwen-tts` - Base voice cloning system
- `vibevoice[asr]` - ASR from GitHub
- `vendor/vibevoice-tts/` - TTS (vendored)

## Usage

### ASR (Prep Samples Tab)
1. Load audio file
2. Select "VibeVoice ASR" from dropdown
3. Click "Transcribe Audio"
4. Output includes speaker labels: `[Speaker 0] text...`
5. When saving as sample, speaker labels are automatically removed

### TTS (Long-Form TTS Tab)
1. Enter long-form script (up to 90 minutes worth)
2. Upload 1-4 voice samples (3-10 seconds each)
3. Select model size (1.5B or Large)
4. Adjust CFG scale (3.0 recommended)
5. Click "Generate Long-Form Audio"
6. Wait for generation (can take several minutes)

## VRAM Management

The app automatically manages VRAM:
- Loading ASR → Unloads all TTS models
- Loading TTS → Unloads all ASR models
- VibeVoice ASR: ~14GB VRAM (7B model)
- VibeVoice TTS: ~3GB (1.5B) or ~6GB (Large)
- Qwen3-TTS: ~3GB (1.7B) or ~1GB (0.6B)

## Model Hosting

### ASR
- Hosted by Microsoft on HuggingFace
- Downloaded automatically on first use
- Cached to `~/.cache/huggingface/`

### TTS
- Hosted by FranckyB on HuggingFace
- Downloaded automatically on first use
- Uses deduplication (fast upload/download)

## Architecture

```
Voice Clone Studio
├── Qwen3-TTS (Base/VoiceDesign/CustomVoice)
│   └── Voice cloning, design, premium voices
├── VibeVoice ASR
│   └── Transcription with speaker diarization
└── VibeVoice TTS (Vendored)
    └── Long-form multi-speaker generation
```

## Future-Proofing

### Why Vendored?
Microsoft removed TTS code from their repo due to misuse. To prevent breakage:
- TTS code is vendored in `vendor/vibevoice-tts/`
- Models hosted on personal HuggingFace account
- Full control and independence

### Updating Models
If you need to re-upload models:

```bash
# Login to HuggingFace
hf login

# Upload model
hf upload FranckyB/VibeVoice-1.5B /path/to/model/ --create
```

## Tips

### For Best ASR Results:
- Use VibeVoice ASR for long audio (up to 60 min)
- Use Whisper for short clips with language hints
- Speaker labels are preserved in transcription display
- Labels removed when saving as voice sample

### For Best TTS Results:
- Use Large model for longer scripts (more stable)
- Voice samples should be 3-10 seconds of clear speech
- Model may add background music spontaneously
- Higher CFG scale = more adherence to voice samples
- Generation can take 5-15 minutes for long scripts

## Troubleshooting

### "VibeVoice ASR not available"
```bash
pip install git+https://github.com/microsoft/VibeVoice.git#egg=vibevoice[asr]
```

### "VibeVoice TTS not available"
```bash
pip install -e vendor/vibevoice-tts/
```

### CUDA Out of Memory
- Close other GPU applications
- Use smaller model sizes (0.6B for Qwen, 1.5B for VibeVoice)
- Models automatically unload each other to free VRAM

### Model Download Fails
- Check internet connection
- Verify HuggingFace is accessible
- Models cached to `~/.cache/huggingface/`
- Delete cache and retry if corrupted

## License

Both VibeVoice components are MIT licensed:
- Original copyright: Microsoft Corporation
- Our modifications: Same MIT license
- Attribution preserved in vendor/vibevoice-tts/LICENSE

## Credits

- **Microsoft** - Original VibeVoice development
- **rsxdalv** - TTS fork preservation
- **Qwen Team** - Qwen3-TTS models
- **FranckyB** - Model hosting on HuggingFace
