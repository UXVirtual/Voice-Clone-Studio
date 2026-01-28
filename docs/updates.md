# Version History

## January 27, 2026

#### Version 0.6.5 - Improved Conversation Tool
- **Three Conversation Modes** - Qwen CustomVoice (9 preset speakers), Qwen Base (8 custom voice samples), and VibeVoice (4 custom voice samples)
- **VRAM Optimization** - Automatic model unloading when switching between conversation modes
- **Global Model Management** - Unload All Models button for manual VRAM cleanup
- **Voice Sample Persistence** - Dropdowns remember your selected voice samples across sessions
- **Improved Conversation Options** - Inspired by [ComfyUI-Qwen-TTS](https://github.com/flybirdxx/ComfyUI-Qwen-TTS)

#### Version 0.6.0 - Enhanced Model Support & Settings
- **VibeVoice Large 4-bit** - Added support for quantized 4-bit VibeVoice Large model for reduced VRAM usage
- **Settings Tab** - New centralized settings interface with configurable folder paths
- **Attention Mechanism Selection** - Choose between SAGE (fastest), Flash Attention 2, SDPA, or Eager with automatic fallback
- **Low CPU Memory Option** - Toggle to reduce CPU memory usage during model loading (all models)
- **UI Improvements** - Reorganized Voice Clone tab with conditional visibility and better layout
- **Refresh Voice Samples** - Added button in Conversation tab to refresh voice sample dropdowns

#### Version 0.5.5 - Platform Support & Requirements
- Consolidated requirements into single universal `requirements.txt` with platform markers
- Added platform-specific setup scripts (Windows/Linux)
- Enhanced setup automation

## January 26, 2026

#### Version 0.5.1 - Training Infrastructure
- Removed bundled Qwen3-TTS module (now using PyPI package)
- Added Qwen fine-tuning scripts for custom voice training
- Checkpoint management with configurable save intervals

#### Version 0.5.0 - UI Polish & Help System
- **Help Guide Tab** - Comprehensive in-app documentation with 8 topic sections
- **Modular Help System** - Extracted help content to separate `ui_help.py` module
- **Better Text Formatting** - Markdown rendering with scrollable containers
- gr.HTML styling improvements with container/padding support
- Label color matching enhancements

#### Version 0.4.0 - Custom Voice Training
- Added **Train Model** tab for fine-tuning custom voices
- Complete training pipeline with validation, data preparation, and model training
- **Batch Transcription** - Process 50-100+ audio files in one click
- Support for both 0.6B and 1.7B base models
- Real-time training progress monitoring with live loss values
- Checkpoint management - compare different training epochs
- Integration with Voice Presets tab for using trained models
- Dataset organization system with `datasets/` folder structure
- Automatic audio format conversion (24kHz 16-bit mono)

## January 25, 2026

#### Version 0.3.5 - Style Instructions
- Added Style Instructions support in Conversation for Qwen model (Unsupported by VibeVoice)

#### Version 0.3.0 - Enhanced Media Support
- **Video File Support** - Upload video files (.mp4, .mov, .avi, .mkv, etc.) to Prep Samples tab
- **Automatic Audio Extraction** - Uses ffmpeg to extract audio from video files for voice cloning
- **Improved Workflow** - Added Clear button to quickly reset the audio editor
- Enhanced media handling and file upload capabilities

## January 24, 2026

#### Version 0.2.0 - VibeVoice Integration
- Added **VibeVoice TTS** support for long-form multi-speaker generation (up to 90 minutes)
- Added **VibeVoice ASR** as alternative transcription engine alongside Whisper
- Conversation tab now supports both Qwen (9 preset voices) and VibeVoice (custom samples) engines
- Multi-speaker conversation support with up to 4 custom voices
- Added Output History management
- Removed Clone Design tab

## January 23, 2026

#### Version 0.1.0 - Initial Release
- Voice cloning with Qwen3-TTS (Base, CustomVoice, VoiceDesign models)
- Whisper-powered automatic transcription
- Sample preparation toolkit (trim, normalize, mono conversion)
- Voice prompt caching for faster generation
- Seed control for reproducible outputs
- 9 premium preset voices (Qwen CustomVoice)
- Voice Design from text descriptions
- Conversation mode with multi-speaker support
