# Voice Clone Studio

A Gradio-based web UI for voice cloning and voice design, powered by [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) and [VibeVoice](https://github.com/microsoft/VibeVoice).
Supports both Whisper or VibeVoice-asr for automatic Transcription.

![Voice Clone Studio](https://img.shields.io/badge/Voice%20Clone%20Studio-Powered%20by%20Qwen3--TTS-blue) ![VibeVoice](https://img.shields.io/badge/VibeVoice-%20TTS-green) ![VibeVoice](https://img.shields.io/badge/VibeVoice-%20ASR-green)

## Features

### Voice Clone
Clone voices from your own audio samples. Just provide a short reference audio clip with its transcript, and generate new speech in that voice.
**Choose Your Engine:**
- **Qwen Small/Fast or VibeVoice Small/Fast** - 

- **Voice prompt caching** - First generation processes the sample, subsequent ones are instant
- **Seed control** - Reproducible results with saved seeds
- **Metadata tracking** - Each output saves generation info (sample, seed, text)

### Conversation
Create multi-speaker dialogues using either Qwen's premium voices or your own custom voice samples using VibeVoice:

**Choose Your Engine:**
- **Qwen** - Fast generation with 9 preset voices, optimized for their native languages
- **VibeVoice** - High-quality custom voices, up to 90 minutes continuous, perfect for podcasts/audiobooks

**Unified Script Format:**
Write scripts using `[N]:` format - works seamlessly with both engines:
```
[1]: Hey, how's it going?
[2]: I'm doing great, thanks for asking!
[3]: Mind if I join this conversation?
```

**Qwen Mode:**
- Mix any of the 9 premium speakers
- Adjustable pause duration between lines
- Fast generation with cached prompts

**Speaker Mapping:**
- [1] = Vivian, [2] = Serena, [3] = Uncle_Fu, [4] = Dylan, [5] = Eric
- [6] = Ryan, [7] = Aiden, [8] = Ono_Anna, [9] = Sohee

**VibeVoice Mode:**
- **Up to 90 minutes** of continuous speech
- **Up to 4 distinct speakers** using your own voice samples
- Cross-lingual support
- May spontaneously add background music/sounds for realism
- Numbers beyond 4 wrap around (5→1, 6→2, 7→3, 8→4, etc.)

Perfect for:
- Podcasts
- Audiobooks
- Long-form conversations
- Multi-speaker narratives

**Models:**
- **Small** - Faster generation (Qwen: 0.6B, VibeVoice: 1.5B)
- **Large** - Best quality (Qwen: 1.7B, VibeVoice: Large model)


### Voice Presets
Generate with premium pre-built voices with optional style instructions using Qwen3-TTS Custom Model:

| Speaker | Description | Language |
|---------|-------------|----------|
| Vivian | Bright, slightly edgy young female | Chinese |
| Serena | Warm, gentle young female | Chinese |
| Uncle_Fu | Seasoned male, low mellow timbre | Chinese |
| Dylan | Youthful Beijing male, clear natural | Chinese (Beijing) |
| Eric | Lively Chengdu male, husky brightness | Chinese (Sichuan) |
| Ryan | Dynamic male, strong rhythmic drive | English |
| Aiden | Sunny American male, clear midrange | English |
| Ono_Anna | Playful Japanese female, light nimble | Japanese |
| Sohee | Warm Korean female, rich emotion | Korean |

- Style instructions supported (emotion, tone, speed)
- Each speaker works best in native language but supports all

### Voice Design
Create voices from natural language descriptions - no audio needed, using Qwen3-TTS Voice Design Model:

- Describe age, gender, emotion, accent, speaking style
- Generate unique voices matching your description

### Train Custom Voices
Fine-tune your own custom voice models with your training data:

- **Dataset Management** - Organize training samples in the `datasets/` folder
- **Audio Preparation** - Auto-converts to 24kHz 16-bit mono format
- **Training Pipeline** - Complete 3-step workflow (validation → extract codes → train)
- **Epoch Selection** - Compare different training checkpoints
- **Live Progress** - Real-time training logs and loss monitoring
- **Voice Presets Integration** - Use trained models alongside premium speakers

**Requirements:**
- CUDA GPU required
- Multiple audio samples with transcripts
- Training time: ~10-30 minutes depending on dataset size

**Workflow:**
1. Prepare audio files (WAV/MP3) and organize in `datasets/YourSpeakerName/` folder
2. Use **Batch Transcribe** to automatically transcribe all files at once
3. Review and edit individual transcripts as needed
4. Configure training parameters (model size, epochs, learning rate)
5. Monitor training progress in real-time
6. Use trained model in Voice Presets tab

### Prep Samples
Full audio preparation workspace:

- **Trim** - Use waveform selection to cut audio
- **Normalize** - Balance audio levels
- **Convert to Mono** - Ensure single-channel audio
- **Transcribe** - Whisper or VibeVoice ASR automatic transcription
- **Batch Transcribe** - Process entire folders of audio files at once
- **Save as Sample** - One-click sample creation

### Output History
View, play back, and manage your previously generated audio files.

---
## Installation

### Prerequisites

- Python 3.12+ (recommended for all platforms)
- CUDA-compatible GPU (recommended: 8GB+ VRAM)
- **SOX**  (Sound eXchange) - Required for audio processing
- **FFMPEG** - Multimedia framework required for audio format conversion
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) (optional but recommended)

**Note for Linux users:** The Linux installation skips `openai-whisper` (compatibility issues). VibeVoice ASR is used for transcription instead.

**Experiencing installation issues on Linux?** See [LINUX_INSTALL.md](LINUX_INSTALL.md) for detailed troubleshooting.

### Setup

#### Quick Setup (Windows)

1. Clone the repository:
```bash
git clone https://github.com/FranckyB/Voice-Clone-Studio.git
cd Voice-Clone-Studio
```

2. Run the setup script:
```bash
setup.bat
```

This will automatically:
- Install SOX (audio processing)
- Create virtual environment
- Install PyTorch with CUDA support
- Install all dependencies
- Display your Python version
- Show instructions for optional Flash Attention 2 installation

#### Quick Setup (Linux)

1. Clone the repository:
```bash
git clone https://github.com/FranckyB/Voice-Clone-Studio.git
cd Voice-Clone-Studio
```

2. Make the setup script executable and run it:
```bash
chmod +x setup-linux.sh
./setup-linux.sh
```

This will automatically:
- Detect your Python version
- Create virtual environment
- Install PyTorch with CUDA support
- Install all dependencies (using appropriate requirements file)
- Handle ONNX Runtime installation issues
- Warn about Whisper compatibility if needed

#### Manual Setup (All Platforms)

1. Clone the repository:
```bash
git clone https://github.com/FranckyB/Voice-Clone-Studio.git
cd Voice-Clone-Studio
```

2. Create a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/MacOs
source venv/bin/activate
```

3. (NVIDIA GPU) Install PyTorch with CUDA support:
```bash
# Linux/Windows
pip install torch==2.9.1 torchaudio --index-url https://download.pytorch.org/whl/cu130
```

4. Install dependencies:
```bash
# Windows
pip install -r requirements-windows.txt

# Linux (skips Whisper, uses VibeVoice ASR instead)
pip install -r requirements-linux.txt
```

#### Linux-Specific Issues & Solutions

**For detailed Linux troubleshooting and step-by-step fixes, see [LINUX_INSTALL.md](LINUX_INSTALL.md)**

**Python Version:** Use Python 3.12+ on Linux. The Linux installation uses VibeVoice ASR for transcription (openai-whisper is skipped due to compatibility issues).

**Issue: ONNX Runtime Build Failures**

If you see ONNX runtime installation errors on Linux, try the nightly build:

```bash
# Install dependencies first
pip install coloredlogs flatbuffers numpy packaging protobuf sympy

# Try nightly build of onnxruntime
pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime

# Then install qwen-tts
pip install qwen-tts --no-deps
pip install librosa soundfile sox einops gradio diffusers markdown
```

5. Install Sox

```bash
# Windows
winget install -e --id ChrisBagwell.SoX

# Linux
# Debian/Ubuntu
sudo apt install sox libsox-dev
# Fedora/RHEL
sudo dnf install sox sox-devel

# MacOs
brew install sox
```

6. Install ffmpeg

```bash
# Windows
winget install -e --id Gyan.FFmpeg

# Linux
# Debian/Ubuntu
sudo apt install ffmpeg
# Fedora/RHEL
sudo dnf install ffmpeg

# MacOs
brew install ffmpeg
```

7. (Optional) Install Flash Attention 2 for better performance:
```bash
# Option 1 - Build from source (requires C++ compiler):
pip install flash-attn --no-build-isolation

# Option 2 - Use prebuilt wheel (faster, recommended):
# Download from: https://github.com/bdashore3/flash-attention/releases
# Then: pip install downloaded-wheel-file.whl
```

## Usage

### Launch the UI

```bash
python voice_clone_studio.py
```

Or use the batch file (Windows):
```bash
launch.bat
```

The UI will open at `http://127.0.0.1:7860`

### Prepare Voice Samples

1. Go to the **Prep Samples** tab
2. Upload or record audio (3-10 seconds of clear speech)
3. Trim and normalize as needed
4. Transcribe or manually enter the text
5. Save as a sample with a name

### Clone a Voice

1. Go to the **Voice Clone** tab
2. Select your sample from the dropdown
3. Enter the text you want to speak
4. Click Generate

### Design a Voice

1. Go to the **Voice Design** tab
2. Enter the text to speak
3. Describe the voice (e.g., "Young female, warm and friendly, slight British accent")
4. Click Generate

## Project Structure

```
Qwen3-TTS-Voice-Clone-Studio/
├── voice_clone_ui.py      # Main Gradio application
├── requirements-windows.txt  # Python dependencies (Windows)
├── requirements-linux.txt    # Python dependencies (Linux)
├── __Launch_UI.bat        # Windows launcher
├── samples/               # Voice samples (.wav + .txt pairs)
│   └── example.wav
│   └── example.txt
├── output/                # Generated audio outputs
├── vendor                 # Included Technology
│   └── vibevoice_asr      # newest version of vibevoice with asr support
│   └── vibevoice_tts      # prior version of vibevoice with tts support
```

## Models Used

Each tab lets you choose between model sizes:

| Model | Sizes | Use Case |
|-------|-------|----------|
| **Base** | Small, Large | Voice cloning from samples |
| **CustomVoice** | Small, Large | Premium speakers with style control |
| **VoiceDesign** | 1.7B only | Voice design from descriptions |
| **VibeVoice** | Small, Large | Long-form multi-speaker (up to 90 min) |
| **Whisper** | Medium | Audio transcription |

- **Small** = Faster, less VRAM (Qwen: 0.6B ~4GB, VibeVoice: 1.5B)
- **Large** = Better quality, more expressive (Qwen: 1.7B ~8GB, VibeVoice: Large model)

Models are automatically downloaded on first use via HuggingFace.

## Tips

- **Reference Audio**: Use clear, noise-free recordings (3-10 seconds)
- **Transcripts**: Should exactly match what's spoken in the audio
- **Caching**: Voice prompts are cached - first generation is slow, subsequent ones are fast
- **Seeds**: Use the same seed to reproduce identical outputs

## Troubleshooting

### Installation Issues

**Q: I get "llvmlite" or "numba" build errors**
- This is caused by `openai-whisper` on Linux
- **Solution (Linux)**: Use `requirements-linux.txt` which skips Whisper
- **Windows**: This shouldn't happen, but you can skip Whisper and use VibeVoice ASR

**Q: ONNX Runtime fails to install on Linux**
```bash
# Try these steps in order:
pip install onnxruntime  # Standard version
# If that fails, try nightly:
pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime
```

**Q: Dependency conflicts with qwen-tts and onnxruntime**
```bash
# Install qwen-tts without dependencies first, then install others separately:
pip install qwen-tts --no-deps
pip install onnxruntime librosa torchaudio soundfile sox einops
```

### Runtime Issues

**Q: Out of memory errors**
- Use smaller model sizes (0.6B instead of 1.7B)
- Reduce batch size in training
- Close other GPU-intensive applications

**Q: Transcription not working**
- **Linux**: Use VibeVoice ASR (Whisper not included on Linux)
- **Windows**: Use either Whisper or VibeVoice ASR
- Both transcription engines work great!

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

This project is based on and uses code from:
- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)**    - Apache 2.0 License (Alibaba)
- **[VibeVoice](https://github.com/microsoft/VibeVoice)** - MIT License
- **[Gradio](https://gradio.app/)**                       - Apache 2.0 License
- **[OpenAI Whisper](https://github.com/openai/whisper)** - MIT License

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba
- [VibeVoice](https://github.com/microsoft/VibeVoice) by Microsoft
- [Gradio](https://gradio.app/) for the web UI framework
- [OpenAI Whisper](https://github.com/openai/whisper) for transcription


## Versions

**Version 0.5.5** - UI Polish
- Added Custom Confirmation pop up Dialog in js for delete taks.
- Added Custom File list display Dialog. 
- (Why were both of these not built in Gradio?!)

**Version 0.5.1** - UI Polish & Help System
- **Help Guide Tab** - Comprehensive in-app documentation with 8 topic sections (First draft)
- **Modular Help System** - Extracted help content to separate `ui_help.py` module
- **Better Text Formatting** - Markdown rendering with scrollable containers for help content

**Version 0.4.0** - Custom Voice Training
- Added **Train Model** tab for fine-tuning custom voices
- Complete training pipeline with validation, data preparation, and model training
- **Batch Transcription** - Process 50-100+ audio files in one click
- Support for both 0.6B and 1.7B base models
- Real-time training progress monitoring with live loss values
- Checkpoint management - compare different training epochs
- Integration with Voice Presets tab for using trained models
- Dataset organization system with `datasets/` folder structure
- Automatic audio format conversion (24kHz 16-bit mono)
- Training progress tracking and error handling

**Version 0.3.5** - Style Instructions
- Added Style Instructions support in Conversation for Qwen model. (Unsupported by VibeVoice)

**Version 0.30** - Enhanced Media Support
- Video File Support - Upload video files (.mp4, .mov, .avi, .mkv, etc.) to Prep Samples tab
- Automatic Audio Extraction - Uses ffmpeg to extract audio from video files for voice cloning
- Improved Workflow - Added Clear button to quickly reset the audio editor

**Version 0.2** - VibeVoice Integration
- Added **VibeVoice TTS** support for long-form multi-speaker generation (up to 90 minutes)
- Added **VibeVoice ASR** as alternative transcription engine alongside Whisper
- Conversation tab now supports both Qwen (9 preset voices) and VibeVoice (custom samples) engines
- Multi-speaker conversation support with up to 4 custom voices

**Version 0.1** - Initial Release
- Voice cloning with Qwen3-TTS (Base, CustomVoice, VoiceDesign models)
- Whisper-powered automatic transcription
- Sample preparation toolkit (trim, normalize, mono conversion)
- Voice prompt caching for faster generation
- Seed control for reproducible outputs
