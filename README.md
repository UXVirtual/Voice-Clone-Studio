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

- Python 3.10+ (recommended for all platforms)
- CUDA-compatible GPU (recommended: 8GB+ VRAM)
- **SOX**  (Sound eXchange) - Required for audio processing
- **FFMPEG** - Multimedia framework required for audio format conversion
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) (optional but recommended)

**Note for Linux users:** The Linux installation skips `openai-whisper` (compatibility issues). VibeVoice ASR is used for transcription instead.

### Setup

#### Quick Setup (Windows)

1. Clone the repository:
```bash
git clone https://github.com/FranckyB/Voice-Clone-Studio.git
cd Voice-Clone-Studio
```

2. Run the setup script:
```bash
setup-windows.bat
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
- Install all dependencies (using requirements file)
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
# All platforms (Windows, Linux, macOS)
pip install -r requirements.txt
```

**Note:** The requirements file uses platform markers to automatically install the correct packages:
- Windows: Includes `openai-whisper` for transcription
- Linux/macOS: Excludes `openai-whisper` (uses VibeVoice ASR instead)

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

7. (Optional) Install FlashAttention 2  for faster generation:
**Note:** The application automatically detects and uses the best available attention mechanism. Configure in Settings tab: `auto` (recommended) → `flash_attention_2` → `sdpa` → `eager`

## Troubleshooting
For troubleshooting solutions, see [docs/troubleshooting.md](docs/troubleshooting.md).

#### Docker Setup (Windows)

1. **Install NVIDIA Drivers (Windows Side)**
   - Install the latest standard NVIDIA driver (Game Ready or Studio) for Windows from the [NVIDIA Drivers page](https://www.nvidia.com/Download/index.aspx).
   - **Crucial:** Do *not* try to install NVIDIA drivers inside your WSL Linux terminal. It will conflict with the host driver.

2. **Update WSL 2**
   - Open **PowerShell** as Administrator and ensure your WSL kernel is up to date:
     ```powershell
     wsl --update
     ```
   - (If you don't have WSL installed yet, run `wsl --install` and restart your computer).

3. **Configure Docker Desktop**
   - Install the latest version of **Docker Desktop for Windows**.
   - Open Docker Desktop **Settings** (gear icon).
   - Under **General**, ensure **"Use the WSL 2 based engine"** is checked.
   - Under **Resources > WSL Integration**, ensure the switch is enabled for your default Linux distro (e.g., Ubuntu).

4. **Run with Docker Compose**
   - Run the following command in the repository root:
     ```powershell
     docker-compose up --build
     ```
   - The application will be accessible at `http://127.0.0.1:7860`.

### Running Tests (Docker)

To verify the installation and features (like the DeepFilterNet denoiser), runs the integration tests inside the container:

```powershell
# Run the Denoiser Integration Test
docker-compose exec voice-clone-studio python tests/integration_test_denoiser.py
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
├── requirements.txt       # Python dependencies
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
| **Qwen3-TTS Base** | Small, Large | Voice cloning from samples |
| **Qwen3-TTS CustomVoice** | Small, Large | Premium speakers with style control |
| **Qwen3-TTS VoiceDesign** | 1.7B only | Voice design from descriptions |
| **VibeVoice-TTS** | Small, Large | Voice cloning & Long-form multi-speaker (up to 90 min) |
| **VibeVoice-ASR** | Large | Audio transcription |
| **Whisper** | Medium | Audio transcription |

- **Small** = Faster, less VRAM (Qwen: 0.6B ~4GB, VibeVoice: 1.5B)
- **Large** = Better quality, more expressive (Qwen: 1.7B ~8GB, VibeVoice: Large model)
- **4 Bit Quantized** version of the Large model is also included for VibeVoice.

Models are automatically downloaded on first use via HuggingFace.

## Tips

- **Reference Audio**: Use clear, noise-free recordings (3-10 seconds)
- **Transcripts**: Should exactly match what's spoken in the audio
- **Caching**: Voice prompts are cached - first generation is slow, subsequent ones are fast
- **Seeds**: Use the same seed to reproduce identical outputs

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

## Updates

For detailed version history and release notes, see [docs/updates.md](docs/updates.md).

**Latest Version:** 0.6.0 - Enhanced Model Support & Settings (January 27, 2026)

