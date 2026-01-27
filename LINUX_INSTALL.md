# Linux Installation Guide

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/FranckyB/Voice-Clone-Studio.git
cd Voice-Clone-Studio

# Run the automated setup
chmod +x setup-linux.sh
./setup-linux.sh

# If automated setup fails, try manual:
python3 -m venv venv
source venv/bin/activate
pip install torch==2.9.1 torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-linux.txt
```

## ⚠️ Common Errors - Quick Reference

| Error | Quick Fix |
|-------|-----------|
| `RuntimeError: Cannot install on Python version 3.12.10` | Use `requirements-linux.txt` - no Python version issues! |
| `Failed to build llvmlite` | Use `requirements-linux.txt` (skips Whisper) |
| `ONNX Runtime conflicts` | `pip install qwen-tts --no-deps` then `pip install onnxruntime` |
| `onnxruntime build failed` | Use nightly build (see below) |

## 📋 What You Need to Know

1. **Use Python 3.12+**: Recommended for all platforms
2. **No Whisper on Linux**: VibeVoice ASR is used instead (works great!)
3. **Use requirements-linux.txt**: Specifically designed for Linux (requirements-windows.txt is for Windows only)
4. **ONNX Runtime**: If standard install fails, try nightly build
5. **GPU is Optional**: Works on CPU (just slower)

---

## Detailed Troubleshooting

If you're experiencing installation issues on Linux, follow this guide.

## Quick Fix (Most Common Issues)

### Recommended Setup

**Use Python 3.12+** and the Linux-specific requirements:

```bash
pip install -r requirements-linux.txt
```

✅ **Note:** Linux installation uses VibeVoice ASR for transcription (openai-whisper is skipped due to compatibility issues). VibeVoice ASR works great!

---

### Issue 1: ONNX Runtime Installation Failures

**Error:** Build failures, dependency conflicts with qwen-tts

**Solution A - Try CPU version first:**
```bash
pip install coloredlogs flatbuffers numpy packaging protobuf sympy
pip install qwen-tts --no-deps
pip install onnxruntime
pip install librosa torchaudio soundfile sox einops gradio diffusers markdown
```

**Solution B - Use nightly build (if above fails):**
```bash
pip install coloredlogs flatbuffers numpy packaging protobuf sympy
pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime
pip install qwen-tts --no-deps
pip install librosa torchaudio soundfile sox einops gradio diffusers markdown
```

**Solution C - GPU version (after getting CPU working):**
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

---

### Issue 2: llvmlite Build Failures

**Error:** `Failed to build llvmlite==0.36.0`

**Cause:** This is from openai-whisper's dependencies

**Solution:** Use requirements-linux.txt which skips Whisper entirely:
```bash
pip install -r requirements-linux.txt
```

---

## Automated Setup (Recommended)

Use the provided setup script (works with Python 3.12+):

```bash
chmod +x setup-linux.sh
./setup-linux.sh
```

This script automatically:
- Creates virtual environment
- Installs PyTorch with CUDA
- Uses requirements-linux.txt (no Whisper)
- Handles ONNX Runtime issues

---

## Manual Clean Installation

If all else fails, try a completely clean install:

```bash
# 1. Remove old environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install PyTorch first
pip install torch==2.9.1 torchaudio --index-url https://download.pytorch.org/whl/cu130

# 4. Install core dependencies one by one
pip install gradio
pip install numpy
pip install librosa
pip install soundfile
pip install sox
pip install einops
pip install diffusers
pip install markdown

# 5. Try ONNX Runtime
pip install onnxruntime
# If that fails:
# pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime

# 6. Install qwen-tts without dependencies
pip install qwen-tts --no-deps

# 7. Done! (No Whisper on Linux)
```

---

## ✅ Verify Installation

```bash
python3 -c "import torch; import onnxruntime; import gradio; print('✅ All core dependencies OK')"
```

## 🆘 Still Having Issues?

Open a GitHub issue with:
1. Your Python version: `python3 --version`
2. Your Linux distro: `cat /etc/os-release`
3. Full error message
4. Output of: `pip list`
