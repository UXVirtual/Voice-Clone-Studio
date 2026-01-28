#### Linux-Specific Issues & Solutions

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

### Installation Issues

**Q: I get "llvmlite" or "numba" build errors**
- This is caused by `openai-whisper` on Linux
- **Solution (Linux)**: uninstall Whisper and use VibeVoice ASR
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
