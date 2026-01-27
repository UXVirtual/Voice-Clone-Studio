# Vendored Dependencies

This directory contains third-party code included directly in our repository for stability and independence.

## vibevoice/

**Source:** https://github.com/microsoft/VibeVoice  
**License:** MIT  
**Purpose:** Complete VibeVoice package with both ASR (transcription with speaker diarization) and TTS (long-form multi-speaker synthesis)  
**Reason for vendoring:** Combines ASR and TTS functionality in a single unified package

### Features

**ASR (Automatic Speech Recognition):**
- Transcription with speaker diarization
- Multi-speaker conversation detection
- High-quality speech-to-text

**TTS (Text-to-Speech):**
- Long-form multi-speaker synthesis (up to 90 minutes)
- Up to 4 distinct speakers
- Natural turn-taking and prosody
- Spontaneous background music/sounds (context-aware)
- Cross-lingual support

### Models

**ASR Model:**
- `microsoft/VibeVoice-ASR` - Hosted on HuggingFace

**TTS Models:**
- `FranckyB/VibeVoice-1.5B` - Faster, good quality
- `FranckyB/VibeVoice-Large` - More stable, best quality

Models automatically download from HuggingFace on first use and are cached locally.

### Usage in App

- **Prep Samples Tab:** Uses VibeVoice ASR for transcription with speaker diarization
- **Conversation Tab:** Uses VibeVoice TTS for 90-minute continuous multi-speaker synthesis

### Attribution

VibeVoice is licensed under MIT License. Original copyright:
- Copyright (c) Microsoft Corporation

See LICENSE file in vibevoice/ directory for full license text.

## Installation

This vendor code is automatically available via sys.path manipulation in the main application.
No separate pip install is required.
