# Voice Clone Studio - Refactoring Plan

## Current Status
- **File**: `voice_clone_studio.py` (2868 lines)
- **Structure**: Monolithic single file
- **Problem**: Hard to maintain, difficult to add new features

## Target Structure

```
voice_clone_studio/
├── __init__.py (✓ Created)
├── constants.py (✓ Created - needs completion)  
├── main.py (entry point - to be migrated)
├── ui.py (Gradio UI assembly)
├── models/
│   ├── __init__.py
│   ├── qwen_tts.py (Qwen TTS model management)
│   ├── vibevoice_tts.py (VibeVoice TTS functions)
│   └── transcription.py (Whisper + VibeVoice ASR)
├── tabs/
│   ├── __init__.py
│   ├── voice_clone.py (Tab 1)
│   ├── conversation.py (Tab 2)
│   ├── voice_presets.py (Tab 3)
│   ├── voice_design.py (Tab 4)
│   ├── prep_samples.py (Tab 5)
│   ├── output_history.py (Tab 6)
│   └── finetune_dataset.py (Tab 7 - NEW)
└── utils/
    ├── __init__.py
    ├── config.py (✓ Created)
    ├── audio.py (audio processing utilities)
    ├── cache.py (prompt caching)
    └── helpers.py (misc helper functions)
```

## Migration Steps

### Phase 1: Extract Utilities (Non-breaking)
1. [✓] Create directory structure
2. [✓] Create constants.py
3. [✓] Create config.py
4. [ ] Extract audio processing functions to utils/audio.py
5. [ ] Extract helper functions to utils/helpers.py
6. [ ] Extract caching logic to utils/cache.py

### Phase 2: Extract Model Management
7. [ ] Move Qwen model loading/generation to models/qwen_tts.py
8. [ ] Move VibeVoice model functions to models/vibevoice_tts.py
9. [ ] Move transcription (Whisper/ASR) to models/transcription.py

### Phase 3: Extract Tab Functions
10. [ ] Extract Voice Clone tab to tabs/voice_clone.py
11. [ ] Extract Conversation tab to tabs/conversation.py
12. [ ] Extract Voice Presets tab to tabs/voice_presets.py
13. [ ] Extract Voice Design tab to tabs/voice_design.py
14. [ ] Extract Prep Samples tab to tabs/prep_samples.py
15. [ ] Extract Output History tab to tabs/output_history.py

### Phase 4: Create UI Assembly
16. [ ] Create ui.py that assembles all tabs
17. [ ] Create main.py entry point
18. [ ] Update Launch_UI.bat to use new main.py

### Phase 5: Testing & Cleanup
19. [ ] Test all features work identically
20. [ ] Remove old voice_clone_studio.py
21. [ ] Update README with new structure

## Current Decision

Rather than rushing the refactor now, let's:
1. **Add the Finetune Dataset feature** to the existing monolithic file
2. **Plan dedicated refactoring session** after feature is tested
3. **Keep refactoring plan** for reference

This ensures:
- New feature delivered quickly
- Refactoring done carefully without bugs
- Clear plan for future maintenance

## Notes
- voice_clone_studio.py should remain functional during transition
- All imports must be backwards compatible
- Config file format must remain unchanged
- Directory structure must remain the same for users
