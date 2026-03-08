from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import Optional, Literal
import io
import soundfile as sf
import numpy as np
import logging
from pathlib import Path
import asyncio
import re

# Impor utils to find models
from modules.core_components.ai_models.model_utils import get_trained_vibevoice_models

logger = logging.getLogger(__name__)

# Global lock to ensure sequential processing
generation_lock = asyncio.Lock()

def clean_input_text(text: str) -> str:
    """
    Remove citations and artifacts from OpenWebUI or other AI outputs.
    """
    # Remove standard citation numbers like [1], [2], [1, 2]
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    # Remove citations with source like [Source] or [source]
    text = re.sub(r'\[[Ss]ource\]', '', text)
    # Remove markdown link syntax but keep text: [text](http...) -> text
    # This handles cases where citations might be links
    # text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text) 
    # Actually, for citations we likely want to remove the whole thing if it's a citation.
    
    # Remove patterns that look like file references in brackets e.g. [filename.pdf]
    text = re.sub(r'\[[\w\s-]+\.(?:pdf|txt|md|docx)\]', '', text, flags=re.IGNORECASE)

    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: Optional[str] = "default"
    response_format: Optional[Literal['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm']] = 'mp3'
    speed: Optional[float] = 1.0

def create_v1_router(tts_manager, trained_models_dir: Path, user_config: dict, samples_dir: Path = None):
    router = APIRouter()


    @router.get("/v1/audio/voices")
    async def get_voices():
        """
        Get available voices (models).
        Compatible with endpoints expecting /v1/audio/voices (like Open WebUI generic/custom integrations).
        """
        try:
            models = get_trained_vibevoice_models(trained_models_dir)
            voices = []
            for m in models:
                voices.append({
                    "voice_id": m['display_name'],
                    "name": m['display_name'],
                    # Optional metadata
                    "category": "vibevoice", 
                })
            return {"voices": voices}
        except Exception as e:
            logger.error(f"Error listing voices: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/v1/models")
    async def get_models():
        """
        Get available models.
        Standard OpenAI compatible endpoint.
        """
        try:
            models = get_trained_vibevoice_models(trained_models_dir)
            data = []
            import time
            current_time = int(time.time())
            
            for m in models:
                data.append({
                    "id": m['display_name'],
                    "object": "model",
                    "created": current_time,
                    "owned_by": "voice-clone-studio",
                })
            return {"object": "list", "data": data}
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/v1/audio/speech")
    async def generate_speech(request: SpeechRequest):
        try:
            # 1. Validate Model
            # Get available trained VibeVoice models
            models = get_trained_vibevoice_models(trained_models_dir)
            
            # Find the model by name (folder name or display name)
            selected_model = None
            
            # Helper to find model in list
            def find_model(name):
                for m in models:
                    if m['display_name'] == name or m['path'].name == name:
                        return m
                return None

            # Priority 1: Check if 'voice' parameter matches a known model
            if request.voice and request.voice != "default":
                selected_model = find_model(request.voice)

            # Priority 2: Check if 'model' parameter matches a known model (fallback or if voice is generic/default)
            if not selected_model:
                selected_model = find_model(request.model)
            
            if not selected_model:
                # If model is generic OpenAI placeholder, user likely provided invalid voice
                if request.model in ["tts-1", "tts-1-hd"]:
                     raise HTTPException(status_code=404, detail=f"Voice '{request.voice}' not found. Available voices: {[m['display_name'] for m in models]}")
                
                raise HTTPException(status_code=404, detail=f"Model/Voice '{request.model}' not found. Available models: {[m['display_name'] for m in models]}")

            # 2. Generate Audio
            # We use the tts_manager to generate audio
            # Note: generate_with_trained_vibevoice is synchronous, so it might block the event loop.
            # In a production app, run this in a threadpool. For this plan, we run distinct.
            
            model_path = selected_model['path']
            print(f"API Request: Generating for voice {selected_model['display_name']} (model: {request.model})")

            # Search for best matching sample in samples_dir (Only for HD model)
            voice_sample_path = None
            if request.model == "tts-1-hd" and samples_dir and samples_dir.exists():
                speaker_name = selected_model.get('speaker_name', '')
                display_name = selected_model.get('display_name', '')
                
                # Targets to match against (speaker name first, then full display name)
                # Filter out empty strings
                targets = [t for t in [speaker_name, display_name] if t]
                
                if targets:
                    # Get all audio files
                    all_samples = list(samples_dir.glob("*"))
                    audio_samples = [f for f in all_samples if f.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']]
                    
                    found = False
                    # Strategy 1: Exact match on stem
                    for target in targets:
                        for f in audio_samples:
                            if f.stem.lower() == target.lower():
                                voice_sample_path = f
                                print(f"API: Found exact sample match for '{target}': {f.name}")
                                found = True
                                break
                        if found: break
                    
                    # Strategy 2: Starts with (e.g. Jessie_01.wav for Jessie)
                    if not found:
                        for target in targets:
                            for f in audio_samples:
                                if f.stem.lower().startswith(target.lower() + "_") or f.stem.lower().startswith(target.lower() + "-"):
                                    voice_sample_path = f
                                    print(f"API: Found prefix sample match for '{target}': {f.name}")
                                    found = True
                                    break
                            if found: break

                    # Strategy 3: Partial containment (e.g. Jessie in Jessie-VibeVoice)
                    if not found:
                        for target in targets:
                            # Skip very short targets to avoid false matches
                            if len(target) < 3: continue
                            
                            for f in audio_samples:
                                # Check if sample name is in target (e.g. Jessie.wav in Jessie-VibeVoice)
                                if f.stem.lower() in target.lower():
                                    voice_sample_path = f
                                    print(f"API: Found substring sample match (sample in target) for '{target}': {f.name}")
                                    found = True
                                    break
                                # Check if target is in sample name (e.g. Jessie in Best_Jessie_Sample.wav)
                                if target.lower() in f.stem.lower():
                                    voice_sample_path = f
                                    print(f"API: Found substring sample match (target in sample) for '{target}': {f.name}")
                                    found = True
                                    break
                            if found: break

            # Clean input text of citations and artifacts
            cleaned_input = clean_input_text(request.input)

            # Streaming generation is not easily supported by the current manager structure without refactoring,
            # so we generate primarily and then encode.
            
            gen_kwargs = {
                "text": cleaned_input,
                "language": "en", # Default to English for now, or infer/add param
                "checkpoint_path": model_path,
                "temperature": 0.7, # Default decent parameters
                "do_sample": True,
                "user_config": user_config
            }
            
            if voice_sample_path:
                gen_kwargs['voice_sample_path'] = str(voice_sample_path)
            
            # Queue execution to prevent VRAM overlap
            queue_position = 0 if not generation_lock.locked() else 1 # Rough estimate
            if generation_lock.locked():
                print(f"API: GPU busy. Request from {request.model} entered queue.")
            
            async with generation_lock:
                print(f"API: Processing generation for {request.model}...")
                # Run synchronous generation in a thread to unblock the event loop (so pings/health checks work)
                audio_data, sample_rate = await asyncio.to_thread(tts_manager.generate_with_trained_vibevoice, **gen_kwargs)
                print(f"API: Generation complete for {request.model}.")

            # 3. Convert to requested format
            buffer = io.BytesIO()
            format_mapping = {
                'wav': 'WAV',
                'flac': 'FLAC',
                'ogg': 'OGG',
                'mp3': 'MP3' 
            }
            
            # soundfile supports WAV, FLAC, OGG primarily. MP3 requires extra libs explicitly sometimes,
            # but let's try standard export.
            # If request.response_format is mp3, we might need to fallback or use pydub if available?
            # Soundfile 0.12+ supports MP3 writing if installed correctly.
            
            sf_format = format_mapping.get(request.response_format, 'WAV') # Default to WAV if mapping missing (pcm, aac, opus -> wav for now)
            
            # Handle PCM raw
            if request.response_format == 'pcm':
                # Return raw bytes
                # Convert float32 [-1, 1] to int16 for standard usage or keep float bytes
                audio_int16 = (audio_data * 32767).astype(np.int16)
                return Response(content=audio_int16.tobytes(), media_type="application/octet-stream")

            # Handle others via soundfile
            try:
                sf.write(buffer, audio_data, sample_rate, format=sf_format)
                buffer.seek(0)
                media_type = f"audio/{request.response_format}"
            except Exception as e:
                # Fallback to WAV if requested format fails
                print(f"Format {sf_format} failed: {e}. Falling back to WAV.")
                buffer = io.BytesIO()
                sf.write(buffer, audio_data, sample_rate, format='WAV')
                buffer.seek(0)
                media_type = "audio/wav"

            return Response(content=buffer.read(), media_type=media_type)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return router
