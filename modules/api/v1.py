from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import Optional, Literal
import io
import soundfile as sf
import numpy as np
import logging
from pathlib import Path

# Impor utils to find models
from modules.core_components.ai_models.model_utils import get_trained_vibevoice_models

logger = logging.getLogger(__name__)

class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: Optional[str] = "default"
    response_format: Optional[Literal['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm']] = 'mp3'
    speed: Optional[float] = 1.0

def create_v1_router(tts_manager, trained_models_dir: Path, user_config: dict):
    router = APIRouter()

    @router.post("/v1/audio/speech")
    async def generate_speech(request: SpeechRequest):
        try:
            # 1. Validate Model
            # Get available trained VibeVoice models
            models = get_trained_vibevoice_models(trained_models_dir)
            
            # Find the model by name (folder name or display name)
            selected_model = None
            for m in models:
                if m['display_name'] == request.model or m['path'].name == request.model:
                    selected_model = m
                    break
            
            if not selected_model:
                raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found. Available models: {[m['display_name'] for m in models]}")

            # 2. Generate Audio
            # We use the tts_manager to generate audio
            # Note: generate_with_trained_vibevoice is synchronous, so it might block the event loop.
            # In a production app, run this in a threadpool. For this plan, we run distinct.
            
            model_path = selected_model['path']
            print(f"API Request: Generating for model {request.model} ({model_path})")

            # Streaming generation is not easily supported by the current manager structure without refactoring,
            # so we generate primarily and then encode.
            
            audio_data, sample_rate = tts_manager.generate_with_trained_vibevoice(
                text=request.input,
                language="en", # Default to English for now, or infer/add param
                checkpoint_path=model_path,
                temperature=0.7, # Default decent parameters
                do_sample=True,
                user_config=user_config
            )

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
