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
import markdown
from html.parser import HTMLParser

# Impor utils to find models
from modules.core_components.ai_models.model_utils import get_trained_vibevoice_models

logger = logging.getLogger(__name__)

# Global lock to ensure sequential processing
generation_lock = asyncio.Lock()

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text_parts = []
    
    def handle_data(self, d):
        self.text_parts.append(d)
        
    def handle_endtag(self, tag):
        # Add space after block elements to prevent concatenation
        if tag in ['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'br', 'tr', 'blockquote']:
            self.text_parts.append(' ')
            
    def get_data(self):
        return "".join(self.text_parts)

def strip_markdown(text: str) -> str:
    # Convert markdown to HTML
    try:
        html = markdown.markdown(text)
        # Strip HTML tags
        s = MLStripper()
        s.feed(html)
        return s.get_data()
    except Exception as e:
        logger.warning(f"Markdown stripping failed: {e}")
        return text

def clean_input_text(text: str) -> str:
    """
    Remove citations, artifacts, and markdown links from OpenWebUI or other AI outputs.
    """
    # First, strip markdown (handles links [text](url) -> text, bold, etc.)
    text = strip_markdown(text)

    # Remove standard citation numbers like [1], [2], [1, 2]
    # Note: strip_markdown might have left these alone as they aren't md links
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    # Remove citations with source like [Source] or [source]
    text = re.sub(r'\[[Ss]ource\]', '', text)
    
    # Remove patterns that look like file references in brackets e.g. [filename.pdf]
    text = re.sub(r'\[[\w\s-]+\.(?:pdf|txt|md|docx)\]', '', text, flags=re.IGNORECASE)

    # Clean up extra whitespace introduced by stripping
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: Optional[str] = "default"
    response_format: Optional[Literal['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm']] = 'mp3'
    speed: Optional[float] = 1.0
    stream: Optional[bool] = False

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
            
            # Built-in Qwen3 speakers
            for q in ['Vivian', 'Serena', 'Uncle_Fu', 'Dylan', 'Eric', 'Ryan', 'Aiden', 'Ono_Anna', 'Sohee']:
                voices.append({"voice_id": q, "name": q, "category": "qwen3"})
                
            # Built-in VibeVoice speakers (and OpenAI fallback mappings)
            for o in ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer', 'Carter', 'Davis', 'Emma', 'Frank', 'Grace', 'Mike', 'Samuel']:
                voices.append({"voice_id": o, "name": o, "category": "vibevoice_stream"})

            # Trained VibeVoice models
            for m in models:
                voices.append({
                    "voice_id": m['display_name'],
                    "name": m['display_name'],
                    "category": "vibevoice_trained", 
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
            
            # Base abstract models
            for base in ["tts-1", "tts-1-hd", "qwen3"]:
                data.append({"id": base, "object": "model", "created": current_time, "owned_by": "voice-clone-studio"})
            
            # Plus trained options just in case UI expects them as 'models'
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
            # 1. Route Request based on model & voice
            vibe_streaming_voices = ['Carter', 'Davis', 'Emma', 'Frank', 'Grace', 'Mike', 'Samuel']
            qwen_voices = ['Vivian', 'Serena', 'Uncle_Fu', 'Dylan', 'Eric', 'Ryan', 'Aiden', 'Ono_Anna', 'Sohee']
            
            # OpenAI voice map to VibeVoice streams
            openai_to_vibevoice = {
                "alloy": "Samuel",
                "echo": "Mike",
                "fable": "Carter",
                "onyx": "Davis",
                "nova": "Emma",
                "shimmer": "Grace"
            }
            
            # OpenAI voice map to Qwen3 voices
            openai_to_qwen3 = {
                "alloy": "Ryan",
                "echo": "Eric",
                "fable": "Dylan",
                "onyx": "Uncle_Fu",
                "nova": "Vivian",
                "shimmer": "Serena"
            }
            
            req_model = request.model or ""
            req_voice = request.voice or "default"
            req_voice_lower = req_voice.lower()
            
            generation_mode = None
            selected_trained_model = None
            
            # Check for trained VibeVoice model first if specific request made
            trained_models = get_trained_vibevoice_models(trained_models_dir)
            for m in trained_models:
                if m.get('display_name') == req_voice or m.get('display_name') == req_model or Path(m.get('path', '')).name == req_voice:
                    selected_trained_model = m
                    generation_mode = "trained_vibevoice"
                    break
            
            if not generation_mode:
                if req_model in ["qwen3", "tts-1-hd"]:
                    generation_mode = "qwen3"
                elif req_model == "tts-1" or req_voice_lower in openai_to_vibevoice or req_voice in vibe_streaming_voices:
                    generation_mode = "vibevoice_stream"
                else:
                    raise HTTPException(status_code=404, detail=f"Model/Voice '{req_voice}' not found.")

            # Clean input text of citations and artifacts
            print(f"API Input Text (Raw): {request.input}", flush=True)
            cleaned_input = clean_input_text(request.input)
            print(f"API Input Text (Cleaned): {cleaned_input}", flush=True)

            if request.stream and generation_mode == "vibevoice_stream":
                from fastapi.responses import StreamingResponse
                
                async def stream_generator():
                    # Wait for lock inside the generator so the request stays alive
                    if generation_lock.locked():
                        print(f"API: GPU busy. Request from {request.model} entered queue.")
                        
                    async with generation_lock:
                        from modules.vibevoice_tts.modular.streamer import AsyncAudioStreamer
                        streamer = AsyncAudioStreamer(batch_size=1)
                        
                        target_speaker = openai_to_vibevoice.get(req_voice_lower, req_voice)
                        if target_speaker not in vibe_streaming_voices:
                            target_speaker = "Samuel"

                        print(f"API: Streaming VibeVoice record: {target_speaker}")
                        
                        task = asyncio.create_task(
                            asyncio.to_thread(
                                tts_manager.generate_vibevoice_streaming,
                                text=cleaned_input,
                                voice_name=target_speaker,
                                audio_streamer=streamer
                            )
                        )
                        
                        if request.response_format != 'pcm':
                            import wave, io
                            buffer = io.BytesIO()
                            with wave.open(buffer, 'wb') as wav_file:
                                wav_file.setnchannels(1)
                                wav_file.setsampwidth(2)
                                wav_file.setframerate(24000)
                                wav_file.setnframes(0xFFFFFFF) # Fake large
                            yield buffer.getvalue()
                            
                        try:
                            async for chunk in streamer.get_stream(0):
                                if chunk is None: break
                                if hasattr(chunk, 'cpu'): chunk = chunk.cpu().numpy()
                                chunk = chunk.squeeze()
                                if chunk.ndim > 1:
                                    chunk = chunk[0]
                                audio_int16 = (chunk * 32767).astype(np.int16)
                                yield audio_int16.tobytes()
                        except Exception as e:
                            print(f"Streaming error: {e}")
                        finally:
                            await task
                            
                return StreamingResponse(
                    stream_generator(),
                    media_type="audio/wav" if request.response_format != 'pcm' else "application/octet-stream"
                )

            if generation_lock.locked():
                print(f"API: GPU busy. Request from {request.model} entered queue.")
            
            async with generation_lock:
                print(f"API: Processing [{generation_mode}] generation for {request.model} with voice {request.voice}...")
                audio_data, sample_rate = None, None

                if generation_mode == "qwen3":
                    # Map standard OpenAI voices if used
                    target_speaker = openai_to_qwen3.get(req_voice_lower, req_voice)
                    
                    # Match target speaker case-insensitively
                    matched = False
                    for q in qwen_voices:
                        if q.lower() == target_speaker.lower():
                            target_speaker = q
                            matched = True
                            break
                            
                    if not matched:
                        target_speaker = "Ryan" # Safe fallback
                        
                    print(f"API: Generating Qwen3 voice: {target_speaker}")
                    audio_data, sample_rate = await asyncio.to_thread(
                        tts_manager.generate_custom_voice,
                        text=cleaned_input,
                        language="english",
                        speaker=target_speaker
                    )

                elif generation_mode == "vibevoice_stream":
                    target_speaker = openai_to_vibevoice.get(req_voice_lower, req_voice)
                    if target_speaker not in vibe_streaming_voices:
                        target_speaker = "Samuel" # Safe fallback
                    
                    print(f"API: Generating VibeVoice Streaming voice: {target_speaker}")
                    audio_data, sample_rate = await asyncio.to_thread(
                        tts_manager.generate_vibevoice_streaming,
                        text=cleaned_input,
                        voice_name=target_speaker
                    )

                elif generation_mode == "trained_vibevoice":
                    model_path = selected_trained_model['path']
                    print(f"API: Generating Trained VibeVoice: {selected_trained_model['display_name']}")
                    
                    # Search for best matching sample in samples_dir (Only for HD model)
                    voice_sample_path = None
                    if request.model == "tts-1-hd" and samples_dir and samples_dir.exists():
                        speaker_name = selected_trained_model.get('speaker_name', '')
                        display_name = selected_trained_model.get('display_name', '')
                        targets = [t for t in [speaker_name, display_name] if t]
                        
                        if targets:
                            all_samples = list(samples_dir.glob("*"))
                            audio_samples = [f for f in all_samples if f.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']]
                            found = False
                            for target in targets:
                                for f in audio_samples:
                                    if f.stem.lower() == target.lower():
                                        voice_sample_path = f
                                        found = True; break
                                if found: break
                                
                            if not found:
                                for target in targets:
                                    for f in audio_samples:
                                        if f.stem.lower().startswith(target.lower() + "_") or f.stem.lower().startswith(target.lower() + "-"):
                                            voice_sample_path = f
                                            found = True; break
                                    if found: break
                                    
                            if not found:
                                for target in targets:
                                    if len(target) < 3: continue
                                    for f in audio_samples:
                                        if f.stem.lower() in target.lower() or target.lower() in f.stem.lower():
                                            voice_sample_path = f
                                            found = True; break
                                    if found: break
                    
                    gen_kwargs = {
                        "text": cleaned_input,
                        "language": "en",
                        "checkpoint_path": model_path,
                        "temperature": 0.7,
                        "do_sample": True,
                        "user_config": user_config
                    }
                    if voice_sample_path:
                        gen_kwargs['voice_sample_path'] = str(voice_sample_path)
                        print(f"API: Conditional sample found: {voice_sample_path.name}")
                        
                    audio_data, sample_rate = await asyncio.to_thread(
                        tts_manager.generate_with_trained_vibevoice,
                        **gen_kwargs
                    )

            # 3. Convert to requested format
            if isinstance(audio_data, np.ndarray):
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                audio_data = audio_data.squeeze()
                if audio_data.ndim > 1:
                    audio_data = audio_data[0]

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
