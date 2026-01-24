import torch
import soundfile as sf
import gradio as gr
from qwen_tts import Qwen3TTSModel
from pathlib import Path
from datetime import datetime
import numpy as np

# Directories
SAMPLES_DIR = Path(__file__).parent / "samples"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Global model cache
_tts_model = None
_whisper_model = None

# Supported languages for TTS
LANGUAGES = [
    "Auto", "English", "Chinese", "Japanese", "Korean", 
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian"
]


def get_tts_model():
    """Lazy-load the TTS model."""
    global _tts_model
    if _tts_model is None:
        print("Loading Qwen3-TTS model...")
        _tts_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        print("TTS model loaded!")
    return _tts_model


def get_whisper_model():
    """Lazy-load the Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model...")
        import whisper
        _whisper_model = whisper.load_model("medium")
        print("Whisper model loaded!")
    return _whisper_model


def get_available_samples():
    """Find all .wav files in samples folder that have matching .txt files."""
    if not SAMPLES_DIR.exists():
        return []
    
    samples = []
    for wav_file in sorted(SAMPLES_DIR.glob("*.wav")):
        txt_file = wav_file.with_suffix(".txt")
        if txt_file.exists():
            ref_text = txt_file.read_text(encoding="utf-8").strip()
            samples.append({
                "name": wav_file.stem,
                "wav_path": str(wav_file),
                "txt_path": str(txt_file),
                "ref_text": ref_text
            })
    return samples


def get_sample_choices():
    """Get sample names for dropdown."""
    samples = get_available_samples()
    return [s["name"] for s in samples]


def get_output_files():
    """Get list of generated output files."""
    if not OUTPUT_DIR.exists():
        return []
    files = sorted(OUTPUT_DIR.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True)
    return [str(f) for f in files]


def on_sample_select(sample_name):
    """When a sample is selected, show its reference text and audio."""
    samples = get_available_samples()
    for s in samples:
        if s["name"] == sample_name:
            return s["wav_path"], s["ref_text"]
    return None, ""


def generate_audio(sample_name, text_to_generate, language, progress=gr.Progress()):
    """Generate audio using voice cloning."""
    if not sample_name:
        return None, "❌ Please select a voice sample first."
    
    if not text_to_generate or not text_to_generate.strip():
        return None, "❌ Please enter text to generate."
    
    # Find the selected sample
    samples = get_available_samples()
    sample = None
    for s in samples:
        if s["name"] == sample_name:
            sample = s
            break
    
    if not sample:
        return None, f"❌ Sample '{sample_name}' not found."
    
    try:
        progress(0.1, desc="Loading model...")
        model = get_tts_model()
        
        progress(0.3, desc="Generating audio...")
        wavs, sr = model.generate_voice_clone(
            text=text_to_generate.strip(),
            language=language if language != "Auto" else "Auto",
            ref_audio=sample["wav_path"],
            ref_text=sample["ref_text"],
        )
        
        progress(0.8, desc="Saving audio...")
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() else "_" for c in sample_name)
        output_file = OUTPUT_DIR / f"{safe_name}_{timestamp}.wav"
        
        sf.write(str(output_file), wavs[0], sr)
        
        progress(1.0, desc="Done!")
        return str(output_file), f"✅ Audio saved to: {output_file.name}"
        
    except Exception as e:
        return None, f"❌ Error generating audio: {str(e)}"


def refresh_samples():
    """Refresh the sample dropdown."""
    choices = get_sample_choices()
    return gr.update(choices=choices, value=choices[0] if choices else None)


def refresh_outputs():
    """Refresh the output file list."""
    files = get_output_files()
    return gr.update(choices=files, value=files[0] if files else None)


def load_output_audio(file_path):
    """Load a selected output file for playback."""
    if file_path and Path(file_path).exists():
        return file_path
    return None


def transcribe_audio(audio_file, whisper_language, progress=gr.Progress()):
    """Transcribe audio using Whisper."""
    if audio_file is None:
        return "❌ Please upload or record an audio file."
    
    try:
        progress(0.2, desc="Loading Whisper model...")
        model = get_whisper_model()
        
        progress(0.4, desc="Transcribing...")
        
        # Handle both file path and tuple (for recorded audio)
        if isinstance(audio_file, tuple):
            sr, audio_data = audio_file
            # Save to temp file for whisper
            temp_path = OUTPUT_DIR / "temp_recording.wav"
            # Normalize if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / 32768.0
            sf.write(str(temp_path), audio_data, sr)
            audio_path = str(temp_path)
        else:
            audio_path = audio_file
        
        # Transcribe
        options = {}
        if whisper_language and whisper_language != "Auto-detect":
            lang_code = {
                "English": "en", "Chinese": "zh", "Japanese": "ja", 
                "Korean": "ko", "German": "de", "French": "fr",
                "Russian": "ru", "Portuguese": "pt", "Spanish": "es", 
                "Italian": "it"
            }.get(whisper_language, None)
            if lang_code:
                options["language"] = lang_code
        
        result = model.transcribe(audio_path, **options)
        
        progress(1.0, desc="Done!")
        
        detected_lang = result.get("language", "unknown")
        transcription = result["text"].strip()
        
        return f"[Detected language: {detected_lang}]\n\n{transcription}"
        
    except Exception as e:
        return f"❌ Error transcribing: {str(e)}"


def save_as_sample(audio_file, transcription, sample_name):
    """Save transcribed audio as a new sample."""
    if not audio_file:
        return "❌ No audio file to save.", gr.update()
    
    if not transcription or transcription.startswith("❌"):
        return "❌ Please transcribe the audio first.", gr.update()
    
    if not sample_name or not sample_name.strip():
        return "❌ Please enter a sample name.", gr.update()
    
    # Clean sample name
    clean_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in sample_name).strip()
    clean_name = clean_name.replace(" ", "_")
    
    if not clean_name:
        return "❌ Invalid sample name.", gr.update()
    
    try:
        # Handle audio file
        if isinstance(audio_file, tuple):
            sr, audio_data = audio_file
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / 32768.0
        else:
            audio_data, sr = sf.read(audio_file)
        
        # Save wav file
        wav_path = SAMPLES_DIR / f"{clean_name}.wav"
        sf.write(str(wav_path), audio_data, sr)
        
        # Save text file (remove the detected language line if present)
        text_content = transcription
        if text_content.startswith("[Detected language:"):
            # Remove the first line
            lines = text_content.split("\n")
            text_content = "\n".join(lines[2:]).strip()
        
        txt_path = SAMPLES_DIR / f"{clean_name}.txt"
        txt_path.write_text(text_content, encoding="utf-8")
        
        # Refresh samples dropdown
        choices = get_sample_choices()
        
        return f"✅ Sample saved as '{clean_name}'", gr.update(choices=choices)
        
    except Exception as e:
        return f"❌ Error saving sample: {str(e)}", gr.update()


def create_ui():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="Qwen3-TTS Voice Clone Studio", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🎙️ Qwen3-TTS Voice Clone Studio
        
        Clone voices using Qwen3-TTS. Select a voice sample, enter your text, and generate speech!
        """)
        
        with gr.Tabs():
            # ============== TAB 1: Voice Clone ==============
            with gr.TabItem("🎤 Voice Clone"):
                with gr.Row():
                    # Left column - Sample selection
                    with gr.Column(scale=1):
                        gr.Markdown("### 1️⃣ Select Voice Sample")
                        
                        sample_dropdown = gr.Dropdown(
                            choices=get_sample_choices(),
                            label="Voice Sample",
                            info="Select a voice to clone"
                        )
                        refresh_samples_btn = gr.Button("🔄 Refresh Samples", size="sm")
                        
                        sample_audio = gr.Audio(
                            label="Reference Audio",
                            type="filepath",
                            interactive=False
                        )
                        
                        sample_text = gr.Textbox(
                            label="Reference Text",
                            lines=3,
                            interactive=False,
                            info="Transcript of the reference audio"
                        )
                    
                    # Right column - Generation
                    with gr.Column(scale=1):
                        gr.Markdown("### 2️⃣ Generate Speech")
                        
                        text_input = gr.Textbox(
                            label="Text to Generate",
                            placeholder="Enter the text you want to speak in the cloned voice...",
                            lines=5
                        )
                        
                        language_dropdown = gr.Dropdown(
                            choices=LANGUAGES,
                            value="English",
                            label="Language",
                            info="Language of the text to generate"
                        )
                        
                        generate_btn = gr.Button("🚀 Generate Audio", variant="primary", size="lg")
                        
                        status_text = gr.Textbox(label="Status", interactive=False)
                        
                        output_audio = gr.Audio(
                            label="Generated Audio",
                            type="filepath"
                        )
                
                gr.Markdown("---")
                gr.Markdown("### 📂 Output History")
                
                with gr.Row():
                    output_dropdown = gr.Dropdown(
                        choices=get_output_files(),
                        label="Previous Outputs",
                        info="Select a previously generated file to play"
                    )
                    refresh_outputs_btn = gr.Button("🔄 Refresh", size="sm")
                
                history_audio = gr.Audio(
                    label="Playback",
                    type="filepath"
                )
                
                # Event handlers for Voice Clone tab
                sample_dropdown.change(
                    on_sample_select,
                    inputs=[sample_dropdown],
                    outputs=[sample_audio, sample_text]
                )
                
                refresh_samples_btn.click(
                    refresh_samples,
                    outputs=[sample_dropdown]
                )
                
                generate_btn.click(
                    generate_audio,
                    inputs=[sample_dropdown, text_input, language_dropdown],
                    outputs=[output_audio, status_text]
                ).then(
                    refresh_outputs,
                    outputs=[output_dropdown]
                )
                
                refresh_outputs_btn.click(
                    refresh_outputs,
                    outputs=[output_dropdown]
                )
                
                output_dropdown.change(
                    load_output_audio,
                    inputs=[output_dropdown],
                    outputs=[history_audio]
                )
            
            # ============== TAB 2: Transcription ==============
            with gr.TabItem("📝 Transcribe Audio"):
                gr.Markdown("""
                ### Transcribe Audio with Whisper
                
                Upload or record audio to generate a transcript. You can then save it as a new voice sample!
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 1️⃣ Provide Audio")
                        
                        audio_input = gr.Audio(
                            label="Audio Input",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        
                        whisper_language = gr.Dropdown(
                            choices=["Auto-detect"] + LANGUAGES[1:],  # Skip "Auto" use "Auto-detect"
                            value="Auto-detect",
                            label="Language Hint",
                            info="Optional: specify language for better accuracy"
                        )
                        
                        transcribe_btn = gr.Button("📝 Transcribe", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 2️⃣ Transcription Result")
                        
                        transcription_output = gr.Textbox(
                            label="Transcription",
                            lines=8,
                            interactive=True,
                            info="You can edit the transcription if needed"
                        )
                
                gr.Markdown("---")
                gr.Markdown("### 3️⃣ Save as Voice Sample")
                
                with gr.Row():
                    new_sample_name = gr.Textbox(
                        label="Sample Name",
                        placeholder="Enter a name for this voice sample...",
                        scale=2
                    )
                    save_sample_btn = gr.Button("💾 Save as Sample", variant="secondary", scale=1)
                
                save_status = gr.Textbox(label="Status", interactive=False)
                
                # Event handlers for Transcription tab
                transcribe_btn.click(
                    transcribe_audio,
                    inputs=[audio_input, whisper_language],
                    outputs=[transcription_output]
                )
                
                save_sample_btn.click(
                    save_as_sample,
                    inputs=[audio_input, transcription_output, new_sample_name],
                    outputs=[save_status, sample_dropdown]
                )
        
        gr.Markdown("""
        ---
        **Tips:**
        - For best results, use clear reference audio (3-10 seconds)
        - The reference text should match what's spoken in the reference audio
        - Use the Transcribe tab to create reference text for audio clips without transcripts
        """)
    
    return app


if __name__ == "__main__":
    print(f"Samples directory: {SAMPLES_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Found {len(get_sample_choices())} samples")
    
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )
