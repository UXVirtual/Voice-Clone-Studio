import types
import sys
import logging
import warnings

def apply_patches():
    """Apply compatibility patches for DeepFilterNet vs Torchaudio 2.1+

    # --- DeepFilterNet / Torchaudio Compatibility Shim ---
    # DeepFilterNet relies on 'torchaudio.backend.common.AudioMetaData' which was removed in torchaudio 2.1+.
    # We monkey-patch sys.modules to satisfy the import before DeepFilterNet tries to import it.
    """

    # Suppress Torchaudio 2.x warning triggered by DeepFilterNet using old resampling method name
    warnings.filterwarnings("ignore", message='.*"sinc_interpolation" resampling method name is being deprecated.*', category=UserWarning)

    try:
        import torchaudio

        # Ensure the backend module structure exists
        if not hasattr(torchaudio, 'backend'):
            torchaudio.backend = types.ModuleType('torchaudio.backend')
            sys.modules['torchaudio.backend'] = torchaudio.backend

        # Ensure the backend.common module exists
        if not hasattr(torchaudio.backend, 'common'):
            common_module = types.ModuleType('torchaudio.backend.common')
            torchaudio.backend.common = common_module
            sys.modules['torchaudio.backend.common'] = common_module

            # Find or Define AudioMetaData
            if hasattr(torchaudio, 'AudioMetaData'):
                # Aliasing directly if available
                common_module.AudioMetaData = torchaudio.AudioMetaData
                AudioMetaData = torchaudio.AudioMetaData
            else:
                # Define a compatible dummy class if not found
                from dataclasses import dataclass

                @dataclass
                class AudioMetaData:
                    sample_rate: int
                    num_frames: int
                    num_channels: int
                    bits_per_sample: int
                    encoding: str
                common_module.AudioMetaData = AudioMetaData

            # Use the class we resolved or created
            AudioMetaDataClass = common_module.AudioMetaData

        # Patch torchaudio.info if missing (Required by DeepFilterNet)
        if not hasattr(torchaudio, 'info'):
            try:
                import soundfile as sf
                # Resolve AudioMetaData again just in case we skipped the block above
                if hasattr(torchaudio, 'AudioMetaData'):
                    AudioMetaDataClass = torchaudio.AudioMetaData
                elif hasattr(torchaudio.backend.common, 'AudioMetaData'):
                    AudioMetaDataClass = torchaudio.backend.common.AudioMetaData
                else:
                    # Capture the definition from above if it was local
                    pass

                def info_patch(filepath, **kwargs):
                    # print(f"DEBUG: Checking file: {filepath}")
                    import os
                    if not os.path.exists(filepath):
                        raise RuntimeError(f"File not found: {filepath}")
                    if os.path.getsize(filepath) == 0:
                        raise RuntimeError(f"File is empty: {filepath}")

                    try:
                        import soundfile as sf
                        # Try soundfile first (fast)
                        sf_info = sf.info(filepath)
                        return AudioMetaDataClass(
                            sample_rate=sf_info.samplerate,
                            num_frames=sf_info.frames,
                            num_channels=sf_info.channels,
                            bits_per_sample=0,
                            encoding=sf_info.subtype
                        )
                    except Exception as sf_error:
                        # Fallback to torchaudio.load (slower, reads whole file)
                        # Used when soundfile fails (e.g. some MP3s or permission issues)
                        load_error = None
                        try:
                            # Avoid infinite recursion if load calls info (unlikely if we are patching info)
                            waveform, sample_rate = torchaudio.load(filepath)
                            return AudioMetaDataClass(
                                sample_rate=sample_rate,
                                num_frames=waveform.shape[1],
                                num_channels=waveform.shape[0],
                                bits_per_sample=32,  # Assumption for loaded tensors
                                encoding="Unknown"
                            )
                        except Exception as e:
                            load_error = e

                        # Fallback to ffprobe (robust system call)
                        try:
                            import subprocess
                            import json
                            cmd = [
                                "ffprobe",
                                "-v", "error",
                                "-print_format", "json",
                                "-show_streams",
                                "-show_format",
                                str(filepath)
                            ]
                            # Run without check=True to capture stderr manually
                            result = subprocess.run(cmd, capture_output=True, text=True)

                            if result.returncode != 0:
                                raise RuntimeError(f"FFprobe failed with code {result.returncode}. Stderr: {result.stderr}")

                            data = json.loads(result.stdout)

                            # Find audio stream
                            audio_stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'audio'), None)
                            if not audio_stream:
                                raise RuntimeError("No audio stream found via ffprobe")

                            sr = int(audio_stream.get('sample_rate'))
                            channels = int(audio_stream.get('channels'))
                            # Approximate frames if not explicit
                            frames = audio_stream.get('nb_samples')
                            if not frames:
                                duration = float(data.get('format', {}).get('duration', 0))
                                frames = int(duration * sr)
                            else:
                                frames = int(frames)

                            return AudioMetaDataClass(
                                sample_rate=sr,
                                num_frames=frames,
                                num_channels=channels,
                                bits_per_sample=0,
                                encoding=audio_stream.get('codec_name', 'unknown')
                            )
                        except Exception as ff_error:
                            raise RuntimeError(f"Patched torchaudio.info failed for {filepath}. SF: {sf_error} | Load: {load_error} | FF: {ff_error}")

                # Capture the info patch function locally so we can assign it
                info_patch_func = info_patch

            except ImportError:
                print("Warning: soundfile not installed, cannot patch torchaudio.info")
                info_patch_func = None

        # Patch torchaudio.load if using TorchCodec (missing)
        # DeepFilterNet uses torchaudio.load underneath
        original_tochaudio_load = torchaudio.load

        def load_patch(filepath, **kwargs):
            try:
                # Try original first, in case they fixed it or it works for wavs
                return original_tochaudio_load(filepath, **kwargs)
            except Exception as e:
                msg = str(e)
                # Check for the specific TorchCodec error or generic backend failure
                if "TorchCodec" in msg or "backend" in msg.lower() or "no module" in msg.lower():
                    # Fallback to soundfile loading converting to Tensor
                    try:
                        import soundfile as sf
                        import torch

                        # soundfile.read returns (data, samplerate)
                        # data is (frames, channels) numpy array
                        data, samplerate = sf.read(filepath)

                        # Convert to torch tensor
                        # torchaudio.load expects (channels, frames)
                        # Ensure float32
                        if data.ndim == 1:
                            # Mono
                            tensor = torch.from_numpy(data).float().unsqueeze(0)
                        else:
                            # Stereo/Multi (frames, channels) -> (channels, frames)
                            tensor = torch.from_numpy(data.T).float()

                        return tensor, samplerate
                    except Exception as sf_e:
                        raise RuntimeError(f"Patched torchaudio.load failed: Original error: {e} | SoundFile fallback error: {sf_e}")
                else:
                    # Re-raise if it's some other error
                    raise e

        torchaudio.load = load_patch
        # print("Applied torchaudio.load compatibility patch.")

        # Patch torchaudio.save if using TorchCodec (missing)
        original_torchaudio_save = torchaudio.save

        def save_patch(filepath, src, sample_rate, **kwargs):
            try:
                original_torchaudio_save(filepath, src, sample_rate, **kwargs)
            except Exception as e:
                msg = str(e)
                if "TorchCodec" in msg or "backend" in msg.lower() or "no module" in msg.lower():
                    try:
                        import soundfile as sf
                        # src is Tensor (channels, frames). Convert to numpy (frames, channels)
                        if src.is_cuda:
                            src = src.cpu()
                        data = src.detach().numpy()

                        if data.ndim == 2:
                            data = data.T  # Transpose to (frames, channels)

                        sf.write(filepath, data, sample_rate)
                    except Exception as sf_e:
                        raise RuntimeError(f"Patched torchaudio.save failed: Original error: {e} | SoundFile fallback error: {sf_e}")
                else:
                    raise e

        torchaudio.save = save_patch

        # Patch torchaudio.info too
        if info_patch_func:
            torchaudio.info = info_patch_func
            # print("Applied torchaudio.info compatibility patch using soundfile.")

    except Exception as e:
        print(f"Warning: Failed to apply torchaudio compatibility patches: {e}")

if __name__ == "__main__":
    apply_patches()
