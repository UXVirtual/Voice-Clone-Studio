import sys
import os
import numpy as np
import soundfile as sf
import warnings

# Add parent directory to path to find 'patches' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_dummy_audio(filename="test_input.wav", sr=48000, duration=1.0):
    """Create a 1-second sine wave audio file."""
    print(f"🎵 Generating dummy audio: {filename} ({sr}Hz)")
    t = np.linspace(0, duration, int(sr * duration))
    # 440Hz sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(filename, audio, sr)
    return filename

def run_integration_test():
    print("==================================================")
    print("    DEEPFILTERNET INTEGRATION TEST HARNESS        ")
    print("==================================================")

    # 1. Apply Patches
    print("\n[Step 1] Applying system patches...")
    try:
        from patches import deepfilternet_torchaudio_patch
        deepfilternet_torchaudio_patch.apply_patches()
        print("✅ Patches applied successfully.")
    except Exception as e:
        print(f"❌ Failed to apply patches: {e}")
        return

    # 2. Import Libraries
    print("\n[Step 2] Importing DeepFilterNet & Torchaudio...")
    try:
        import torch
        import torchaudio
        from df.enhance import init_df, enhance, save_audio
        from df.io import load_audio as df_load_audio
        print(f"✅ Torchaudio Version: {torchaudio.__version__}")
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return

    # 3. Model Initialization
    print("\n[Step 3] Initializing DeepFilterNet Model...")
    try:
        # init_df logic matching voice_clone_studio.py
        res = init_df()
        if isinstance(res, tuple):
            model, state, params = res
            print("✅ Model loaded (Tuple unpacking successful)")
        else:
            model = res
            state = None
            params = None
            print("✅ Model loaded (Single object)")

        target_sr = params.sr if params and hasattr(params, 'sr') else 48000
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return

    # 4. Audio Loading (The standard failure point)
    print("\n[Step 4] Testing Audio Loading...")
    dummy_file = create_dummy_audio()

    # Test A: Direct Torchaudio Load
    print("\n   [Subtest A] Direct torchaudio.load()...")
    try:
        w, s = torchaudio.load(dummy_file)
        print(f"   ✅ torchaudio.load() SUCCESS. Shape: {w.shape}, SR: {s}")
    except Exception as e:
        print(f"   ❌ torchaudio.load() FAILED: {e}")

    # Test B: DeepFilterNet Load
    print("\n   [Subtest B] DeepFilterNet load_audio()...")
    try:
        audio, _ = df_load_audio(dummy_file, sr=target_sr)
        print(f"   ✅ df_load_audio() SUCCESS. Shape: {audio.shape}")
    except Exception as e:
        print(f"   ❌ df_load_audio() FAILED: {e}")
        print("   ⚠️  ABORTING TEST - Cannot proceed without audio.")
        return

    # 5. Enhancement
    print("\n[Step 5] Running Enhancement (Inference)...")
    try:
        enhanced_audio = enhance(model, df_state=state, audio=audio)
        print(f"✅ Enhancement SUCCESS. Output shape: {enhanced_audio.shape}")
    except Exception as e:
        print(f"❌ Enhancement FAILED: {e}")
        return

    # 6. Saving
    print("\n[Step 6] Saving Output...")
    output_file = "test_output.wav"
    try:
        save_audio(output_file, enhanced_audio, target_sr)
        print(f"✅ Saved to {output_file}")
    except Exception as e:
        print(f"❌ Saving FAILED: {e}")
        return

    print("\n==================================")
    print("    TEST COMPLETED SUCCESSFULLY   ")
    print("==================================")

    # Cleanup
    if os.path.exists(dummy_file):
        os.remove(dummy_file)
    if os.path.exists(output_file):
        os.remove(output_file)

if __name__ == "__main__":
    run_integration_test()
