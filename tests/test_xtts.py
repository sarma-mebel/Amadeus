#!/usr/bin/env python3
"""
XTTS-v2 Test Script for Japanese TTS

Tests XTTS-v2 with Japanese text to verify installation and measure performance.
"""

import os
import time
from pathlib import Path

# Accept XTTS non-commercial license (CPML)
os.environ["COQUI_TOS_AGREED"] = "1"

# ---- PyTorch 2.6 safe-load fix for Coqui XTTS ----
import torch

try:
    from TTS.tts.configs.xtts_config import XttsConfig

    # Allow-list the class used inside the checkpoint
    torch.serialization.add_safe_globals([XttsConfig])
    print("[patch] Registered XttsConfig in torch safe_globals")
except Exception as e:
    print("[patch] Could not register safe_globals:", e)
# ---------------------------------------------------


from TTS.api import TTS


def test_xtts_japanese():
    """Test XTTS-v2 with Japanese text."""
    print("=" * 70)
    print("XTTS-v2 Japanese TTS Test")
    print("=" * 70)
    print()

    # Test texts
    test_cases = [
        ("short", "こんにちは。"),
        ("medium", "私の名前はクリスティーナです。"),
        ("long", "タイムマシンの研究をしています。物理学と科学が大好きです。"),
    ]

    print("Loading XTTS-v2 model...")
    print("This may take a while on first run (downloading model)...")
    start_load = time.perf_counter()

    try:
        # Initialize XTTS-v2
        # Using multilingual model which supports Japanese
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

        load_time = time.perf_counter() - start_load
        print(f"✅ Model loaded in {load_time:.2f}s")
        print()

        # List available languages
        print(
            "Available languages:",
            tts.languages if hasattr(tts, "languages") else "Check TTS documentation",
        )
        print()

        # Test with each case
        for test_name, text in test_cases:
            print(f"Test: {test_name}")
            print(f"  Text: {text}")

            output_path = f"test_xtts_{test_name}_ja.wav"

            try:
                start_time = time.perf_counter()

                # Generate speech
                # XTTS-v2 requires a reference audio for voice cloning
                # For initial testing, we'll use the model's default voice
                # Note: You'll need to provide speaker_wav and language for best results
                tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    language="ja",  # Japanese language code
                )

                synthesis_time = time.perf_counter() - start_time

                # Calculate audio duration
                import soundfile as sf

                audio, sr = sf.read(output_path)
                audio_duration = len(audio) / sr

                # Calculate RTF
                rtf = (
                    synthesis_time / audio_duration
                    if audio_duration > 0
                    else float("inf")
                )

                print(f"  ⏱️  Synthesis time: {synthesis_time:.3f}s")
                print(f"  🔊 Audio duration: {audio_duration:.3f}s")
                print(f"  📊 RTF: {rtf:.2f}x")
                print(
                    f"  {'✅' if rtf < 1.5 else '⚠️'} Real-time: {'YES' if rtf < 1.5 else 'MARGINAL' if rtf < 3.0 else 'NO'}"
                )
                print(f"  💾 Saved to: {output_path}")
                print()

            except Exception as exc:
                print(f"  ❌ Error: {exc}")
                print()

        print("=" * 70)
        print("Test Complete!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("1. Listen to the generated audio files")
        print("2. If quality is good, proceed with Kurisu voice training")
        print("3. For voice cloning, you'll need:")
        print("   - Reference audio (your 3-hour Kurisu dataset)")
        print("   - Transcriptions for the audio")
        print()

    except Exception as exc:
        print(f"❌ Failed to load XTTS-v2: {exc}")
        print()
        print("Troubleshooting:")
        print("- Ensure TTS package is installed: pip install TTS")
        print("- Check internet connection (model downloads on first run)")
        print("- Try updating TTS: pip install --upgrade TTS")
        return False

    return True


def test_xtts_with_voice_cloning():
    """
    Test XTTS-v2 with voice cloning capability.

    This function demonstrates how voice cloning will work once you have
    your Kurisu reference audio ready.
    """
    print("=" * 70)
    print("XTTS-v2 Voice Cloning Test (Reference Example)")
    print("=" * 70)
    print()

    print("Voice cloning with XTTS-v2 requires:")
    print("1. A reference audio file (WAV, 6-30 seconds recommended)")
    print("2. Language specification")
    print("3. Text to synthesize")
    print()
    print("Example code:")
    print()
    print("```python")
    print("tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2')")
    print("tts.tts_to_file(")
    print("    text='こんにちは、私はクリスティーナです。',")
    print("    file_path='output.wav',")
    print("    speaker_wav='kurisu_reference.wav',  # Your reference audio")
    print("    language='ja'")
    print(")")
    print("```")
    print()
    print("For best results with your 3-hour dataset:")
    print("- Use dataset preparation script (coming next)")
    print("- Fine-tune model on full dataset")
    print("- This will create a custom Kurisu voice model")
    print()


if __name__ == "__main__":
    print("\n")
    success = test_xtts_japanese()

    if success:
        print("\n")
        test_xtts_with_voice_cloning()
