#!/usr/bin/env python3
"""
Minimal XTTS-v2 Test - Testing one short Japanese phrase
"""

import os
import time

# Accept XTTS non-commercial license (CPML)
os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.api import TTS

print("=" * 70)
print("XTTS-v2 Minimal Japanese Test")
print("=" * 70)
print()

print("Loading XTTS-v2 model...")
start_time = time.perf_counter()

try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    load_time = time.perf_counter() - start_time
    print(f"✅ Model loaded in {load_time:.2f}s")
    print()

    # Test with one short Japanese phrase
    test_text = "こんにちは。"
    output_path = "test_xtts_minimal.wav"

    print(f"Testing Japanese synthesis...")
    print(f"Text: {test_text}")

    start_synth = time.perf_counter()

    # Generate speech - for testing without reference audio
    # XTTS v2 can use the default speaker embedding
    tts.tts_to_file(text=test_text, file_path=output_path, language="ja")

    synth_time = time.perf_counter() - start_synth

    # Calculate audio duration
    import soundfile as sf

    audio, sr = sf.read(output_path)
    audio_duration = len(audio) / sr
    rtf = synth_time / audio_duration if audio_duration > 0 else float("inf")

    print(f"✅ Synthesis complete!")
    print(f"  ⏱️  Synthesis time: {synth_time:.3f}s")
    print(f"  🔊 Audio duration: {audio_duration:.3f}s")
    print(f"  📊 RTF: {rtf:.2f}x")
    print(
        f"  {'✅' if rtf < 1.5 else '⚠️'} Real-time: {'YES' if rtf < 1.5 else 'MARGINAL' if rtf < 3.0 else 'NO'}"
    )
    print(f"  💾 Saved to: {output_path}")
    print()
    print("=" * 70)
    print("✅ Test completed successfully!")
    print("=" * 70)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
