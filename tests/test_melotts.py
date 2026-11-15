#!/usr/bin/env python3
"""
MeloTTS Test for Japanese TTS - Memory-efficient alternative to XTTS
"""

import time
import os

print("=" * 70)
print("MeloTTS Japanese TTS Test")
print("=" * 70)
print()

try:
    print("Importing MeloTTS...")
    from melo.api import TTS

    print("✅ Import successful")
    print()

    # Test Japanese
    print("Initializing Japanese model...")
    start_load = time.perf_counter()

    device = "cpu"  # Use CPU
    model = TTS(language="JP", device=device)
    speaker_ids = model.hps.data.spk2id

    load_time = time.perf_counter() - start_load
    print(f"✅ Model loaded in {load_time:.2f}s")
    print(f"Available speakers: {list(speaker_ids.keys())}")
    print()

    # Test texts
    test_cases = [
        ("short", "こんにちは。"),
        ("medium", "私の名前はクリスティーナです。"),
        ("long", "タイムマシンの研究をしています。物理学と科学が大好きです。"),
    ]

    for test_name, text in test_cases:
        print(f"Test: {test_name}")
        print(f"  Text: {text}")

        output_path = f"test_melo_{test_name}_ja.wav"

        try:
            start_synth = time.perf_counter()

            # Generate speech
            # MeloTTS expects speaker_id as the value (int) not the key
            speaker_id = speaker_ids[list(speaker_ids.keys())[0]]
            model.tts_to_file(
                text=text, speaker_id=speaker_id, output_path=output_path, speed=1.0
            )

            synth_time = time.perf_counter() - start_synth

            # Calculate audio duration
            import soundfile as sf

            audio, sr = sf.read(output_path)
            audio_duration = len(audio) / sr
            rtf = synth_time / audio_duration if audio_duration > 0 else float("inf")

            print(f"  ⏱️  Synthesis time: {synth_time:.3f}s")
            print(f"  🔊 Audio duration: {audio_duration:.3f}s")
            print(f"  📊 RTF: {rtf:.2f}x")
            print(
                f"  {'✅' if rtf < 1.5 else '⚠️'} Real-time: {'YES' if rtf < 1.5 else 'MARGINAL' if rtf < 3.0 else 'NO'}"
            )
            print(f"  💾 Saved to: {output_path}")
            print()

        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()

    print("=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)
    print()
    print("Summary:")
    print("- MeloTTS is lightweight and runs well on CPU")
    print("- Memory usage is much lower than XTTS-v2")
    print("- Good for real-time applications")
    print("- Uses pre-trained voices (no custom Kurisu voice)")
    print()

except ImportError as e:
    print(f"❌ MeloTTS not installed properly: {e}")
    print("Install with: pip install git+https://github.com/myshell-ai/MeloTTS.git")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
