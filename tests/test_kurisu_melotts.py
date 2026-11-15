#!/usr/bin/env python3
"""Test the trained Kurisu MeloTTS voice.

This script tests the custom trained MeloTTS model for Kurisu's voice.
Run this after training completes and you've copied the checkpoint and config.

Usage:
    python test_kurisu_melotts.py
"""

#!/usr/bin/env python3
"""Test the trained Kurisu MeloTTS voice.

This script tests the custom trained MeloTTS model for Kurisu's voice.
Run this after training completes and you've copied the checkpoint and config.

Usage:
    python test_kurisu_melotts.py
"""

import time
import pytest
from pathlib import Path


def test_kurisu_melotts():
    print("=" * 70)
    print("Kurisu MeloTTS Voice Test")
    print("=" * 70)
    print()

    # Check if model files exist
    config_path = Path("models/kurisu_voice/config.json")
    checkpoint_path = Path("models/kurisu_voice/G_20000.pth")

    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        pytest.skip(f"Config not found: {config_path}")
        return

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")
        return

    print("✅ Found model files:")
    print(f"   Config: {config_path}")
    print(f"   Checkpoint: {checkpoint_path}")
    print()

    try:
        print("Loading MeloTTS provider...")
        from amadeus.providers.tts.melotts_provider import MeloTTSProvider

        start_load = time.perf_counter()
        tts = MeloTTSProvider(
            config_path=str(config_path),
            checkpoint_path=str(checkpoint_path),
            language="JP",
            device="cpu",  # Change to "cuda" if you have GPU
        )
        load_time = time.perf_counter() - start_load
        print(f"✅ Model loaded in {load_time:.2f}s")
        print()

        # Test Japanese sentences (Kurisu-style)
        test_texts = [
            ("greeting", "こんにちは、私はクリスティーナです。"),
            ("research", "タイムマシンの研究をしています。"),
            ("tsundere", "バカ！何を言っているの？"),
            ("scientific", "物理学と科学が大好きです。"),
            ("casual", "よろしくね。"),
        ]

        print("Running synthesis tests...")
        print("=" * 70)
        print()

        for test_name, text in test_texts:
            print(f"Test: {test_name}")
            print(f"  Text (JP): {text}")

            try:
                start_synth = time.perf_counter()

                # Synthesize speech
                audio_bytes = tts.synthesize(text, speaking_rate=1.0)

                synth_time = time.perf_counter() - start_synth

                # Calculate RTF (Real-Time Factor)
                # Approximate audio duration from file size (rough estimate)
                # For accurate RTF, we'd need to parse WAV header
                estimated_duration = len(audio_bytes) / (44100 * 2)  # Rough estimate
                rtf = (
                    synth_time / estimated_duration
                    if estimated_duration > 0
                    else float("inf")
                )

                print(f"  ⏱️  Synthesis time: {synth_time:.3f}s")
                print(f"  💾 Audio size: {len(audio_bytes):,} bytes")
                print(f"  📊 RTF (estimated): {rtf:.2f}x")
                print(
                    f"  {'✅' if rtf < 1.5 else '⚠️'} Real-time: {'YES' if rtf < 1.5 else 'MARGINAL' if rtf < 3.0 else 'NO'}"
                )

                # Save to file
                output_file = f"test_kurisu_{test_name}.wav"
                with open(output_file, "wb") as f:
                    f.write(audio_bytes)
                print(f"  💾 Saved to: {output_file}")
                print()

            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback

                traceback.print_exc()
                print()
                pytest.fail(f"Synthesis failed: {e}")

        print("=" * 70)
        print("✅ Test Complete!")
        print("=" * 70)

    except ImportError as e:
        print(f"❌ MeloTTS not installed properly: {e}")
        pytest.skip(f"MeloTTS not installed: {e}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Test failed: {e}")


if __name__ == "__main__":
    # For manual running, we can just call the function
    # We mock pytest.skip/fail to just print/exit
    class MockPytest:
        @staticmethod
        def skip(msg):
            print(f"SKIPPED: {msg}")
            exit(0)

        @staticmethod
        def fail(msg):
            print(f"FAILED: {msg}")
            exit(1)

    # Temporarily replace pytest with mock for manual run
    original_pytest = pytest
    pytest = MockPytest()
    try:
        test_kurisu_melotts()
    finally:
        pytest = original_pytest
