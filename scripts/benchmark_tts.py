#!/usr/bin/env python3
"""
TTS Performance Benchmark Script

Tests the current TTS providers to measure synthesis latency and determine
if real-time performance is achievable for the Amadeus voice assistant.
"""

import time
import sys
from pathlib import Path

# Add amadeus to path
sys.path.insert(0, str(Path(__file__).parent))

from amadeus.config import Config
from amadeus.cli import load_tts


# Test cases with varying complexity
TEST_CASES = {
    "short_japanese": "こんにちは。",
    "medium_japanese": "テストです。今日はいい天気ですね。",
    "long_japanese": "私の名前はクリスティーナです。タイムマシンの研究をしています。物理学と科学が大好きです。",
    "very_long_japanese": "ラボメンナンバー004、牧瀬紅莉栖です。アメリカの青葉大学脳科学研究所を経て、現在はビクトル・コンドリア大学の研究員として活動しています。専門は脳の記憶システムに関する研究です。",
    "short_english": "Hello there.",
    "medium_english": "This is a test. The weather is nice today.",
    "long_english": "My name is Christina. I research time machines. I love physics and science.",
}


def benchmark_tts_provider(provider_name: str, config: Config):
    """Benchmark a specific TTS provider."""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {provider_name}")
    print(f"{'='*70}\n")

    # Load TTS provider
    try:
        config.providers.tts = provider_name
        tts = load_tts(config)
    except Exception as exc:
        print(f"❌ Failed to load {provider_name}: {exc}")
        return None

    results = []

    for test_name, text in TEST_CASES.items():
        print(f"Test: {test_name}")
        print(f"  Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"  Length: {len(text)} chars")

        try:
            # Measure synthesis time
            start_time = time.perf_counter()
            audio_bytes = tts.synthesize(text)
            end_time = time.perf_counter()

            synthesis_time = end_time - start_time

            # Calculate audio duration (assuming WAV format)
            import io
            import wave
            with io.BytesIO(audio_bytes) as buffer:
                with wave.open(buffer, "rb") as wav:
                    frames = wav.getnframes()
                    rate = wav.getframerate()
                    audio_duration = frames / rate

            # Calculate Real-Time Factor (RTF)
            # RTF < 1.0 means faster than real-time (good for real-time apps)
            # RTF = 1.0 means exactly real-time
            # RTF > 1.0 means slower than real-time (bad for real-time apps)
            rtf = synthesis_time / audio_duration if audio_duration > 0 else float('inf')

            # Determine if suitable for real-time
            real_time_suitable = rtf < 1.5  # Allow some overhead

            print(f"  ⏱️  Synthesis time: {synthesis_time:.3f}s")
            print(f"  🔊 Audio duration: {audio_duration:.3f}s")
            print(f"  📊 RTF: {rtf:.2f}x")
            print(f"  {'✅' if real_time_suitable else '❌'} Real-time: {'YES' if real_time_suitable else 'NO'}")
            print()

            results.append({
                "test": test_name,
                "text_length": len(text),
                "synthesis_time": synthesis_time,
                "audio_duration": audio_duration,
                "rtf": rtf,
                "real_time_suitable": real_time_suitable,
            })

        except Exception as exc:
            print(f"  ❌ Error: {exc}\n")
            results.append({
                "test": test_name,
                "error": str(exc),
            })

    return results


def print_summary(provider_name: str, results: list):
    """Print summary statistics."""
    if not results:
        return

    print(f"\n{'='*70}")
    print(f"SUMMARY: {provider_name}")
    print(f"{'='*70}\n")

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    if successful:
        avg_rtf = sum(r["rtf"] for r in successful) / len(successful)
        avg_synthesis_time = sum(r["synthesis_time"] for r in successful) / len(successful)
        real_time_count = sum(1 for r in successful if r["real_time_suitable"])

        print(f"✅ Successful tests: {len(successful)}/{len(results)}")
        print(f"📊 Average RTF: {avg_rtf:.2f}x")
        print(f"⏱️  Average synthesis time: {avg_synthesis_time:.3f}s")
        print(f"🚀 Real-time suitable: {real_time_count}/{len(successful)} tests")
        print()

        if avg_rtf < 1.0:
            print("🎉 EXCELLENT: This TTS is faster than real-time!")
            print("   Recommended for real-time web app use.")
        elif avg_rtf < 1.5:
            print("✅ GOOD: This TTS is acceptable for real-time use.")
            print("   May have slight delays, but usable for web app.")
        elif avg_rtf < 3.0:
            print("⚠️  MARGINAL: This TTS may feel sluggish.")
            print("   Consider optimization or alternative models.")
        else:
            print("❌ SLOW: This TTS is too slow for real-time use.")
            print("   Strongly recommend switching to a faster model (e.g., MeloTTS).")

    if failed:
        print(f"\n❌ Failed tests: {len(failed)}")
        for r in failed:
            print(f"  - {r['test']}: {r['error']}")

    print()


def main():
    """Main benchmark function."""
    print("=" * 70)
    print("TTS PERFORMANCE BENCHMARK FOR AMADEUS")
    print("=" * 70)
    print("\nThis script will test the current TTS providers to measure")
    print("synthesis latency and real-time suitability.\n")

    # Load config
    config = Config.load()

    # List of providers to test
    providers_to_test = []

    # Always test the currently configured provider
    current_provider = config.providers.tts
    providers_to_test.append(current_provider)

    # Test HuggingFace TTS (mio/amadeus) if available
    if current_provider != "huggingface" and config.providers.huggingface_tts_path:
        providers_to_test.append("huggingface")

    # Test pyttsx3 as a baseline (fast but low quality)
    if current_provider != "pyttsx3":
        providers_to_test.append("pyttsx3")

    # Run benchmarks
    all_results = {}
    for provider in providers_to_test:
        results = benchmark_tts_provider(provider, config)
        if results:
            all_results[provider] = results
            print_summary(provider, results)

    # Final comparison
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70 + "\n")

        for provider, results in all_results.items():
            successful = [r for r in results if "error" not in r]
            if successful:
                avg_rtf = sum(r["rtf"] for r in successful) / len(successful)
                print(f"{provider:20s}: RTF {avg_rtf:.2f}x")
        print()


if __name__ == "__main__":
    main()
