#!/usr/bin/env python3
"""
Benchmark pymlask emotion analysis performance on CPU.
Tests inference speed, memory usage, and accuracy for Japanese text.
"""

import time
import psutil
import os
from mlask import MLAsk

def benchmark_pymlask():
    """Benchmark pymlask performance on CPU."""
    print("=" * 70)
    print("pymlask Performance Benchmark (CPU)")
    print("=" * 70)
    print()

    # Test texts (typical Kurisu responses)
    test_texts = [
        "とても嬉しいです！",  # Very happy!
        "ふん、当然よ",  # Hmph, of course (tsundere)
        "悲しいね...",  # It's sad...
        "これは面白い実験だ",  # This is an interesting experiment
        "怒っているわけじゃない！",  # I'm not angry!
        "びっくりした！",  # I'm surprised!
        "そうね、落ち着いて考えましょう",  # Let's think calmly
        "科学的に考えれば分かるでしょう",  # You'll understand if you think scientifically
        "まったく、どうしてこうなるの？",  # Seriously, why does this happen?
        "やった！成功したわ！",  # Yes! It succeeded!
    ]

    print(f"Test dataset: {len(test_texts)} Japanese texts")
    print()

    # Initialize MLAsk
    print("Initializing MLAsk...")
    init_start = time.perf_counter()
    mlask = MLAsk()
    init_time = (time.perf_counter() - init_start) * 1000
    print(f"✅ Initialization time: {init_time:.2f}ms")
    print()

    # Memory usage
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory usage: {memory_before:.2f}MB")
    print()

    # Warm-up run
    print("Warming up...")
    for text in test_texts[:3]:
        mlask.analyze(text)
    print("✅ Warm-up complete")
    print()

    # Benchmark inference speed
    print("=" * 70)
    print("Inference Speed Benchmark")
    print("=" * 70)
    print()

    inference_times = []

    print("Running individual inferences:")
    for i, text in enumerate(test_texts, 1):
        start = time.perf_counter()
        result = mlask.analyze(text)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        inference_times.append(elapsed)

        emotions = result.get('emotion', {})
        print(f"{i}. \"{text}\"")
        print(f"   Time: {elapsed:.3f}ms")
        print(f"   Emotions: {list(emotions.keys()) if emotions else ['neutral']}")
        print()

    # Statistics
    avg_time = sum(inference_times) / len(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)

    print("=" * 70)
    print("Performance Statistics")
    print("=" * 70)
    print()
    print(f"Average inference time: {avg_time:.3f}ms")
    print(f"Minimum inference time: {min_time:.3f}ms")
    print(f"Maximum inference time: {max_time:.3f}ms")
    print()

    # Throughput test
    print("=" * 70)
    print("Throughput Test (1000 inferences)")
    print("=" * 70)
    print()

    test_iterations = 1000
    throughput_start = time.perf_counter()

    for i in range(test_iterations):
        text = test_texts[i % len(test_texts)]
        mlask.analyze(text)

    throughput_time = time.perf_counter() - throughput_start
    throughput_per_sec = test_iterations / throughput_time
    avg_latency = (throughput_time / test_iterations) * 1000

    print(f"Total time: {throughput_time:.3f}s")
    print(f"Throughput: {throughput_per_sec:.1f} inferences/second")
    print(f"Average latency: {avg_latency:.3f}ms")
    print()

    # Memory after benchmark
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_delta = memory_after - memory_before

    print("=" * 70)
    print("Memory Analysis")
    print("=" * 70)
    print()
    print(f"Memory before: {memory_before:.2f}MB")
    print(f"Memory after: {memory_after:.2f}MB")
    print(f"Memory increase: {memory_delta:.2f}MB")
    print()

    # Real-time suitability
    print("=" * 70)
    print("Real-Time Suitability Assessment")
    print("=" * 70)
    print()

    real_time_threshold = 10  # 10ms is good for real-time
    acceptable_threshold = 50  # 50ms is acceptable

    if avg_time < real_time_threshold:
        status = "✅ EXCELLENT - Real-time capable"
    elif avg_time < acceptable_threshold:
        status = "✅ GOOD - Acceptable for real-time"
    else:
        status = "⚠️ SLOW - May cause noticeable delay"

    print(f"Average latency: {avg_time:.3f}ms")
    print(f"Status: {status}")
    print()

    if avg_time < acceptable_threshold:
        print("✅ pymlask is suitable for real-time emotion analysis!")
        print("   Can be integrated into response pipeline with negligible delay")
    else:
        print("⚠️ Consider optimization or alternative approach")

    print()

    # Emotion detection examples
    print("=" * 70)
    print("Emotion Detection Examples")
    print("=" * 70)
    print()

    example_texts = {
        "Happy": "とても嬉しいです！大成功だわ！",
        "Angry": "ふざけないで！怒ってるのよ！",
        "Sad": "とても悲しい...涙が出そう...",
        "Surprised": "えっ！？びっくりした！",
        "Neutral": "そうですね、科学的に考えましょう",
    }

    for category, text in example_texts.items():
        result = mlask.analyze(text)
        emotions = result.get('emotion', {})

        print(f"{category}: \"{text}\"")
        print(f"  Detected: {list(emotions.keys()) if emotions else ['none']}")
        print()

    print("=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)
    print()

    return {
        "avg_inference_time_ms": avg_time,
        "throughput_per_sec": throughput_per_sec,
        "memory_usage_mb": memory_after,
        "is_realtime_suitable": avg_time < acceptable_threshold
    }


if __name__ == "__main__":
    results = benchmark_pymlask()

    # Save results
    import json
    with open("emotion_benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to: emotion_benchmark_results.json")
