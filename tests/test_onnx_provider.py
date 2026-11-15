#!/usr/bin/env python3
"""Test the ONNX TTS provider for quality and performance."""

import time
from pathlib import Path

# Test ONNX provider
print("=" * 60)
print("Testing MeloTTS ONNX Provider")
print("=" * 60)

from amadeus.providers.tts.melotts_onnx_provider import MeloTTSONNXProvider

model_path = "models/kurisu_voice_onnx/model.onnx"
tokens_path = "models/kurisu_voice_onnx/tokens.txt"

print(f"\nInitializing ONNX provider...")
print(f"  Model: {model_path}")
print(f"  Tokens: {tokens_path}")

start_init = time.time()
onnx_provider = MeloTTSONNXProvider(
    model_path=model_path,
    tokens_path=tokens_path,
    speaker_id=0,
)
init_time = time.time() - start_init

print(f"✅ Initialization complete ({init_time:.2f}s)")
print(f"  Sample rate: {onnx_provider.sample_rate} Hz")

# Test synthesis with a Japanese sentence
test_text = "こんにちは、私は牧瀬紅莉栖です。"
print(f"\nSynthesizing: '{test_text}'")

start_synth = time.time()
audio_bytes = onnx_provider.synthesize(test_text, speaking_rate=1.4)
synth_time = time.time() - start_synth

print(f"✅ Synthesis complete ({synth_time:.2f}s)")
print(f"  Audio size: {len(audio_bytes):,} bytes")

# Save output for manual inspection
output_path = Path("test_onnx_output.wav")
with open(output_path, "wb") as f:
    f.write(audio_bytes)
print(f"  Saved to: {output_path}")

print("\n" + "=" * 60)
print("ONNX Provider Test Complete!")
print("=" * 60)
print(f"\n📊 Performance Summary:")
print(f"  Initialization: {init_time:.2f}s")
print(f"  Synthesis: {synth_time:.2f}s")
print(f"  Total: {init_time + synth_time:.2f}s")
