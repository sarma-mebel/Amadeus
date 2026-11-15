#!/usr/bin/env python3
"""
Clone Kurisu's voice with Qwen3-TTS (zero-shot) using the reference clip
in models/kurisu_voice_qwen, and synthesize new Japanese text. One-off
utility; the assistant integration lives in
amadeus/providers/tts/qwen3_tts_provider.py.

Usage:
    python clone_kurisu_voice.py --text "何よ、その顔は。" --out output/kurisu_line.wav
"""

import argparse
import os

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

REF_AUDIO = "models/kurisu_voice_qwen/kurisu_ref.wav"
REF_TEXT = "ビクトルコンドリア大学の精神生理学研究所がビジュアルリビルディング"
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Target Japanese text to synthesize")
    parser.add_argument("--out", default="clone_output/kurisu_clone.wav")
    parser.add_argument("--ref_audio", default=REF_AUDIO)
    parser.add_argument("--ref_text", default=REF_TEXT)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    wavs, sr = model.generate_voice_clone(
        text=args.text,
        language="Japanese",
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
    )

    sf.write(args.out, wavs[0], sr)
    print(f"Wrote {args.out} ({len(wavs[0]) / sr:.2f}s @ {sr}Hz)")


if __name__ == "__main__":
    main()
