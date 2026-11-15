#!/usr/bin/env python3
"""
Transcribe Kurisu audio clips using Whisper-large-v3.
Run this script on A40 GPU for fast transcription.

Usage:
    python transcribe_kurisu_clips.py --clips_dir kurisu_dataset/clips --output transcriptions.json
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import whisper

def transcribe_clips(clips_dir, output_file, model_name="large-v3", device="cuda", language="ja"):
    """
    Transcribe all audio clips using Whisper.

    Args:
        clips_dir: Directory containing audio clips
        output_file: Output JSON file for transcriptions
        model_name: Whisper model to use (large-v3 recommended)
        device: Device to use (cuda for GPU, cpu for CPU)
        language: Language code (ja for Japanese)
    """
    print("=" * 70)
    print("Kurisu Audio Transcription with Whisper-large-v3")
    print("=" * 70)
    print()

    clips_path = Path(clips_dir)
    if not clips_path.exists():
        print(f"Error: Clips directory '{clips_dir}' not found")
        return

    # Get all WAV files
    audio_files = sorted(clips_path.glob("*.wav"))
    print(f"Found {len(audio_files)} audio clips")
    print()

    # Load Whisper model
    print(f"Loading Whisper model: {model_name}")
    print(f"Device: {device}")
    model = whisper.load_model(model_name, device=device)
    print("✅ Model loaded successfully")
    print()

    # Transcribe all clips
    transcriptions = {}
    errors = []

    print("Transcribing clips...")
    print("(This may take a while depending on GPU speed)")
    print()

    for audio_file in tqdm(audio_files, desc="Transcribing"):
        try:
            # Transcribe with Whisper
            result = model.transcribe(
                str(audio_file),
                language=language,
                task="transcribe",
                fp16=(device == "cuda"),  # Use FP16 on GPU for speed
                verbose=False
            )

            transcriptions[audio_file.name] = {
                "text": result["text"].strip(),
                "language": result.get("language", language),
                "segments": result.get("segments", []),
            }

        except Exception as e:
            errors.append({
                "file": audio_file.name,
                "error": str(e)
            })
            print(f"\n⚠️  Error transcribing {audio_file.name}: {e}")

    print()
    print("=" * 70)
    print("Transcription Complete!")
    print("=" * 70)
    print()
    print(f"Total clips: {len(audio_files)}")
    print(f"Successfully transcribed: {len(transcriptions)}")
    print(f"Errors: {len(errors)}")
    print()

    # Save transcriptions to JSON
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "transcriptions": transcriptions,
            "errors": errors,
            "model": model_name,
            "language": language,
            "total_clips": len(audio_files),
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ Transcriptions saved to: {output_file}")
    print()

    # Show some sample transcriptions
    print("Sample transcriptions:")
    print("-" * 70)
    for i, (filename, data) in enumerate(list(transcriptions.items())[:5]):
        print(f"{i+1}. {filename}")
        print(f"   Text: {data['text']}")
        print()

    if errors:
        print(f"⚠️  {len(errors)} errors occurred. Check the output JSON for details.")
        print()

    return transcriptions


def create_melotts_metadata(transcriptions_file, clips_dir, output_file="metadata.list",
                            speaker_name="Kurisu", language="JP"):
    """
    Create MeloTTS metadata.list file from transcriptions.

    Format: path/to/audio.wav|speaker_name|language|text
    """
    print("=" * 70)
    print("Creating MeloTTS Metadata File")
    print("=" * 70)
    print()

    # Load transcriptions
    with open(transcriptions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    transcriptions = data['transcriptions']

    # Create metadata.list
    clips_path = Path(clips_dir)
    metadata_lines = []

    for filename, trans_data in transcriptions.items():
        audio_path = clips_path / filename
        text = trans_data['text']

        # Skip empty transcriptions
        if not text or text.strip() == "":
            continue

        # Format: audio_path|speaker|language|text
        metadata_line = f"{audio_path}|{speaker_name}|{language}|{text}"
        metadata_lines.append(metadata_line)

    # Save metadata.list
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(metadata_lines))

    print(f"✅ Created metadata.list with {len(metadata_lines)} entries")
    print(f"   Saved to: {output_file}")
    print()
    print("Sample entries:")
    print("-" * 70)
    for line in metadata_lines[:3]:
        print(line)
    print()

    return metadata_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe Kurisu clips with Whisper-large-v3")
    parser.add_argument("--clips_dir", default="kurisu_dataset/clips",
                       help="Directory containing audio clips")
    parser.add_argument("--output", default="kurisu_dataset/transcriptions.json",
                       help="Output JSON file for transcriptions")
    parser.add_argument("--model", default="large-v3",
                       help="Whisper model to use (default: large-v3)")
    parser.add_argument("--device", default="cuda",
                       help="Device to use: cuda or cpu (default: cuda)")
    parser.add_argument("--language", default="ja",
                       help="Language code (default: ja for Japanese)")
    parser.add_argument("--create-metadata", action="store_true",
                       help="Also create metadata.list file after transcription")
    parser.add_argument("--metadata-output", default="kurisu_dataset/metadata.list",
                       help="Output path for metadata.list")

    args = parser.parse_args()

    # Transcribe clips
    transcriptions = transcribe_clips(
        clips_dir=args.clips_dir,
        output_file=args.output,
        model_name=args.model,
        device=args.device,
        language=args.language
    )

    # Create metadata.list if requested
    if args.create_metadata and transcriptions:
        create_melotts_metadata(
            transcriptions_file=args.output,
            clips_dir=args.clips_dir,
            output_file=args.metadata_output
        )

        print("=" * 70)
        print("All Done!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("1. Review metadata.list to ensure quality")
        print("2. Run MeloTTS preprocessing to generate config.json")
        print("3. Start training on A40 GPU")
        print()
