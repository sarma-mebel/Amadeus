#!/usr/bin/env python3
"""
Dataset Preparation Script for Kurisu Voice Training

Prepares 3-hour WAV audio for XTTS-v2 fine-tuning by:
1. Splitting into segments
2. Cleaning/normalizing audio
3. Generating transcriptions (if needed)
4. Creating dataset structure for training

Usage:
    python prepare_kurisu_dataset.py --input kurisu_3hour.wav --output_dir kurisu_dataset
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from tqdm import tqdm


def load_audio(file_path):
    """Load audio file."""
    print(f"Loading audio from: {file_path}")
    audio, sr = sf.read(file_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        print("  Converting stereo to mono...")
        audio = np.mean(audio, axis=1)

    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(audio) / sr / 60:.2f} minutes")
    print(f"  Audio shape: {audio.shape}")

    return audio, sr


def normalize_audio(audio):
    """Normalize audio to -3dB peak."""
    peak = np.abs(audio).max()
    if peak > 0:
        # Normalize to 0.7 (-3dB)
        audio = audio * (0.7 / peak)
    return audio


def detect_silence(audio, sr, threshold=-40, min_silence_len=0.5):
    """
    Detect silence periods in audio.

    Args:
        audio: Audio signal
        sr: Sample rate
        threshold: Silence threshold in dB
        min_silence_len: Minimum silence length in seconds

    Returns:
        List of (start, end) tuples in samples
    """
    # Convert to dB
    audio_db = 20 * np.log10(np.abs(audio) + 1e-10)

    # Find silent regions
    is_silent = audio_db < threshold
    min_silence_samples = int(min_silence_len * sr)

    silence_regions = []
    in_silence = False
    silence_start = 0

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            silence_start = i
            in_silence = True
        elif not silent and in_silence:
            silence_end = i
            if silence_end - silence_start >= min_silence_samples:
                silence_regions.append((silence_start, silence_end))
            in_silence = False

    return silence_regions


def split_audio_by_silence(audio, sr, max_segment_len=30, min_segment_len=3):
    """
    Split audio into segments using silence detection.

    Args:
        audio: Audio signal
        sr: Sample rate
        max_segment_len: Maximum segment length in seconds
        min_segment_len: Minimum segment length in seconds

    Returns:
        List of audio segments
    """
    print("Detecting silence and splitting audio...")

    # Detect silence
    silence_regions = detect_silence(audio, sr)

    if not silence_regions:
        print("  No silence detected, splitting by fixed length...")
        # Fall back to fixed-length splitting
        return split_audio_fixed_length(audio, sr, max_segment_len)

    print(f"  Found {len(silence_regions)} silence regions")

    segments = []
    current_start = 0
    max_samples = int(max_segment_len * sr)
    min_samples = int(min_segment_len * sr)

    for silence_start, silence_end in silence_regions:
        # If segment is long enough, save it
        segment_len = silence_start - current_start

        if segment_len >= min_samples:
            # If segment is too long, split it
            if segment_len > max_samples:
                # Split into multiple segments
                num_splits = int(np.ceil(segment_len / max_samples))
                split_size = segment_len // num_splits

                for i in range(num_splits):
                    split_start = current_start + i * split_size
                    split_end = min(split_start + split_size, silence_start)
                    segments.append(audio[split_start:split_end])
            else:
                # Save as single segment
                segments.append(audio[current_start:silence_start])

        # Move to next segment
        current_start = silence_end

    # Add final segment if long enough
    if len(audio) - current_start >= min_samples:
        segments.append(audio[current_start:])

    print(f"  Created {len(segments)} segments")
    return segments


def split_audio_fixed_length(audio, sr, segment_len=30):
    """Split audio into fixed-length segments."""
    segment_samples = int(segment_len * sr)
    segments = []

    for i in range(0, len(audio), segment_samples):
        segment = audio[i:i + segment_samples]
        if len(segment) >= int(3 * sr):  # Minimum 3 seconds
            segments.append(segment)

    return segments


def save_segments(segments, sr, output_dir):
    """Save audio segments to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(exist_ok=True)

    print(f"Saving {len(segments)} segments to {wavs_dir}...")

    segment_info = []

    for i, segment in enumerate(tqdm(segments)):
        # Normalize segment
        segment = normalize_audio(segment)

        # Save segment
        filename = f"kurisu_{i:04d}.wav"
        filepath = wavs_dir / filename
        sf.write(filepath, segment, sr)

        # Store info
        duration = len(segment) / sr
        segment_info.append({
            "filename": filename,
            "duration": duration,
            "text": ""  # Placeholder for transcription
        })

    return segment_info


def create_metadata_file(segment_info, output_dir):
    """Create metadata file for XTTS-v2 training."""
    output_dir = Path(output_dir)

    # Create metadata.csv for Coqui TTS format
    metadata_path = output_dir / "metadata.csv"

    print(f"Creating metadata file: {metadata_path}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        for info in segment_info:
            # Format: filename|text
            f.write(f"{info['filename']}|{info['text']}\n")

    print(f"  Wrote {len(segment_info)} entries")
    print()
    print("⚠️  IMPORTANT: You need to add transcriptions!")
    print("  Edit metadata.csv and add Japanese text for each audio file.")
    print("  Format: filename.wav|Japanese transcription text")
    print()

    # Also save as JSON for easier editing
    json_path = output_dir / "segments.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segment_info, f, indent=2, ensure_ascii=False)

    print(f"  Also saved as JSON: {json_path}")


def create_dataset_structure(output_dir):
    """Create necessary dataset directory structure."""
    output_dir = Path(output_dir)

    # Create directories
    (output_dir / "wavs").mkdir(parents=True, exist_ok=True)

    print(f"Dataset structure created in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Kurisu voice dataset for XTTS-v2 training")
    parser.add_argument("--input", required=True, help="Input WAV file (3-hour Kurisu audio)")
    parser.add_argument("--output_dir", default="kurisu_dataset", help="Output directory for dataset")
    parser.add_argument("--max_segment_len", type=int, default=30, help="Maximum segment length in seconds")
    parser.add_argument("--min_segment_len", type=int, default=3, help="Minimum segment length in seconds")
    parser.add_argument("--silence_threshold", type=int, default=-40, help="Silence threshold in dB")

    args = parser.parse_args()

    print("=" * 70)
    print("Kurisu Voice Dataset Preparation")
    print("=" * 70)
    print()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    create_dataset_structure(output_dir)
    print()

    # Load audio
    audio, sr = load_audio(input_path)
    print()

    # Resample if needed (XTTS-v2 works best with 22050 Hz)
    target_sr = 22050
    if sr != target_sr:
        print(f"Resampling from {sr} Hz to {target_sr} Hz...")
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        print()

    # Split audio
    segments = split_audio_by_silence(
        audio, sr,
        max_segment_len=args.max_segment_len,
        min_segment_len=args.min_segment_len
    )
    print()

    # Calculate statistics
    total_duration = sum(len(seg) / sr for seg in segments)
    avg_duration = total_duration / len(segments)

    print(f"Dataset statistics:")
    print(f"  Total segments: {len(segments)}")
    print(f"  Total duration: {total_duration / 60:.2f} minutes")
    print(f"  Average segment length: {avg_duration:.2f} seconds")
    print()

    # Save segments
    segment_info = save_segments(segments, sr, output_dir)
    print()

    # Create metadata
    create_metadata_file(segment_info, output_dir)

    print("=" * 70)
    print("Dataset Preparation Complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print(f"1. Add transcriptions to: {output_dir}/metadata.csv")
    print("2. Verify audio quality in: {output_dir}/wavs/")
    print("3. Run training script with prepared dataset")
    print()
    print("For transcription, you can:")
    print("- Manually transcribe each segment")
    print("- Use Whisper for automatic transcription (recommended)")
    print("- Use existing subtitles/scripts if available")
    print()


if __name__ == "__main__":
    main()
