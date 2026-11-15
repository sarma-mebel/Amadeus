#!/usr/bin/env python3
"""
Split Kurisu audio file into individual clips for TTS training.
Uses silence detection to create natural speech segments.
"""

import os
import sys
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence
import soundfile as sf

def split_audio_file(input_file, output_dir, min_silence_len=500, silence_thresh=-40,
                     min_clip_length=1000, max_clip_length=10000):
    """
    Split audio file into individual clips based on silence detection.

    Args:
        input_file: Path to input WAV file
        output_dir: Directory to save clips
        min_silence_len: Minimum silence length to split on (ms)
        silence_thresh: Silence threshold in dB
        min_clip_length: Minimum clip length to keep (ms)
        max_clip_length: Maximum clip length to keep (ms)
    """
    print("=" * 70)
    print("Kurisu Audio Splitting for MeloTTS Training")
    print("=" * 70)
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading audio file: {input_file}")
    audio = AudioSegment.from_wav(input_file)

    total_duration = len(audio) / 1000  # Convert to seconds
    print(f"Total duration: {total_duration:.2f}s ({total_duration/3600:.2f} hours)")
    print(f"Sample rate: {audio.frame_rate} Hz")
    print(f"Channels: {audio.channels}")
    print()

    print("Splitting on silence...")
    print(f"  Min silence length: {min_silence_len}ms")
    print(f"  Silence threshold: {silence_thresh}dB")
    print(f"  Min clip length: {min_clip_length}ms")
    print(f"  Max clip length: {max_clip_length}ms")
    print()

    # Split on silence
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=500,  # Keep 500ms of silence at edges
    )

    print(f"Found {len(chunks)} potential clips")
    print()

    # Filter and save clips
    saved_clips = []
    skipped_short = 0
    skipped_long = 0

    print("Filtering and saving clips...")
    for i, chunk in enumerate(chunks):
        chunk_len = len(chunk)

        # Skip clips that are too short or too long
        if chunk_len < min_clip_length:
            skipped_short += 1
            continue
        if chunk_len > max_clip_length:
            skipped_long += 1
            continue

        # Normalize audio level
        chunk = chunk.normalize()

        # Save clip
        clip_filename = output_path / f"kurisu_{i:05d}.wav"
        chunk.export(clip_filename, format="wav")
        saved_clips.append(clip_filename)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(chunks)} clips...")

    print()
    print("=" * 70)
    print("Splitting Complete!")
    print("=" * 70)
    print()
    print(f"Total clips found: {len(chunks)}")
    print(f"Clips saved: {len(saved_clips)}")
    print(f"Skipped (too short): {skipped_short}")
    print(f"Skipped (too long): {skipped_long}")
    print(f"Output directory: {output_dir}")
    print()

    # Calculate total duration of saved clips
    total_saved_duration = sum(len(AudioSegment.from_wav(clip)) for clip in saved_clips[:100]) / 1000 * (len(saved_clips) / 100)
    print(f"Estimated total duration of saved clips: {total_saved_duration:.2f}s ({total_saved_duration/3600:.2f} hours)")
    print()

    # Save clip list for reference
    clips_list_file = output_path / "clips_list.txt"
    with open(clips_list_file, 'w') as f:
        for clip in saved_clips:
            f.write(f"{clip.name}\n")

    print(f"Clip list saved to: {clips_list_file}")
    print()

    return saved_clips


if __name__ == "__main__":
    input_file = "Kurisu_voice_44100_mono.wav"
    output_dir = "kurisu_dataset/clips"

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)

    # Split audio
    clips = split_audio_file(
        input_file=input_file,
        output_dir=output_dir,
        min_silence_len=500,  # 500ms minimum silence
        silence_thresh=-40,   # -40dB threshold
        min_clip_length=1000, # 1 second minimum
        max_clip_length=20000 # 20 seconds maximum
    )

    print("Next steps:")
    print("1. Review some clips to verify quality")
    print("2. Run Whisper-large-v3 transcription on all clips")
    print("3. Create metadata.list file")
    print()
