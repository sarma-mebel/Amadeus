#!/usr/bin/env python3
"""Slice a raw voice-line recording into clone-ready reference candidates.

Takes a long audio file (a game voice-line collection, a talk segment, an
anime compilation), finds utterances via silence detection, keeps segments
in the sweet spot for the qwen3 zero-shot clone (4-16s), transcribes each
with faster-whisper (Japanese), and writes the top candidates plus a
transcript sheet into the target directory:

    python scripts/prep_voice_candidates.py source.wav characters/packs/rem/candidates

Pick the best-sounding candidate BY EAR, rename it voice_ref.wav, copy its
transcript into character.yaml's voice.ref_text, and re-run the pack
installer. Candidates are ranked by a crude cleanliness score (speech level
vs background floor) — the ranking narrows the listening, it doesn't
replace it.

Private use only, same stance as everything else in static/models/.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

MIN_LEN = 4.0
MAX_LEN = 16.0
TOP_N = 6


def detect_segments(src: Path, silence_db: float = -35.0, min_silence: float = 0.45):
    """Utterance windows between silences, via ffmpeg silencedetect."""
    result = subprocess.run(
        [
            "ffmpeg", "-i", str(src), "-af",
            f"silencedetect=noise={silence_db}dB:d={min_silence}",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    ends = [float(m) for m in re.findall(r"silence_end: ([0-9.]+)", result.stderr)]
    starts = [float(m) for m in re.findall(r"silence_start: ([0-9.]+)", result.stderr)]
    duration_match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", result.stderr)
    total = 0.0
    if duration_match:
        h, m, s = duration_match.groups()
        total = int(h) * 3600 + int(m) * 60 + float(s)
    # Speech spans run from each silence_end to the next silence_start.
    spans = []
    speech_starts = [0.0] + ends
    speech_ends = starts + [total]
    for begin, end in zip(speech_starts, speech_ends):
        if end - begin >= 1.0:
            spans.append((begin, end))
    # Merge adjacent spans into candidate windows within MAX_LEN.
    merged = []
    cur_start, cur_end = None, None
    for begin, end in spans:
        if cur_start is None:
            cur_start, cur_end = begin, end
        elif end - cur_start <= MAX_LEN:
            cur_end = end
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = begin, end
    if cur_start is not None:
        merged.append((cur_start, cur_end))
    return [(b, e) for b, e in merged if MIN_LEN <= e - b <= MAX_LEN]


def rms_stats(path: Path) -> tuple[float, float]:
    """(mean_volume_db, background_floor_db) via ffmpeg volumedetect and a
    5th-percentile proxy from astats — used for the cleanliness score."""
    result = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    mean = re.search(r"mean_volume: (-?[\d.]+) dB", result.stderr)
    peak = re.search(r"max_volume: (-?[\d.]+) dB", result.stderr)
    return (
        float(mean.group(1)) if mean else -99.0,
        float(peak.group(1)) if peak else 0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", help="long source audio (wav/m4a/...)")
    parser.add_argument("out_dir", help="candidates output directory")
    parser.add_argument("--top", type=int, default=TOP_N)
    args = parser.parse_args()

    src = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = detect_segments(src)
    if not windows:
        print("No candidate windows found (source too dense or too quiet?)")
        return 1
    print(f"{len(windows)} candidate windows; extracting + transcribing…")

    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8")

    rows = []
    for index, (begin, end) in enumerate(windows):
        clip = out_dir / f"cand_{index:02d}.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src),
                "-ss", f"{begin:.2f}", "-to", f"{end:.2f}",
                "-ar", "24000", "-ac", "1", str(clip),
            ],
            capture_output=True,
        )
        mean_db, peak_db = rms_stats(clip)
        segments, _ = model.transcribe(str(clip), language="ja")
        text = "".join(s.text for s in segments).strip()
        # Cleanliness proxy: louder mean (clear speech) with headroom, and a
        # real transcript. BGM-heavy clips tend to transcribe to garbage or
        # music tags.
        score = mean_db + (5 if len(text) >= 8 else -20)
        rows.append({
            "file": clip.name,
            "start": round(begin, 2),
            "duration": round(end - begin, 2),
            "mean_db": mean_db,
            "transcript": text,
            "score": round(score, 1),
        })
        print(f"  {clip.name} [{begin:7.1f}s +{end-begin:4.1f}s] {text[:48]}")

    rows.sort(key=lambda r: -r["score"])
    keep = rows[: args.top]
    keep_files = {r["file"] for r in keep}
    for row in rows:
        if row["file"] not in keep_files:
            (out_dir / row["file"]).unlink(missing_ok=True)

    sheet = out_dir / "transcripts.json"
    sheet.write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTop {len(keep)} kept in {out_dir} — pick ONE by ear, rename it")
    print("voice_ref.wav, and copy its transcript into character.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
