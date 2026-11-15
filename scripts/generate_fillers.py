#!/usr/bin/env python3
"""Generate the filler backchannel clips ("thinking sounds") in her voice.

Synthesizes a small pool of short, neutral Japanese thinking lines (んー…,
そうね…) with the TTS provider from your config and writes them to
``static/fillers/`` next to a ``manifest.json``. The frontend plays one at
random when a reply's first audio takes longer than ~600ms, so the wait
reads as her thinking instead of the app hanging; no manifest means the
feature simply stays off.

The clips are voice-specific (they ARE her voice): re-run this after
changing the TTS provider, the reference clip, or the character pack. The
output directory is gitignored — generated audio never belongs in the repo.

Usage:
    python scripts/generate_fillers.py [--config config.local.yaml]
                                       [--out static/fillers] [--force]

With the qwen3_tts voice this needs the GPU environment; expect a few
seconds per clip plus the usual model warmup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amadeus.config import Config  # noqa: E402
from amadeus.provider_loader import load_tts  # noqa: E402

# Short, neutral beats that sit naturally under any reply that follows —
# never content-bearing, never an answer. Brief on purpose: a filler is a
# beat, not a sentence, and short zero-shot generations synthesize fastest.
FILLER_LINES: list[tuple[str, str]] = [
    ("hmm", "んー…"),
    ("sou-ne", "そうね…"),
    ("eeto", "ええと…"),
    ("uun", "うーん…"),
    ("chotto-matte", "ちょっと待って。"),
    ("kangaesasete", "少し考えさせて。"),
    ("sou-nee", "うーん、そうねえ…"),
    ("ima-kangaeteru", "待ちなさい、今考えてるの。"),
]

# Listening backchannels: soft acknowledgements she voices WHILE the user is
# talking into the mic (frontend audio/listeningReactions.ts). Even shorter
# and quieter in register than the fillers — a listener's "mm", not a reply.
BACKCHANNEL_LINES: list[tuple[str, str]] = [
    ("bc-un", "うん。"),
    ("bc-ee", "ええ。"),
    ("bc-un-un", "うんうん。"),
    ("bc-fumu", "ふむ。"),
    ("bc-hee", "へえ。"),
    ("bc-naruhodo", "なるほど。"),
]


def generate_fillers(
    tts: Any,
    out_dir: Path,
    *,
    force: bool = False,
    speaking_rate: float = 1.0,
    pitch_shift: float = 0.0,
) -> list[dict[str, str]]:
    """Synthesize every filler line into ``out_dir`` and write the manifest.

    Idempotent: existing clips are kept unless ``force``. A line that fails
    to synthesize is reported and left out of the manifest (the frontend
    only ever sees clips that really exist) without failing the rest.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def synthesize_lines(lines: list[tuple[str, str]]) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        for slug, text in lines:
            wav_path = out_dir / f"{slug}.wav"
            if wav_path.exists() and not force:
                print(f"  kept    {wav_path.name} ({text})")
                entries.append({"file": wav_path.name, "text": text})
                continue
            try:
                audio = tts.synthesize(
                    text, speaking_rate=speaking_rate, pitch_shift=pitch_shift
                )
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  FAILED  {wav_path.name}: {exc}", file=sys.stderr)
                continue
            wav_path.write_bytes(audio)
            print(f"  wrote   {wav_path.name} ({text})")
            entries.append({"file": wav_path.name, "text": text})
        return entries

    entries = synthesize_lines(FILLER_LINES)
    backchannels = synthesize_lines(BACKCHANNEL_LINES)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"version": 1, "fillers": entries, "backchannels": backchannels},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Manifest: {manifest_path} "
        f"({len(entries)} fillers, {len(backchannels)} backchannels)"
    )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate filler backchannel clips")
    parser.add_argument("--config", default=None, help="config YAML path")
    parser.add_argument(
        "--out",
        default=None,
        help="output directory (default: <repo>/static/fillers)",
    )
    parser.add_argument(
        "--force", action="store_true", help="regenerate clips that already exist"
    )
    args = parser.parse_args()

    config = Config.load(Path(args.config) if args.config else None)
    out_dir = (
        Path(args.out)
        if args.out
        else Path(__file__).resolve().parent.parent / "static" / "fillers"
    )

    print(f"Loading TTS provider '{config.providers.tts}'…")
    try:
        tts = load_tts(config)
    except Exception as exc:  # noqa: BLE001 - a script wants a message, not a trace
        print(
            f"Could not load TTS provider '{config.providers.tts}': {exc}\n"
            "Configure a real voice first (providers.tts).",
            file=sys.stderr,
        )
        return 1

    entries = generate_fillers(
        tts,
        out_dir,
        force=args.force,
        speaking_rate=config.persona.speaking_rate,
        pitch_shift=config.persona.pitch_shift,
    )
    return 0 if entries else 1


if __name__ == "__main__":
    raise SystemExit(main())
