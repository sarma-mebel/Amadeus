#!/usr/bin/env python3
"""Generate the non-verbal vocalization clips ([laugh]/[sigh]/...) in her voice.

Synthesizes a small pool of laughs, sighs, gasps, giggles, and thinking hums
with the TTS provider from your config and writes them to
``static/nonverbals/`` next to a ``manifest.json``. When the clips exist, the
streaming chat advertises the matching tags to the LLM and a standalone
"[laugh]" line in a reply splices one of these clips into the audio stream
at exactly that point; no manifest means the feature simply stays off.

The clips are voice-specific (they ARE her voice): re-run this after
changing the TTS provider, the reference clip, or the character pack. The
output directory is gitignored — generated audio never belongs in the repo.

Vet the results by ear before shipping: zero-shot TTS renders laughter with
varying success. Delete any wav that sounds wrong and re-run with --force
for just that gap, or edit the line texts below; the manifest only ever
lists clips that really exist.

Usage:
    python scripts/generate_nonverbals.py [--config config.local.yaml]
                                          [--out static/nonverbals] [--force]

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
from amadeus.web.streaming.parser import NONVERBAL_KINDS  # noqa: E402

# A few variants per kind so repeated laughs don't sound like a soundboard.
# Kind names must stay within parser.NONVERBAL_KINDS or the backend will
# never load them (verified below).
NONVERBAL_LINES: dict[str, list[str]] = {
    "laugh": ["あはは!", "あはははっ。", "ふふ、あははっ。"],
    "giggle": ["ふふっ。", "ふふふ…。"],
    "sigh": ["はぁ…。", "ふぅ…。"],
    "gasp": ["えっ!?", "はっ!?"],
    "hmm": ["うーん…。", "んー…。"],
}


def generate_nonverbals(
    tts: Any,
    out_dir: Path,
    *,
    force: bool = False,
    speaking_rate: float = 1.0,
    pitch_shift: float = 0.0,
) -> dict[str, list[str]]:
    """Synthesize every non-verbal variant and write the manifest.

    Idempotent: existing clips are kept unless ``force``. A variant that
    fails to synthesize is reported and left out of the manifest (the
    backend only ever sees clips that really exist) without failing the
    rest.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    clips: dict[str, list[str]] = {}
    for kind, texts in NONVERBAL_LINES.items():
        if kind not in NONVERBAL_KINDS:
            print(f"  SKIPPED {kind}: not a parser-recognized kind", file=sys.stderr)
            continue
        for index, text in enumerate(texts, start=1):
            wav_path = out_dir / f"{kind}_{index}.wav"
            if wav_path.exists() and not force:
                print(f"  kept    {wav_path.name} ({text})")
                clips.setdefault(kind, []).append(wav_path.name)
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
            clips.setdefault(kind, []).append(wav_path.name)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"version": 1, "clips": clips}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    total = sum(len(v) for v in clips.values())
    print(f"Manifest: {manifest_path} ({total} clips, kinds: {sorted(clips)})")
    return clips


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate non-verbal vocalization clips"
    )
    parser.add_argument("--config", default=None, help="config YAML path")
    parser.add_argument(
        "--out",
        default=None,
        help="output directory (default: <repo>/static/nonverbals)",
    )
    parser.add_argument(
        "--force", action="store_true", help="regenerate clips that already exist"
    )
    args = parser.parse_args()

    config = Config.load(Path(args.config) if args.config else None)
    out_dir = (
        Path(args.out)
        if args.out
        else Path(__file__).resolve().parent.parent / "static" / "nonverbals"
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

    clips = generate_nonverbals(
        tts,
        out_dir,
        force=args.force,
        speaking_rate=config.persona.speaking_rate,
        pitch_shift=config.persona.pitch_shift,
    )
    return 0 if clips else 1


if __name__ == "__main__":
    raise SystemExit(main())
