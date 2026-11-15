"""Download the smart-turn-v3 end-of-turn model for duplex endpointing.

Fetches the ONNX model (~8MB, BSD-2) from the pipecat-ai/smart-turn-v3
HuggingFace repo into models/smart_turn/, where
amadeus/web/streaming/smart_turn.py auto-discovers it. Idempotent: an
existing .onnx in the target directory is left alone unless --force.

Usage:
    python scripts/fetch_smart_turn.py [--dest models/smart_turn] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = "pipecat-ai/smart-turn-v3"
API_URL = f"https://huggingface.co/api/models/{REPO}"
RESOLVE_URL = f"https://huggingface.co/{REPO}/resolve/main/"


def _list_onnx_files() -> list[str]:
    with urllib.request.urlopen(API_URL, timeout=30) as response:
        data = json.load(response)
    return [
        s["rfilename"]
        for s in data.get("siblings", [])
        if str(s.get("rfilename", "")).endswith(".onnx")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="models/smart_turn")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    existing = sorted(dest.glob("*.onnx"))
    if existing and not args.force:
        print(f"Already present: {existing[0]} (use --force to re-download)")
        return 0

    try:
        files = _list_onnx_files()
    except Exception as exc:
        print(f"Could not list {REPO}: {exc}", file=sys.stderr)
        return 1
    if not files:
        print(
            f"No .onnx files listed in {REPO} — repo layout changed?", file=sys.stderr
        )
        return 1
    # Prefer an int8/quantized build when several exist (CPU target).
    files.sort(key=lambda name: (("int8" not in name and "quant" not in name), name))
    name = files[0]
    target = dest / Path(name).name
    print(f"Downloading {name} -> {target}")
    try:
        urllib.request.urlretrieve(RESOLVE_URL + name, target)  # noqa: S310
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Done ({target.stat().st_size / 1e6:.1f} MB). Duplex calls will use it "
        "on the next backend start (web.duplex_smart_turn)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
