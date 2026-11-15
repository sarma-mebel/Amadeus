"""Download the ONNX speech-emotion-recognition model for vocal tone analysis.

Mirrors scripts/fetch_smart_turn.py's shape: fetches the model (plus its
classifier sidecar) into models/vocal_emotion/, where
amadeus/vocal_emotion.py auto-discovers it. Idempotent: an existing .onnx in
the target directory is left alone unless --force.

The pinned default is the emotion2vec+ base ONNX export
(ziyu12345/emotion2vec_plus_base_onnx, ~373MB) — emotion2vec (ACL 2024,
arXiv 2312.15185) is a speech-emotion foundation model fine-tuned on ~4,800
hours of real-world multilingual speech, far more robust on natural
conversational audio than acted-corpus SER classifiers. Its export takes a
raw 16 kHz waveform (normalization folded into the graph) and emits backbone
features; the tiny classifier head rides alongside as emotion2vec_head.json,
which this script downloads too — amadeus/vocal_emotion.py REFUSES to load a
model without a head/labels sidecar (label order must never be guessed).

Attribution: the emotion2vec+ weights are distributed under the FunASR Model
Open Source License (free use/modification/sharing; attribution required, no
non-commercial clause). Keep the model name visible in any redistribution.

A different repo can be substituted with --repo. For logits-style models
(e.g. the Apache-2.0 runner-up onnx-community/
wav2vec2-base-Speech_Emotion_Recognition-ONNX, 95MB int8), the script
derives labels.json from the repo's config.json id2label when no head
sidecar exists. With no model fetched, or with vocal_emotion.mode:
"heuristic", Amadeus runs the dependency-free prosody heuristic, which is
the fully-functional default either way.

Usage:
    python scripts/fetch_vocal_emotion.py [--repo <hf-org>/<hf-repo>] [--dest models/vocal_emotion] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

DEFAULT_REPO = "ziyu12345/emotion2vec_plus_base_onnx"

ATTRIBUTION_NOTE = (
    "Model: emotion2vec+ base (FunAudioLLM/emotion2vec, ACL 2024, "
    "arXiv 2312.15185), ONNX export by ziyu12345. Weights are under the "
    "FunASR Model Open Source License — attribution required; keep the "
    "model name visible if you redistribute."
)


def _repo_files(repo: str) -> list[str]:
    api_url = f"https://huggingface.co/api/models/{repo}"
    with urllib.request.urlopen(api_url, timeout=30) as response:
        data = json.load(response)
    return [str(s.get("rfilename", "")) for s in data.get("siblings", [])]


def _download(repo: str, name: str, target: Path) -> None:
    resolve_url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    print(f"Downloading {name} -> {target}")
    urllib.request.urlretrieve(resolve_url, target)  # noqa: S310
    print(f"  done ({target.stat().st_size / 1e6:.1f} MB)")


def _write_labels_from_config(repo: str, files: list[str], dest: Path) -> bool:
    """Synthesize labels.json from a transformers-style config.json id2label.

    Only used for --repo overrides whose export has no head sidecar — the
    wrapper needs an index-ordered label list from SOMEWHERE, and id2label
    is the authoritative source for logits-style classifier exports.
    """
    if "config.json" not in files:
        return False
    url = f"https://huggingface.co/{repo}/resolve/main/config.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            config = json.load(response)
    except Exception as exc:
        print(f"Could not read config.json from {repo}: {exc}", file=sys.stderr)
        return False
    id2label = config.get("id2label")
    if not isinstance(id2label, dict) or not id2label:
        return False
    try:
        ordered = [id2label[str(i)] for i in range(len(id2label))]
    except KeyError:
        return False
    target = dest / "labels.json"
    target.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    print(f"Wrote {target} from config.json id2label: {ordered}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"HuggingFace repo id hosting the SER ONNX model (default: "
        f"{DEFAULT_REPO}). A substitute repo needs either a *head.json "
        "sidecar or a config.json with id2label — vet its input contract "
        "against amadeus/vocal_emotion.py's VocalEmotionModel first.",
    )
    parser.add_argument("--dest", default="models/vocal_emotion")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    existing = sorted(dest.glob("*.onnx"))
    if existing and not args.force:
        print(f"Already present: {existing[0]} (use --force to re-download)")
        return 0

    try:
        files = _repo_files(args.repo)
    except Exception as exc:
        print(f"Could not list {args.repo}: {exc}", file=sys.stderr)
        return 1
    onnx_files = [f for f in files if f.endswith(".onnx")]
    if not onnx_files:
        print(
            f"No .onnx files listed in {args.repo} — repo layout changed?",
            file=sys.stderr,
        )
        return 1
    # Prefer a quantized artifact when the repo offers one (smaller, faster
    # on CPU); the pinned default repo ships a single fp32 file.
    onnx_files.sort(
        key=lambda name: (("int8" not in name and "quant" not in name), name)
    )
    model_name = onnx_files[0]

    try:
        _download(args.repo, model_name, dest / Path(model_name).name)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    # Classifier sidecar: any *head.json in the repo (the pinned export's
    # emotion2vec_head.json), else labels.json, else derive labels from
    # config.json. Without one of these the backend refuses the model.
    sidecars = [f for f in files if f.endswith("head.json") or f == "labels.json"]
    for name in sidecars:
        try:
            _download(args.repo, name, dest / Path(name).name)
        except Exception as exc:
            print(f"Sidecar download failed ({name}): {exc}", file=sys.stderr)
            return 1
    if not sidecars and not _write_labels_from_config(args.repo, files, dest):
        print(
            f"WARNING: {args.repo} has no *head.json/labels.json and no "
            "usable config.json id2label — amadeus/vocal_emotion.py will "
            "REFUSE this model until a sidecar exists. Write labels.json "
            "(an index-ordered JSON list of class names) by hand after "
            "verifying the model's label order.",
            file=sys.stderr,
        )

    if args.repo == DEFAULT_REPO:
        print(ATTRIBUTION_NOTE)
    print(
        "Done. vocal_emotion.mode 'auto' (the default) picks the model up "
        "on the next backend start; 'model' makes it exclusive (no "
        "heuristic fallback)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
