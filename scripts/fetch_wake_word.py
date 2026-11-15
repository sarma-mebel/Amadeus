"""Download an openWakeWord model (plus its shared feature extractors) for
the local wake-word listener.

Fetches the chosen wake model AND the two feature-extractor graphs
(melspectrogram.onnx, embedding_model.onnx — shared by every wake model)
into models/wake_word/, where amadeus/web/streaming/wake_word.py
auto-discovers them. Idempotent: existing files are left alone unless
--force.

There is no pretrained "Kurisu"/"Amadeus" phrase — openWakeWord ships a
handful of community models ("hey jarvis" is the default here) and custom
phrases can be trained with its notebook (see docs/wake-word.md). Drop a
custom .onnx into models/wake_word/ (or set wake_word.model) to use one.

Prefers openwakeword's own downloader when the [wake] extra is installed
(it knows the canonical asset URLs); falls back to the pinned GitHub
release assets otherwise.

Usage:
    python scripts/fetch_wake_word.py [--model hey_jarvis_v0.1]
                                      [--dest models/wake_word] [--force]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# openWakeWord's model assets live on this pinned GitHub release (Apache-2.0).
_RELEASE_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
_FEATURE_MODELS = ("melspectrogram.onnx", "embedding_model.onnx")


def _download(name: str, target: Path) -> None:
    url = _RELEASE_URL + name
    print(f"Downloading {url} -> {target}")
    urllib.request.urlretrieve(url, target)  # noqa: S310


def _fetch_via_sdk(model: str, dest: Path) -> bool:
    """Try openwakeword's own downloader; False = fall back to direct URLs."""
    try:
        from openwakeword import utils
    except ImportError:
        return False
    try:
        utils.download_models(model_names=[model], target_directory=str(dest))
        return True
    except TypeError:
        # Older/newer SDK without target_directory — use the direct path.
        return False
    except Exception as exc:
        print(f"openwakeword downloader failed ({exc}); trying direct URLs")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="hey_jarvis_v0.1",
        help="openWakeWord model name (release asset stem, e.g. hey_jarvis_v0.1)",
    )
    parser.add_argument("--dest", default="models/wake_word")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    model_file = f"{args.model}.onnx"
    wanted = [model_file, *_FEATURE_MODELS]
    missing = [n for n in wanted if args.force or not (dest / n).is_file()]
    if not missing:
        print(f"Already present in {dest}: {', '.join(wanted)} (--force to refetch)")
        return 0

    if _fetch_via_sdk(args.model, dest):
        still_missing = [n for n in wanted if not (dest / n).is_file()]
        missing = still_missing
        if not missing:
            print(f"Done via openwakeword downloader -> {dest}")
            _print_next_steps()
            return 0

    failures = 0
    for name in missing:
        try:
            _download(name, dest / name)
        except Exception as exc:
            print(f"Download failed for {name}: {exc}", file=sys.stderr)
            failures += 1
    if failures:
        return 1
    print(f"Done -> {dest}")
    _print_next_steps()
    return 0


def _print_next_steps() -> None:
    print(
        "Enable with wake_word.enabled: true in config (plus the Settings "
        "toggle in the web UI) and restart the backend. Needs the [wake] "
        'extra: pip install "amadeus-kurisu[wake]".'
    )


if __name__ == "__main__":
    raise SystemExit(main())
