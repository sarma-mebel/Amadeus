"""Pre-recorded non-verbal vocalization clips (laugh/sigh/...).

A ``[laugh]`` line in the tagged grammar resolves to one of these clips,
spliced into the ordered streaming audio at its seq slot. Clips are
voice-specific generated assets (``scripts/generate_nonverbals.py``) living
in the gitignored ``static/nonverbals/`` next to a manifest:

    {"version": 1, "clips": {"laugh": ["laugh_1.wav", ...], "sigh": [...]}}

No manifest (or an empty one) means the feature is silently off: the parser
still strips the tags, the prompt never advertises them, and no audio plays
— the same missing-assets contract as the frontend filler clips.
"""

from __future__ import annotations

import json
import os
import random
import threading
from pathlib import Path
from typing import Dict, List, Optional

from ...logging_utils import get_logger
from .parser import NONVERBAL_KINDS

LOGGER = get_logger(__name__)

_MAX_CLIP_BYTES = 2 * 1024 * 1024  # a one-shot laugh should be ~1s of WAV


class NonVerbalLibrary:
    """In-memory clip pool keyed by kind, with a no-immediate-repeat pick."""

    def __init__(self, clips: Dict[str, List[bytes]], mime: str = "audio/wav") -> None:
        self._clips = {kind: variants for kind, variants in clips.items() if variants}
        self.mime = mime
        self._last_pick: Dict[str, int] = {}

    @property
    def kinds(self) -> List[str]:
        return sorted(self._clips)

    def __bool__(self) -> bool:
        return bool(self._clips)

    def pick(self, kind: str) -> Optional[bytes]:
        """One clip for ``kind``, avoiding the previously played variant."""
        variants = self._clips.get(kind)
        if not variants:
            return None
        if len(variants) == 1:
            return variants[0]
        last = self._last_pick.get(kind)
        choices = [i for i in range(len(variants)) if i != last]
        index = random.choice(choices)
        self._last_pick[kind] = index
        return variants[index]

    @classmethod
    def load(cls, root: Path) -> "NonVerbalLibrary":
        """Read the manifest + clip bytes; malformed entries are skipped.

        Only kinds the parser recognizes (``NONVERBAL_KINDS``) are loaded —
        a clip for an unknown tag could never be triggered anyway.
        """
        manifest_path = root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls({})
        except Exception as exc:  # noqa: BLE001 - malformed manifest = feature off
            LOGGER.warning("Unreadable non-verbal manifest %s: %s", manifest_path, exc)
            return cls({})
        clips: Dict[str, List[bytes]] = {}
        raw = manifest.get("clips")
        if not isinstance(raw, dict):
            return cls({})
        for kind, files in raw.items():
            if kind not in NONVERBAL_KINDS or not isinstance(files, list):
                continue
            for name in files:
                if not isinstance(name, str):
                    continue
                path = root / Path(name).name  # flat dir; no traversal
                try:
                    if path.stat().st_size > _MAX_CLIP_BYTES:
                        LOGGER.warning("Skipping oversized non-verbal clip %s", path)
                        continue
                    clips.setdefault(kind, []).append(path.read_bytes())
                except OSError as exc:
                    LOGGER.warning("Skipping non-verbal clip %s: %s", path, exc)
        library = cls(clips)
        if library:
            LOGGER.info(
                "Loaded non-verbal clips: %s",
                {k: len(v) for k, v in clips.items()},
            )
        return library


_cached: Optional[NonVerbalLibrary] = None
_cache_lock = threading.Lock()


def _static_dir() -> Path:
    """Resolve the served static dir the same way web/app.py mounts it."""
    override = os.getenv("AMADEUS_STATIC_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent.parent.parent / "static"


def get_nonverbal_library() -> NonVerbalLibrary:
    """Process-wide library, loaded once (clips are small and immutable).

    Regenerating clips (new voice, new refs) needs a server restart, same
    as swapping the TTS reference clip itself.
    """
    global _cached
    if _cached is None:
        with _cache_lock:
            if _cached is None:
                _cached = NonVerbalLibrary.load(_static_dir() / "nonverbals")
    return _cached


def reset_nonverbal_library_for_test() -> None:
    global _cached
    _cached = None
