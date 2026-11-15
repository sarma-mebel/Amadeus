"""Non-verbal vocalization clips: library loading and audio-stream splicing."""

from __future__ import annotations

import json
from pathlib import Path

from amadeus.web.streaming import AudioChunk, NonVerbalLibrary, TTSPipeline


def _write_library(root: Path, clips: dict[str, list[str]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for files in clips.values():
        for name in files:
            (root / name).write_bytes(f"wav:{name}".encode())
    (root / "manifest.json").write_text(
        json.dumps({"version": 1, "clips": clips}), encoding="utf-8"
    )


def test_library_loads_known_kinds_and_skips_junk(tmp_path: Path) -> None:
    _write_library(
        tmp_path,
        {
            "laugh": ["laugh_1.wav", "laugh_2.wav"],
            "sigh": ["sigh_1.wav"],
            "yodel": ["yodel_1.wav"],  # not a parser-recognized kind
        },
    )
    library = NonVerbalLibrary.load(tmp_path)
    assert library.kinds == ["laugh", "sigh"]
    assert library.pick("sigh") == b"wav:sigh_1.wav"
    assert library.pick("yodel") is None
    assert bool(library)


def test_library_missing_manifest_is_silently_empty(tmp_path: Path) -> None:
    library = NonVerbalLibrary.load(tmp_path / "nowhere")
    assert not library
    assert library.kinds == []
    assert library.pick("laugh") is None


def test_library_missing_clip_file_is_skipped(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"version": 1, "clips": {"laugh": ["ghost.wav"]}}),
        encoding="utf-8",
    )
    library = NonVerbalLibrary.load(tmp_path)
    assert not library


def test_pick_avoids_immediate_repeat(tmp_path: Path) -> None:
    _write_library(tmp_path, {"laugh": ["laugh_1.wav", "laugh_2.wav"]})
    library = NonVerbalLibrary.load(tmp_path)
    picks = [library.pick("laugh") for _ in range(10)]
    assert all(p is not None for p in picks)
    for previous, current in zip(picks, picks[1:]):
        assert previous != current


async def test_pipeline_prerendered_clip_keeps_stream_order() -> None:
    """A spliced clip at seq 2 must play after seq 1 and before seq 3, even
    though it needs no synthesis and is 'ready' instantly."""
    import time

    def synth(text: str) -> bytes:
        time.sleep(0.03 if text == "1" else 0.0)
        return f"audio-{text}".encode()

    emitted: list[AudioChunk] = []

    async def emit(chunk: AudioChunk) -> None:
        emitted.append(chunk)

    pipeline = TTSPipeline(synth, emit, workers=3)
    pipeline.submit(1, "1")
    pipeline.submit_prerendered(2, b"laugh-bytes")
    pipeline.submit(3, "3")
    await pipeline.close()

    assert [c.seq for c in emitted] == [1, 2, 3]
    import base64

    assert base64.b64decode(emitted[1].data_b64) == b"laugh-bytes"


async def test_pipeline_prerendered_after_close_is_dropped() -> None:
    async def emit(chunk: AudioChunk) -> None:  # pragma: no cover - never called
        raise AssertionError("nothing should emit")

    pipeline = TTSPipeline(lambda text: b"x", emit)
    await pipeline.close()
    pipeline.submit_prerendered(1, b"late")  # must not raise or emit
