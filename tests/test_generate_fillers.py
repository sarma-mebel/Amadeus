"""Tests for scripts/generate_fillers.py (provider faked, no synthesis)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "generate_fillers", REPO_ROOT / "scripts" / "generate_fillers.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTTS:
    def __init__(self, fail_texts: set[str] | None = None):
        self.fail_texts = fail_texts or set()
        self.calls: list[str] = []

    def synthesize(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ) -> bytes:
        self.calls.append(text)
        if text in self.fail_texts:
            raise RuntimeError("synth boom")
        return f"audio:{text}".encode("utf-8")


def test_generates_all_clips_and_manifest(tmp_path):
    module = _load_script_module()
    tts = FakeTTS()

    entries = module.generate_fillers(tts, tmp_path)

    assert len(entries) == len(module.FILLER_LINES)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert len(manifest["fillers"]) == len(module.FILLER_LINES)
    for entry in manifest["fillers"]:
        wav = tmp_path / entry["file"]
        assert wav.exists()
        assert wav.read_bytes() == f"audio:{entry['text']}".encode("utf-8")


def test_failed_line_is_left_out_of_manifest(tmp_path):
    module = _load_script_module()
    failing_text = module.FILLER_LINES[0][1]
    tts = FakeTTS(fail_texts={failing_text})

    entries = module.generate_fillers(tts, tmp_path)

    assert len(entries) == len(module.FILLER_LINES) - 1
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    texts = {entry["text"] for entry in manifest["fillers"]}
    assert failing_text not in texts
    # The failed clip's wav was never written.
    failed_slug = module.FILLER_LINES[0][0]
    assert not (tmp_path / f"{failed_slug}.wav").exists()


def test_existing_clips_kept_unless_force(tmp_path):
    module = _load_script_module()
    all_lines = len(module.FILLER_LINES) + len(module.BACKCHANNEL_LINES)

    first = FakeTTS()
    module.generate_fillers(first, tmp_path)
    assert len(first.calls) == all_lines

    second = FakeTTS()
    entries = module.generate_fillers(second, tmp_path)
    assert second.calls == []  # everything already on disk
    assert len(entries) == len(module.FILLER_LINES)

    third = FakeTTS()
    module.generate_fillers(third, tmp_path, force=True)
    assert len(third.calls) == all_lines


def test_backchannels_written_to_manifest(tmp_path):
    module = _load_script_module()
    tts = FakeTTS()

    module.generate_fillers(tts, tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["backchannels"]) == len(module.BACKCHANNEL_LINES)
    for entry in manifest["backchannels"]:
        wav = tmp_path / entry["file"]
        assert wav.exists()
        assert wav.read_bytes() == f"audio:{entry['text']}".encode("utf-8")
    # Slug namespaces stay disjoint so a backchannel can never clobber a
    # filler wav in the shared directory.
    filler_files = {entry["file"] for entry in manifest["fillers"]}
    assert filler_files.isdisjoint({e["file"] for e in manifest["backchannels"]})
