"""Tests for the GPU runtime pack fetcher (amadeus/gpu_runtime.py).

No live network: the fake release is served from a local directory via
file:// URLs (urllib.request.urlopen supports file:// natively, so this
exercises the real download/verify/extract code path without any HTTP
server or mocking of urlopen itself).
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

from amadeus import gpu_runtime, telemetry


@pytest.fixture(autouse=True)
def _isolated_gpu_runtime(monkeypatch):
    """Every test gets a fresh controller singleton and clean telemetry, and
    never touches the real AMADEUS_GPU_RUNTIME_* env unless it sets it."""
    monkeypatch.delenv("AMADEUS_GPU_RUNTIME_AUTOFETCH", raising=False)
    monkeypatch.delenv("AMADEUS_GPU_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("AMADEUS_GPU_RUNTIME_URL_BASE", raising=False)
    gpu_runtime._reset_controller_for_tests()
    telemetry.reset()
    original_path = list(sys.path)
    yield
    sys.path[:] = original_path
    gpu_runtime._reset_controller_for_tests()
    telemetry.reset()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_fake_pack(release_dir: Path, *, extra_zip_entries=None) -> dict:
    """Build a tiny two-part fake pack (real zip, real sha256s) under
    release_dir, mirroring the packaging agent's exact contract shape."""
    release_dir.mkdir(parents=True, exist_ok=True)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("site-packages/torch/__init__.py", "# fake torch\n")
        archive.writestr("site-packages/torch/lib/placeholder.txt", "dll placeholder")
        archive.writestr("sox/sox.exe", "fake sox binary")
        for entry in extra_zip_entries or []:
            archive.writestr(entry, "malicious")
    zip_bytes = zip_buffer.getvalue()

    midpoint = max(1, len(zip_bytes) // 2)
    part1 = zip_bytes[:midpoint]
    part2 = zip_bytes[midpoint:]

    (release_dir / "gpu-runtime.zip.part01").write_bytes(part1)
    (release_dir / "gpu-runtime.zip.part02").write_bytes(part2)

    manifest = {
        "version": gpu_runtime.VERSION,
        "created": "2026-07-31T00:00:00Z",
        "zip_sha256": _sha256_bytes(zip_bytes),
        "total_size": len(zip_bytes),
        "parts": [
            {
                "name": "gpu-runtime.zip.part01",
                "size": len(part1),
                "sha256": _sha256_bytes(part1),
            },
            {
                "name": "gpu-runtime.zip.part02",
                "size": len(part2),
                "sha256": _sha256_bytes(part2),
            },
        ],
        "layout": {"site_packages": "site-packages", "sox": "sox"},
        "torch": "2.5.0+cu121",
    }
    (release_dir / gpu_runtime.MANIFEST_NAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest


def _file_url(path: Path) -> str:
    return path.resolve().as_uri()


# --------------------------------------------------------------------------
# gpu_likely_present / dir helpers
# --------------------------------------------------------------------------


def test_gpu_likely_present_true_when_nvidia_smi_on_path(monkeypatch):
    monkeypatch.setattr(gpu_runtime.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    assert gpu_runtime.gpu_likely_present() is True


def test_gpu_likely_present_false_when_absent(monkeypatch):
    monkeypatch.setattr(gpu_runtime.shutil, "which", lambda name: None)
    assert gpu_runtime.gpu_likely_present() is False


def test_runtime_dir_honors_dir_override(tmp_path, monkeypatch):
    override = tmp_path / "custom-gpu-dir"
    monkeypatch.setenv("AMADEUS_GPU_RUNTIME_DIR", str(override))
    assert gpu_runtime.runtime_dir(tmp_path / "app-data") == override


def test_runtime_dir_defaults_under_app_data(tmp_path):
    app_data = tmp_path / "app-data"
    assert gpu_runtime.runtime_dir(app_data) == app_data / "gpu-runtime"


def test_installed_pack_dir_none_without_sentinel(tmp_path):
    app_data = tmp_path / "app-data"
    pack_dir = app_data / "gpu-runtime" / gpu_runtime.VERSION
    pack_dir.mkdir(parents=True)
    (pack_dir / "site-packages").mkdir()
    assert gpu_runtime.installed_pack_dir(app_data) is None


def test_installed_pack_dir_present_with_sentinel(tmp_path):
    app_data = tmp_path / "app-data"
    pack_dir = app_data / "gpu-runtime" / gpu_runtime.VERSION
    pack_dir.mkdir(parents=True)
    (pack_dir / ".complete").write_text("", encoding="utf-8")
    assert gpu_runtime.installed_pack_dir(app_data) == pack_dir


# --------------------------------------------------------------------------
# ensure_pack: gating no-ops
# --------------------------------------------------------------------------


def test_ensure_pack_noop_when_autofetch_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("AMADEUS_GPU_RUNTIME_AUTOFETCH", "0")
    monkeypatch.setattr(gpu_runtime, "gpu_likely_present", lambda: True)
    called = []
    monkeypatch.setattr(gpu_runtime, "_run_fetch", lambda *a, **k: called.append(True))

    gpu_runtime.ensure_pack(tmp_path / "app-data", background=False)

    assert not called
    assert gpu_runtime.status()["state"] == "disabled"


def test_ensure_pack_noop_when_gpu_not_present(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_runtime, "gpu_likely_present", lambda: False)
    called = []
    monkeypatch.setattr(gpu_runtime, "_run_fetch", lambda *a, **k: called.append(True))

    gpu_runtime.ensure_pack(tmp_path / "app-data", background=False)

    assert not called
    assert gpu_runtime.status()["state"] == "absent"


def test_ensure_pack_noop_when_already_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_runtime, "gpu_likely_present", lambda: True)
    app_data = tmp_path / "app-data"
    pack_dir = app_data / "gpu-runtime" / gpu_runtime.VERSION
    pack_dir.mkdir(parents=True)
    (pack_dir / ".complete").write_text("", encoding="utf-8")
    called = []
    monkeypatch.setattr(gpu_runtime, "_run_fetch", lambda *a, **k: called.append(True))

    gpu_runtime.ensure_pack(app_data, background=False)

    assert not called
    assert gpu_runtime.status()["state"] == "ready_restart"


# --------------------------------------------------------------------------
# End-to-end happy path
# --------------------------------------------------------------------------


def test_ensure_pack_happy_path_end_to_end(tmp_path, monkeypatch):
    release_dir = tmp_path / "release"
    _build_fake_pack(release_dir)
    monkeypatch.setenv("AMADEUS_GPU_RUNTIME_URL_BASE", _file_url(release_dir))
    monkeypatch.setattr(gpu_runtime, "gpu_likely_present", lambda: True)

    app_data = tmp_path / "app-data"
    gpu_runtime.ensure_pack(app_data, background=False)

    pack_dir = gpu_runtime.installed_pack_dir(app_data)
    assert pack_dir is not None
    assert (pack_dir / "site-packages" / "torch" / "__init__.py").exists()
    assert (pack_dir / "sox" / "sox.exe").exists()
    assert not (gpu_runtime.runtime_dir(app_data) / "tmp").exists()

    snap = gpu_runtime.status()
    assert snap["state"] == "ready_restart"
    assert snap["version"] == gpu_runtime.VERSION

    counters = telemetry.snapshot()
    assert counters.get("gpu_runtime.download_started") == 1
    assert counters.get("gpu_runtime.download_completed") == 1
    assert "gpu_runtime.download_failed" not in counters


def test_ensure_pack_background_thread_completes(tmp_path, monkeypatch):
    release_dir = tmp_path / "release"
    _build_fake_pack(release_dir)
    monkeypatch.setenv("AMADEUS_GPU_RUNTIME_URL_BASE", _file_url(release_dir))
    monkeypatch.setattr(gpu_runtime, "gpu_likely_present", lambda: True)

    app_data = tmp_path / "app-data"
    gpu_runtime.ensure_pack(app_data, background=True)

    # Join the daemon thread the controller kicked off (there's exactly one
    # gpu-runtime-fetch thread; joining by name avoids a fixed sleep).
    import threading

    for thread in threading.enumerate():
        if thread.name == "gpu-runtime-fetch":
            thread.join(timeout=10)

    assert gpu_runtime.installed_pack_dir(app_data) is not None
    assert gpu_runtime.status()["state"] == "ready_restart"


# --------------------------------------------------------------------------
# Corruption / retry / failure paths
# --------------------------------------------------------------------------


def test_corrupted_part_retries_then_fails(tmp_path, monkeypatch):
    release_dir = tmp_path / "release"
    manifest = _build_fake_pack(release_dir)
    # Corrupt part02 on disk so its sha256 no longer matches the manifest.
    part2 = release_dir / "gpu-runtime.zip.part02"
    part2.write_bytes(b"corrupted-bytes-not-matching-manifest")
    monkeypatch.setenv("AMADEUS_GPU_RUNTIME_URL_BASE", _file_url(release_dir))
    monkeypatch.setattr(gpu_runtime, "gpu_likely_present", lambda: True)

    app_data = tmp_path / "app-data"
    gpu_runtime.ensure_pack(app_data, background=False)

    assert gpu_runtime.installed_pack_dir(app_data) is None
    snap = gpu_runtime.status()
    assert snap["state"] == "failed"
    assert snap["detail"]

    counters = telemetry.snapshot()
    # 2 retries beyond the first attempt (_PART_MAX_RETRIES=2), all failing
    # since the file never changes.
    assert counters.get("gpu_runtime.part_retries") == gpu_runtime._PART_MAX_RETRIES
    assert counters.get("gpu_runtime.download_failed") == 1


def test_bad_zip_sha256_fails_and_cleans_up(tmp_path, monkeypatch):
    release_dir = tmp_path / "release"
    manifest_path_dir = release_dir
    _build_fake_pack(release_dir)
    manifest_file = manifest_path_dir / gpu_runtime.MANIFEST_NAME
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["zip_sha256"] = "0" * 64  # deliberately wrong
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("AMADEUS_GPU_RUNTIME_URL_BASE", _file_url(release_dir))
    monkeypatch.setattr(gpu_runtime, "gpu_likely_present", lambda: True)

    app_data = tmp_path / "app-data"
    gpu_runtime.ensure_pack(app_data, background=False)

    assert gpu_runtime.installed_pack_dir(app_data) is None
    base_dir = gpu_runtime.runtime_dir(app_data)
    # Nothing installed, no leftover partial/tmp dirs.
    assert not (base_dir / gpu_runtime.VERSION).exists()
    assert not (base_dir / f"{gpu_runtime.VERSION}.partial").exists()
    assert not (base_dir / "tmp").exists()

    snap = gpu_runtime.status()
    assert snap["state"] == "failed"
    assert "sha256" in snap["detail"]


def test_manifest_fetch_failure_sets_failed_state(tmp_path, monkeypatch):
    # Point at a URL base with no manifest present at all.
    empty_dir = tmp_path / "empty-release"
    empty_dir.mkdir()
    monkeypatch.setenv("AMADEUS_GPU_RUNTIME_URL_BASE", _file_url(empty_dir))
    monkeypatch.setattr(gpu_runtime, "gpu_likely_present", lambda: True)

    gpu_runtime.ensure_pack(tmp_path / "app-data", background=False)

    snap = gpu_runtime.status()
    assert snap["state"] == "failed"
    counters = telemetry.snapshot()
    assert counters.get("gpu_runtime.download_failed") == 1


# --------------------------------------------------------------------------
# Zip traversal safety
# --------------------------------------------------------------------------


def test_traversal_entry_rejected(tmp_path, monkeypatch):
    release_dir = tmp_path / "release"
    _build_fake_pack(release_dir, extra_zip_entries=["../../evil.txt"])
    monkeypatch.setenv("AMADEUS_GPU_RUNTIME_URL_BASE", _file_url(release_dir))
    monkeypatch.setattr(gpu_runtime, "gpu_likely_present", lambda: True)

    app_data = tmp_path / "app-data"
    gpu_runtime.ensure_pack(app_data, background=False)

    assert gpu_runtime.installed_pack_dir(app_data) is None
    snap = gpu_runtime.status()
    assert snap["state"] == "failed"
    assert "escapes" in snap["detail"]
    # Nothing should have leaked outside the app-data tree.
    assert not (tmp_path / "evil.txt").exists()


# --------------------------------------------------------------------------
# Old-version cleanup
# --------------------------------------------------------------------------


def test_old_version_dir_removed_after_install(tmp_path, monkeypatch):
    release_dir = tmp_path / "release"
    _build_fake_pack(release_dir)
    monkeypatch.setenv("AMADEUS_GPU_RUNTIME_URL_BASE", _file_url(release_dir))
    monkeypatch.setattr(gpu_runtime, "gpu_likely_present", lambda: True)

    app_data = tmp_path / "app-data"
    base_dir = gpu_runtime.runtime_dir(app_data)
    stale = base_dir / "gpu-runtime-v0-stale"
    stale.mkdir(parents=True)
    (stale / "site-packages").mkdir()

    gpu_runtime.ensure_pack(app_data, background=False)

    assert gpu_runtime.installed_pack_dir(app_data) is not None
    assert not stale.exists()


# --------------------------------------------------------------------------
# activate()
# --------------------------------------------------------------------------


def _install_fake_pack_dir(app_data: Path) -> Path:
    pack_dir = app_data / "gpu-runtime" / gpu_runtime.VERSION
    site_packages = pack_dir / "site-packages"
    torch_lib = site_packages / "torch" / "lib"
    sox_dir = pack_dir / "sox"
    torch_lib.mkdir(parents=True)
    sox_dir.mkdir(parents=True)
    (pack_dir / ".complete").write_text("", encoding="utf-8")
    return pack_dir


def test_activate_appends_site_packages_and_prepends_sox_path(tmp_path, monkeypatch):
    app_data = tmp_path / "app-data"
    pack_dir = _install_fake_pack_dir(app_data)
    monkeypatch.setenv("PATH", "/usr/bin")

    activated = gpu_runtime.activate(app_data)

    assert activated is True
    site_packages_str = str(pack_dir / "site-packages")
    assert site_packages_str in sys.path
    # Appended, not prepended — frozen modules must resolve first.
    assert sys.path[-1] == site_packages_str
    assert str(pack_dir / "sox") in gpu_runtime.os.environ["PATH"].split(
        gpu_runtime.os.pathsep
    )
    assert gpu_runtime.status()["state"] == "active"


def test_activate_is_idempotent(tmp_path, monkeypatch):
    app_data = tmp_path / "app-data"
    pack_dir = _install_fake_pack_dir(app_data)
    monkeypatch.setenv("PATH", "/usr/bin")

    assert gpu_runtime.activate(app_data) is True
    assert gpu_runtime.activate(app_data) is True

    site_packages_str = str(pack_dir / "site-packages")
    assert sys.path.count(site_packages_str) == 1
    path_entries = gpu_runtime.os.environ["PATH"].split(gpu_runtime.os.pathsep)
    assert path_entries.count(str(pack_dir / "sox")) == 1


def test_activate_no_pack_installed_is_noop(tmp_path):
    app_data = tmp_path / "app-data"
    assert gpu_runtime.activate(app_data) is False
    # No state recorded — controller stays uninitialized.
    assert gpu_runtime.is_initialized() is False


def test_activate_uses_add_dll_directory_when_available(tmp_path, monkeypatch):
    app_data = tmp_path / "app-data"
    pack_dir = _install_fake_pack_dir(app_data)
    calls = []
    monkeypatch.setattr(
        gpu_runtime.os, "add_dll_directory", lambda p: calls.append(p), raising=False
    )

    gpu_runtime.activate(app_data)

    assert str(pack_dir / "site-packages" / "torch" / "lib") in calls
    assert str(pack_dir / "sox") in calls


# --------------------------------------------------------------------------
# status() / is_initialized()
# --------------------------------------------------------------------------


def test_is_initialized_false_before_any_call():
    assert gpu_runtime.is_initialized() is False


def test_is_initialized_true_after_ensure_pack(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_runtime, "gpu_likely_present", lambda: False)
    gpu_runtime.ensure_pack(tmp_path / "app-data", background=False)
    assert gpu_runtime.is_initialized() is True


def test_status_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_runtime, "gpu_likely_present", lambda: False)
    gpu_runtime.ensure_pack(tmp_path / "app-data", background=False)
    snap = gpu_runtime.status()
    assert set(snap.keys()) == {"state", "progress", "detail", "version"}
    assert snap["version"] == gpu_runtime.VERSION


# --------------------------------------------------------------------------
# server.py wiring smoke tests
# --------------------------------------------------------------------------


def _make_desktop_assets(root: Path) -> None:
    from amadeus.desktop.runtime import REQUIRED_ASSET_PATHS

    for relative in REQUIRED_ASSET_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("asset", encoding="utf-8")


def _run_server_main(tmp_path, monkeypatch, *, prefer_qwen: bool):
    import amadeus.desktop.server as server

    resources = tmp_path / "resources"
    app_data = tmp_path / "app-data"
    _make_desktop_assets(resources)

    activate_calls = []
    ensure_calls = []
    monkeypatch.setattr(
        server.gpu_runtime, "activate", lambda d: activate_calls.append(d)
    )
    monkeypatch.setattr(
        server.gpu_runtime,
        "ensure_pack",
        lambda d, **kw: ensure_calls.append((d, kw)),
    )
    monkeypatch.setattr(
        server, "uvicorn", type("U", (), {"run": staticmethod(lambda *a, **k: None)})
    )
    monkeypatch.setitem(
        sys.modules,
        "amadeus.web.app",
        type("M", (), {"create_app": staticmethod(lambda: object())}),
    )

    argv = [
        "--port",
        "8123",
        "--config",
        str(tmp_path / "config.yaml"),
        "--frontend-dir",
        str(tmp_path / "frontend"),
        "--static-dir",
        str(tmp_path / "static"),
        "--resources-dir",
        str(resources),
        "--app-data-dir",
        str(app_data),
        "--write-default-config",
    ]
    if prefer_qwen:
        argv.append("--prefer-qwen")

    exit_code = server.main(argv)
    return exit_code, activate_calls, ensure_calls


def test_server_main_calls_gpu_runtime_when_prefer_qwen(tmp_path, monkeypatch):
    exit_code, activate_calls, ensure_calls = _run_server_main(
        tmp_path, monkeypatch, prefer_qwen=True
    )
    assert exit_code == 0
    assert len(activate_calls) == 1
    assert len(ensure_calls) == 1
    assert ensure_calls[0][1] == {"background": True}


def test_server_main_skips_gpu_runtime_without_prefer_qwen(tmp_path, monkeypatch):
    exit_code, activate_calls, ensure_calls = _run_server_main(
        tmp_path, monkeypatch, prefer_qwen=False
    )
    assert exit_code == 0
    assert not activate_calls
    assert not ensure_calls


def test_server_main_survives_gpu_runtime_exception(tmp_path, monkeypatch):
    import amadeus.desktop.server as server

    def _boom(*a, **k):
        raise RuntimeError("network is on fire")

    resources = tmp_path / "resources"
    app_data = tmp_path / "app-data"
    _make_desktop_assets(resources)
    monkeypatch.setattr(server.gpu_runtime, "activate", _boom)
    monkeypatch.setattr(server.gpu_runtime, "ensure_pack", _boom)
    monkeypatch.setattr(
        server, "uvicorn", type("U", (), {"run": staticmethod(lambda *a, **k: None)})
    )
    monkeypatch.setitem(
        sys.modules,
        "amadeus.web.app",
        type("M", (), {"create_app": staticmethod(lambda: object())}),
    )

    argv = [
        "--port",
        "8123",
        "--config",
        str(tmp_path / "config.yaml"),
        "--frontend-dir",
        str(tmp_path / "frontend"),
        "--static-dir",
        str(tmp_path / "static"),
        "--resources-dir",
        str(resources),
        "--app-data-dir",
        str(app_data),
        "--write-default-config",
        "--prefer-qwen",
    ]

    # Must not raise — startup survives even if both gpu_runtime calls blow up.
    exit_code = server.main(argv)
    assert exit_code == 0


def test_extract_pack_normalizes_backslash_entry_names(tmp_path):
    """PowerShell 5.1's .NET Framework zip writes backslash entry names (the
    real gpu-runtime-v1 pack shipped this way before the script fix); the
    extractor must produce proper nested paths from either separator style."""
    import zipfile

    from amadeus.gpu_runtime import _extract_pack

    zip_path = tmp_path / "pack.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("site-packages\\torch\\version.py", "x = 1\n")
        z.writestr("sox\\sox.exe", b"fake")
        z.writestr("site-packages/plain/ok.py", "y = 2\n")

    dest = tmp_path / "out"
    _extract_pack(zip_path, dest)
    assert (dest / "site-packages" / "torch" / "version.py").read_text() == "x = 1\n"
    assert (dest / "sox" / "sox.exe").read_bytes() == b"fake"
    assert (dest / "site-packages" / "plain" / "ok.py").exists()
    # No literal backslash-named files anywhere
    assert not [p for p in dest.rglob("*") if "\\" in p.name]
