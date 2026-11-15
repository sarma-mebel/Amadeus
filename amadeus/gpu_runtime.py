"""GPU runtime pack: fetch CUDA torch + qwen deps after a slim install.

The GPU desktop installer used to bundle CUDA torch (~2.5-3GB) directly,
which made the NSIS installer too large to build and the release asset too
large for GitHub's 2GB cap (breaking auto-update for that variant). The slim
GPU installer ships WITHOUT the CUDA venv and instead fetches a "GPU runtime
pack" — a split zip on a GitHub release — on first launch, same pattern as
the ~8.5GB HF voice weights that already download at first run
(``providers/tts/qwen3_tts_provider.py``). The pack is extracted under app
data and activated onto ``sys.path`` so the frozen PyInstaller backend can
import ``torch``/``qwen`` from it without having bundled either.

Cross-agent contract (packaging side builds to this exact spec):
- Release tag ``gpu-runtime-v2``; default URL base
  ``https://github.com/zxKyouma/Amadeus/releases/download/gpu-runtime-v2``.
- Assets: ``gpu-runtime-manifest.json`` + ``gpu-runtime.zip.partNN`` (each
  <=1.9GB; concatenated in order they form one zip).
- Manifest: ``{"version", "created", "zip_sha256", "total_size",
  "parts": [{"name", "size", "sha256"}, ...], "layout": {"site_packages",
  "sox"}, "torch"}``.
- Installed layout: ``<app_data>/gpu-runtime/gpu-runtime-v2/site-packages/``,
  ``.../sox/``, with a ``.complete`` sentinel written last.
- The pack's ``site-packages`` is the FULL GPU venv site-packages plus a
  stdlib overlay (top-level ``Lib/*.py`` + ``DLLs/*.pyd``) — NOT a diff
  against the CPU venv. v1 was a diff and it shipped a pack whose ``import
  torch`` died inside the frozen sidecar (PyInstaller freezes only what its
  analysis reaches, so "the CPU venv has it" never implied "the sidecar has
  it" — numpy/tokenizers/filelock and even stdlib pickletools/filecmp/
  cProfile were all missing at runtime). Activation APPENDS to ``sys.path``,
  so frozen modules always win and pack/sidecar overlap is harmless.
- Env overrides: ``AMADEUS_GPU_RUNTIME_AUTOFETCH=0`` disables auto-download;
  ``AMADEUS_GPU_RUNTIME_DIR`` overrides the base dir; the base URL is
  overridden with ``AMADEUS_GPU_RUNTIME_URL_BASE`` (tests point this at a
  local file:// or http server).

Degrade loudly: every failure path counts a ``gpu_runtime.*`` telemetry
counter and logs a warning. A failed download leaves state "failed" with a
detail string; there is no persistent backoff — the next process launch is
the retry cadence (this is a desktop app, not a long-lived server).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import telemetry

_LOGGER = logging.getLogger(__name__)

VERSION = "gpu-runtime-v2"
DEFAULT_URL_BASE = (
    "https://github.com/zxKyouma/Amadeus/releases/download/gpu-runtime-v2"
)
MANIFEST_NAME = "gpu-runtime-manifest.json"

_AUTOFETCH_ENV = "AMADEUS_GPU_RUNTIME_AUTOFETCH"
_DIR_ENV = "AMADEUS_GPU_RUNTIME_DIR"
_URL_BASE_ENV = "AMADEUS_GPU_RUNTIME_URL_BASE"

_PART_MAX_RETRIES = 2
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024


# --------------------------------------------------------------------------
# Pure path/probe helpers
# --------------------------------------------------------------------------


def gpu_likely_present() -> bool:
    """Torch-free NVIDIA probe: is ``nvidia-smi`` on PATH?

    NVIDIA drivers ship ``nvidia-smi`` into System32 (Windows) / PATH
    (Linux), so its presence is a cheap, dependency-free signal that *some*
    NVIDIA GPU + driver is installed — used for the config-bake decision and
    as the download gate. It is NOT proof CUDA actually works; the
    definitive check stays torch-based at provider load time
    (``desktop/runtime.py::cuda_available``), which the slim build cannot
    perform until the pack is activated.
    """

    return shutil.which("nvidia-smi") is not None


def runtime_dir(app_data_dir: Path | str) -> Path:
    """Base directory for GPU runtime packs, honoring the dir override."""

    override = os.environ.get(_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path(app_data_dir).expanduser() / "gpu-runtime"


def _url_base() -> str:
    return os.environ.get(_URL_BASE_ENV) or DEFAULT_URL_BASE


def installed_pack_dir(app_data_dir: Path | str) -> Optional[Path]:
    """The versioned pack dir, only when its ``.complete`` sentinel exists."""

    candidate = runtime_dir(app_data_dir) / VERSION
    if (candidate / ".complete").exists():
        return candidate
    return None


def _autofetch_enabled() -> bool:
    return os.environ.get(_AUTOFETCH_ENV) != "0"


# --------------------------------------------------------------------------
# Activation
# --------------------------------------------------------------------------


def activate(app_data_dir: Path | str) -> bool:
    """Make an installed pack importable in this process. Idempotent.

    Appends (never prepends) the pack's ``site-packages`` to ``sys.path`` so
    frozen/bundled modules always win over the downloaded pack — the
    PyInstaller payload's own stdlib/numpy/etc must never be shadowed.
    Registers DLL search directories for ``torch/lib`` and the pack's
    ``sox`` dir (Windows only; guarded via ``hasattr`` since
    ``os.add_dll_directory`` doesn't exist on other platforms), and prepends
    the ``sox`` dir to PATH — the ``sox`` Python wrapper discovers the
    binary via a bare ``os.popen('sox -h')`` PATH lookup, the same reason
    ``desktop/server.py::_ensure_bundled_sox_on_path`` exists for the
    full-GPU build's bundled sox.

    Returns whether activation happened (False when no pack is installed).
    """

    pack_dir = installed_pack_dir(app_data_dir)
    if pack_dir is None:
        return False

    try:
        site_packages = pack_dir / "site-packages"
        site_packages_str = str(site_packages)
        if site_packages.exists() and site_packages_str not in sys.path:
            sys.path.append(site_packages_str)

        torch_lib = site_packages / "torch" / "lib"
        sox_dir = pack_dir / "sox"
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None:
            if torch_lib.exists():
                add_dll_directory(str(torch_lib))
            if sox_dir.exists():
                add_dll_directory(str(sox_dir))
        if sox_dir.exists():
            sox_str = str(sox_dir)
            path = os.environ.get("PATH", "")
            if sox_str not in path.split(os.pathsep):
                os.environ["PATH"] = sox_str + os.pathsep + path

        _controller().mark_active()
        return True
    except Exception as exc:  # pragma: no cover - defensive
        _LOGGER.warning("gpu_runtime: activation failed: %s", exc)
        telemetry.incr("gpu_runtime.activation_failed")
        return False


# --------------------------------------------------------------------------
# Fetch controller
# --------------------------------------------------------------------------


@dataclass
class _State:
    state: str = "absent"
    progress: Optional[float] = None
    detail: Optional[str] = None
    active: bool = False


@dataclass
class _Controller:
    """Process-wide singleton tracking one pack fetch's lifecycle."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    _state: _State = field(default_factory=_State)
    _initialized: bool = False

    def mark_active(self) -> None:
        with self.lock:
            self._initialized = True
            self._state.active = True
            if self._state.state in ("absent", "ready_restart"):
                self._state.state = "active"

    def _set(self, **kwargs) -> None:
        with self.lock:
            self._initialized = True
            for key, value in kwargs.items():
                setattr(self._state, key, value)

    def snapshot(self) -> dict:
        with self.lock:
            s = self._state
            return {
                "state": s.state,
                "progress": s.progress,
                "detail": s.detail,
                "version": VERSION,
            }

    def is_initialized(self) -> bool:
        with self.lock:
            return self._initialized

    def is_active(self) -> bool:
        with self.lock:
            return self._state.active


_controller_lock = threading.Lock()
_controller_instance: Optional[_Controller] = None


def _controller() -> _Controller:
    global _controller_instance
    with _controller_lock:
        if _controller_instance is None:
            _controller_instance = _Controller()
        return _controller_instance


def _reset_controller_for_tests() -> None:
    """Test-only: drop the singleton so each test starts clean."""

    global _controller_instance
    with _controller_lock:
        _controller_instance = None


def is_initialized() -> bool:
    """Whether the fetch controller has ever recorded state this process.

    Lets ``/api/status`` omit the ``gpu_runtime`` key entirely on web/CPU
    deployments that never call ``ensure_pack``/``activate`` — no dead
    section in the UI for builds that don't have this feature.
    """

    controller = _controller_instance
    return controller is not None and controller.is_initialized()


def status() -> dict:
    """Current fetch/activation state for ``/api/status``."""

    return _controller().snapshot()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_to(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out, length=_DOWNLOAD_CHUNK_BYTES)


def _fetch_manifest(url_base: str) -> dict:
    url = f"{url_base}/{MANIFEST_NAME}"
    with urllib.request.urlopen(url, timeout=30) as response:
        # utf-8-sig: PowerShell 5.1 writes JSON with a UTF-8 BOM by default,
        # and a BOM'd manifest crashed plain utf-8 json.loads (found on the
        # first real pack build). -sig decodes both BOM'd and BOM-less.
        return json.loads(response.read().decode("utf-8-sig"))


def _validate_member(name: str, dest_root: Path) -> Path:
    """Reject zip entries with absolute paths or ``..`` traversal.

    The pack is our own build artifact, but it rides an unauthenticated CDN
    URL (a plain GitHub release asset) — the zip's own sha256 is verified
    BEFORE extraction starts, and every member path is still validated on
    the way out, mirroring ``characters.py::_extract_pack_zip``.
    """

    if name.startswith(("/", "\\")) or (":" in name and sys.platform == "win32"):
        raise ValueError(f"Runtime pack entry has an absolute path: {name!r}")
    target = (dest_root / name).resolve()
    if not str(target).startswith(str(dest_root.resolve())):
        raise ValueError(f"Runtime pack entry escapes the archive: {name!r}")
    return target


def _extract_pack(zip_path: Path, dest_root: Path) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            # Normalize separators before validating/extracting: PowerShell
            # 5.1's .NET Framework zip writes backslash entry names (the zip
            # spec says '/'), which extractall would faithfully turn into
            # literal 'a\b\c' filenames on non-Windows and which the real
            # v1 pack actually ships. Manual extraction from the normalized
            # name is correct on every platform and for both styles.
            name = info.filename.replace("\\", "/")
            target = _validate_member(name, dest_root)
            if name.endswith("/") or info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)


def _cleanup_other_versions(base_dir: Path, keep: str) -> None:
    if not base_dir.exists():
        return
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name in (keep, "tmp") or child.name.endswith(".partial"):
            continue
        try:
            shutil.rmtree(child, ignore_errors=True)
        except Exception:  # pragma: no cover - best-effort cleanup
            pass


def _run_fetch(app_data_dir: Path | str) -> None:
    controller = _controller()
    base_dir = runtime_dir(app_data_dir)
    tmp_dir = base_dir / "tmp"
    url_base = _url_base()

    controller._set(state="downloading", progress=0.0, detail=None)
    telemetry.incr("gpu_runtime.download_started")

    try:
        manifest = _fetch_manifest(url_base)
    except Exception as exc:
        _LOGGER.warning("gpu_runtime: manifest fetch failed: %s", exc)
        telemetry.incr("gpu_runtime.download_failed")
        controller._set(
            state="failed", progress=None, detail=f"manifest fetch failed: {exc}"
        )
        return

    parts = manifest.get("parts") or []
    total_size = int(manifest.get("total_size") or sum(p.get("size", 0) for p in parts))
    zip_sha256 = manifest.get("zip_sha256")
    layout = manifest.get("layout") or {"site_packages": "site-packages", "sox": "sox"}

    tmp_dir.mkdir(parents=True, exist_ok=True)
    part_paths: list[Path] = []
    bytes_done = 0

    try:
        for part in parts:
            name = part["name"]
            expected_size = int(part.get("size") or 0)
            expected_sha256 = part.get("sha256")
            part_path = tmp_dir / name

            attempt = 0
            while True:
                attempt += 1
                try:
                    _download_to(f"{url_base}/{name}", part_path)
                    actual_size = part_path.stat().st_size
                    actual_sha256 = _sha256_file(part_path)
                    if expected_size and actual_size != expected_size:
                        raise ValueError(
                            f"size mismatch for {name}: got {actual_size}, "
                            f"expected {expected_size}"
                        )
                    if expected_sha256 and actual_sha256 != expected_sha256:
                        raise ValueError(f"sha256 mismatch for {name}")
                    break
                except Exception as exc:
                    if attempt > _PART_MAX_RETRIES:
                        raise
                    _LOGGER.warning(
                        "gpu_runtime: part %s failed (attempt %d/%d): %s",
                        name,
                        attempt,
                        _PART_MAX_RETRIES + 1,
                        exc,
                    )
                    telemetry.incr("gpu_runtime.part_retries")

            part_paths.append(part_path)
            bytes_done += part_path.stat().st_size
            progress = (bytes_done / total_size) if total_size else None
            controller._set(state="downloading", progress=progress, detail=None)

        # Concatenate parts into one zip, in manifest order.
        zip_path = tmp_dir / "gpu-runtime.zip"
        with zip_path.open("wb") as out:
            for part_path in part_paths:
                with part_path.open("rb") as handle:
                    shutil.copyfileobj(handle, out, length=_DOWNLOAD_CHUNK_BYTES)

        if zip_sha256:
            actual = _sha256_file(zip_path)
            if actual != zip_sha256:
                raise ValueError(
                    f"assembled zip sha256 mismatch: got {actual}, expected {zip_sha256}"
                )

        controller._set(state="extracting", progress=None, detail=None)
        version = str(manifest.get("version") or VERSION)
        partial_dir = base_dir / f"{version}.partial"
        if partial_dir.exists():
            shutil.rmtree(partial_dir, ignore_errors=True)
        _extract_pack(zip_path, partial_dir)

        final_dir = base_dir / version
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        partial_dir.rename(final_dir)
        (final_dir / ".complete").write_text(
            layout.get("torch", "") or "", encoding="utf-8"
        )

        _cleanup_other_versions(base_dir, keep=version)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        telemetry.incr("gpu_runtime.download_completed")
        controller._set(state="ready_restart", progress=1.0, detail=None)
    except Exception as exc:
        _LOGGER.warning("gpu_runtime: pack fetch failed: %s", exc)
        telemetry.incr("gpu_runtime.download_failed")
        controller._set(state="failed", progress=None, detail=str(exc))
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:  # pragma: no cover - best-effort cleanup
            pass


def ensure_pack(app_data_dir: Path | str, *, background: bool = True) -> None:
    """Kick off the first-run GPU runtime pack download if needed.

    No-op when: a pack is already installed, ``AMADEUS_GPU_RUNTIME_AUTOFETCH
    =0``, or ``gpu_likely_present()`` is False (no visible NVIDIA driver —
    don't spend 2-3GB of bandwidth on a machine that can't use CUDA anyway).
    Runs in a daemon thread by default so app startup never blocks on a
    multi-gigabyte download; the app runs on the ONNX voice fallback in the
    meantime and picks up the pack on the next launch (see ``activate``).
    """

    controller = _controller()

    if installed_pack_dir(app_data_dir) is not None:
        controller._set(state="active" if controller.is_active() else "ready_restart")
        return

    if not _autofetch_enabled():
        controller._set(state="disabled", progress=None, detail=None)
        return

    if not gpu_likely_present():
        controller._set(
            state="absent", progress=None, detail="no NVIDIA driver detected"
        )
        return

    if background:
        thread = threading.Thread(
            target=_run_fetch,
            args=(app_data_dir,),
            daemon=True,
            name="gpu-runtime-fetch",
        )
        thread.start()
    else:
        _run_fetch(app_data_dir)
