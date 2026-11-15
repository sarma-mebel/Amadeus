"""Sidecar entrypoint for the packaged desktop backend."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn

try:
    from .. import gpu_runtime
    from .runtime import (
        build_desktop_config,
        reconcile_desktop_config,
        validate_required_assets,
        write_desktop_config,
    )
except ImportError:  # pragma: no cover - PyInstaller script entrypoint
    from amadeus import gpu_runtime
    from amadeus.desktop.runtime import (
        build_desktop_config,
        reconcile_desktop_config,
        validate_required_assets,
        write_desktop_config,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Amadeus desktop backend sidecar."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frontend-dir", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--resources-dir", type=Path, required=True)
    parser.add_argument("--app-data-dir", type=Path, required=True)
    parser.add_argument(
        "--write-default-config",
        action="store_true",
        help="Create a desktop config in app data if one does not already exist.",
    )
    parser.add_argument(
        "--prefer-qwen",
        action="store_true",
        help=(
            "GPU installer variant: prefer the Qwen3 cloned voice when a CUDA "
            "GPU and the bundled reference clip are present (falls back to the "
            "ONNX voice otherwise). Only affects a freshly written default "
            "config."
        ),
    )
    return parser


def _ensure_bundled_sox_on_path() -> None:
    """Put the bundled ``sox.exe`` on PATH for the GPU build.

    The GPU sidecar bundles the native ``sox`` binary (qwen-tts shells out to
    it) into the PyInstaller payload dir (``sys._MEIPASS``), but the ``sox``
    Python wrapper discovers it with a bare ``os.popen('sox -h')`` — a PATH
    lookup. Without this, PATH doesn't include the payload dir and sox is
    reported "could not be found", silently breaking Qwen3 voice synthesis.
    Must run before anything imports the ``sox`` module.
    """

    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return  # not running as a PyInstaller bundle (dev/from-source)
    if not (Path(meipass) / "sox.exe").exists():
        return  # CPU build: no bundled sox, nothing to do
    os.environ["PATH"] = meipass + os.pathsep + os.environ.get("PATH", "")


def main(argv: list[str] | None = None) -> int:
    _ensure_bundled_sox_on_path()
    args = _parser().parse_args(argv)

    validation = validate_required_assets(args.resources_dir)
    if not validation.ok:
        missing = "\n".join(f"- {path}" for path in validation.missing)
        print(
            f"Desktop package is missing required assets under {validation.root}:\n{missing}",
            file=sys.stderr,
        )
        return 2

    if args.write_default_config and not args.config.exists():
        config = build_desktop_config(
            app_data_dir=args.app_data_dir,
            resources_dir=args.resources_dir,
            prefer_qwen=args.prefer_qwen,
        )
        write_desktop_config(args.config, config)
    elif args.config.exists():
        # The config is written once, so one inherited from a previous
        # install can point resource paths at a tree that no longer exists
        # (silently killing the voice) or keep a relative cache dir that
        # resolves into Program Files. Heal what is provably dead, loudly.
        try:
            reconcile_desktop_config(
                args.config,
                app_data_dir=args.app_data_dir,
                resources_dir=args.resources_dir,
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"desktop config reconcile failed: {exc}", file=sys.stderr)
            from amadeus import telemetry

            telemetry.incr("desktop.config_reconcile_failed")

    if args.prefer_qwen:
        # Slim GPU installer: activate a previously-installed runtime pack
        # (torch/qwen deps) before any provider loads, then kick off the
        # first-run download in the background if one isn't installed yet.
        # Never let a gpu_runtime failure take down startup — the app must
        # still come up on the ONNX voice fallback either way.
        try:
            gpu_runtime.activate(args.app_data_dir)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"gpu_runtime.activate failed: {exc}", file=sys.stderr)
            from amadeus import telemetry

            telemetry.incr("gpu_runtime.activation_failed")
        try:
            gpu_runtime.ensure_pack(args.app_data_dir, background=True)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"gpu_runtime.ensure_pack failed: {exc}", file=sys.stderr)
            from amadeus import telemetry

            telemetry.incr("gpu_runtime.download_failed")

    os.environ["AMADEUS_CONFIG"] = str(args.config)
    os.environ["AMADEUS_USER_CONFIG"] = str(args.config)
    os.environ["KURISU_FRONTEND_DIR"] = str(args.frontend_dir)
    os.environ["AMADEUS_STATIC_DIR"] = str(args.static_dir)
    # We call create_app() ourselves below — without this flag the import of
    # amadeus.web.app ALSO builds the eager module-level app, loading every
    # provider twice. In the GPU build that meant two Qwen3-TTS models on one
    # 8GB card: VRAM full, synthesis ~25x slower (1100ms/step vs 45ms).
    os.environ["AMADEUS_NO_MODULE_APP"] = "1"

    from amadeus.web.app import create_app

    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        log_level=os.getenv("AMADEUS_LOG_LEVEL", "info"),
        # No per-request access log in the packaged sidecar: stdout/stderr are
        # pipes drained by the Tauri shell, and every console write is a
        # liability there — if the drain ever stops, a full pipe blocks the
        # event loop inside logging flush() and the whole backend wedges
        # (seen live on v0.15.0; the frontend polls make access logs a steady
        # flood). App-level logging still goes to the rotating file handler.
        access_log=False,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
