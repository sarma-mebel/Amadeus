#!/usr/bin/env python3
"""Environment doctor: checks this machine against Amadeus's known dev requirements.

Stdlib-only by design (it has to run BEFORE you know whether the project's
own dependencies are even installed). Prints a PASS/FAIL/WARN table and
exits nonzero only if something FAILed — WARNs are informational (e.g.
optional voice assets not present) and don't fail the run.

Usage:
    python scripts/doctor.py
    ~/miniconda3/envs/amadeus/bin/python scripts/doctor.py

See docs/dev-setup.md for what each check means and how to fix a FAIL/WARN.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Callable, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


class Result(NamedTuple):
    name: str
    status: str
    detail: str


def _run(cmd: list[str], timeout: float = 5.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out.strip()
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, "timed out"


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_python_version() -> Result:
    major, minor, micro = sys.version_info[:3]
    version = f"{major}.{minor}.{micro}"
    if (major, minor) == (3, 10):
        return Result("Python version", PASS, f"{version} (matches dev env)")
    if major == 3 and minor >= 9:
        return Result(
            "Python version",
            WARN,
            f"{version} (pyproject requires-python >=3.9, but the dev env "
            "used for validation is 3.10 — some extras may behave differently)",
        )
    return Result("Python version", FAIL, f"{version} (need >=3.9, dev env is 3.10)")


def _import_version(module: str) -> tuple[bool, str]:
    # Some optional deps (qwen-tts, the `sox` pip package it pulls in, etc.)
    # print warnings straight to stdout/stderr on import. Suppress that here
    # so the doctor's own table stays clean; it doesn't affect the result.
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - report any import failure as absent
        return False, str(exc)
    version = getattr(mod, "__version__", None)
    if version is None:
        version = getattr(mod, "version", "installed (no __version__)")
    return True, str(version)


def check_fastapi() -> Result:
    ok, detail = _import_version("fastapi")
    if not ok:
        return Result(
            "fastapi",
            FAIL,
            "not importable — install with `pip install -e \".[web]\"` "
            "(the test suite importorskip's this; most tests are skipped without it)",
        )
    return Result("fastapi", PASS, detail)


def check_torch() -> Result:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return Result(
            "torch",
            WARN,
            f"not importable ({exc}) — only needed for local/[memory]/qwen-tts extras",
        )
    version = getattr(torch, "__version__", "unknown")
    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda_ok = False
    if cuda_ok:
        return Result("torch", PASS, f"{version}, CUDA available")
    if "+cpu" in version or "cu" not in version:
        return Result(
            "torch",
            WARN,
            f"{version}, CUDA NOT available — this looks like a CPU-only build. "
            "qwen3 voice will run ~10x slower or fail. Install from the PyTorch "
            "CUDA index (see environment.yml).",
        )
    return Result(
        "torch",
        WARN,
        f"{version}, CUDA reports unavailable on this machine (no GPU, or "
        "driver mismatch) — voice synthesis will be slow",
    )


def check_transformers() -> Result:
    ok, detail = _import_version("transformers")
    if not ok:
        return Result(
            "transformers",
            WARN,
            "not importable — only needed for local/[memory]/qwen-tts extras",
        )
    pinned = _find_transformers_pin()
    if pinned and detail != pinned:
        return Result(
            "transformers",
            WARN,
            f"{detail} installed, but requirements-lock.txt pins {pinned} "
            "(an upgrade past the pin previously broke qwen-tts compatibility)",
        )
    return Result("transformers", PASS, detail)


def _find_transformers_pin() -> str | None:
    lock = REPO_ROOT / "requirements-lock.txt"
    if not lock.exists():
        return None
    try:
        for line in lock.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.lower().startswith("transformers=="):
                return line.split("==", 1)[1]
    except OSError:
        return None
    return None


def check_qwen_tts() -> Result:
    have_qwen, qwen_detail = _import_version("qwen_tts")
    have_faster, faster_detail = _import_version("faster_qwen3_tts")
    if have_faster:
        return Result("qwen3 TTS engine", PASS, f"faster-qwen3-tts {faster_detail} (fast path)")
    if have_qwen:
        return Result(
            "qwen3 TTS engine",
            WARN,
            f"only qwen-tts {qwen_detail} present, faster-qwen3-tts missing — "
            "TTS will work but run roughly 10x slower",
        )
    return Result(
        "qwen3 TTS engine",
        WARN,
        "neither qwen-tts nor faster-qwen3-tts importable — only needed if "
        "providers.tts: qwen3_tts (install the [qwen-tts] extra)",
    )


def check_sox_binary() -> Result:
    path = shutil.which("sox")
    if path:
        return Result("sox (system binary)", PASS, path)
    return Result(
        "sox (system binary)",
        WARN,
        "not on PATH — required by the qwen3 TTS audio pipeline "
        "(`sudo apt install sox` on Debian/Ubuntu/WSL); unrelated to the "
        "pip `sox`/`soxr` packages",
    )


def check_node() -> Result:
    code, out = _run(["node", "--version"])
    if code != 0:
        return Result("node", FAIL, "not found on PATH")
    version = out.strip().lstrip("v")
    try:
        major = int(version.split(".", 1)[0])
    except ValueError:
        return Result("node", WARN, f"unparsable version output: {out!r}")
    # frontend/package.json has no engines field pinning a minimum; Node 20
    # LTS is what CI and docs standardize on, so warn (not fail) below it.
    if major >= 20:
        return Result("node", PASS, f"v{version}")
    return Result("node", WARN, f"v{version} (project standardizes on Node 20 LTS)")


def check_npm() -> Result:
    code, out = _run(["npm", "--version"])
    if code != 0:
        return Result("npm", FAIL, "not found on PATH")
    return Result("npm", PASS, out.strip())


def check_frontend_node_modules() -> Result:
    path = REPO_ROOT / "frontend" / "node_modules"
    if path.is_dir():
        return Result("frontend/node_modules", PASS, "present")
    return Result(
        "frontend/node_modules",
        WARN,
        "missing — run `npm install` (or `npm ci`) inside frontend/",
    )


def _candidate_cargo_paths() -> list[Path]:
    candidates = []
    home = Path.home()
    candidates.append(home / ".cargo" / "bin" / "cargo")
    # WSL: the Windows-side cargo is usable for `cargo check` even though
    # this repo's Tauri shell is normally built on native Windows.
    for user_dir in Path("/mnt/c/Users").glob("*"):
        candidates.append(user_dir / ".cargo" / "bin" / "cargo.exe")
    return candidates


def check_cargo() -> Result:
    path = shutil.which("cargo")
    if path:
        code, out = _run([path, "--version"])
        if code == 0:
            return Result("cargo", PASS, f"{path} ({out.strip()})")
    for candidate in _candidate_cargo_paths():
        if candidate.exists():
            code, out = _run([str(candidate), "--version"])
            if code == 0:
                return Result("cargo", PASS, f"{candidate} ({out.strip()})")
            return Result("cargo", WARN, f"found at {candidate} but failed to run")
    return Result(
        "cargo",
        WARN,
        "not found on PATH or in common WSL/Windows locations — only needed "
        "for frontend/src-tauri desktop builds (`cargo check`)",
    )


def check_config_local() -> Result:
    path = REPO_ROOT / "config.local.yaml"
    if path.exists():
        return Result("config.local.yaml", PASS, "present")
    return Result(
        "config.local.yaml",
        WARN,
        "missing — copy config/config.example.yaml to config.local.yaml "
        "to run against real providers (the app still runs on dummy "
        "providers without it)",
    )


def check_voice_ref_clip() -> Result:
    path = REPO_ROOT / "models" / "kurisu_voice_qwen" / "kurisu_ref.wav"
    if path.exists():
        size_kb = path.stat().st_size // 1024
        return Result("qwen3 voice ref clip", PASS, f"present ({size_kb} KB)")
    return Result(
        "qwen3 voice ref clip",
        WARN,
        "models/kurisu_voice_qwen/kurisu_ref.wav missing — gitignored "
        "runtime state, only needed for providers.tts: qwen3_tts "
        "(voice-optional; other TTS providers work without it)",
    )


def check_nonverbal_assets() -> Result:
    """Pre-generated [laugh]/[sigh]/... clips (scripts/generate_nonverbals.py).

    Gitignored voice-specific assets — missing entirely is a normal "feature
    off" state, not a problem. NonVerbalLibrary (amadeus/web/streaming/
    nonverbals.py) is a process-wide singleton loaded once, so regenerating
    clips needs a backend restart to pick up.
    """
    root = REPO_ROOT / "static" / "nonverbals"
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return Result(
            "non-verbal clips (static/nonverbals/)",
            WARN,
            "manifest.json missing — feature off (generate with "
            "`python scripts/generate_nonverbals.py`); NonVerbalLibrary is a "
            "process singleton, restart the backend after generating",
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - malformed manifest = feature off
        return Result(
            "non-verbal clips (static/nonverbals/)",
            WARN,
            f"manifest.json unreadable ({exc}) — feature off",
        )
    clips = manifest.get("clips")
    if not isinstance(clips, dict):
        return Result(
            "non-verbal clips (static/nonverbals/)",
            WARN,
            "manifest.json has no 'clips' object — feature off",
        )
    total = 0
    missing: list[str] = []
    for kind, files in clips.items():
        if not isinstance(files, list):
            continue
        for name in files:
            if not isinstance(name, str):
                continue
            total += 1
            if not (root / Path(name).name).exists():
                missing.append(f"{kind}/{name}")
    if total == 0:
        return Result(
            "non-verbal clips (static/nonverbals/)",
            WARN,
            "manifest.json present but lists no clips — feature off",
        )
    if missing:
        return Result(
            "non-verbal clips (static/nonverbals/)",
            WARN,
            f"{total} clip(s) in manifest, {len(missing)} missing on disk: "
            f"{', '.join(missing[:5])}"
            + (", ..." if len(missing) > 5 else "")
            + " — NonVerbalLibrary is a process singleton, restart the "
            "backend after regenerating",
        )
    return Result(
        "non-verbal clips (static/nonverbals/)",
        PASS,
        f"{total} clip(s) across {len(clips)} kind(s) — NonVerbalLibrary is a "
        "process singleton, restart the backend after regenerating",
    )


def check_filler_assets() -> Result:
    """Pre-generated filler/backchannel clips (scripts/generate_fillers.py).

    Gitignored voice-specific assets consumed by frontend/src/audio/
    fillerPlayer.ts (fetched client-side, not by the backend) — missing
    entirely is a normal "feature off" state.
    """
    root = REPO_ROOT / "static" / "fillers"
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return Result(
            "filler clips (static/fillers/)",
            WARN,
            "manifest.json missing — feature off (generate with "
            "`python scripts/generate_fillers.py`)",
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - malformed manifest = feature off
        return Result(
            "filler clips (static/fillers/)",
            WARN,
            f"manifest.json unreadable ({exc}) — feature off",
        )
    total = 0
    missing: list[str] = []
    for key in ("fillers", "backchannels"):
        entries = manifest.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("file")
            if not isinstance(name, str):
                continue
            total += 1
            if not (root / Path(name).name).exists():
                missing.append(f"{key}/{name}")
    if total == 0:
        return Result(
            "filler clips (static/fillers/)",
            WARN,
            "manifest.json present but lists no clips — feature off",
        )
    if missing:
        return Result(
            "filler clips (static/fillers/)",
            WARN,
            f"{total} clip(s) in manifest, {len(missing)} missing on disk: "
            f"{', '.join(missing[:5])}" + (", ..." if len(missing) > 5 else ""),
        )
    return Result("filler clips (static/fillers/)", PASS, f"{total} clip(s) present")


def check_githooks() -> Result:
    code, out = _run(["git", "-C", str(REPO_ROOT), "config", "--get", "core.hooksPath"])
    if code == 0 and out.strip() == ".githooks":
        return Result("secret-scan hook", PASS, "core.hooksPath = .githooks")
    return Result(
        "secret-scan hook",
        WARN,
        "core.hooksPath not set to .githooks — run "
        "`git config core.hooksPath .githooks` to enable the pre-commit secret scan",
    )


def check_port_8000() -> Result:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        in_use = sock.connect_ex(("127.0.0.1", 8000)) == 0
    finally:
        sock.close()
    if in_use:
        return Result(
            "port 8000",
            WARN,
            "something is already listening on 127.0.0.1:8000 — uvicorn "
            "will fail to bind; on this kind of WSL setup it's often a "
            "Windows-side process squatting the port, not a stale backend. "
            "Use --port to bind elsewhere or free the port first.",
        )
    return Result("port 8000", PASS, "free")


CHECKS: list[Callable[[], Result]] = [
    check_python_version,
    check_fastapi,
    check_torch,
    check_transformers,
    check_qwen_tts,
    check_sox_binary,
    check_node,
    check_npm,
    check_frontend_node_modules,
    check_cargo,
    check_config_local,
    check_voice_ref_clip,
    check_nonverbal_assets,
    check_filler_assets,
    check_githooks,
    check_port_8000,
]


def main() -> int:
    results = []
    for check in CHECKS:
        try:
            results.append(check())
        except Exception as exc:  # noqa: BLE001 - a broken check must not crash the doctor
            results.append(Result(check.__name__, FAIL, f"check raised: {exc!r}"))

    name_width = max(len(r.name) for r in results) + 2
    status_width = 4

    print(f"Amadeus environment doctor — {sys.executable}\n")
    for r in results:
        print(f"[{r.status:<{status_width}}] {r.name:<{name_width}} {r.detail}")

    fails = [r for r in results if r.status == FAIL]
    warns = [r for r in results if r.status == WARN]
    print(f"\n{len(results)} checks: {len(results) - len(fails) - len(warns)} passed, "
          f"{len(warns)} warned, {len(fails)} failed.")

    if fails:
        print("\nFAIL means the check found a real blocker — see the detail above.")
        return 1
    if warns:
        print("\nWARNs are informational (often expected on WSL / a fresh checkout).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
