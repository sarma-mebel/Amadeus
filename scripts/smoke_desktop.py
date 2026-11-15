#!/usr/bin/env python3
"""Smoke-test a packaged/built desktop app end to end.

Boots the real Tauri exe, waits for the backend sidecar, discovers its
(random) port from the sidecar process's listening sockets, and then
verifies the contract that manual shakedowns keep re-checking by hand:

- the app window actually exists (a boot crash shows up here)
- /healthz answers ok and reports the voice state
- /api/status shows a REAL llm provider (not the silent dummy fallback)
- optionally, one real chat turn round-trips (isolated `smoke-test` user
  id, so your own memory store is never touched; costs one live LLM call)

Run it after each local release build:

    python scripts/smoke_desktop.py                       # checks only
    python scripts/smoke_desktop.py --chat                # + one live turn
    python scripts/smoke_desktop.py --timeout 360         # GPU build warmup

Designed to run from WSL (launches the Windows exe via interop; process
introspection AND all HTTP go through powershell.exe, because WSL's
127.0.0.1 is not the Windows 127.0.0.1 the app binds). Webview-internal checks
(ACLs, SPA state) are NOT covered: WebView2 ignores
WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS when the host passes explicit
browser args, so packaged builds cannot expose CDP — that part stays a
manual shakedown item.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_EXE = REPO / "frontend" / "src-tauri" / "target" / "release" / "amadeus-desktop.exe"


def http_get_json(url: str, timeout: int = 10) -> dict | None:
    """HTTP through the Windows side: WSL's 127.0.0.1 is not the Windows
    127.0.0.1 the packaged app binds (verified live — WSL curl connects to
    the forwarded socket but no HTTP ever completes)."""
    raw = powershell(
        f"try {{ (Invoke-WebRequest -Uri '{url}' -TimeoutSec {timeout} "
        "-UseBasicParsing).Content } catch { '' }"
    ).strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def http_post_json(url: str, payload: dict, timeout: int = 180) -> dict | None:
    body = json.dumps(payload).replace("'", "''")
    raw = powershell(
        f"try {{ Invoke-RestMethod -Uri '{url}' -Method Post -ContentType "
        f"'application/json' -Body '{body}' -TimeoutSec {timeout} | "
        "ConvertTo-Json -Depth 6 -Compress } catch { '' }"
    ).strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def powershell(command: str) -> str:
    # Force UTF-8 on the pipe: replies contain Japanese, and the default
    # console codepage garbles it (and crashed utf-8 decoding once).
    wrapped = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; " + command
    result = subprocess.run(
        ["powershell.exe", "-ExecutionPolicy", "Bypass", "-Command", wrapped],
        capture_output=True,
    )
    return result.stdout.decode("utf-8", errors="replace")


def wait_for(predicate, timeout: float, interval: float = 2.0, label: str = ""):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise TimeoutError(f"timed out after {timeout:.0f}s waiting for {label}")


def backend_port() -> int | None:
    """The sidecar binds one random localhost port; read it off netstat by
    the amadeus-backend PID.

    Checks EVERY amadeus-backend PID: the onefile exe runs as a
    parent(bootstrap)+child(python) pair and Get-Process ordering is not
    guaranteed — taking the first PID intermittently polled the
    non-listening parent for the whole window and failed a smoke whose
    backend was actually healthy (v0.15.1, twice)."""
    pid_text = powershell(
        "(Get-Process amadeus-backend -ErrorAction SilentlyContinue).Id"
    ).strip()
    if not pid_text:
        return None
    net = powershell("netstat -ano -p TCP | Select-String 'LISTENING'")
    for pid in pid_text.split():
        match = re.search(rf"127\.0\.0\.1:(\d+)\s.*LISTENING\s+{pid}\s*$", net, re.M)
        if match:
            return int(match.group(1))
    return None


def window_present() -> bool:
    title = powershell(
        "(Get-Process amadeus-desktop -ErrorAction SilentlyContinue).MainWindowTitle"
    ).strip()
    return bool(title)


def kill_app() -> None:
    """Blunt teardown by image name (never by pattern — a pattern can match
    the invoking shell itself; that one has bitten before)."""
    for image in ("amadeus-desktop.exe", "amadeus-backend.exe"):
        subprocess.run(
            ["cmd.exe", "/c", "taskkill", "/IM", image, "/T", "/F"],
            capture_output=True,
        )


def preexisting_instances() -> str:
    """Any amadeus process already running before the smoke boots its exe.

    Discovery (backend_port) matches by process NAME, so a running installed
    app poisons every check — the v0.15.0 smoke happily health-checked the
    user's installed 0.13.0 companion (its version, its provider) and then
    kill_app() closed it. Refuse to run instead."""
    return powershell(
        "(Get-Process amadeus-desktop,amadeus-backend -ErrorAction "
        "SilentlyContinue | Select-Object -ExpandProperty ProcessName) -join ','"
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Packaged desktop app smoke test")
    parser.add_argument("--exe", default=str(DEFAULT_EXE), help="app exe to boot")
    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="seconds to wait for backend health (GPU build: use 360)",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="also run one real chat turn (isolated smoke-test user id)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="leave the app running after the checks",
    )
    parser.add_argument(
        "--out",
        default=str(REPO / "logs" / "smoke"),
        help="directory for the report",
    )
    args = parser.parse_args()

    exe = Path(args.exe)
    if not exe.exists():
        print(f"ERROR: exe not found: {exe}", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"exe": str(exe), "steps": {}}
    passed_all = True

    def step(name: str, passed: bool, detail: str = "") -> None:
        nonlocal passed_all
        passed_all = passed_all and passed
        report["steps"][name] = {"ok": passed, "detail": detail}
        print(f"  {'✓' if passed else '✗'} {name}" + (f" — {detail}" if detail else ""))

    running = preexisting_instances()
    if running:
        print(
            f"REFUSING TO RUN: amadeus already running ({running}). Close the "
            "installed/desktop app first — the smoke discovers the sidecar by "
            "process name, so it would test the running instance instead of "
            "this build, and teardown would kill it."
        )
        return 1

    print(f"Booting {exe.name}…")
    process = subprocess.Popen(
        [str(exe)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        # 180s, not 60: the onefile sidecar extracts a ~180MB payload into
        # %TEMP% (a dir Defender still scans) before python even starts —
        # measured 67s spawn-to-first-log on v0.15.1, which blew a 60s window
        # and killed the app mid-boot while it was actually healthy.
        port = wait_for(backend_port, timeout=180, label="the backend sidecar to bind")
        origin = f"http://127.0.0.1:{port}"
        step("sidecar spawned and listening", True, origin)

        def healthy():
            data = http_get_json(f"{origin}/healthz", timeout=3)
            return data if data and data.get("status") == "ok" else None

        health = wait_for(
            healthy, timeout=args.timeout, label="/healthz to pass"
        )
        step(
            "healthz ok",
            True,
            f"voice={health.get('voice')} version={health.get('version')}",
        )

        step("app window present", window_present())

        status = http_get_json(f"{origin}/api/status", timeout=10) or {}
        llm = status.get("llm", {})
        step(
            "llm provider is real",
            not llm.get("is_dummy", True),
            f"{llm.get('provider')} ({llm.get('model')})",
        )
        counters = status.get("counters", {})
        noisy = {k: v for k, v in counters.items() if "failure" in k and v}
        step("no failure counters at boot", not noisy, json.dumps(noisy) if noisy else "")

        if args.chat:
            data = http_post_json(
                f"{origin}/api/chat",
                {
                    "message": "Quick systems check — say one short line back.",
                    "user_id": "smoke-test",
                    "audio_enabled": False,
                },
            )
            reply = (data or {}).get("reply", "")
            step("chat turn round-trips", bool(reply.strip()), reply[:80])
    except TimeoutError as exc:
        step("boot sequence", False, str(exc))
    finally:
        if not args.keep:
            kill_app()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()

    (out_dir / "smoke_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n{'SMOKE PASSED' if passed_all else 'SMOKE FAILED'} — report in {out_dir}")
    return 0 if passed_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
