#!/usr/bin/env python3
"""Long-session soak harness for Amadeus / Kurisu.

Drives hundreds of realistic chat turns against a LIVE running server and
reports degradation over time: latency drift, RSS growth, VRAM creep, and
error accumulation. This is the safety net for the "CUDA OOM after an
evening" bug class that the unit suite (single-turn, mocked providers) can
never see.

This is NOT a pytest file. It needs a running server:

    uvicorn amadeus.web.app:app --port 8000
    python scripts/soak_test.py --turns 200
    python scripts/soak_test.py --url http://127.0.0.1:8091 --turns 50 --post

Each turn appends one JSONL record to --out (default soak_out/): wall-clock
timing, the server's own `timings` breakdown from the WS done frame
(ttft_ms / first_sentence_ms / ttfa_ms / total_ms), reply length, frame
counts, and error info. Every --status-every turns it snapshots
GET /api/status counters, server RSS, and GPU VRAM. A failed turn is logged
and the run continues; surviving the marathon is the point.

Exit code is non-zero when any degradation verdict fires (latency drift,
RSS/VRAM growth past the thresholds below) or more than 10% of turns failed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx

try:  # Ships with uvicorn[standard]; the soak client may run elsewhere.
    from websockets.sync.client import connect as ws_connect

    HAVE_WEBSOCKETS = True
except ImportError:  # pragma: no cover - environment-dependent
    ws_connect = None
    HAVE_WEBSOCKETS = False

try:
    import psutil

    HAVE_PSUTIL = True
except ImportError:  # pragma: no cover - environment-dependent
    psutil = None
    HAVE_PSUTIL = False

# --------------------------------------------------------------------------
# Degradation thresholds (the verdict constants)
# --------------------------------------------------------------------------

# p90 latency of the last quartile of turns vs the first quartile: growth past
# this percentage is called out as drift.
DRIFT_WARN_PCT = 25.0
# Server RSS growth first snapshot -> last snapshot, in MB.
RSS_WARN_MB = 300.0
# GPU VRAM growth first snapshot -> last snapshot, in MB.
VRAM_WARN_MB = 512.0
# Fraction of failed turns above which the run itself is judged unhealthy.
ERROR_RATE_WARN = 0.10
# Minimum ok-turn count before quartile drift is worth computing at all.
MIN_TURNS_FOR_DRIFT = 8

# Built-in rotating prompt list: casual, technical, and emotional prompts so
# summarization, recall, and the emotion path all get exercised. Includes
# repeat greetings (the persona has an escalation rule for those) and a few
# long-reply prompts to stress TTS/summary volume.
DEFAULT_PROMPTS = [
    "Hey Kurisu.",
    "What are you thinking about right now?",
    "Can you explain how a hash table handles collisions?",
    "I had a rough day at work, honestly.",
    "Tell me something interesting about neuroscience.",
    "Hey.",
    "What's the difference between a process and a thread?",
    "Do you ever get bored between our conversations?",
    "Walk me through how you'd design a caching layer for a chat app. Take your time and be thorough.",
    "I finally fixed that bug I was fighting all week!",
    "What do you remember about what I told you earlier?",
    "Explain quantum entanglement like I'm a first-year student.",
    "Hey Kurisu.",
    "Honestly I've been feeling kind of anxious lately.",
    "What's your take on time travel paradoxes? Give me the long version.",
    "Recommend me something to read this weekend.",
    "How does garbage collection work in Python?",
    "You'd tell me if I was being annoying, right?",
    "Summarize what we've talked about so far.",
    "Good night thoughts? I'm about to log off soon but not yet.",
]


class TurnError(Exception):
    """One turn failed. Carries whether a POST retry is worth attempting."""

    def __init__(self, message: str, fallback_to_post: bool = True):
        super().__init__(message)
        self.fallback_to_post = fallback_to_post


# --------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_soak_script.py)
# --------------------------------------------------------------------------


def percentile(values: List[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile. Returns None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = max(1, min(len(ordered), round(pct / 100.0 * len(ordered))))
    return float(ordered[rank - 1])


def latency_percentiles(values: List[float]) -> Dict[str, Optional[float]]:
    return {
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "mean": round(statistics.fmean(values), 1) if values else None,
        "n": len(values),
    }


def quartile_split(values: List[float]) -> Dict[str, List[float]]:
    """First and last quartile of a chronologically ordered series."""
    quarter = max(1, len(values) // 4)
    return {"first": values[:quarter], "last": values[-quarter:]}


def drift_verdict(
    label: str,
    first_p90: Optional[float],
    last_p90: Optional[float],
    threshold_pct: float = DRIFT_WARN_PCT,
) -> Optional[str]:
    """Human verdict line for one latency metric, or None when undecidable.

    Returns e.g. "latency p90 drifted +42% (1200ms -> 1704ms)" past the
    threshold, or an explicit "no significant drift" line otherwise.
    """
    if not first_p90 or last_p90 is None:
        return None
    pct = (last_p90 - first_p90) / first_p90 * 100.0
    detail = f"({first_p90:.0f}ms -> {last_p90:.0f}ms)"
    if pct >= threshold_pct:
        return f"DRIFT: {label} p90 drifted {pct:+.0f}% {detail}"
    return f"ok: {label} p90 {pct:+.0f}% {detail}, no significant drift"


def growth_verdict(
    label: str,
    first_mb: Optional[float],
    last_mb: Optional[float],
    threshold_mb: float,
) -> Optional[str]:
    """Human verdict line for a first->last memory delta, or None if unknown."""
    if first_mb is None or last_mb is None:
        return None
    delta = last_mb - first_mb
    detail = f"({first_mb:.0f}MB -> {last_mb:.0f}MB)"
    if delta >= threshold_mb:
        return f"GROWTH: {label} grew {delta:+.0f}MB over the run {detail}"
    return f"ok: {label} {delta:+.0f}MB {detail}"


def latency_averages(counters: Dict[str, Any]) -> Dict[str, float]:
    """Turn the latency.*_ms_sum / latency.*_count pairs into averages."""
    averages: Dict[str, float] = {}
    for key, value in counters.items():
        match = re.fullmatch(r"latency\.(\w+)_ms_sum", key)
        if not match:
            continue
        count = counters.get(f"latency.{match.group(1)}_count") or 0
        if count:
            averages[match.group(1)] = round(value / count, 1)
    return averages


def build_summary(
    turns: List[Dict[str, Any]],
    snapshots: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-turn records + periodic snapshots into the final report."""
    ok_turns = [t for t in turns if t.get("ok")]
    errors = [t for t in turns if not t.get("ok")]

    def metric_series(key: str) -> List[float]:
        series = []
        for t in ok_turns:
            timings = t.get("timings") or {}
            value = timings.get(key)
            if value is None and key == "total_ms":
                value = t.get("client_ms")  # POST turns have no server timings
            if value is not None:
                series.append(float(value))
        return series

    summary: Dict[str, Any] = {
        "turns_total": len(turns),
        "turns_ok": len(ok_turns),
        "errors": len(errors),
        "error_turns": [{"turn": t["turn"], "error": t.get("error")} for t in errors][
            :20
        ],
        "latency": {},
        "verdicts": [],
    }

    verdicts: List[str] = summary["verdicts"]
    for label, key in (("turn total", "total_ms"), ("first audio", "ttfa_ms")):
        series = metric_series(key)
        if not series:
            continue
        block: Dict[str, Any] = {"overall": latency_percentiles(series)}
        if len(series) >= MIN_TURNS_FOR_DRIFT:
            quartiles = quartile_split(series)
            block["first_quartile"] = latency_percentiles(quartiles["first"])
            block["last_quartile"] = latency_percentiles(quartiles["last"])
            verdict = drift_verdict(
                label,
                block["first_quartile"]["p90"],
                block["last_quartile"]["p90"],
            )
            if verdict:
                verdicts.append(verdict)
        summary["latency"][key] = block

    def snapshot_series(key: str) -> List[float]:
        return [s[key] for s in snapshots if s.get(key) is not None]

    rss = snapshot_series("rss_mb")
    summary["rss_mb"] = {"first": rss[0], "last": rss[-1]} if rss else None
    verdict = growth_verdict(
        "server RSS", rss[0] if rss else None, rss[-1] if rss else None, RSS_WARN_MB
    )
    if verdict:
        verdicts.append(verdict)

    vram = snapshot_series("vram_mb")
    summary["vram_mb"] = {"first": vram[0], "last": vram[-1]} if vram else None
    verdict = growth_verdict(
        "GPU VRAM", vram[0] if vram else None, vram[-1] if vram else None, VRAM_WARN_MB
    )
    if verdict:
        verdicts.append(verdict)

    if snapshots:
        last_counters = snapshots[-1].get("counters") or {}
        summary["server_latency_averages"] = latency_averages(last_counters)
        slow = last_counters.get("tts.qwen3_slow_fallback")
        if slow:
            verdicts.append(f"note: tts.qwen3_slow_fallback = {slow}")

    if turns:
        rate = len(errors) / len(turns)
        if rate > ERROR_RATE_WARN:
            verdicts.append(
                f"ERRORS: {len(errors)}/{len(turns)} turns failed ({rate:.0%})"
            )
        else:
            verdicts.append(f"ok: {len(errors)}/{len(turns)} turns failed")

    summary["healthy"] = not any(
        v.startswith(("DRIFT:", "GROWTH:", "ERRORS:")) for v in verdicts
    )
    return summary


# --------------------------------------------------------------------------
# Transport: one turn over WS (with POST fallback) or plain POST
# --------------------------------------------------------------------------


def ws_url_for(base_url: str, token: Optional[str]) -> str:
    parts = urlsplit(base_url)
    scheme = "wss" if parts.scheme == "https" else "ws"
    query = f"token={token}" if token else ""
    return urlunsplit((scheme, parts.netloc, "/api/chat/stream", query, ""))


def run_ws_turn(
    base_url: str,
    prompt: str,
    session_id: Optional[str],
    user_id: str,
    token: Optional[str],
    turn_timeout: float,
) -> Dict[str, Any]:
    """One streamed turn. Returns the per-turn record fields; raises TurnError."""
    record: Dict[str, Any] = {
        "transport": "ws",
        "tokens": 0,
        "sentences": 0,
        "audio_frames": 0,
    }
    started = time.monotonic()
    deadline = started + turn_timeout
    try:
        with ws_connect(
            ws_url_for(base_url, token), open_timeout=15, max_size=None
        ) as ws:
            payload = {"type": "chat", "message": prompt, "user_id": user_id}
            if session_id:
                payload["session_id"] = session_id
            ws.send(json.dumps(payload))
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TurnError(
                        f"turn timed out after {turn_timeout:.0f}s",
                        fallback_to_post=False,
                    )
                frame = json.loads(ws.recv(timeout=remaining))
                ftype = frame.get("type")
                if ftype == "session":
                    record["session_id"] = frame.get("session_id")
                elif ftype == "token":
                    record["tokens"] += 1
                elif ftype == "sentence":
                    record["sentences"] += 1
                elif ftype == "audio":
                    record["audio_frames"] += 1
                elif ftype == "error":
                    raise TurnError(
                        str(frame.get("message")),
                        fallback_to_post=bool(frame.get("fallback_to_post"))
                        # After the session frame flowed, a retry hits the
                        # same provider error (mirrors the frontend rule).
                        and "session_id" not in record,
                    )
                elif ftype == "done":
                    response = frame.get("response") or {}
                    record["client_ms"] = int(1000 * (time.monotonic() - started))
                    record["timings"] = frame.get("timings")
                    record["reply_chars"] = len(response.get("reply") or "")
                    record["tts_failed"] = frame.get("tts_failed", 0)
                    record["cancelled"] = bool(frame.get("cancelled"))
                    if response.get("session_id"):
                        record["session_id"] = response["session_id"]
                    return record
                # tool_progress / thought frames are counted nowhere on purpose
    except TurnError:
        raise
    except Exception as exc:
        # Connect refused, closed socket, malformed frame: the frontend falls
        # back to POST on any WS failure, so the soak does too.
        raise TurnError(f"{type(exc).__name__}: {exc}") from exc


def run_post_turn(
    client: httpx.Client,
    base_url: str,
    prompt: str,
    session_id: Optional[str],
    user_id: str,
    turn_timeout: float,
) -> Dict[str, Any]:
    started = time.monotonic()
    payload: Dict[str, Any] = {"message": prompt, "user_id": user_id}
    if session_id:
        payload["session_id"] = session_id
    try:
        response = client.post(
            f"{base_url}/api/chat", json=payload, timeout=turn_timeout
        )
        response.raise_for_status()
        body = response.json()
    except Exception as exc:
        raise TurnError(f"{type(exc).__name__}: {exc}", fallback_to_post=False) from exc
    return {
        "transport": "post",
        "client_ms": int(1000 * (time.monotonic() - started)),
        "timings": None,  # server timings ride the WS done frame only
        "reply_chars": len(body.get("reply") or ""),
        "sentences": len(body.get("sentences") or []),
        "audio_frames": 1 if body.get("audio") else 0,
        "session_id": body.get("session_id"),
    }


# --------------------------------------------------------------------------
# Periodic snapshots: /api/status counters, RSS, VRAM
# --------------------------------------------------------------------------


def find_server_pid(port: int) -> Optional[int]:
    """Best-effort uvicorn/amadeus process discovery. Needs psutil."""
    if not HAVE_PSUTIL:
        return None
    own = {psutil.Process().pid}
    candidates = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.pid in own:
                continue
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if "soak_test" in cmdline:
                continue
            if ("uvicorn" in cmdline or "amadeus" in cmdline) and (
                str(port) in cmdline or "amadeus.web" in cmdline
            ):
                candidates.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return candidates[0] if candidates else None


def rss_mb(pid: Optional[int]) -> Optional[float]:
    """Resident set size of the server process, in MB. None when unknowable."""
    if pid is None:
        return None
    if HAVE_PSUTIL:
        try:
            return round(psutil.Process(pid).memory_info().rss / (1024 * 1024), 1)
        except Exception:
            return None
    try:  # /proc fallback keeps Linux working without psutil
        status = Path(f"/proc/{pid}/status").read_text()
        match = re.search(r"VmRSS:\s+(\d+)\s+kB", status)
        return round(int(match.group(1)) / 1024, 1) if match else None
    except OSError:
        return None


def vram_mb() -> Optional[float]:
    """Total GPU memory used across GPUs via nvidia-smi. Silent skip without it."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        values = [float(line.strip()) for line in out.splitlines() if line.strip()]
        return sum(values) if values else None
    except Exception:
        return None


def take_snapshot(
    client: httpx.Client, base_url: str, turn: int, pid: Optional[int]
) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "turn": turn,
        "ts": datetime.now(timezone.utc).isoformat(),
        "rss_mb": rss_mb(pid),
        "vram_mb": vram_mb(),
        "counters": None,
        "latency_avg": None,
    }
    try:
        status = client.get(f"{base_url}/api/status", timeout=15).json()
        counters = status.get("counters") or {}
        snapshot["counters"] = counters
        snapshot["latency_avg"] = latency_averages(counters)
    except Exception as exc:
        snapshot["status_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------


def load_prompts(prompt_file: Optional[str]) -> List[str]:
    if not prompt_file:
        return DEFAULT_PROMPTS
    lines = Path(prompt_file).read_text(encoding="utf-8").splitlines()
    prompts = [line.strip() for line in lines if line.strip()]
    if not prompts:
        raise SystemExit(f"Prompt file {prompt_file} has no prompts")
    return prompts


def run_soak(args: argparse.Namespace) -> Dict[str, Any]:
    base_url = args.url.rstrip("/")
    prompts = load_prompts(args.prompt_file)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    turns_path = out_dir / "turns.jsonl"
    snapshots_path = out_dir / "snapshots.jsonl"

    use_ws = args.transport == "ws"
    if use_ws and not HAVE_WEBSOCKETS:
        print("websockets not installed; running POST-only", flush=True)
        use_ws = False

    port = urlsplit(base_url).port or 80
    pid = args.server_pid or find_server_pid(port)
    if pid:
        print(f"Tracking server pid {pid} for RSS", flush=True)
    elif not HAVE_PSUTIL and not args.server_pid:
        print("psutil not installed and no --server-pid; skipping RSS", flush=True)

    client = httpx.Client()
    session_id: Optional[str] = None
    turns: List[Dict[str, Any]] = []
    snapshots: List[Dict[str, Any]] = []

    def snapshot(turn_index: int) -> None:
        snap = take_snapshot(client, base_url, turn_index, pid)
        snapshots.append(snap)
        with snapshots_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(snap) + "\n")
        print(
            f"  [snapshot @{turn_index}] rss={snap['rss_mb']}MB "
            f"vram={snap['vram_mb']}MB latency_avg={snap['latency_avg']}",
            flush=True,
        )

    snapshot(0)
    for index in range(args.turns):
        prompt = prompts[index % len(prompts)]
        record: Dict[str, Any] = {
            "turn": index,
            "ts": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "ok": True,
        }
        started = time.monotonic()
        try:
            if use_ws:
                try:
                    result = run_ws_turn(
                        base_url,
                        prompt,
                        session_id,
                        args.user_id,
                        args.token,
                        args.turn_timeout,
                    )
                except TurnError as exc:
                    if not exc.fallback_to_post:
                        raise
                    record["ws_error"] = str(exc)
                    result = run_post_turn(
                        client,
                        base_url,
                        prompt,
                        session_id,
                        args.user_id,
                        args.turn_timeout,
                    )
            else:
                result = run_post_turn(
                    client,
                    base_url,
                    prompt,
                    session_id,
                    args.user_id,
                    args.turn_timeout,
                )
            record.update(result)
            session_id = result.get("session_id") or session_id
        except TurnError as exc:
            record["ok"] = False
            record["error"] = str(exc)
            record["client_ms"] = int(1000 * (time.monotonic() - started))
            print(f"turn {index} FAILED: {exc}", flush=True)

        turns.append(record)
        with turns_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

        if record["ok"]:
            timings = record.get("timings") or {}
            total = timings.get("total_ms") or record.get("client_ms")
            print(
                f"turn {index}: {record['transport']} total={total}ms "
                f"reply={record.get('reply_chars')}ch "
                f"sentences={record.get('sentences')} "
                f"audio={record.get('audio_frames')}",
                flush=True,
            )
        if (index + 1) % args.status_every == 0 and index + 1 < args.turns:
            snapshot(index + 1)
        if args.delay > 0 and index + 1 < args.turns:
            time.sleep(args.delay)

    snapshot(args.turns)
    client.close()

    summary = build_summary(turns, snapshots)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n=== Soak summary ===")
    print(f"turns: {summary['turns_ok']}/{summary['turns_total']} ok")
    for key, block in summary.get("latency", {}).items():
        overall = block["overall"]
        line = (
            f"{key}: p50={overall['p50']:.0f}ms p90={overall['p90']:.0f}ms "
            f"p99={overall['p99']:.0f}ms (n={overall['n']})"
        )
        if "first_quartile" in block:
            line += (
                f" | first-quartile p90={block['first_quartile']['p90']:.0f}ms"
                f" last-quartile p90={block['last_quartile']['p90']:.0f}ms"
            )
        print(line)
    if summary.get("server_latency_averages"):
        print(f"server latency averages: {summary['server_latency_averages']}")
    if summary.get("rss_mb"):
        print(f"rss: {summary['rss_mb']['first']}MB -> {summary['rss_mb']['last']}MB")
    if summary.get("vram_mb"):
        print(
            f"vram: {summary['vram_mb']['first']}MB -> {summary['vram_mb']['last']}MB"
        )
    print("verdicts:")
    for verdict in summary["verdicts"]:
        print(f"  {verdict}")
    print("healthy" if summary["healthy"] else "NOT healthy")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Soak-test a running Amadeus server over many turns"
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running server (default %(default)s)",
    )
    parser.add_argument(
        "--turns", type=int, default=200, help="Turns to run (default %(default)s)"
    )
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument(
        "--ws",
        dest="transport",
        action="store_const",
        const="ws",
        help="Stream over WebSocket with automatic POST fallback (default)",
    )
    transport.add_argument(
        "--post",
        dest="transport",
        action="store_const",
        const="post",
        help="Use POST /api/chat only (no server-side timings)",
    )
    parser.set_defaults(transport="ws")
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to sleep between turns (default %(default)s)",
    )
    parser.add_argument(
        "--out",
        default="soak_out",
        help="Artifact directory for turns.jsonl / snapshots.jsonl / summary.json",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="File with one prompt per line (default: built-in rotating list)",
    )
    parser.add_argument(
        "--status-every",
        type=int,
        default=20,
        help="Snapshot /api/status + RSS + VRAM every N turns (default %(default)s)",
    )
    parser.add_argument(
        "--server-pid",
        type=int,
        default=None,
        help="Server PID for RSS tracking (default: psutil name match)",
    )
    parser.add_argument(
        "--turn-timeout",
        type=float,
        default=120.0,
        help="Per-turn timeout in seconds (default %(default)s)",
    )
    parser.add_argument(
        "--user-id", default="soak", help="user_id sent with every turn"
    )
    parser.add_argument(
        "--token", default=None, help="API token when web.api_token is set"
    )
    args = parser.parse_args()

    summary = run_soak(args)
    print_summary(summary)
    return 0 if summary["healthy"] else 1


if __name__ == "__main__":
    sys.exit(main())
