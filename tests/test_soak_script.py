"""Tests for scripts/soak_test.py pure helpers. No server, no network."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "soak_test", REPO_ROOT / "scripts" / "soak_test.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


soak = _load_script_module()


def test_percentile_empty_and_single():
    assert soak.percentile([], 90) is None
    assert soak.percentile([42.0], 50) == 42.0
    assert soak.percentile([42.0], 99) == 42.0


def test_percentile_nearest_rank():
    values = [float(v) for v in range(1, 101)]  # 1..100
    assert soak.percentile(values, 50) == 50.0
    assert soak.percentile(values, 90) == 90.0
    assert soak.percentile(values, 99) == 99.0


def test_quartile_split_order_preserved():
    values = [float(v) for v in range(40)]
    quartiles = soak.quartile_split(values)
    assert quartiles["first"] == values[:10]
    assert quartiles["last"] == values[-10:]


def test_quartile_split_tiny_series():
    quartiles = soak.quartile_split([1.0, 2.0, 3.0])
    assert quartiles["first"] == [1.0]
    assert quartiles["last"] == [3.0]


def test_drift_verdict_flags_growth():
    verdict = soak.drift_verdict("turn total", 1000.0, 1420.0)
    assert verdict.startswith("DRIFT:")
    assert "+42%" in verdict


def test_drift_verdict_no_drift():
    verdict = soak.drift_verdict("turn total", 1000.0, 1050.0)
    assert verdict.startswith("ok:")
    assert "no significant drift" in verdict


def test_drift_verdict_undecidable():
    assert soak.drift_verdict("turn total", None, 1200.0) is None
    assert soak.drift_verdict("turn total", 0, 1200.0) is None
    assert soak.drift_verdict("turn total", 1000.0, None) is None


def test_growth_verdict():
    assert soak.growth_verdict("server RSS", 500.0, 900.0, 300.0).startswith("GROWTH:")
    assert soak.growth_verdict("server RSS", 500.0, 550.0, 300.0).startswith("ok:")
    assert soak.growth_verdict("server RSS", None, 550.0, 300.0) is None


def test_latency_averages_from_counters():
    counters = {
        "latency.ttft_ms_sum": 3000,
        "latency.ttft_count": 3,
        "latency.turn_total_ms_sum": 9000,
        "latency.turn_total_count": 3,
        "latency.orphan_ms_sum": 500,  # no matching count: skipped
        "tts.qwen3_slow_fallback": 1,
    }
    averages = soak.latency_averages(counters)
    assert averages == {"ttft": 1000.0, "turn_total": 3000.0}


def _turn(index, total_ms, ok=True, transport="ws"):
    record = {
        "turn": index,
        "ok": ok,
        "transport": transport,
        "timings": {"total_ms": total_ms} if transport == "ws" else None,
    }
    if transport == "post":
        record["client_ms"] = total_ms
    if not ok:
        record["error"] = "boom"
        record["timings"] = None
    return record


def test_build_summary_healthy_run():
    turns = [_turn(i, 1000 + (i % 5)) for i in range(40)]
    snapshots = [
        {"turn": 0, "rss_mb": 500.0, "vram_mb": 4000.0, "counters": {}},
        {"turn": 40, "rss_mb": 520.0, "vram_mb": 4010.0, "counters": {}},
    ]
    summary = soak.build_summary(turns, snapshots)
    assert summary["healthy"] is True
    assert summary["turns_ok"] == 40
    assert summary["errors"] == 0
    assert summary["latency"]["total_ms"]["overall"]["n"] == 40
    assert summary["rss_mb"] == {"first": 500.0, "last": 520.0}


def test_build_summary_flags_latency_drift():
    # First quartile ~1000ms, last quartile ~2000ms: past DRIFT_WARN_PCT.
    turns = [_turn(i, 1000.0 if i < 30 else 2000.0) for i in range(40)]
    summary = soak.build_summary(turns, [])
    assert summary["healthy"] is False
    assert any(v.startswith("DRIFT:") for v in summary["verdicts"])


def test_build_summary_flags_rss_growth_and_errors():
    turns = [_turn(i, 1000.0, ok=(i % 2 == 0)) for i in range(20)]
    snapshots = [
        {"turn": 0, "rss_mb": 500.0, "counters": {}},
        {"turn": 20, "rss_mb": 900.0, "counters": {"tts.qwen3_slow_fallback": 2}},
    ]
    summary = soak.build_summary(turns, snapshots)
    assert summary["healthy"] is False
    verdicts = summary["verdicts"]
    assert any(v.startswith("GROWTH: server RSS") for v in verdicts)
    assert any(v.startswith("ERRORS:") for v in verdicts)
    assert any("qwen3_slow_fallback" in v for v in verdicts)


def test_build_summary_post_turns_use_client_ms():
    turns = [_turn(i, 800.0, transport="post") for i in range(10)]
    summary = soak.build_summary(turns, [])
    assert summary["latency"]["total_ms"]["overall"]["p50"] == 800.0


def test_ws_url_for():
    assert (
        soak.ws_url_for("http://127.0.0.1:8000", None)
        == "ws://127.0.0.1:8000/api/chat/stream"
    )
    assert (
        soak.ws_url_for("https://example.com", "s3cret")
        == "wss://example.com/api/chat/stream?token=s3cret"
    )
