"""Runtime health counters: increments at the wired points + /api/status exposure."""

from pathlib import Path

import pytest

from amadeus import telemetry
from amadeus.config import Config


@pytest.fixture(autouse=True)
def _clean_counters():
    telemetry.reset()
    yield
    telemetry.reset()


def test_incr_gauge_snapshot_reset():
    telemetry.incr("a")
    telemetry.incr("a", 2)
    telemetry.set_gauge("last_b", 7)
    snap = telemetry.snapshot()
    assert snap == {"a": 3, "last_b": 7}
    # Snapshot is a copy — mutating it doesn't touch the live values.
    snap["a"] = 99
    assert telemetry.snapshot()["a"] == 3
    telemetry.reset()
    assert telemetry.snapshot() == {}


def test_duration_percentiles_computed_on_snapshot():
    """p50/p95 (nearest-rank over the sample window) appear in snapshots."""
    for ms in range(1, 101):  # 1..100, window keeps the last 64: 37..100
        telemetry.record_duration("turn.first_audio_ms", float(ms))
    entry = telemetry.duration_snapshot()["turn.first_audio_ms"]
    # Aggregates still cover ALL samples, not just the window.
    assert entry["count"] == 100
    assert entry["total_ms"] == pytest.approx(5050.0)
    assert entry["max_ms"] == pytest.approx(100.0)
    assert entry["last_ms"] == pytest.approx(100.0)
    # Window is 37..100 (64 samples): nearest-rank p50 = 68, p95 = 97.
    assert entry["p50_ms"] == pytest.approx(68.0)
    assert entry["p95_ms"] == pytest.approx(97.0)


def test_duration_percentiles_single_sample():
    telemetry.record_duration("stt.request_ms", 42.0)
    entry = telemetry.duration_snapshot()["stt.request_ms"]
    assert entry["p50_ms"] == pytest.approx(42.0)
    assert entry["p95_ms"] == pytest.approx(42.0)


def test_duration_window_is_bounded():
    """Old samples fall out of the percentile window; max_ms is all-time."""
    telemetry.record_duration("k", 10_000.0)  # ancient disaster
    for _ in range(64):
        telemetry.record_duration("k", 5.0)
    entry = telemetry.duration_snapshot()["k"]
    # The 10s outlier was evicted from the 64-sample window...
    assert entry["p50_ms"] == pytest.approx(5.0)
    assert entry["p95_ms"] == pytest.approx(5.0)
    # ...but the all-time aggregates still remember it.
    assert entry["max_ms"] == pytest.approx(10_000.0)
    assert entry["count"] == 65
    # Internal window never grows past its bound.
    assert len(telemetry._SAMPLES["k"]) == 64


def test_duration_snapshot_empty_and_reset():
    assert telemetry.duration_snapshot() == {}
    telemetry.record_duration("k", 1.0)
    assert "p50_ms" in telemetry.duration_snapshot()["k"]
    telemetry.reset()
    assert telemetry.duration_snapshot() == {}
    assert telemetry._SAMPLES == {}


def test_turn_context_records_recall_gauge(tmp_path: Path):
    from amadeus.memory.context import assemble_turn_context
    from amadeus.memory.manager import MemoryManager

    config = Config()
    config.memory.persist_path = None
    config.memory.profile_path = None
    config.memory.enable_long_term = False  # no embedder: recall always empty
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    manager = MemoryManager(config)

    assemble_turn_context(manager, "system", "hello there")
    snap = telemetry.snapshot()
    assert snap["memory.turns"] == 1
    assert snap["memory.last_recall_items"] == 0
    assert snap["memory.recall_empty_turns"] == 1


def test_summary_error_counts(tmp_path: Path):
    """A summary job whose LLM call failed inside the job increments the counter."""
    from amadeus.memory.manager import MemoryManager

    class ExplodingLLM:
        name = "boom"

        def generate(self, messages, **kwargs):
            raise RuntimeError("provider down")

    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None
    config.memory.profile_path = None
    config.memory.enable_long_term = False
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    config.memory.summary_trigger_messages = 2
    manager = MemoryManager(config, summary_provider=ExplodingLLM())
    manager.add_message("user", "one")
    manager.add_message("assistant", "two")
    manager.flush()

    assert telemetry.snapshot().get("memory.summary_failures", 0) >= 1


def test_tagged_repair_counts(monkeypatch):
    from amadeus.web.streaming.parser import ensure_tagged_reply

    class TaggingLLM:
        def generate(self, messages, **kwargs):
            return "[happy] Fixed. || 直った。"

    out = ensure_tagged_reply(TaggingLLM(), "system prompt", "plain untagged reply")
    assert "Fixed." in out
    snap = telemetry.snapshot()
    assert snap["web.tagged_repairs"] == 1
    assert "web.tagged_repair_failures" not in snap

    class BrokenLLM:
        def generate(self, messages, **kwargs):
            raise RuntimeError("nope")

    ensure_tagged_reply(BrokenLLM(), "system prompt", "plain untagged reply")
    snap = telemetry.snapshot()
    assert snap["web.tagged_repairs"] == 2
    assert snap["web.tagged_repair_failures"] == 1


async def test_status_exposes_counters(tmp_path: Path):
    pytest.importorskip("fastapi")
    import httpx

    from amadeus.web.app import create_app

    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    telemetry.incr("memory.summary_failures", 3)

    app = create_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        response = await client.get("/api/status")
    assert response.status_code == 200
    counters = response.json()["counters"]
    assert counters["memory.summary_failures"] == 3
