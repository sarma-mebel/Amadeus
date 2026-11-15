"""Insight consolidation: store semantics, reply parsing, the SKIP-vs-
failure retry contract, scheduler due-predicate/idempotence, prompt
injection through prepare_turn, and the read-only /api/insights endpoint."""

from __future__ import annotations

import datetime as dt

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from amadeus.config import Config
from amadeus.insights import (
    Insight,
    InsightSet,
    InsightStore,
    collect_insight_facts,
    format_insights_block,
    generate_insights,
    parse_insights_reply,
)
from amadeus.memory.profile import UserProfile
from amadeus.persona import PersonaSettings, build_system_prompt
from amadeus.providers.llm.base import ChatMessage
from amadeus.web.app import create_app
from amadeus.web.scheduler import BackgroundScheduler


class RecordingProvider:
    name = "recording"

    def __init__(
        self,
        reply: str = "style: Keep answers short and concrete.\nrhythm: They usually show up late at night; don't comment on it.",
    ) -> None:
        self.reply = reply
        self.calls: list[list[ChatMessage]] = []
        self.fail = False

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        if self.fail:
            raise RuntimeError("provider down")
        return self.reply


def _insight_set(date: dt.date, texts=("Keep answers short.",)) -> InsightSet:
    return InsightSet(
        date=date.isoformat(),
        insights=[Insight(text=text, kind="style") for text in texts],
        generated_at=f"{date.isoformat()}T05:00:00",
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_store_roundtrip_and_per_user_keying(tmp_path):
    store = InsightStore(str(tmp_path / "insights.json"))
    today = dt.date(2026, 7, 30)
    assert store.put(None, _insight_set(today, ("default guidance.",))) is True
    assert store.put("alice", _insight_set(today, ("alice guidance.",))) is True

    reloaded = InsightStore(str(tmp_path / "insights.json"))
    assert reloaded.get(None).insights[0].text == "default guidance."
    assert reloaded.get("alice").insights[0].text == "alice guidance."


def test_store_put_refuses_same_date_but_replaces_next_day(tmp_path):
    store = InsightStore(str(tmp_path / "insights.json"))
    today = dt.date(2026, 7, 30)
    assert store.put(None, _insight_set(today)) is True
    assert store.put(None, _insight_set(today, ("duplicate.",))) is False
    assert store.has_for_date(None, today) is True
    tomorrow = today + dt.timedelta(days=1)
    assert store.put(None, _insight_set(tomorrow, ("newer.",))) is True
    assert store.get(None).insights[0].text == "newer."


def test_store_empty_set_finalizes_the_day(tmp_path):
    """A stored empty set (a model SKIP) counts as done for the date."""
    store = InsightStore(str(tmp_path / "insights.json"))
    today = dt.date(2026, 7, 30)
    assert store.put(None, InsightSet(date=today.isoformat())) is True
    assert store.has_for_date(None, today) is True
    assert format_insights_block(store.get(None)) is None


def test_store_corrupt_file_starts_clean(tmp_path):
    path = tmp_path / "insights.json"
    path.write_text("{ not json", encoding="utf-8")
    store = InsightStore(str(path))
    assert store.get(None) is None
    assert store.put(None, _insight_set(dt.date(2026, 7, 30))) is True


# ---------------------------------------------------------------------------
# Parsing + generation contract
# ---------------------------------------------------------------------------


def test_parse_valid_lines_with_kinds_and_bullets():
    parsed = parse_insights_reply(
        "- style: Keep answers short and concrete.\n"
        "boundary: Don't push about the thesis deadline.\n"
        "* weird: Unknown kinds fold into other.\n"
        "no colon here but still long enough to keep\n"
    )
    assert parsed is not None
    assert [entry.kind for entry in parsed] == ["style", "boundary", "other", "other"]
    assert parsed[0].text == "Keep answers short and concrete."


def test_parse_skip_and_garbage_return_none():
    assert parse_insights_reply("SKIP") is None
    assert parse_insights_reply("skip — not enough yet") is None
    assert parse_insights_reply("") is None
    assert parse_insights_reply(None) is None
    assert parse_insights_reply("ok\n-\nx: y") is None  # only fragments


def test_parse_caps_at_max():
    reply = "\n".join(
        f"style: Guidance number {i} that is long enough." for i in range(12)
    )
    parsed = parse_insights_reply(reply)
    assert parsed is not None and len(parsed) == 7


def _facts(session_count=5):
    profile = UserProfile(
        facts={"favorite_topic": "time travel theory"},
        session_count=session_count,
        open_threads=["finish the draft"],
        last_episode="talked about the draft",
    )
    profile.record_tone(-0.6, dt.date(2026, 7, 28))
    profile.record_tone(0.5, dt.date(2026, 7, 29))
    return collect_insight_facts(profile=profile)


def test_collect_insight_facts_renders_tones_as_labels():
    facts = _facts()
    joined = "\n".join(facts)
    assert "Conversations so far: 5." in joined
    assert "low, upbeat" in joined
    assert "favorite_topic" in joined
    assert collect_insight_facts(profile=None) == []


def test_generate_skip_returns_empty_list_and_failure_returns_none():
    provider = RecordingProvider(reply="SKIP")
    assert generate_insights(provider=provider, facts=_facts()) == []

    failing = RecordingProvider()
    failing.fail = True
    assert generate_insights(provider=failing, facts=_facts()) is None


def test_generate_young_profile_makes_no_call():
    provider = RecordingProvider()
    assert generate_insights(provider=provider, facts=_facts(session_count=1)) is None
    assert provider.calls == []


def test_generate_feeds_previous_set_forward():
    provider = RecordingProvider()
    previous = _insight_set(dt.date(2026, 7, 29), ("Keep answers short.",))
    result = generate_insights(provider=provider, facts=_facts(), previous=previous)
    assert result is not None and len(result) == 2
    prompt_text = provider.calls[0][1].content
    assert "Keep answers short." in prompt_text


# ---------------------------------------------------------------------------
# Dismissal (the transparency surface's write half)
# ---------------------------------------------------------------------------


def test_dismiss_removes_the_insight_and_records_it(tmp_path):
    store = InsightStore(str(tmp_path / "insights.json"))
    store.put(
        None, _insight_set(dt.date(2026, 8, 1), ("Keep it short.", "Tease less."))
    )

    assert store.dismiss(None, "Tease less.") is True

    current = store.get(None)
    assert [entry.text for entry in current.insights] == ["Keep it short."]
    assert current.dismissed == ["Tease less."]
    # Survives a reload — the whole point is that it outlives the process.
    assert InsightStore(str(tmp_path / "insights.json")).get(None).dismissed == [
        "Tease less."
    ]


def test_dismiss_is_truthful_about_misses(tmp_path):
    store = InsightStore(str(tmp_path / "insights.json"))
    store.put(None, _insight_set(dt.date(2026, 8, 1)))

    assert store.dismiss(None, "never said this") is False
    assert store.dismiss(None, "  ") is False
    assert store.dismiss("nobody", "Keep answers short.") is False


def test_dismissals_carry_forward_across_consolidations(tmp_path):
    """THE CONTRACT. Each nightly set REPLACES the last, so without an
    explicit merge the user's deletion would live exactly one day and
    tonight's job would restate it — making the delete button a lie."""
    store = InsightStore(str(tmp_path / "insights.json"))
    store.put(
        None, _insight_set(dt.date(2026, 8, 1), ("Keep it short.", "Tease less."))
    )
    store.dismiss(None, "Tease less.")

    store.put(None, _insight_set(dt.date(2026, 8, 2), ("Keep it short.",)))

    assert store.get(None).dismissed == ["Tease less."]


def test_generate_tells_the_model_not_to_restate_dismissals():
    provider = RecordingProvider()
    previous = _insight_set(dt.date(2026, 7, 29), ("Keep answers short.",))
    previous.dismissed = ["Tease less."]

    generate_insights(provider=provider, facts=_facts(), previous=previous)

    prompt_text = provider.calls[0][1].content
    assert "REMOVED" in prompt_text
    assert "Tease less." in prompt_text


# ---------------------------------------------------------------------------
# Prompt injection
# ---------------------------------------------------------------------------


def test_format_block_and_prompt_placement():
    block = format_insights_block(
        _insight_set(dt.date(2026, 7, 30), ("Keep it short.",))
    )
    assert block is not None
    assert "never recite" in block
    assert "- Keep it short." in block
    assert format_insights_block(None) is None
    # The user can read this exact list in the memory panel, so they may
    # quote one back. Denying a note they are looking at would be a lie;
    # reciting the rest is the behavior "never recite" exists to prevent.
    assert "never deny it, never recite the others" in block

    prompt = build_system_prompt(PersonaSettings(), insights_block=block)
    assert "Keep it short." in prompt
    # Static-region placement: the block must precede the per-turn clock
    # line (the dynamic tail), so the cached prefix still covers it.
    assert prompt.index("Keep it short.") < prompt.index("Now: ")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def _scheduler_config(tmp_path) -> Config:
    config = Config()
    config.insights.enabled = True
    config.insights.hour = 5
    config.insights.store_path = str(tmp_path / "insights.json")
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    return config


def _seed_profile(config, *, last_seen: dt.datetime, session_count: int = 5):
    from pathlib import Path

    profile = UserProfile(
        facts={"work": "neuroimaging research"},
        open_threads=["finish the draft"],
        last_episode="talked about the draft",
        last_seen=last_seen.isoformat(),
        session_count=session_count,
    )
    profile.save_atomic(Path(config.memory.profile_path))


def _make_scheduler(monkeypatch, tmp_path, now: dt.datetime, **profile_kwargs):
    config = _scheduler_config(tmp_path)
    _seed_profile(config, last_seen=now - dt.timedelta(days=1), **profile_kwargs)
    recorder = RecordingProvider()

    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: recorder)
    app = create_app(config)
    store = app.state.session_store
    store.llm = recorder
    monkeypatch.setattr(
        "amadeus.engine.build_summary_provider", lambda cfg, llm: recorder
    )
    scheduler = BackgroundScheduler(store, config, clock=lambda: now)
    return scheduler, store, recorder, config


@pytest.mark.asyncio
async def test_scheduler_before_hour_noop(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 30, 4, 30)
    scheduler, store, recorder, _ = _make_scheduler(monkeypatch, tmp_path, now)
    await scheduler._insights_job(now)
    assert store.insight_store.get(None) is None
    assert recorder.calls == []


@pytest.mark.asyncio
async def test_scheduler_consolidates_once_per_day(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 30, 5, 10)
    scheduler, store, recorder, config = _make_scheduler(monkeypatch, tmp_path, now)
    await scheduler._insights_job(now)
    current = store.insight_store.get(None)
    assert current is not None and current.date == "2026-07-30"
    assert [entry.kind for entry in current.insights] == ["style", "rhythm"]
    assert len(recorder.calls) == 1

    # Same-day second tick and a restart: no regeneration.
    await scheduler._insights_job(now + dt.timedelta(minutes=5))
    fresh = BackgroundScheduler(store, config, clock=lambda: now)
    await fresh._insights_job(now + dt.timedelta(hours=2))
    assert len(recorder.calls) == 1

    # Next day: a new consolidation replaces the set.
    await scheduler._insights_job(now + dt.timedelta(days=1))
    assert len(recorder.calls) == 2
    assert store.insight_store.get(None).date == "2026-07-31"


@pytest.mark.asyncio
async def test_scheduler_stores_empty_set_on_model_skip(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 30, 5, 10)
    scheduler, store, recorder, _ = _make_scheduler(monkeypatch, tmp_path, now)
    recorder.reply = "SKIP"
    await scheduler._insights_job(now)
    current = store.insight_store.get(None)
    assert current is not None and current.insights == []
    # The SKIP finalized the day — no re-ask on the next tick.
    await scheduler._insights_job(now + dt.timedelta(minutes=1))
    assert len(recorder.calls) == 1


@pytest.mark.asyncio
async def test_scheduler_retries_after_provider_failure(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 30, 5, 10)
    scheduler, store, recorder, _ = _make_scheduler(monkeypatch, tmp_path, now)
    recorder.fail = True
    await scheduler._tick()  # must not raise
    assert store.insight_store.get(None) is None
    recorder.fail = False
    await scheduler._insights_job(now)
    assert store.insight_store.get(None) is not None


@pytest.mark.asyncio
async def test_scheduler_young_profile_makes_no_call(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 30, 5, 10)
    scheduler, store, recorder, _ = _make_scheduler(
        monkeypatch, tmp_path, now, session_count=1
    )
    await scheduler._insights_job(now)
    assert recorder.calls == []
    assert store.insight_store.get(None) is None


@pytest.mark.asyncio
async def test_scheduler_disabled_noop(monkeypatch, tmp_path):
    now = dt.datetime(2026, 7, 30, 5, 10)
    scheduler, store, recorder, config = _make_scheduler(monkeypatch, tmp_path, now)
    config.insights.enabled = False
    await scheduler._insights_job(now)
    assert recorder.calls == []


# ---------------------------------------------------------------------------
# Engine injection + endpoint
# ---------------------------------------------------------------------------


def test_prepare_turn_injects_insights_block(monkeypatch, tmp_path):
    config = _scheduler_config(tmp_path)
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: RecordingProvider())
    app = create_app(config)
    store = app.state.session_store
    store.insight_store.put(
        None, _insight_set(dt.date.today(), ("Prefers blunt answers.",))
    )

    _, session = store.get(None, None)
    setup = session.prepare_turn("hey", PersonaSettings())
    assert "Prefers blunt answers." in setup.system_prompt


def test_insights_endpoint_read_only(monkeypatch, tmp_path):
    config = _scheduler_config(tmp_path)
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: RecordingProvider())
    app = create_app(config)
    store = app.state.session_store
    store.insight_store.put(
        None, _insight_set(dt.date(2026, 7, 30), ("Keep it short.",))
    )

    client = TestClient(app)
    payload = client.get("/api/insights").json()
    assert payload["insights"]["date"] == "2026-07-30"
    assert payload["insights"]["insights"] == [
        {"text": "Keep it short.", "kind": "style"}
    ]


def test_insights_dismiss_endpoint_removes_and_persists(monkeypatch, tmp_path):
    config = _scheduler_config(tmp_path)
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: RecordingProvider())
    app = create_app(config)
    store = app.state.session_store
    store.insight_store.put(
        None, _insight_set(dt.date(2026, 7, 30), ("Keep it short.", "Tease less."))
    )

    client = TestClient(app)
    response = client.post("/api/insights/dismiss", json={"text": "Tease less."})
    assert response.json() == {"dismissed": True}

    payload = client.get("/api/insights").json()
    assert payload["insights"]["insights"] == [
        {"text": "Keep it short.", "kind": "style"}
    ]
    assert payload["insights"]["dismissed"] == ["Tease less."]

    # Dismissing something already gone is a truthful false, not a 500.
    assert client.post(
        "/api/insights/dismiss", json={"text": "Tease less."}
    ).json() == {"dismissed": False}


def test_insights_dismiss_endpoint_disabled_returns_false(monkeypatch, tmp_path):
    config = _scheduler_config(tmp_path)
    config.insights.enabled = False
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: RecordingProvider())
    app = create_app(config)
    client = TestClient(app)
    assert client.post("/api/insights/dismiss", json={"text": "x"}).json() == {
        "dismissed": False
    }


def test_insights_endpoint_disabled_returns_null(monkeypatch, tmp_path):
    config = _scheduler_config(tmp_path)
    config.insights.enabled = False
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: RecordingProvider())
    app = create_app(config)
    client = TestClient(app)
    assert client.get("/api/insights").json() == {"insights": None}
