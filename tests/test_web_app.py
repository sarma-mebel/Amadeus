import pytest

pytest.importorskip("fastapi")

import httpx

from amadeus.config import Config
from amadeus.web.app import create_app


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.mark.asyncio
async def test_web_app_serves_index(monkeypatch, tmp_path):
    # Point the frontend bundle at an empty dir so the inline template is served
    # regardless of whether frontend/dist has been built locally.
    monkeypatch.setenv("KURISU_FRONTEND_DIR", str(tmp_path / "no-bundle"))
    app = create_app(_isolated_config(tmp_path))

    async with _client(app) as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert "<title>Kurisu</title>" in response.text


def _isolated_config(tmp_path) -> Config:
    """Config whose persistence lands in tmp_path, not the repo's live state."""
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    return config


def test_session_store_prewarms_melotts_onnx(monkeypatch, tmp_path):
    import amadeus.web.app as app_module

    class FakeTTS:
        def __init__(self):
            self.calls = []

        def synthesize(self, text, *, speaking_rate=1.0, pitch_shift=0.0):
            self.calls.append((text, speaking_rate, pitch_shift))
            return b"wav"

    fake_tts = FakeTTS()
    config = _isolated_config(tmp_path)
    config.audio.enable_voice = True
    config.providers.tts = "melotts_onnx"
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: fake_tts)

    app = create_app(config)

    assert app.state.session_store.tts is fake_tts
    assert fake_tts.calls == [("テスト", 1.0, 0.0)]


@pytest.mark.asyncio
async def test_web_app_chat_endpoint_roundtrip(tmp_path):
    app = create_app(_isolated_config(tmp_path))

    async with _client(app) as client:
        response = await client.post("/api/chat", json={"message": "Hello Kurisu"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"]
    # No API key in the isolated config -> DummyProvider's in-character
    # no-key reply (English side of the tagged bilingual format).
    assert "API key" in payload["reply"]
    assert payload["timings"]["total_ms"] >= 0
    assert payload["timings"]["llm_ms"] >= 0


@pytest.mark.asyncio
async def test_memory_endpoint_actions(tmp_path):
    config = _isolated_config(tmp_path)
    config.memory.enable_fact_extraction = True  # activates the profile
    config.memory.enable_long_term = False  # no embedder needed in tests
    app = create_app(config)

    async with _client(app) as client:
        overview = await client.post(
            "/api/memory", json={"session_id": "s1", "action": "overview"}
        )
        assert overview.status_code == 200
        body = overview.json()
        assert body["status"] == "ok"
        assert body["overview"]["profile"]["session_count"] >= 1

        remember = await client.post(
            "/api/memory",
            json={"session_id": "s1", "action": "remember", "text": "likes tea"},
        )
        assert remember.json()["status"] == "unavailable"  # long-term off

        store = app.state.session_store
        session = store.sessions["s1"]
        session.memory.profile.facts["favorite_drink"] = "black tea"

        forgotten = await client.post(
            "/api/memory",
            json={"session_id": "s1", "action": "forget", "text": "favorite_drink"},
        )
        assert forgotten.json() == {
            "status": "ok",
            "detail": "Forgotten: favorite_drink: black tea",
        }

        missing = await client.post(
            "/api/memory", json={"session_id": "s1", "action": "forget", "text": "zzz"}
        )
        assert missing.json()["status"] == "not_found"

        bad = await client.post(
            "/api/memory", json={"session_id": "s1", "action": "explode"}
        )
        assert bad.status_code == 400


@pytest.mark.asyncio
async def test_memory_endpoint_pin_edit_forget_id(tmp_path):
    from amadeus.memory.manager import EmbeddingProvider

    class DummyEmbedder(EmbeddingProvider):
        def embed_text(self, text):
            return [float(len(text))]

        def similarity(self, a, b):
            return 1.0

    config = _isolated_config(tmp_path)
    config.memory.enable_fact_extraction = True
    config.memory.enable_long_term = True
    app = create_app(config)
    store = app.state.session_store

    async with _client(app) as client:
        # Prime the session so store.sessions["s1"] exists, then attach an
        # embedder so long-term memory actually functions in this test.
        await client.post(
            "/api/memory", json={"session_id": "s1", "action": "overview"}
        )
        session = store.sessions["s1"]
        session.memory.embedder = DummyEmbedder()
        session.memory.set_embedder(DummyEmbedder())

        remember = await client.post(
            "/api/memory",
            json={
                "session_id": "s1",
                "action": "remember",
                "text": "User likes hiking",
                "pinned": True,
            },
        )
        assert remember.json()["status"] == "ok"
        records = list(session.memory.vector_store.dump())
        assert len(records) == 1
        assert records[0].metadata["pinned"] is True
        record_id = records[0].id

        # 400s on missing record_id.
        for action in ("pin", "unpin", "edit", "forget_id"):
            missing = await client.post(
                "/api/memory", json={"session_id": "s1", "action": action}
            )
            assert missing.status_code == 400

        unpin = await client.post(
            "/api/memory",
            json={"session_id": "s1", "action": "unpin", "record_id": record_id},
        )
        assert unpin.json() == {"status": "ok"}
        assert session.memory.vector_store.get(record_id).metadata["pinned"] is False

        pin = await client.post(
            "/api/memory",
            json={"session_id": "s1", "action": "pin", "record_id": record_id},
        )
        assert pin.json() == {"status": "ok"}
        assert session.memory.vector_store.get(record_id).metadata["pinned"] is True

        pin_missing = await client.post(
            "/api/memory",
            json={"session_id": "s1", "action": "pin", "record_id": "nonexistent"},
        )
        assert pin_missing.json()["status"] == "not_found"

        # forget_id on a pinned record is protected (G4): deleting a pinned
        # memory by id must not silently succeed — the UI flow is
        # unpin-then-delete.
        forget_id_pinned = await client.post(
            "/api/memory",
            json={"session_id": "s1", "action": "forget_id", "record_id": record_id},
        )
        assert forget_id_pinned.json() == {
            "status": "protected",
            "detail": "That memory is pinned — unpin it first.",
        }
        assert len(session.memory.vector_store.dump()) == 1

        # Editing a pinned record IS allowed (edit is not a pin-check site).
        edit = await client.post(
            "/api/memory",
            json={
                "session_id": "s1",
                "action": "edit",
                "record_id": record_id,
                "text": "User likes mountain biking",
            },
        )
        assert edit.json() == {"status": "ok"}
        updated = session.memory.vector_store.get(record_id)
        assert updated.text == "User likes mountain biking"
        assert updated.id == record_id
        assert updated.metadata["pinned"] is True

        edit_no_text = await client.post(
            "/api/memory",
            json={"session_id": "s1", "action": "edit", "record_id": record_id},
        )
        assert edit_no_text.status_code == 400

        overview = await client.post(
            "/api/memory", json={"session_id": "s1", "action": "overview"}
        )
        assert overview.status_code == 200
        assert "fact_records" in overview.json()["overview"]
        assert len(overview.json()["overview"]["fact_records"]) == 1

        # Unpin, then forget_id succeeds normally.
        await client.post(
            "/api/memory",
            json={"session_id": "s1", "action": "unpin", "record_id": record_id},
        )
        forget_id = await client.post(
            "/api/memory",
            json={"session_id": "s1", "action": "forget_id", "record_id": record_id},
        )
        assert forget_id.json() == {
            "status": "ok",
            "detail": "Forgotten: User likes mountain biking",
        }
        assert session.memory.vector_store.dump() == []


@pytest.mark.asyncio
async def test_memory_delete_refreshes_sibling_sessions(tmp_path):
    """Synapse-map delete must reach SIBLING live sessions' RAM, not just
    the addressed one — a sibling still holding the record resurrects it on
    its own next whole-file persist (reproduced 2026-07-28: 'I deleted
    memories in the synapse map and they didn't go through')."""
    from amadeus.memory.manager import EmbeddingProvider

    class DummyEmbedder(EmbeddingProvider):
        def embed_text(self, text):
            return [float(len(text))]

        def similarity(self, a, b):
            return 1.0

    config = _isolated_config(tmp_path)
    config.memory.enable_fact_extraction = True
    config.memory.enable_long_term = True
    app = create_app(config)
    store = app.state.session_store

    async with _client(app) as client:
        # Session s1 stores a memory.
        await client.post(
            "/api/memory", json={"session_id": "s1", "action": "overview"}
        )
        s1 = store.sessions["s1"]
        s1.memory.set_embedder(DummyEmbedder())
        remember = await client.post(
            "/api/memory",
            json={"session_id": "s1", "action": "remember", "text": "hates cilantro"},
        )
        assert remember.json()["status"] == "ok"
        record_id = s1.memory.vector_store.dump()[0].id

        # Session s2 comes alive over the same files and sees the record.
        await client.post(
            "/api/memory", json={"session_id": "s2", "action": "overview"}
        )
        s2 = store.sessions["s2"]
        s2.memory.set_embedder(DummyEmbedder())
        assert len(s2.memory.vector_store.dump()) == 1

        # Delete via s1 — the sibling's RAM copy must go too.
        forget = await client.post(
            "/api/memory",
            json={"session_id": "s1", "action": "forget_id", "record_id": record_id},
        )
        assert forget.json()["status"] == "ok"
        assert s2.memory.vector_store.dump() == []

        # s2's own next store write must not resurrect the deleted record
        # on disk (this was the bug: last-writer-wins full-file rewrite).
        assert s2.memory.remember("likes trains")
        # Check DISK state (via a reload) instead of trusting either RAM copy.
        s1.memory.reload_long_term_from_disk()
        assert [rec.text for rec in s1.memory.vector_store.dump()] == ["likes trains"]

        forget_id_missing = await client.post(
            "/api/memory",
            json={"session_id": "s1", "action": "forget_id", "record_id": record_id},
        )
        assert forget_id_missing.json()["status"] == "not_found"


@pytest.mark.asyncio
async def test_greeting_endpoint(tmp_path, monkeypatch):
    from amadeus.providers.llm.base import LLMProvider

    class GreetingLLM(LLMProvider):
        name = "greeting-fake"
        model = None

        def generate(self, messages, **kwargs):  # type: ignore[override]
            return "Late again. I kept your fMRI question warm."

    # Default: disabled — endpoint answers but declines to greet.
    app = create_app(_isolated_config(tmp_path))
    async with _client(app) as client:
        resp = await client.post("/api/greeting", json={"session_id": "g1"})
    assert resp.status_code == 200
    assert resp.json()["greeting"] is None

    # Enabled with a usable provider: greets once, commits to memory.
    import amadeus.web.app as app_module

    monkeypatch.setattr(app_module, "load_llm", lambda cfg: GreetingLLM())
    config = _isolated_config(tmp_path / "enabled")
    config.memory.enable_greeting = True
    app2 = create_app(config)
    async with _client(app2) as client2:
        body = (await client2.post("/api/greeting", json={"session_id": "g2"})).json()
    assert body["greeting"] == "Late again. I kept your fMRI question warm."
    assert body["emotion"]

    session = app2.state.session_store.sessions["g2"]
    assert session.memory.short_term[-1].content == body["greeting"]
    # Same fresh-session gate as the engine: no double greeting.
    async with _client(app2) as client2:
        again = (await client2.post("/api/greeting", json={"session_id": "g2"})).json()
    assert again["greeting"] is None


def test_session_store_evicts_idle_sessions(tmp_path):
    config = _isolated_config(tmp_path)
    config.web.session_ttl_hours = 1.0
    app = create_app(config)
    store = app.state.session_store

    sid, session = store.get(None, "user-a")
    assert sid in store.sessions
    assert store.user_index["user-a"] == sid

    # Age the session past the TTL, then any store access sweeps it.
    session.last_activity -= 2 * 3600
    other_sid, _ = store.get(None, "user-b")
    assert sid not in store.sessions
    assert "user-a" not in store.user_index
    assert other_sid in store.sessions


def test_evicted_user_reloads_memory_from_disk(tmp_path):
    config = _isolated_config(tmp_path)
    config.web.session_ttl_hours = 1.0
    app = create_app(config)
    store = app.state.session_store

    sid, session = store.get(None, "user-a")
    # A completed pair: a store ending on a lone user message is healed away
    # on reload as a failed-turn orphan (see MemoryManager._load).
    session.memory.add_message("user", "remember the lab meeting")
    session.memory.add_message("assistant", "Noted. The lab meeting.")
    session.memory.flush()
    persist_path = session.persist_path

    session.last_activity -= 2 * 3600
    new_sid, new_session = store.get(None, "user-a")
    assert new_sid != sid
    # Same per-user persistence path -> the transcript survives eviction.
    assert new_session.persist_path == persist_path
    assert any("lab meeting" in msg.content for msg in new_session.memory.short_term)


def test_session_ttl_zero_disables_eviction(tmp_path):
    config = _isolated_config(tmp_path)
    config.web.session_ttl_hours = 0
    app = create_app(config)
    store = app.state.session_store

    sid, session = store.get(None, None)
    session.last_activity -= 365 * 24 * 3600
    store.get(None, "someone-else")
    assert sid in store.sessions


@pytest.mark.asyncio
async def test_chat_endpoint_provider_failure_returns_502(tmp_path):
    """A provider exception surfaces as a structured 502, not a raw 500."""
    app = create_app(_isolated_config(tmp_path))
    store = app.state.session_store

    class ExplodingLLM:
        def generate(self, messages, **kwargs):
            raise RuntimeError("429 rate limited")

    async with _client(app) as client:
        # Materialize the session first, then swap in the failing provider.
        first = await client.post("/api/chat", json={"message": "hi"})
        session_id = first.json()["session_id"]
        session = store.sessions[session_id]
        original_llm = session.llm
        session.llm = ExplodingLLM()

        response = await client.post(
            "/api/chat", json={"message": "boom", "session_id": session_id}
        )
        assert response.status_code == 502
        detail = response.json()["detail"]
        assert "rate limited" in detail["error"]
        # Additive classification field: this failure message reads as a
        # rate limit, so clients can caption it honestly.
        assert detail["kind"] == "rate_limited"

        # The failed turn must not have been committed to memory: restore the
        # provider and confirm the session still answers cleanly.
        session.llm = original_llm
        retry = await client.post(
            "/api/chat", json={"message": "hello again", "session_id": session_id}
        )
    assert retry.status_code == 200
    assert all("boom" not in str(m) for m in session.memory.short_term)


def test_startup_warns_when_requested_llm_degrades_to_dummy(tmp_path, caplog):
    config = _isolated_config(tmp_path)
    config.providers.llm = "openai"  # no API key in the isolated config
    config.openai_api_key = None
    with caplog.at_level("WARNING", logger="amadeus.web.app"):
        create_app(config)
    assert any("DEGRADED" in rec.message for rec in caplog.records)


def test_startup_silent_when_dummy_is_explicit(tmp_path, caplog):
    config = _isolated_config(tmp_path)
    config.providers.llm = "dummy"
    with caplog.at_level("WARNING", logger="amadeus.web.app"):
        create_app(config)
    assert not any("DEGRADED" in rec.message for rec in caplog.records)
