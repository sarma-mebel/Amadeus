"""Session management API: list/new/delete/transcript + disk persistence."""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("fastapi")

import httpx

from amadeus.config import Config
from amadeus.web.app import ChatRequest, create_app


def _config(tmp_path) -> Config:
    config = Config()
    config.audio.enable_voice = False
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.transcripts_path = str(tmp_path / "transcripts.json")
    config.memory.enable_long_term = False
    return config


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.mark.asyncio
async def test_session_lifecycle(tmp_path):
    app = create_app(_config(tmp_path))
    async with _client(app) as client:
        # Fresh session appears in the list immediately, empty.
        created = await client.post("/api/sessions/new", json={"user_id": "u1"})
        assert created.status_code == 200
        sid = created.json()["session_id"]
        assert sid

        listing = (await client.get("/api/sessions", params={"user_id": "u1"})).json()
        assert listing["sessions"] == [
            {
                "session_id": sid,
                "title": "New conversation",
                "created_at": pytest.approx(listing["sessions"][0]["created_at"]),
                "updated_at": pytest.approx(listing["sessions"][0]["updated_at"]),
                "message_count": 0,
            }
        ]

        # A chat turn into that explicit session records both messages.
        chat = await client.post(
            "/api/chat",
            json={
                "message": "Tell me about dopamine",
                "session_id": sid,
                "user_id": "u1",
            },
        )
        assert chat.status_code == 200
        assert chat.json()["session_id"] == sid

        listing = (await client.get("/api/sessions", params={"user_id": "u1"})).json()
        (summary,) = listing["sessions"]
        assert summary["title"] == "Tell me about dopamine"
        assert summary["message_count"] == 2
        assert summary["updated_at"] >= summary["created_at"]

        transcript = (
            await client.get(
                f"/api/sessions/{sid}/transcript", params={"user_id": "u1"}
            )
        ).json()
        messages = transcript["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[0]["content"] == "Tell me about dopamine"
        assert messages[0]["ja"] is None
        assert messages[1]["content"]
        # POST path stashes emotion metadata alongside the reply.
        assert messages[1]["emotion"] is not None
        assert messages[0]["timestamp"] <= messages[1]["timestamp"]

        # Delete drops both the in-memory session and the persisted transcript.
        deleted = await client.delete(f"/api/sessions/{sid}", params={"user_id": "u1"})
        assert deleted.json() == {"ok": True}
        assert sid not in app.state.session_store.sessions

        listing = (await client.get("/api/sessions", params={"user_id": "u1"})).json()
        assert listing["sessions"] == []
        missing = await client.get(
            f"/api/sessions/{sid}/transcript", params={"user_id": "u1"}
        )
        assert missing.status_code == 404


@pytest.mark.asyncio
async def test_sessions_sorted_by_recency_and_scoped_per_user(tmp_path):
    app = create_app(_config(tmp_path))
    async with _client(app) as client:
        first = (await client.post("/api/sessions/new", json={"user_id": "u1"})).json()[
            "session_id"
        ]
        second = (
            await client.post("/api/sessions/new", json={"user_id": "u1"})
        ).json()["session_id"]
        # Chatting into the *older* session makes it the most recent.
        await client.post(
            "/api/chat", json={"message": "hi", "session_id": first, "user_id": "u1"}
        )

        ordered = (await client.get("/api/sessions", params={"user_id": "u1"})).json()
        assert [s["session_id"] for s in ordered["sessions"]] == [first, second]

        # Another user sees none of it.
        other = (await client.get("/api/sessions", params={"user_id": "u2"})).json()
        assert other["sessions"] == []


@pytest.mark.asyncio
async def test_title_truncated_to_sixty_chars(tmp_path):
    app = create_app(_config(tmp_path))
    long_message = "dopamine " * 20
    async with _client(app) as client:
        chat = await client.post(
            "/api/chat", json={"message": long_message, "user_id": "u1"}
        )
        assert chat.status_code == 200
        listing = (await client.get("/api/sessions", params={"user_id": "u1"})).json()
    title = listing["sessions"][0]["title"]
    assert len(title) <= 60
    assert title.startswith("dopamine")


@pytest.mark.asyncio
async def test_transcripts_survive_backend_restart(tmp_path):
    config = _config(tmp_path)
    app = create_app(config)
    async with _client(app) as client:
        chat = await client.post(
            "/api/chat", json={"message": "Remember this one", "user_id": "u1"}
        )
        sid = chat.json()["session_id"]

    # Simulated restart: a brand-new app/store over the same config paths.
    app2 = create_app(config)
    assert app2.state.session_store.sessions == {}
    async with _client(app2) as client2:
        listing = (await client2.get("/api/sessions", params={"user_id": "u1"})).json()
        assert [s["session_id"] for s in listing["sessions"]] == [sid]
        assert listing["sessions"][0]["title"] == "Remember this one"

        transcript = (
            await client2.get(
                f"/api/sessions/{sid}/transcript", params={"user_id": "u1"}
            )
        ).json()
        assert transcript["messages"][0]["content"] == "Remember this one"

        # And chatting into the resumed session keeps appending to it.
        again = await client2.post(
            "/api/chat",
            json={"message": "Continue please", "session_id": sid, "user_id": "u1"},
        )
        assert again.json()["session_id"] == sid
        transcript = (
            await client2.get(
                f"/api/sessions/{sid}/transcript", params={"user_id": "u1"}
            )
        ).json()
        assert len(transcript["messages"]) == 4


@pytest.mark.asyncio
async def test_ws_streaming_turn_records_transcript_once(tmp_path):
    """Both finalization paths flow through complete_turn — exactly once."""
    config = _config(tmp_path)
    app = create_app(config)
    store = app.state.session_store

    session_id, session = store.get(None, "ws-user")
    frames = []

    async def emit(frame):
        frames.append(frame)

    request = ChatRequest.model_validate(
        {"message": "Stream me something", "user_id": "ws-user"}
    )
    await session.handle_stream(request, emit, threading.Event())
    assert frames[-1]["type"] == "done"

    transcript = store.transcripts.get_transcript("ws-user", session_id)
    assert transcript is not None
    assert [m["role"] for m in transcript] == ["user", "assistant"]
    assert transcript[0]["content"] == "Stream me something"


@pytest.mark.asyncio
async def test_transcript_endpoint_includes_turn_id(tmp_path):
    app = create_app(_config(tmp_path))
    async with _client(app) as client:
        chat = await client.post(
            "/api/chat", json={"message": "Tell me about dopamine", "user_id": "u1"}
        )
        sid = chat.json()["session_id"]
        turn_id = chat.json()["turn_id"]
        assert turn_id

        transcript = (
            await client.get(
                f"/api/sessions/{sid}/transcript", params={"user_id": "u1"}
            )
        ).json()
        messages = transcript["messages"]
        assert messages[0]["turn_id"] == turn_id
        assert messages[1]["turn_id"] == turn_id


@pytest.mark.asyncio
async def test_rewind_endpoint_truncates_and_returns_messages(tmp_path):
    app = create_app(_config(tmp_path))
    async with _client(app) as client:
        first = await client.post("/api/chat", json={"message": "one", "user_id": "u1"})
        sid = first.json()["session_id"]
        second = await client.post(
            "/api/chat",
            json={"message": "two", "session_id": sid, "user_id": "u1"},
        )
        second_turn_id = second.json()["turn_id"]

        rewound = await client.post(
            f"/api/sessions/{sid}/rewind",
            json={"user_id": "u1", "turn_id": second_turn_id},
        )
        assert rewound.status_code == 200
        body = rewound.json()
        assert body["session_id"] == sid
        assert body["status"] == "ok"
        assert len(body["messages"]) == 2

        transcript = (
            await client.get(
                f"/api/sessions/{sid}/transcript", params={"user_id": "u1"}
            )
        ).json()
        assert len(transcript["messages"]) == 2
        assert transcript["messages"][0]["content"] == "one"


@pytest.mark.asyncio
async def test_rewind_endpoint_unknown_turn_returns_404(tmp_path):
    app = create_app(_config(tmp_path))
    async with _client(app) as client:
        created = await client.post("/api/sessions/new", json={"user_id": "u1"})
        sid = created.json()["session_id"]

        rewound = await client.post(
            f"/api/sessions/{sid}/rewind",
            json={"user_id": "u1", "turn_id": "nonexistent"},
        )
        assert rewound.status_code == 404


@pytest.mark.asyncio
async def test_amend_endpoint_updates_reply(tmp_path):
    app = create_app(_config(tmp_path))
    async with _client(app) as client:
        chat = await client.post("/api/chat", json={"message": "hi", "user_id": "u1"})
        sid = chat.json()["session_id"]
        turn_id = chat.json()["turn_id"]

        amended = await client.post(
            f"/api/sessions/{sid}/amend",
            json={"user_id": "u1", "turn_id": turn_id, "content": "Edited reply."},
        )
        assert amended.status_code == 200
        assert amended.json()["status"] == "ok"

        transcript = (
            await client.get(
                f"/api/sessions/{sid}/transcript", params={"user_id": "u1"}
            )
        ).json()
        assert transcript["messages"][-1]["content"] == "Edited reply."


@pytest.mark.asyncio
async def test_amend_endpoint_stale_turn_returns_409(tmp_path):
    app = create_app(_config(tmp_path))
    async with _client(app) as client:
        first = await client.post("/api/chat", json={"message": "one", "user_id": "u1"})
        sid = first.json()["session_id"]
        first_turn_id = first.json()["turn_id"]
        await client.post(
            "/api/chat",
            json={"message": "two", "session_id": sid, "user_id": "u1"},
        )

        amended = await client.post(
            f"/api/sessions/{sid}/amend",
            json={"user_id": "u1", "turn_id": first_turn_id, "content": "Nope."},
        )
        assert amended.status_code == 409


@pytest.mark.asyncio
async def test_amend_endpoint_survives_trailing_idle_chatter_message(tmp_path):
    """A trailing assistant line appended outside the normal turn flow (idle
    chatter) must not shadow the turn amend actually targets."""
    app = create_app(_config(tmp_path))
    async with _client(app) as client:
        chat = await client.post("/api/chat", json={"message": "hi", "user_id": "u1"})
        sid = chat.json()["session_id"]
        turn_id = chat.json()["turn_id"]

        store = app.state.session_store
        _sid, session = store.get(sid, "u1")
        session.memory.add_message("assistant", "*idle chatter aside*")

        amended = await client.post(
            f"/api/sessions/{sid}/amend",
            json={"user_id": "u1", "turn_id": turn_id, "content": "Edited reply."},
        )
        assert amended.status_code == 200
        assert amended.json()["status"] == "ok"

        transcript = (
            await client.get(
                f"/api/sessions/{sid}/transcript", params={"user_id": "u1"}
            )
        ).json()
        assert transcript["messages"][-1]["content"] == "Edited reply."

        contents = [m.content for m in session.memory.short_term]
        assert "Edited reply." in contents
        assert "*idle chatter aside*" in contents


@pytest.mark.asyncio
async def test_rewind_endpoint_not_found_status_maps_to_404(tmp_path):
    """rewind_to_turn's fallback-validation failure (F5) surfaces as the same
    404 as an unknown turn id — the route only distinguishes "ok" from
    anything else."""
    app = create_app(_config(tmp_path))
    async with _client(app) as client:
        chat = await client.post("/api/chat", json={"message": "hi", "user_id": "u1"})
        sid = chat.json()["session_id"]
        turn_id = chat.json()["turn_id"]

        store = app.state.session_store
        _sid, session = store.get(sid, "u1")
        # Corrupt the tail so the fallback verification in
        # _rewind_fallback_no_store fails even though turn_id matches.
        session.transcript_store = None
        from amadeus.memory.manager import Message

        session.memory.short_term.append(
            Message(role="user", content="dangling", timestamp=None)
        )

        rewound = await client.post(
            f"/api/sessions/{sid}/rewind",
            json={"user_id": "u1", "turn_id": turn_id},
        )
        assert rewound.status_code == 404
