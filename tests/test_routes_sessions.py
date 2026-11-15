"""amadeus/web/routes_sessions.py — the /api/sessions CRUD router.

test_sessions_api.py covers the deeper turn-level semantics; this file is a
focused pass over the router itself: every route's happy path against a
dummy-provider app, plus the 404/409 error mappings the route layer owns.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

import httpx

from amadeus.config import Config
from amadeus.web.app import create_app


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
async def test_new_list_delete_roundtrip(tmp_path):
    app = create_app(_config(tmp_path))
    async with _client(app) as client:
        created = await client.post("/api/sessions/new", json={"user_id": "u1"})
        assert created.status_code == 200
        sid = created.json()["session_id"]
        assert sid

        listing = (await client.get("/api/sessions", params={"user_id": "u1"})).json()
        assert [s["session_id"] for s in listing["sessions"]] == [sid]

        # Sessions are user-scoped: another user sees nothing.
        other = (await client.get("/api/sessions", params={"user_id": "u2"})).json()
        assert other["sessions"] == []

        deleted = await client.delete(f"/api/sessions/{sid}", params={"user_id": "u1"})
        assert deleted.status_code == 200
        assert deleted.json() == {"ok": True}
        listing = (await client.get("/api/sessions", params={"user_id": "u1"})).json()
        assert listing["sessions"] == []


@pytest.mark.asyncio
async def test_transcript_happy_path_and_unknown_404(tmp_path):
    app = create_app(_config(tmp_path))
    async with _client(app) as client:
        chat = await client.post(
            "/api/chat", json={"message": "hello there", "user_id": "u1"}
        )
        assert chat.status_code == 200
        sid = chat.json()["session_id"]

        transcript = await client.get(
            f"/api/sessions/{sid}/transcript", params={"user_id": "u1"}
        )
        assert transcript.status_code == 200
        messages = transcript.json()["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[0]["content"] == "hello there"

        missing = await client.get(
            "/api/sessions/no-such-session/transcript", params={"user_id": "u1"}
        )
        assert missing.status_code == 404


@pytest.mark.asyncio
async def test_rewind_happy_path_and_unknown_turn_404(tmp_path):
    app = create_app(_config(tmp_path))
    async with _client(app) as client:
        first = await client.post("/api/chat", json={"message": "one", "user_id": "u1"})
        sid = first.json()["session_id"]
        second = await client.post(
            "/api/chat", json={"message": "two", "session_id": sid, "user_id": "u1"}
        )

        rewound = await client.post(
            f"/api/sessions/{sid}/rewind",
            json={"user_id": "u1", "turn_id": second.json()["turn_id"]},
        )
        assert rewound.status_code == 200
        body = rewound.json()
        assert body["session_id"] == sid
        assert body["status"] == "ok"
        assert [m["content"] for m in body["messages"]][0] == "one"

        unknown = await client.post(
            f"/api/sessions/{sid}/rewind",
            json={"user_id": "u1", "turn_id": "nonexistent"},
        )
        assert unknown.status_code == 404


@pytest.mark.asyncio
async def test_amend_happy_path_and_stale_turn_409(tmp_path):
    app = create_app(_config(tmp_path))
    async with _client(app) as client:
        first = await client.post("/api/chat", json={"message": "one", "user_id": "u1"})
        sid = first.json()["session_id"]
        first_turn = first.json()["turn_id"]

        amended = await client.post(
            f"/api/sessions/{sid}/amend",
            json={"user_id": "u1", "turn_id": first_turn, "content": "Edited."},
        )
        assert amended.status_code == 200
        assert amended.json()["status"] == "ok"
        transcript = (
            await client.get(
                f"/api/sessions/{sid}/transcript", params={"user_id": "u1"}
            )
        ).json()
        assert transcript["messages"][-1]["content"] == "Edited."

        # A second turn makes the first turn stale for amending.
        await client.post(
            "/api/chat", json={"message": "two", "session_id": sid, "user_id": "u1"}
        )
        stale = await client.post(
            f"/api/sessions/{sid}/amend",
            json={"user_id": "u1", "turn_id": first_turn, "content": "Nope."},
        )
        assert stale.status_code == 409
