"""Session-management routes: the /api/sessions CRUD."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from .models import AmendRequest, NewSessionRequest, RewindRequest
from .sessions import SessionStore, _run_blocking


def build_router(store: SessionStore) -> APIRouter:
    router = APIRouter()

    # --- Session management -------------------------------------------------
    #
    # In-memory sessions are TTL-evicted; the durable source of truth for
    # these endpoints is the TranscriptStore, which persists across restarts
    # until a session is explicitly deleted.

    @router.get("/api/sessions")
    async def list_sessions(user_id: Optional[str] = None) -> Dict[str, Any]:
        sessions = await _run_blocking(store.transcripts.list_sessions, user_id)
        return {"sessions": sessions}

    @router.post("/api/sessions/new")
    async def new_session(payload: NewSessionRequest) -> Dict[str, str]:
        session_id = await _run_blocking(store.create_session, payload.user_id)
        return {"session_id": session_id}

    @router.delete("/api/sessions/{session_id}")
    async def delete_session(
        session_id: str, user_id: Optional[str] = None
    ) -> Dict[str, bool]:
        await _run_blocking(store.drop_session, session_id, user_id)
        return {"ok": True}

    @router.get("/api/sessions/{session_id}/transcript")
    async def session_transcript(
        session_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        messages = await _run_blocking(
            store.transcripts.get_transcript, user_id, session_id
        )
        if messages is None:
            raise HTTPException(
                status_code=404, detail=f"Unknown session: {session_id}"
            )
        return {"messages": messages}

    # --- Turn-level operations (foundation for regenerate/edit/rewind UI) --

    @router.post("/api/sessions/{session_id}/rewind")
    async def rewind_session(session_id: str, payload: RewindRequest) -> Dict[str, Any]:
        sid, session = store.get(session_id, payload.user_id)
        async with session.handle_lock:
            result = await _run_blocking(session.rewind_to_turn, payload.turn_id)
        if result.get("status") == "not_found":
            raise HTTPException(
                status_code=404, detail=f"Unknown turn: {payload.turn_id}"
            )
        return {"session_id": sid, **result}

    @router.post("/api/sessions/{session_id}/amend")
    async def amend_session(session_id: str, payload: AmendRequest) -> Dict[str, Any]:
        sid, session = store.get(session_id, payload.user_id)
        async with session.handle_lock:
            result = await _run_blocking(
                session.amend_last_assistant,
                payload.turn_id,
                payload.content,
                payload.ja,
            )
        if result.get("status") == "conflict":
            raise HTTPException(
                status_code=409, detail=f"Stale turn: {payload.turn_id}"
            )
        return {"session_id": sid, **result}

    return router
