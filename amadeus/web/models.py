"""Pydantic request/response models for the Amadeus web API."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ImageAttachment(BaseModel):
    """One user-shared image riding on a chat turn (vision). Raw base64
    without a data: prefix; the size validator is a hard server-side cap —
    the client is expected to downscale before upload."""

    data: str = Field(..., description="Base64-encoded image bytes")
    mime: Literal["image/jpeg", "image/png", "image/webp"]

    @field_validator("data")
    @classmethod
    def _size_cap(cls, value: str) -> str:
        # ~4/3 expansion of the configured byte cap; enforced here so an
        # oversized payload fails validation before any decode work. Uses
        # the default cap (config isn't reachable from a pydantic
        # validator) — deliberately generous, the client downscales.
        from ..config import VisionSettings

        max_b64 = int(VisionSettings.max_upload_bytes * 4 / 3) + 4
        if len(value) > max_b64:
            raise ValueError("image attachment too large")
        return value


class ChatRequest(BaseModel):
    """Inbound payload from the web UI."""

    message: str = Field(..., min_length=1, description="User prompt to send to Kurisu")
    images: Optional[List[ImageAttachment]] = Field(
        default=None,
        max_length=1,
        description="User-shared image(s) for this turn (vision); max one",
    )
    session_id: Optional[str] = Field(
        default=None, description="Existing session identifier"
    )
    user_id: Optional[str] = Field(
        default=None, description="Stable identifier for the user/device"
    )
    safe_mode: Optional[bool] = Field(
        default=None, description="Override safe mode toggle"
    )
    tsundere_level: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    scientific_formality: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    audio_enabled: Optional[bool] = Field(
        default=None, description="Whether Kurisu should speak this reply"
    )
    # Deprecated no-ops, accepted so pre-council-removal clients never 422:
    # the council was deleted 2026-07-17 and every request runs the plain
    # chat path regardless of these values.
    council: Optional[bool] = Field(default=None, description="Deprecated; ignored")
    agents: Optional[list[str]] = Field(default=None, description="Deprecated; ignored")
    rounds: Optional[int] = Field(default=None, description="Deprecated; ignored")
    council_mode: Optional[str] = Field(default=None, description="Deprecated; ignored")
    council_phases: Optional[list[dict]] = Field(
        default=None, description="Deprecated; ignored"
    )
    budget_tier: Optional[str] = Field(
        default="balanced", description="Deprecated; ignored"
    )
    defer_audio: bool = Field(
        default=False,
        description=(
            "Skip TTS synthesis for this turn; the client fetches audio "
            "later via POST /api/tts. Audio fields come back null."
        ),
    )


class ResetRequest(BaseModel):
    """Request body when the client wants to clear the conversation."""

    session_id: str
    user_id: Optional[str] = Field(
        default=None, description="Stable identifier for the user/device"
    )
    forget: bool = Field(default=True, description="Whether to erase persisted memory")
    full_forget: bool = Field(
        default=False,
        description="Also erase the long-lived user profile (who-you-are memory)",
    )


class MemoryRequest(BaseModel):
    """User-directed memory controls (remember, forget, pin, edit, overview...)."""

    session_id: Optional[str] = None
    user_id: Optional[str] = Field(
        default=None, description="Stable identifier for the user/device"
    )
    action: str = Field(
        description=(
            "One of: remember, forget, forget_fact, forget_thread, overview, "
            "graph, reset_relationship, pin, unpin, edit, forget_id"
        )
    )
    text: Optional[str] = Field(default=None, description="Payload for remember/forget")
    record_id: Optional[str] = Field(
        default=None, description="Target a specific memory record by id"
    )
    pinned: Optional[bool] = Field(
        default=None, description="Set/clear a memory record's pinned state"
    )


class RewindRequest(BaseModel):
    """Truncate a session's history to before a given turn."""

    user_id: Optional[str] = Field(
        default=None, description="Stable identifier for the user/device"
    )
    turn_id: str = Field(description="Turn id to rewind to (that turn is dropped)")


class AmendRequest(BaseModel):
    """Replace the latest assistant reply's text (edit-in-place)."""

    user_id: Optional[str] = Field(
        default=None, description="Stable identifier for the user/device"
    )
    turn_id: str = Field(description="Turn id of the assistant reply being amended")
    content: str = Field(description="New reply text")
    ja: Optional[str] = Field(default=None, description="New Japanese speech text")


class GreetingRequest(BaseModel):
    """Client asks whether Kurisu wants to open the session herself."""

    session_id: Optional[str] = None
    user_id: Optional[str] = Field(
        default=None, description="Stable identifier for the user/device"
    )


class ScreenPeekRequest(BaseModel):
    """Screen-companion peek (pet mode, opt-in): one screenshot in, one
    short unprompted line out. Same identification as the presence
    endpoints; the image follows the vision attachment contract (only ever
    lives in this request, never persisted)."""

    session_id: Optional[str] = None
    user_id: Optional[str] = Field(
        default=None, description="Stable identifier for the user/device"
    )
    image: ImageAttachment


class ReadAlongRequest(BaseModel):
    """ "Read together" mode: one screenshot per page turn, or an end signal
    closing the reading session. Same session/user identification as the
    other presence endpoints. Exactly one of ``image``/``end`` must be set —
    a page post and a session-end are mutually exclusive requests."""

    session_id: Optional[str] = None
    user_id: Optional[str] = Field(
        default=None, description="Stable identifier for the user/device"
    )
    image: Optional[ImageAttachment] = Field(
        default=None, description="Screenshot of the current page (page-turn path)"
    )
    end: bool = Field(
        default=False, description="Close the reading session (end-of-session path)"
    )

    @model_validator(mode="after")
    def _exactly_one_of_image_or_end(self) -> "ReadAlongRequest":
        if bool(self.image) == bool(self.end):
            raise ValueError(
                "Provide exactly one of 'image' (a page) or 'end' (close the session)"
            )
        return self


class WatchTogetherRequest(BaseModel):
    """ "Watch together" mode: one periodic screen frame, or an end signal
    closing the watch session. ReadAlongRequest's sibling — same
    identification, same image-XOR-end contract, same vision attachment
    privacy contract (the frame only ever lives in this request)."""

    session_id: Optional[str] = None
    user_id: Optional[str] = Field(
        default=None, description="Stable identifier for the user/device"
    )
    image: Optional[ImageAttachment] = Field(
        default=None, description="Periodic screen frame (watch path)"
    )
    end: bool = Field(
        default=False, description="Close the watch session (end-of-session path)"
    )

    @model_validator(mode="after")
    def _exactly_one_of_image_or_end(self) -> "WatchTogetherRequest":
        if bool(self.image) == bool(self.end):
            raise ValueError(
                "Provide exactly one of 'image' (a frame) or 'end' (close the session)"
            )
        return self


class PresenceRequest(BaseModel):
    """Shared shape for the poke and idle-chatter endpoints — same
    session/user identification as the greeting. Poke additionally carries
    the body region the avatar raycast resolved (region-aware reactions)."""

    session_id: Optional[str] = None
    user_id: Optional[str] = Field(
        default=None, description="Stable identifier for the user/device"
    )
    region: Optional[str] = Field(
        default=None,
        # Raycast body regions plus the desktop-pet event regions (window
        # dragged, click-through enabled, press-and-hold head pat, woken
        # from the doze state). Keep this in sync with the frontend's
        # firePoke call sites and presence._POKE_REGION_CONTEXT — an
        # unlisted region 422s here and the pet UI swallows poke failures
        # by design, so the reaction just silently never happens.
        # tests/test_contracts.py asserts all three layers agree.
        pattern="^(head|chest|hands|body|head_pat|dragged|click_through|woke_from_doze)$",
        description="Poked body region from the avatar raycast or pet-window event",
    )


class AmbientRequest(BaseModel):
    """Ambient desktop signal (desktop app): the client noticed a
    noteworthy transition — long focus on one app, battery low, track
    change — and asks whether she has a line about it. Text-only cousin of
    ScreenPeekRequest; presence.ambient_comment owns all gating."""

    session_id: Optional[str] = None
    user_id: Optional[str] = Field(
        default=None, description="Stable identifier for the user/device"
    )
    kind: str = Field(
        # Keep in sync with presence._AMBIENT_KINDS and the frontend's
        # ambient tracker; an unlisted kind 422s here.
        pattern="^(focus|battery|music)$",
        description="Which signal family fired",
    )
    detail: str = Field(
        min_length=1,
        max_length=300,
        description="Human-readable one-line description of the transition",
    )


class FocusRequest(BaseModel):
    """Focus / co-working mode (amadeus/focus_mode.py): start a heads-down
    session, end one early, or poll current status. ``minutes`` is only
    used by "start" (clamped server-side to 10-180; a missing/invalid
    value falls back to ``focus.default_minutes``)."""

    session_id: Optional[str] = None
    user_id: Optional[str] = Field(
        default=None, description="Stable identifier for the user/device"
    )
    action: str = Field(pattern="^(start|end|status)$")
    minutes: Optional[int] = Field(
        default=None, ge=1, le=1440, description="Requested session length (start only)"
    )


class FarewellRequest(BaseModel):
    """The user is leaving: either they said goodbye (one warm send-off
    line) or the window is closing (silent session finalize only)."""

    session_id: Optional[str] = None
    user_id: Optional[str] = Field(
        default=None, description="Stable identifier for the user/device"
    )
    message: Optional[str] = Field(
        default=None, description="The user's goodbye text (goodbye path only)"
    )
    reason: str = Field(default="goodbye", pattern="^(goodbye|window_close)$")


class TTSRequest(BaseModel):
    """Standalone synthesis request for the deferred-audio flow."""

    text: str = Field(..., min_length=1, description="Japanese text to speak")
    emotion: Optional[str] = Field(
        default="neutral",
        description="One of happy/angry/sad/relaxed/surprised/neutral",
    )


class NewSessionRequest(BaseModel):
    """Body for POST /api/sessions/new."""

    user_id: Optional[str] = Field(
        default=None, description="Stable identifier for the user/device"
    )


class RhythmRequest(BaseModel):
    """Body for POST /api/rhythm — one local day's coarse tally.

    Every field is a counter or a bucket. There is deliberately no field for
    a window title, document name, track, artist, or timestamp: the client
    reduces to this shape BEFORE posting so raw titles never cross the
    network (frontend/src/rhythm.ts). Unknown app buckets are dropped
    server-side by RhythmDay.from_dict, so this schema is not the only guard.
    """

    user_id: Optional[str] = Field(default=None)
    date: str = Field(min_length=8, max_length=10, description="ISO local date")
    late_night_blocks: int = Field(default=0, ge=0, le=48)
    long_focus_stretches: int = Field(default=0, ge=0, le=48)
    longest_focus_hours: float = Field(default=0.0, ge=0.0, le=24.0)
    active_hour_buckets: List[int] = Field(default_factory=list, max_length=24)
    music_events: int = Field(default=0, ge=0, le=500)
    battery_low_events: int = Field(default=0, ge=0, le=50)
    app_buckets: Dict[str, int] = Field(default_factory=dict)


class InsightDismissRequest(BaseModel):
    """Body for POST /api/insights/dismiss."""

    user_id: Optional[str] = Field(
        default=None, description="Stable identifier for the user/device"
    )
    text: str = Field(
        min_length=1,
        max_length=400,
        description="Exact text of the insight to remove from the set",
    )


class SentencePayload(BaseModel):
    """One reply sentence: display text, speech text, and emotion."""

    en: str
    ja: str
    emotion: str = "neutral"
    emotion_raw: str | None = None


class ChatResponse(BaseModel):
    """Payload returned back to the web client."""

    session_id: str
    reply: str
    turn_id: Optional[str] = Field(
        default=None, description="Server-generated id for this exchange"
    )
    memories: list[str]
    summary: Optional[str]
    summary_strategy: Optional[str] = Field(
        default=None, description="Source of the summary (llm|truncate)"
    )
    summary_revision: Optional[int] = Field(
        default=None, description="Revision counter for the summary"
    )
    summary_updated_at: Optional[str] = Field(
        default=None, description="ISO timestamp of the last summary update"
    )
    emotion: Optional[str] = Field(
        default="neutral", description="Detected emotion for animation"
    )
    animations: list[str] = Field(
        default_factory=list,
        description="Avatar animations the play_animation tool requested this turn",
    )
    emotion_intensity: Optional[float] = Field(
        default=0.8, description="Expression intensity 0.0-1.0"
    )
    audio: Optional[str] = Field(
        default=None, description="Base64 encoded audio payload"
    )
    audio_mime_type: Optional[str] = Field(
        default=None,
        description="MIME type for the encoded audio clip",
    )
    audio_enabled: bool = Field(default=True, description="Whether audio was generated")
    thought: Optional[str] = Field(
        default=None,
        description="Unspoken [thought] line (display-only, never voiced)",
    )
    sentences: Optional[list[SentencePayload]] = Field(
        default=None,
        description=(
            "Per-sentence en/ja/emotion pairs; drives the client's Japanese "
            "subtitle and the deferred per-sentence TTS path"
        ),
    )
    timings: Optional[dict] = Field(
        default=None, description="Server-side latency breakdown in milliseconds"
    )
    routing_decision: Optional[dict] = Field(
        default=None, description="Agent routing decision details"
    )
    tool_calls: Optional[list[dict]] = Field(
        default=None, description="Tools executed during the session"
    )
