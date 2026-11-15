"""Shared-activity presence: "read together" pages and "watch together" frames.

Extracted from ``presence.py`` for the line budget (same decomposition
pass as presence.py itself out of engine.py). The two features are one
family: the client posts screenshots of a shared activity (a page turn, a
periodic screen frame), she keeps an ephemeral in-RAM NOTE buffer for
continuity, and only occasionally SAYs anything — one LLM call per
post, notes always advance, the SAY half obeys a speak cooldown, and
nothing is ever persisted or committed to the conversation store (the
vision privacy contract, see docs/companion-ethics.md).

Same conventions as presence.py: free functions taking the
:class:`~amadeus.engine.ChatEngine` as first argument, all mutable state
(``_read_along``, ``_watch_together``, the cap-logged flags) lives on the
engine instance, and ``ChatEngine`` keeps thin delegating methods so
callers and tests are untouched.
"""

from __future__ import annotations

import collections
import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from . import moments, telemetry
from .logging_utils import get_logger
from .persona import persona_capsule
from .providers.llm.base import ChatMessage, ImageContent, LLMProvider

if TYPE_CHECKING:  # pragma: no cover - import cycle guard (engine imports us)
    from .engine import ChatEngine

LOGGER = get_logger(__name__)


@dataclass
class _ReadAlongState:
    """In-process, in-RAM-only state for one "read together" session.

    Lives exactly as long as the reading session: created lazily on the
    first page, cleared by read_along_end() (or a fresh session/engine
    instance — nothing here is persisted anywhere). ``notes`` is a rolling
    buffer, not a transcript: old notes silently fall off maxlen so she
    stays "a few pages behind" instead of accreting the whole book.
    """

    notes: "collections.deque[str]"
    pages: int = 0
    last_spoke_at: Optional[dt.datetime] = None


@dataclass
class _WatchTogetherState:
    """In-process, in-RAM-only state for one "watch together" session.

    read_along's sibling: created lazily on the first frame, cleared by
    watch_end() (or a fresh engine instance — never persisted). ``notes``
    is the same rolling few-minutes-back buffer, not a session transcript.
    """

    notes: "collections.deque[str]"
    frames: int = 0
    last_spoke_at: Optional[dt.datetime] = None


def _read_along_provider(engine: "ChatEngine") -> Optional[LLMProvider]:
    """Vision-capable provider for a read-along page, cheap-model first.

    Prefers summary_provider (usually the cheaper model) when it exists
    AND actually supports vision — reading sessions can run many pages,
    so routing them off the main chat model when possible matters more
    here than for a one-off screen_comment glance. Falls back to
    engine.llm, then None when neither can see.
    """
    for candidate in (engine.summary_provider, engine.llm):
        if candidate is None:
            continue
        supports = getattr(candidate, "supports_vision", None)
        if callable(supports) and supports():
            return candidate
    return None


def _read_along_notes_block(
    state: Optional["_ReadAlongState"], *, header: str = "Earlier pages"
) -> str:
    if state is None or not state.notes:
        return ""
    lines = "\n".join(f"{i}. {note}" for i, note in enumerate(state.notes, start=1))
    return f"{header}:\n{lines}"


def _parse_read_along_reply(text: str) -> tuple[str, Optional[str]]:
    """Tolerantly split a NOTE/SAY reply.

    The strict contract asks for a first line prefixed ``NOTE:`` and an
    optional second line prefixed ``SAY:``. Models drift, so: a missing
    ``NOTE:`` prefix falls back to treating the whole first non-empty
    line as the note, and a ``SAY:`` line is only honored when the
    prefix is actually present (an unprefixed second line is far more
    likely to be a continuation of the note than a line she'd say).
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "", None
    first = lines[0]
    if first.upper().startswith("NOTE:"):
        note = first[len("NOTE:") :].strip()
    else:
        note = first
    say: Optional[str] = None
    for ln in lines[1:]:
        if ln.upper().startswith("SAY:"):
            candidate = ln[len("SAY:") :].strip()
            if candidate:
                say = candidate
            break
    return note, say


def read_along_page(
    engine: "ChatEngine", image_mime: str, image_data: str
) -> Optional[str]:
    """Process one page of a "read together" session.

    Gated exactly like screen_comment (vision.enabled + a vision-capable
    provider — see _read_along_provider for the summary-provider
    preference). Makes ONE LLM call per page and ALWAYS records a
    continuity note for it (even when nothing is said aloud): the
    rolling note buffer is what lets her reference "a few pages back"
    later, so it must keep advancing regardless of the speak cooldown.
    Returns the SAY line only when one was produced AND the cooldown has
    elapsed; otherwise the line is swallowed (never lost — the note still
    landed) and None is returned. Never committed to memory: like
    screen_comment, the image and her reaction live and die in this one
    request/process, never in the conversation store.
    """
    if not getattr(getattr(engine.config, "vision", None), "enabled", False):
        return None
    read_along_cfg = getattr(engine.config, "read_along", None)
    if read_along_cfg is None:
        return None
    provider = _read_along_provider(engine)
    if provider is None:
        return None

    if engine._read_along is None:
        engine._read_along = _ReadAlongState(
            notes=collections.deque(
                maxlen=max(1, int(getattr(read_along_cfg, "max_notes", 10) or 10))
            )
        )
    state = engine._read_along

    max_pages = int(getattr(read_along_cfg, "max_pages_per_session", 300) or 300)
    if max_pages > 0 and state.pages >= max_pages:
        if not engine._read_along_cap_logged:
            LOGGER.info(
                "Read-along session hit max_pages_per_session (%d); "
                "further pages are ignored without an LLM call until "
                "read_along_end() clears the session.",
                max_pages,
            )
            engine._read_along_cap_logged = True
        return None

    notes_block = _read_along_notes_block(state)
    prompt = [
        ChatMessage(
            role="system",
            content=(
                f"You are {engine.character_full_name}, reading a book or manga "
                "together with the user, page by page, side by side. This is "
                "companionable reading, not narration — most pages deserve "
                "comfortable silence, not commentary. Only speak up when "
                "something is genuinely reaction-worthy: a twist, a "
                "confession, a great panel, a joke landing. "
                + persona_capsule(getattr(engine.config, "character", None))
                + " Respond in EXACTLY this format, nothing else:\n"
                "NOTE: <one factual line, under 25 words, about what "
                "happens on this page>\n"
                "SAY: <one short in-character line, under 20 words> "
                "(OMIT this line entirely on most pages — only include it "
                "when the page truly earns a reaction)\n"
                "If the page does not look like reading material (private "
                "or sensitive content instead), the NOTE must be a neutral "
                "'not reading material' and the SAY line must be omitted. "
                "No stage directions, no quotation marks."
                + (f"\n\n{notes_block}" if notes_block else "")
            ),
        ),
        ChatMessage(
            role="user",
            content="(the current page)",
            images=[ImageContent(mime=image_mime, data=image_data)],
        ),
    ]
    try:
        raw = (
            provider.generate(
                prompt,
                max_tokens=int(
                    getattr(read_along_cfg, "comment_max_tokens", 800) or 800
                ),
            )
            or ""
        ).strip()
        telemetry.record_llm_usage(provider, "read_along")
    except Exception as exc:  # pragma: no cover - provider runtime errors
        LOGGER.warning("Read-along page generation failed: %s", exc)
        telemetry.incr("presence.read_along_failures")
        return None

    note, say = _parse_read_along_reply(raw)
    # The note buffer always advances, independent of whether anything
    # is spoken — this is the continuity mechanism the cooldown must
    # never starve.
    if note:
        state.notes.append(note)
    state.pages += 1

    if not say:
        return None
    cooldown = float(getattr(read_along_cfg, "speak_cooldown_seconds", 90.0) or 90.0)
    if not engine._presence_cooldown_ready(state.last_spoke_at, cooldown):
        # Swallowed by the cooldown: the note already landed above, only
        # the spoken line is dropped.
        return None
    state.last_spoke_at = dt.datetime.utcnow()
    return say


def read_along_end(engine: "ChatEngine") -> Optional[str]:
    """Close out a "read together" session with an optional sign-off.

    Cooldown-exempt (ending a session is a one-shot, not a spammable
    event). Clears the in-RAM state unconditionally, so a repeated call
    (or a session that never really started) is always safe. Only
    generates a line when at least 5 pages were actually read — a
    two-page glance doesn't deserve a send-off. The opt-in moments
    distillation (see amadeus/moments.py) runs independently of the
    sign-off's own success/length gate, keyed only on the notes buffer
    having enough content — a short in-character sign-off and a durable
    memory are separate concerns with separate thresholds.
    """
    state = engine._read_along
    engine._read_along = None
    engine._read_along_cap_logged = False
    if state is None:
        return None
    moments.maybe_store_moment(
        engine, activity="reading session", notes=list(state.notes)
    )
    if state.pages < 5:
        return None
    provider = engine.summary_provider or engine.llm
    if provider is None:
        return None
    notes_block = _read_along_notes_block(state, header="Notes from the session")
    prompt = [
        ChatMessage(
            role="system",
            content=(
                f"You are {engine.character_full_name}. You and the user just "
                "finished a reading session together (a book or manga), "
                f"{state.pages} pages. Write exactly ONE short in-character "
                "sign-off line (under 20 words) about wrapping up the "
                "reading session, in character — a little fond is fine. "
                + persona_capsule(getattr(engine.config, "character", None))
                + " You may reference something from the notes below if it "
                "fits naturally. No stage directions, no quotation marks."
                + (f"\n\n{notes_block}" if notes_block else "")
            ),
        ),
        ChatMessage(role="user", content="(reading session ended)"),
    ]
    try:
        text = (
            (
                provider.generate(
                    prompt,
                    max_tokens=int(
                        getattr(engine.config.read_along, "comment_max_tokens", 800)
                        or 800
                    ),
                )
                or ""
            )
            .strip()
            .strip('"')
            .strip()
        )
        telemetry.record_llm_usage(provider, "read_along")
    except Exception as exc:  # pragma: no cover - provider runtime errors
        LOGGER.warning("Read-along sign-off generation failed: %s", exc)
        telemetry.incr("presence.read_along_failures")
        return None
    return text or None


def watch_frame(
    engine: "ChatEngine", image_mime: str, image_data: str
) -> Optional[str]:
    """Process one frame of a "watch together" session.

    read_along_page's screen-shaped sibling: the client posts periodic
    frames of whatever the user is watching/playing/working on, she keeps
    an ephemeral NOTE buffer for continuity and only occasionally SAYs
    anything. One LLM call per frame, notes always advance, the SAY half
    obeys the speak cooldown, and — like every vision path — the image and
    her reaction are never committed to memory.
    """
    if not getattr(getattr(engine.config, "vision", None), "enabled", False):
        return None
    watch_cfg = getattr(engine.config, "watch_together", None)
    if watch_cfg is None:
        return None
    provider = _read_along_provider(engine)
    if provider is None:
        return None

    if engine._watch_together is None:
        engine._watch_together = _WatchTogetherState(
            notes=collections.deque(
                maxlen=max(1, int(getattr(watch_cfg, "max_notes", 8) or 8))
            )
        )
    state = engine._watch_together

    max_frames = int(getattr(watch_cfg, "max_frames_per_session", 240) or 240)
    if max_frames > 0 and state.frames >= max_frames:
        if not engine._watch_cap_logged:
            LOGGER.info(
                "Watch-together session hit max_frames_per_session (%d); "
                "further frames are ignored without an LLM call until "
                "watch_end() clears the session.",
                max_frames,
            )
            engine._watch_cap_logged = True
        return None

    notes_block = _read_along_notes_block(state, header="A few minutes back")
    prompt = [
        ChatMessage(
            role="system",
            content=(
                f"You are {engine.character_full_name}, keeping the user "
                "company while they watch, play, or work on something — "
                "you glance at their screen every minute or so. This is "
                "comfortable co-presence, not commentary: quiet is the "
                "default, but watching together means actually reacting "
                "now and then — a short line every few minutes feels like "
                "company, while total silence reads as absence. Speak when "
                "something earns it — a twist on screen, a win or a fail "
                "in a game, something funny or striking, a clear change "
                "of activity — and if your recent notes show several "
                "interesting glances you never reacted to, lean toward "
                "speaking on this one. "
                + persona_capsule(getattr(engine.config, "character", None))
                + " Respond in EXACTLY this format, nothing else:\n"
                "NOTE: <one factual line, under 25 words, about what is "
                "on screen / what they seem to be doing>\n"
                "SAY: <one short in-character line, under 20 words> "
                "(omit this line on ordinary or unchanged frames — never "
                "narrate, never repeat a reaction you already had)\n"
                "If the screen looks private or sensitive (messages, "
                "banking, credentials), the NOTE must be a neutral "
                "'private screen' and the SAY line must be omitted. "
                "No stage directions, no quotation marks."
                + (f"\n\n{notes_block}" if notes_block else "")
            ),
        ),
        ChatMessage(
            role="user",
            content="(a glance at the screen)",
            images=[ImageContent(mime=image_mime, data=image_data)],
        ),
    ]
    try:
        raw = (
            provider.generate(
                prompt,
                max_tokens=int(getattr(watch_cfg, "comment_max_tokens", 800) or 800),
            )
            or ""
        ).strip()
        telemetry.record_llm_usage(provider, "watch_together")
    except Exception as exc:  # pragma: no cover - provider runtime errors
        LOGGER.warning("Watch-together frame generation failed: %s", exc)
        telemetry.incr("presence.watch_together_failures")
        return None

    note, say = _parse_read_along_reply(raw)
    # Continuity notes advance regardless of the speak cooldown, exactly
    # like read_along — silence must never starve the running context.
    if note:
        state.notes.append(note)
    state.frames += 1

    if not say:
        return None
    cooldown = float(getattr(watch_cfg, "speak_cooldown_seconds", 75.0) or 75.0)
    if not engine._presence_cooldown_ready(state.last_spoke_at, cooldown):
        return None
    state.last_spoke_at = dt.datetime.utcnow()
    return say


def watch_end(engine: "ChatEngine") -> Optional[str]:
    """Close out a "watch together" session with an optional sign-off.

    Mirrors read_along_end: cooldown-exempt, clears the in-RAM state
    unconditionally (repeated calls are safe), and only generates a line
    when at least 5 frames actually landed — a one-glance session doesn't
    deserve a send-off. The opt-in moments distillation (amadeus/moments.py)
    runs regardless of that sign-off gate, keyed only on the notes buffer.
    """
    state = engine._watch_together
    engine._watch_together = None
    engine._watch_cap_logged = False
    if state is None:
        return None
    moments.maybe_store_moment(
        engine, activity="watch-together session", notes=list(state.notes)
    )
    if state.frames < 5:
        return None
    provider = engine.summary_provider or engine.llm
    if provider is None:
        return None
    notes_block = _read_along_notes_block(state, header="Notes from the session")
    prompt = [
        ChatMessage(
            role="system",
            content=(
                f"You are {engine.character_full_name}. You just spent a "
                "while keeping the user company while they watched or "
                "worked on something. Write exactly ONE short in-character "
                "line (under 20 words) wrapping up the shared time — a "
                "little fond is fine. "
                + persona_capsule(getattr(engine.config, "character", None))
                + " You may reference something from the notes below if it "
                "fits naturally. No stage directions, no quotation marks."
                + (f"\n\n{notes_block}" if notes_block else "")
            ),
        ),
        ChatMessage(role="user", content="(watch-together session ended)"),
    ]
    try:
        text = (
            (
                provider.generate(
                    prompt,
                    max_tokens=int(
                        getattr(engine.config.watch_together, "comment_max_tokens", 800)
                        or 800
                    ),
                )
                or ""
            )
            .strip()
            .strip('"')
            .strip()
        )
        telemetry.record_llm_usage(provider, "watch_together")
    except Exception as exc:  # pragma: no cover - provider runtime errors
        LOGGER.warning("Watch-together sign-off generation failed: %s", exc)
        telemetry.incr("presence.watch_together_failures")
        return None
    return text or None
