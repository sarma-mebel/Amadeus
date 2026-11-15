"""Full-duplex voice conversation loop.

Gated by ``web.duplex_enabled`` (config; server-side default ON since the
2026-07-27 promotion — the client still opts in per browser via the
localStorage ``kurisu_duplex`` toggle) — with the flag off the
``/api/chat/duplex`` route (routes_chat.py) answers a single error frame and
closes, and nothing here touches the default POST / ``/api/chat/stream``
paths.

Two layers:

* :class:`DuplexController` — a pure(ish) turn-taking state machine fed
  16-bit mono PCM chunks. It owns endpoint detection (energy VAD with a
  ~600ms silence hangover), the utterance buffer, and barge-in detection
  while the assistant is processing/speaking. Time is derived exclusively
  from the audio itself (chunk length / sample rate), never the wall
  clock, so it is deterministic under test.
* :class:`DuplexSocketRunner` — thin async glue that drives one duplex
  WebSocket end to end, including the open handshake: binary frames feed
  the controller, endpoints run STT (through the SessionStore's normal
  ``get_stt`` surface, so the speculative-recall wrapper applies) and then
  the EXISTING streaming turn orchestrator via ``session.handle_stream`` —
  memory commits exactly once per turn and the outbound frames mirror the
  ``/api/chat/stream`` vocabulary (session / token / sentence / audio /
  thought / tool_progress / done / error) plus duplex-only ``state`` and
  ``transcript`` frames.

Protocol (mirrors, but is not identical to, ``/api/chat/stream``):

1. Client connects, server has already accepted the socket.
2. Client sends one JSON text frame: ``{"type": "start", "session_id"?,
   "user_id"?, "sample_rate"?: 16000, "encoding"?: "pcm16"}``. Only
   ``pcm16`` is supported; a mismatched/missing encoding is a protocol
   error. This mirrors ``/api/chat/stream``'s ``{"type": "chat", ...}``
   first-frame handshake.
3. Server replies ``{"type": "session", "session_id": ..., "duplex": true,
   "sample_rate": ..., "encoding": "pcm16"}`` then ``{"type": "state",
   "state": "listening"}``.
4. Client streams binary PCM16 mono frames continuously (mic capture,
   including while she is speaking — barge-in relies on that).
5. On endpoint detection the server transcribes, runs one chat turn through
   the shared orchestrator, and streams back the same frame vocabulary as
   ``/api/chat/stream`` (token/sentence/audio/tool_progress/thought/done),
   interleaved with ``state`` frames tracking listening -> user_speaking ->
   processing -> assistant_speaking -> listening.
6. Text control frames: ``{"type": "stop"}`` ends the session;
   ``{"type": "cancel"}`` is a manual barge-in.

VAD seam: the energy endpointer below is deliberately a self-contained
strategy. A model-based VAD (e.g. silero-vad as an OPTIONAL extra — do not
add it as a required dependency) can slot in by replacing
``DuplexController._rms`` + the threshold logic with per-chunk speech
probabilities; the state machine and the runner do not care where the
"voiced?" verdict comes from.

Client-side echo cancellation is assumed (the frontend mic paths already
request ``echoCancellation``); the stricter barge-in threshold + warmup
window guard against the residual echo burst before the AEC converges,
mirroring the frontend detector in ``frontend/src/audio/vad.ts``.
"""

from __future__ import annotations

import array
import asyncio
import dataclasses
import json
import math
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from ... import telemetry
from ...logging_utils import get_logger
from ..blocking import run_blocking as _run_blocking
from ..models import ChatRequest

LOGGER = get_logger(__name__)

DUPLEX_ENCODING = "pcm16"
DEFAULT_SAMPLE_RATE = 16000

# Endpointer thresholds — same rationale (and values) as the frontend
# hands-free endpointer (frontend/src/audio/handsFree.ts): AGC'd speech on
# an echo-cancelled capture path sits around 0.05-0.2 RMS, room noise under
# 0.01. The hangover is the task's ~600ms "finished talking" silence.
VOICE_THRESHOLD = 0.015
RELEASE_THRESHOLD = 0.010
MIN_VOICED_MS = 250.0
HANGOVER_MS = 600.0
MIN_UTTERANCE_MS = 600.0
MAX_UTTERANCE_MS = 30_000.0
PRE_ROLL_MS = 300.0

# Barge-in (while she is processing/speaking) — stricter than the
# endpointer so residual echo through laptop speakers can't self-trigger.
BARGE_IN_THRESHOLD = 0.045
BARGE_IN_MIN_VOICED_MS = 250.0
BARGE_IN_WARMUP_MS = 500.0

# Verbal backchannels while the USER is speaking ("un", "hm" — the
# turn-taking rhythm cue that distinguishes a person from a walkie-talkie).
# Cues fire at micro-pauses (a short dip that hasn't reached the endpoint
# hangover), only in long-enough utterances, rate-limited per utterance —
# humans backchannel at clause boundaries in sustained speech, not during
# quick exchanges. All timing is audio-derived (deterministic under test).
BACKCHANNEL_LEAD_MS = 2500.0
BACKCHANNEL_GAP_MS = 4000.0
BACKCHANNEL_MAX_PER_UTTERANCE = 2
BACKCHANNEL_DIP_MS = 180.0


@dataclasses.dataclass
class UserSpeaking:
    """The user's voice onset was confirmed (min sustained voiced audio)."""


@dataclasses.dataclass
class Endpoint:
    """One finished utterance: buffered PCM (pre-roll + speech + hangover)."""

    pcm: bytes
    utterance_ms: float
    reason: str  # "silence" | "max-utterance"


@dataclasses.dataclass
class BargeIn:
    """Voiced audio while processing/speaking — cancel the in-flight turn.

    The controller has already transitioned itself to ``user_speaking`` and
    seeded the new utterance buffer when this is emitted.
    """


@dataclasses.dataclass
class BackchannelCue:
    """A micro-pause in sustained user speech — a natural spot for a short
    verbal acknowledgement ("un", "hm"). Advisory: the glue may drop it
    (config off), and the frontend plays a pre-rendered clip only when its
    backchannel assets exist."""


DuplexAction = Any  # UserSpeaking | Endpoint | BargeIn | BackchannelCue


class DuplexController:
    """Turn-taking state machine over a continuous PCM16 mono stream.

    Public ``state``: ``listening`` -> ``user_speaking`` -> ``processing``
    -> ``assistant_speaking`` -> ``listening``, with barge-in transitions
    from ``processing``/``assistant_speaking`` back to ``user_speaking``.
    ``feed`` consumes one binary chunk and returns the actions the glue
    must take; ``assistant_started``/``turn_finished`` are the two external
    notifications (first audio frame emitted / done frame emitted).
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        *,
        voice_threshold: float = VOICE_THRESHOLD,
        release_threshold: float = RELEASE_THRESHOLD,
        min_voiced_ms: float = MIN_VOICED_MS,
        hangover_ms: float = HANGOVER_MS,
        min_utterance_ms: float = MIN_UTTERANCE_MS,
        max_utterance_ms: float = MAX_UTTERANCE_MS,
        pre_roll_ms: float = PRE_ROLL_MS,
        barge_in_threshold: float = BARGE_IN_THRESHOLD,
        barge_in_min_voiced_ms: float = BARGE_IN_MIN_VOICED_MS,
        barge_in_warmup_ms: float = BARGE_IN_WARMUP_MS,
        backchannel_lead_ms: float = BACKCHANNEL_LEAD_MS,
        backchannel_gap_ms: float = BACKCHANNEL_GAP_MS,
        backchannel_max_per_utterance: int = BACKCHANNEL_MAX_PER_UTTERANCE,
        backchannel_dip_ms: float = BACKCHANNEL_DIP_MS,
    ) -> None:
        self.sample_rate = max(1, int(sample_rate))
        self.voice_threshold = voice_threshold
        # Release can never exceed acquire or the hysteresis inverts.
        self.release_threshold = min(release_threshold, voice_threshold)
        self.min_voiced_ms = min_voiced_ms
        self.hangover_ms = hangover_ms
        self.min_utterance_ms = min_utterance_ms
        self.max_utterance_ms = max_utterance_ms
        self.pre_roll_ms = pre_roll_ms
        self.barge_in_threshold = barge_in_threshold
        self.barge_in_min_voiced_ms = barge_in_min_voiced_ms
        self.barge_in_warmup_ms = barge_in_warmup_ms
        self.backchannel_lead_ms = backchannel_lead_ms
        # The dip must sit clear of the hangover, or a cue would fire on
        # every endpoint's way out.
        self.backchannel_dip_ms = min(backchannel_dip_ms, hangover_ms * 0.5)
        self.backchannel_gap_ms = backchannel_gap_ms
        self.backchannel_max_per_utterance = max(0, int(backchannel_max_per_utterance))
        # Semantic end-of-turn layer (smart_turn.py), assigned after
        # construction by the session glue so existing controller-factory
        # fakes keep their signature. predictor(pcm16, sample_rate) ->
        # True (finished) / False (mid-thought: hold the endpoint open up
        # to smart_turn_hold_ms of total silence) / None (unavailable —
        # behave as if the layer didn't exist) — or a concurrent.futures
        # Future resolving to one of those (ThreadedTurnPredictor: the
        # consult runs off the event loop; the endpoint stays open as
        # "pending" until it resolves, bounded by smart_turn_hold_ms).
        # Consulted ONCE per pause, at the moment the energy hangover
        # would have endpointed.
        self.turn_predictor: Optional[Any] = None
        self.smart_turn_hold_ms: float = 2400.0

        self.state = "listening"
        # Internal capture phase within listening: idle -> pending (onset
        # candidate, not yet believed) -> confirmed (== user_speaking).
        self._phase = "idle"
        self._buffer = bytearray()
        # (chunk, dt_ms) ring covering pre-roll (listening) or the barge-in
        # onset run (processing/assistant_speaking), so a confirmed onset
        # never loses its own first syllable.
        self._recent: Deque[tuple[bytes, float]] = deque()
        self._recent_ms = 0.0
        self._pending_run_ms = 0.0
        self._voiced_total_ms = 0.0
        self._silence_ms = 0.0
        self._utterance_ms = 0.0
        # Barge-in tracker (only consulted in processing/assistant_speaking).
        self._bi_warmup_target = 0.0
        self._bi_warmed_ms = 0.0
        self._bi_voiced_ms = 0.0
        # Backchannel tracker (only consulted while user_speaking).
        self._bc_count = 0
        self._bc_last_at_ms: Optional[float] = None
        # Smart-turn per-pause tracker: checked-once flag + active hold +
        # (for future-shaped predictors) the in-flight verdict being awaited.
        self._smart_checked = False
        self._smart_holding = False
        self._smart_pending: Optional[Any] = None

    # -- external notifications from the glue -------------------------------

    def assistant_started(self) -> None:
        """First audio frame of the reply was emitted — she is speaking.

        Resets the barge-in warmup: the first moments of her speech carry a
        residual-echo burst before the client AEC converges.
        """
        if self.state == "processing":
            self.state = "assistant_speaking"
            self._reset_barge_in(warmup_ms=self.barge_in_warmup_ms)

    def turn_finished(self) -> None:
        """The turn's done frame was emitted. Back to listening — unless a
        barge-in already moved us into the next utterance, which must not be
        clobbered by the cancelled turn's own (later) done frame."""
        if self.state in ("processing", "assistant_speaking"):
            self.state = "listening"
            self._phase = "idle"
            self._buffer.clear()
            self._clear_recent()
            self._pending_run_ms = 0.0
            self._voiced_total_ms = 0.0
            self._silence_ms = 0.0

    # -- feed ----------------------------------------------------------------

    def feed(self, chunk: bytes) -> List[DuplexAction]:
        """Consume one PCM16 chunk; returns actions for the glue layer."""
        if len(chunk) < 2:
            return []
        rms = self._rms(chunk)
        dt_ms = 1000.0 * (len(chunk) // 2) / self.sample_rate
        if self.state in ("processing", "assistant_speaking"):
            return self._feed_barge_in(chunk, rms, dt_ms)
        return self._feed_capture(chunk, rms, dt_ms)

    # -- capture path (listening / user_speaking) ---------------------------

    def _feed_capture(
        self, chunk: bytes, rms: float, dt_ms: float
    ) -> List[DuplexAction]:
        actions: List[DuplexAction] = []
        if self._phase == "idle":
            if rms >= self.voice_threshold:
                self._phase = "pending"
                # Onset candidate: promote the pre-roll into the utterance
                # buffer so the first syllable isn't clipped.
                self._buffer.clear()
                for kept, _dt in self._recent:
                    self._buffer.extend(kept)
                self._clear_recent()
                self._buffer.extend(chunk)
                self._pending_run_ms = dt_ms
                self._voiced_total_ms = dt_ms
                self._silence_ms = 0.0
                self._utterance_ms = dt_ms
                # Fall through to the pending-confirmation check below rather
                # than returning immediately: a single chunk whose OWN
                # duration already meets min_voiced_ms (a coarse capture
                # buffer, or a hangover-sized synthetic chunk) must confirm
                # the onset in this same feed() call, not one call later —
                # deferring it a full call lags every state transition (and
                # the resulting Endpoint/BargeIn actions) by one chunk.
                if self._pending_run_ms >= self.min_voiced_ms:
                    self._phase = "confirmed"
                    self.state = "user_speaking"
                    actions.append(UserSpeaking())
            else:
                self._push_recent(chunk, dt_ms, self.pre_roll_ms)
            return actions

        if self._phase == "pending":
            self._buffer.extend(chunk)
            self._utterance_ms += dt_ms
            if rms >= self.release_threshold:
                self._pending_run_ms += dt_ms
                self._voiced_total_ms += dt_ms
                if self._pending_run_ms >= self.min_voiced_ms:
                    self._phase = "confirmed"
                    self.state = "user_speaking"
                    actions.append(UserSpeaking())
            else:
                # Blip (cough/click) — discard silently, back to idle.
                self._settle_to_listening()
            return actions

        # _phase == "confirmed" (state == user_speaking)
        self._buffer.extend(chunk)
        self._utterance_ms += dt_ms
        if self._utterance_ms >= self.max_utterance_ms:
            actions.append(self._endpoint("max-utterance"))
            return actions
        if rms >= self.release_threshold:
            self._silence_ms = 0.0
            self._voiced_total_ms += dt_ms
            # Speech resumed: any smart-turn hold did its job (the pause
            # was indeed mid-thought); the next pause gets a fresh check.
            # A still-in-flight verdict is dropped — its answer no longer
            # matters for a pause that just ended.
            self._smart_checked = False
            self._smart_holding = False
            self._smart_pending = None
            return actions
        prev_silence_ms = self._silence_ms
        self._silence_ms += dt_ms
        # Backchannel cue: fires exactly once as the silence run CROSSES the
        # dip threshold (crossing detection, not level detection, so one
        # pause can't emit a cue per chunk) — a micro-pause well short of the
        # endpoint hangover, in an utterance long enough to be sustained
        # speech, rate-limited per utterance.
        if (
            prev_silence_ms < self.backchannel_dip_ms <= self._silence_ms
            and self._bc_count < self.backchannel_max_per_utterance
            and self._voiced_total_ms >= self.backchannel_lead_ms
            and (
                self._bc_last_at_ms is None
                or self._utterance_ms - self._bc_last_at_ms >= self.backchannel_gap_ms
            )
        ):
            self._bc_count += 1
            self._bc_last_at_ms = self._utterance_ms
            actions.append(BackchannelCue())
        if self._silence_ms < self.hangover_ms:
            return actions
        # Hangover elapsed: endpoint or discard.
        if self._voiced_total_ms >= self.min_utterance_ms:
            # Semantic layer: at the moment the energy VAD would endpoint,
            # ask smart-turn whether the utterance actually sounds finished.
            # "Mid-thought" holds the endpoint open (bounded by
            # smart_turn_hold_ms of total silence — a real stop still ends
            # the turn, just later); True/None endpoint immediately.
            if (
                self.turn_predictor is not None
                and not self._smart_checked
                and self.smart_turn_hold_ms > self.hangover_ms
            ):
                self._smart_checked = True
                try:
                    verdict = self.turn_predictor(bytes(self._buffer), self.sample_rate)
                except Exception:  # pragma: no cover - predictor guards itself
                    verdict = None
                if callable(getattr(verdict, "done", None)):
                    # Future-shaped verdict (ThreadedTurnPredictor): the
                    # ~100ms ONNX consult is running OFF the event loop
                    # instead of blocking feed(). Hold the endpoint open as
                    # "pending" and poll it on the following chunks; the
                    # checked-once contract is unchanged (the consult itself
                    # already fired).
                    self._smart_pending = verdict
                elif verdict is False:
                    telemetry.incr("web.duplex_smart_turn_holds")
                    self._smart_holding = True
            if self._smart_pending is not None:
                fut = self._smart_pending
                if not fut.done():
                    if self._silence_ms >= self.smart_turn_hold_ms:
                        # Inference never came back by the hold deadline —
                        # degrade to the energy-only endpoint rather than
                        # holding the turn open forever behind a wedged
                        # predictor.
                        telemetry.incr("web.duplex_smart_turn_pending_timeouts")
                        self._smart_pending = None
                        actions.append(self._endpoint("smart-turn-timeout"))
                    # Still waiting on inference: keep the endpoint open (a
                    # healthy consult resolves within a chunk or two).
                    return actions
                self._smart_pending = None
                try:
                    verdict = fut.result()
                except Exception:  # pragma: no cover - predictor guards itself
                    verdict = None
                if verdict is False:
                    telemetry.incr("web.duplex_smart_turn_holds")
                    self._smart_holding = True
            if self._smart_holding:
                if self._silence_ms < self.smart_turn_hold_ms:
                    return actions
                telemetry.incr("web.duplex_smart_turn_hold_timeouts")
                actions.append(self._endpoint("smart-turn-timeout"))
                return actions
            actions.append(self._endpoint("silence"))
        else:
            # Too little voiced audio to be speech — drop it, keep listening.
            self._settle_to_listening()
        return actions

    def _endpoint(self, reason: str) -> Endpoint:
        action = Endpoint(
            pcm=bytes(self._buffer), utterance_ms=self._utterance_ms, reason=reason
        )
        self._buffer.clear()
        self._clear_recent()
        self._phase = "idle"
        self._pending_run_ms = 0.0
        self._voiced_total_ms = 0.0
        self._silence_ms = 0.0
        self._utterance_ms = 0.0
        self._bc_count = 0
        self._bc_last_at_ms = None
        self._smart_checked = False
        self._smart_holding = False
        self._smart_pending = None
        self.state = "processing"
        if reason == "max-utterance":
            # The cap cut the speaker off MID-SENTENCE: with the usual
            # warmup_ms=0.0 their continued speech would trip barge-in
            # within ~250ms and pre-cancel the very turn answering the 30s
            # they already said — transcribed, rolled back, never answered.
            # Suppress barge-in entirely until she actually starts speaking
            # (assistant_started() re-arms it with the normal echo warmup);
            # from then on, talking over her is a genuine barge-in again.
            self._reset_barge_in(warmup_ms=math.inf)
        else:
            # No echo is playing yet (she hasn't started speaking), so
            # barge-in needs no warmup here; assistant_started() re-arms it.
            self._reset_barge_in(warmup_ms=0.0)
        return action

    def _settle_to_listening(self) -> None:
        self._phase = "idle"
        self.state = "listening"
        self._buffer.clear()
        self._clear_recent()
        self._pending_run_ms = 0.0
        self._voiced_total_ms = 0.0
        self._silence_ms = 0.0
        self._utterance_ms = 0.0
        self._bc_count = 0
        self._bc_last_at_ms = None
        self._smart_checked = False
        self._smart_holding = False
        self._smart_pending = None

    # -- barge-in path (processing / assistant_speaking) --------------------

    def _feed_barge_in(
        self, chunk: bytes, rms: float, dt_ms: float
    ) -> List[DuplexAction]:
        # Keep a rolling window so the barge-in utterance keeps its onset.
        self._push_recent(chunk, dt_ms, self.pre_roll_ms + self.barge_in_min_voiced_ms)
        if self._bi_warmed_ms < self._bi_warmup_target:
            self._bi_warmed_ms += dt_ms
            return []
        if rms >= self.barge_in_threshold:
            self._bi_voiced_ms += dt_ms
            if self._bi_voiced_ms >= self.barge_in_min_voiced_ms:
                # Triggered: the new utterance starts NOW, seeded from the
                # rolling window (which contains the voiced run itself).
                self._buffer.clear()
                self._utterance_ms = 0.0
                for kept, kept_dt in self._recent:
                    self._buffer.extend(kept)
                    self._utterance_ms += kept_dt
                self._clear_recent()
                self._phase = "confirmed"
                self.state = "user_speaking"
                self._voiced_total_ms = self._bi_voiced_ms
                self._silence_ms = 0.0
                self._pending_run_ms = 0.0
                self._bc_count = 0
                self._bc_last_at_ms = None
                return [BargeIn()]
        else:
            self._bi_voiced_ms = 0.0
        return []

    def _reset_barge_in(self, *, warmup_ms: float) -> None:
        self._bi_warmup_target = warmup_ms
        self._bi_warmed_ms = 0.0
        self._bi_voiced_ms = 0.0
        self._clear_recent()

    # -- helpers -------------------------------------------------------------

    def _push_recent(self, chunk: bytes, dt_ms: float, cap_ms: float) -> None:
        self._recent.append((chunk, dt_ms))
        self._recent_ms += dt_ms
        while self._recent and self._recent_ms > cap_ms:
            _old, old_dt = self._recent.popleft()
            self._recent_ms -= old_dt

    def _clear_recent(self) -> None:
        self._recent.clear()
        self._recent_ms = 0.0

    @staticmethod
    def _rms(chunk: bytes) -> float:
        """Normalized RMS of a PCM16-LE chunk (0..1). Stdlib only — this is
        the seam a model-based VAD (silero) would replace."""
        samples = array.array("h")
        usable = len(chunk) - (len(chunk) % 2)
        samples.frombytes(chunk[:usable])
        if not samples:
            return 0.0
        total = 0
        for value in samples:
            total += value * value
        return math.sqrt(total / len(samples)) / 32768.0


class DuplexProtocolError(Exception):
    """The client's open handshake was missing/malformed."""


class DuplexSocketRunner:
    """Drive one accepted duplex WebSocket to completion, handshake included.

    The websocket only needs ``receive()`` (starlette ASGI-message shaped: a
    dict with ``type`` and optional ``bytes``/``text``) and ``send_json`` —
    tests drive it with a fake. All turn orchestration is delegated to
    ``session.handle_stream`` (the same seam ``/api/chat/stream`` uses), so
    cancel truthfulness — rollback on no-emitted-text cancels, emitted-only
    commits with the ``[interrupted]`` marker — is inherited, not
    reimplemented.

    ``store`` is the ``SessionStore``: session identity (``session_id``/
    ``user_id``) is not known until the client's open handshake arrives, so
    unlike ``/api/chat/stream`` (whose route resolves the session before
    constructing any per-turn glue) this class owns ``store.get(...)``
    itself, inside :meth:`run`.
    """

    def __init__(
        self,
        websocket: Any,
        store: Any,
        config: Any,
        *,
        controller_factory: Any = None,
    ) -> None:
        self.websocket = websocket
        self.store = store
        self.config = config
        self._controller_factory = controller_factory or DuplexController
        self.controller: Optional[DuplexController] = None
        self.session: Any = None
        self.session_id: Optional[str] = None
        self.user_id: Optional[str] = None
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self._send_lock = asyncio.Lock()
        self._dead = False
        # Monotonically increasing turn generation: stale frames from a
        # cancelled turn must not flip the controller's state after a
        # barge-in already moved it into the next utterance.
        self._generation = 0
        self._cancel_event: Optional[threading.Event] = None
        self._turn_tasks: set = set()

    async def emit(self, frame: Dict[str, Any]) -> None:
        async with self._send_lock:
            if self._dead:
                return
            try:
                await self.websocket.send_json(frame)
            except Exception:  # pragma: no cover - client already gone
                # Same degrade-loudly contract as the stream endpoint's
                # emit: a dead transport self-cancels the in-flight turn
                # and is counted, never silent.
                self._dead = True
                telemetry.incr("web.duplex_send_failures")
                if self._cancel_event is not None:
                    self._cancel_event.set()

    async def run(self) -> None:
        """Run the open handshake, then the receive loop, to completion.

        Returns normally on a clean stop/disconnect. Raises
        :class:`DuplexProtocolError` if the handshake never arrives or is
        malformed — the caller (routes_chat.py) is expected to have already
        checked ``web.duplex_enabled``/auth/rate-limit before calling this,
        so a protocol error here is purely a bad client.
        """
        if not await self._handshake():
            return
        telemetry.incr("web.duplex_sessions")
        await self.emit({"type": "state", "state": "listening"})
        try:
            while not self._dead:
                try:
                    message = await self.websocket.receive()
                except Exception:
                    break
                msg_type = message.get("type")
                if msg_type == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data:
                    self._handle_audio(data)
                    continue
                text = message.get("text")
                if not text:
                    continue
                if not self._handle_text(text):
                    break
        finally:
            # A disconnect mid-turn is a barge-in nobody will hear: cancel
            # so the existing machinery finalizes truthfully (partial commit
            # or rollback), then wait for the turn tasks to finish their
            # memory writes before the socket context unwinds.
            if self._cancel_event is not None:
                self._cancel_event.set()
            for task in list(self._turn_tasks):
                try:
                    await task
                except Exception:  # pragma: no cover - task logged its own
                    pass

    # -- handshake -------------------------------------------------------

    async def _handshake(self) -> bool:
        """Receive+validate the client's open frame, resolve the session,
        and reply with the session frame. Returns False (after emitting an
        error frame) on any protocol violation."""
        try:
            message = await self.websocket.receive()
        except Exception:
            return False
        text = message.get("text") if isinstance(message, dict) else None
        if not text:
            await self.emit(
                {
                    "type": "error",
                    "message": "First frame must be a JSON {type: 'start', ...} text frame.",
                    "kind": "protocol_error",
                }
            )
            return False
        try:
            raw = json.loads(text)
        except Exception:
            await self.emit(
                {
                    "type": "error",
                    "message": "First frame is not valid JSON.",
                    "kind": "protocol_error",
                }
            )
            return False
        if not isinstance(raw, dict) or raw.get("type") != "start":
            await self.emit(
                {
                    "type": "error",
                    "message": "First frame must be {type: 'start', ...}.",
                    "kind": "protocol_error",
                }
            )
            return False
        encoding = str(raw.get("encoding") or DUPLEX_ENCODING).lower()
        if encoding != DUPLEX_ENCODING:
            await self.emit(
                {
                    "type": "error",
                    "message": f"Unsupported encoding '{encoding}'; only 'pcm16' is supported.",
                    "kind": "protocol_error",
                }
            )
            return False
        try:
            sample_rate = int(raw.get("sample_rate") or DEFAULT_SAMPLE_RATE)
        except (TypeError, ValueError):
            sample_rate = DEFAULT_SAMPLE_RATE
        if sample_rate <= 0:
            sample_rate = DEFAULT_SAMPLE_RATE
        self.sample_rate = sample_rate
        self.user_id = raw.get("user_id")
        requested_session_id = raw.get("session_id")

        self.session_id, self.session = self.store.get(
            requested_session_id, self.user_id
        )
        self.controller = self._controller_factory(sample_rate=self.sample_rate)
        # Semantic endpoint layer: assigned post-construction (never passed
        # through the factory) so test fakes with a (sample_rate)-only
        # signature keep working. Missing model/onnxruntime → None → the
        # controller behaves exactly as before.
        if hasattr(self.controller, "turn_predictor"):
            from .smart_turn import ThreadedTurnPredictor, load_smart_turn_predictor

            # Wrapped so the ~100ms CPU inference (plus the numpy log-mel
            # over up to 8s of audio) runs on a worker thread: raw predictor
            # calls inside controller.feed() would block the entire event
            # loop at every end-of-utterance pause. The controller handles
            # the future-shaped verdict (see turn_predictor's contract).
            predictor = load_smart_turn_predictor(self.config)
            self.controller.turn_predictor = (
                ThreadedTurnPredictor(predictor) if predictor is not None else None
            )
            self.controller.smart_turn_hold_ms = float(
                getattr(
                    getattr(self.config, "web", None),
                    "duplex_smart_turn_hold_ms",
                    2400.0,
                )
                or 2400.0
            )

        await self.emit(
            {
                "type": "session",
                "session_id": self.session_id,
                "duplex": True,
                "sample_rate": self.sample_rate,
                "encoding": DUPLEX_ENCODING,
            }
        )
        return True

    # -- inbound -------------------------------------------------------------

    def _handle_audio(self, data: bytes) -> None:
        assert self.controller is not None
        for action in self.controller.feed(data):
            if isinstance(action, Endpoint):
                self._start_turn(action)
            elif isinstance(action, BargeIn):
                self._on_barge_in()
            elif isinstance(action, UserSpeaking):
                self._spawn(self.emit({"type": "state", "state": "user_speaking"}))
            elif isinstance(action, BackchannelCue):
                # Advisory frame only — the client owns the actual sound
                # (pre-rendered clip, skipped when its assets don't exist),
                # so a server cue can never talk over synthesis it started.
                if getattr(
                    getattr(self.config, "web", None), "duplex_backchannels", True
                ):
                    telemetry.incr("web.duplex_backchannel_cues")
                    self._spawn(self.emit({"type": "backchannel"}))

    def _handle_text(self, text: str) -> bool:
        """Returns False when the session should end."""
        try:
            msg = json.loads(text)
        except Exception:
            return True
        kind = msg.get("type") if isinstance(msg, dict) else None
        if kind == "stop":
            return False
        if kind == "cancel":
            # Manual barge-in (e.g. the client's own stop button). Mirrors
            # _on_barge_in: bumping the generation supersedes a turn still
            # inside its STT window — _run_turn only binds _cancel_event
            # AFTER transcription returns, so setting the event alone was a
            # no-op for the whole 0.5-2s STT wait. The superseded turn then
            # starts pre-cancelled and its user message is rolled back.
            if self.controller is not None and self.controller.state in (
                "processing",
                "assistant_speaking",
            ):
                self._generation += 1
                if self._cancel_event is not None:
                    self._cancel_event.set()
                # Unlike a voice barge-in (where the controller already moved
                # itself into the next utterance), nothing else returns the
                # controller to listening now that the cancelled turn's own
                # done frame is generation-gated out of _end_turn_to_listening.
                self._spawn(self._end_turn_to_listening(self._generation))
        return True

    def _on_barge_in(self) -> None:
        telemetry.incr("web.duplex_barge_ins")
        # Supersede the in-flight turn even if it hasn't created its cancel
        # event yet (barge-in while STT is still transcribing): bumping the
        # generation makes that turn start pre-cancelled, so its user
        # message is rolled back instead of committed for a reply the user
        # already talked over.
        self._generation += 1
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._spawn(self.emit({"type": "state", "state": "user_speaking"}))

    # -- turns ---------------------------------------------------------------

    def _start_turn(self, endpoint: Endpoint) -> None:
        self._generation += 1
        task = asyncio.get_running_loop().create_task(
            self._run_turn(endpoint, self._generation, time.monotonic())
        )
        self._turn_tasks.add(task)
        task.add_done_callback(self._turn_tasks.discard)

    def _spawn(self, coro: Any) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._turn_tasks.add(task)
        task.add_done_callback(self._turn_tasks.discard)

    async def _end_turn_to_listening(self, generation: int) -> None:
        if generation == self._generation:
            self.controller.turn_finished()
            if self.controller.state == "listening":
                await self.emit({"type": "state", "state": "listening"})

    async def _run_turn(
        self, endpoint: Endpoint, generation: int, endpointed_at: float
    ) -> None:
        await self.emit({"type": "state", "state": "processing"})
        try:
            stt = await asyncio.to_thread(self.store.get_stt)
        except Exception:  # pragma: no cover - loader is defensive already
            stt = None
        if stt is None:
            telemetry.incr("web.duplex_stt_unavailable")
            await self.emit(
                {
                    "type": "error",
                    "message": "Speech-to-text is not available on this server.",
                    "kind": "stt_unavailable",
                }
            )
            await self._end_turn_to_listening(generation)
            return
        try:
            # Through the store's normal STT surface — the speculative-recall
            # wrapper pre-starts the memory lookup off this transcript.
            text = await _run_blocking(
                stt.transcribe_audio, endpoint.pcm, self.sample_rate
            )
        except Exception as exc:
            LOGGER.warning("Duplex STT transcription failed: %s", exc)
            telemetry.incr("web.duplex_stt_failures")
            await self.emit(
                {
                    "type": "error",
                    "message": f"Transcription failed: {exc}",
                    "kind": "stt_error",
                }
            )
            await self._end_turn_to_listening(generation)
            return
        # Endpoint-detected -> transcript-ready latency for this utterance.
        telemetry.record_duration(
            "web.duplex_endpoint_ms", 1000.0 * (time.monotonic() - endpointed_at)
        )
        text = (text or "").strip()
        if not text:
            telemetry.incr("web.duplex_empty_transcripts")
            await self._end_turn_to_listening(generation)
            return
        await self.emit({"type": "transcript", "text": text})

        request = ChatRequest(
            message=text,
            session_id=self.session_id,
            user_id=self.user_id,
            audio_enabled=True,
        )
        cancel_event = threading.Event()
        if generation == self._generation:
            self._cancel_event = cancel_event
        else:
            # A newer utterance already superseded this one before its turn
            # even started (barge-in during STT): run it cancelled so the
            # user message is never committed for a reply nobody wanted.
            cancel_event.set()

        first_audio_seen = False

        async def turn_emit(frame: Dict[str, Any]) -> None:
            nonlocal first_audio_seen
            if (
                frame.get("type") == "audio"
                and not first_audio_seen
                and generation == self._generation
            ):
                first_audio_seen = True
                self.controller.assistant_started()
                await self.emit({"type": "state", "state": "assistant_speaking"})
            await self.emit(frame)
            if frame.get("type") == "done":
                await self._end_turn_to_listening(generation)

        try:
            # handle_stream serializes on the session's handle_lock, commits
            # memory exactly once via the shared _prepare_turn/_finalize
            # seams, and already implements cancel truthfulness.
            await self.session.handle_stream(request, turn_emit, cancel_event)
        except Exception as exc:
            LOGGER.error("Duplex turn failed: %s", exc)
            telemetry.incr("web.duplex_turn_failures")
            from ..routes_chat import _provider_error_kind

            await self.emit(
                {
                    "type": "error",
                    "message": str(exc),
                    "kind": _provider_error_kind(exc),
                }
            )
            await self._end_turn_to_listening(generation)
