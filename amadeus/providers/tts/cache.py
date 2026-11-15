"""Disk cache wrapper for TTS providers.

Canned/recurring lines (greetings, presence lines, short fillers) get
re-synthesized on every turn even though the audio never changes. This
wrapper hashes the synthesis inputs and caches the resulting WAV bytes on
disk, so a slow GPU provider like Qwen3-TTS only pays its ~5-10s cost once
per distinct short line.

Only short texts are cached (``max_chars``) — long, mostly-unique LLM
replies would otherwise just fill the disk with one-shot entries.
"""

from __future__ import annotations

import hashlib
import io
import wave
from pathlib import Path
from typing import Any, Iterator

from ... import telemetry
from ...logging_utils import get_logger
from .base import TTSProvider

LOGGER = get_logger(__name__)


class CachedTTSProvider(TTSProvider):
    """Wraps a TTSProvider with a disk cache keyed on the synthesis inputs.

    Delegates everything to the wrapped provider except ``synthesize``
    (and ``synthesize_async`` when present), which are cached on disk.
    """

    def __init__(
        self,
        provider: Any,
        *,
        cache_dir: str = "cache/tts",
        max_chars: int = 80,
    ) -> None:
        self.provider = provider
        self.cache_dir = Path(cache_dir)
        self.max_chars = max_chars
        self._signature = getattr(provider, "cache_signature", type(provider).__name__)

        # Only expose synthesize_async when the wrapped provider has one —
        # web/app.py's `_synthesize_tts` uses `getattr(tts, "synthesize_async", None)`
        # to decide whether to await it directly or run `synthesize` in a
        # thread, so the attribute must be absent (not merely None) when
        # unsupported.
        inner_async = getattr(provider, "synthesize_async", None)
        if inner_async is not None:
            self.synthesize_async = self._make_synthesize_async(inner_async)

        # Same feature-detection story as synthesize_async: only expose
        # synthesize_stream when the wrapped provider actually has one, so
        # TTSPipeline's getattr(tts, "synthesize_stream", None) check works
        # through the cache wrapper transparently.
        inner_stream = getattr(provider, "synthesize_stream", None)
        if inner_stream is not None:
            self.synthesize_stream = self._make_synthesize_stream(inner_stream)

        # Emotion-conditioned synthesis passes through transparently:
        # callers feature-detect on available_emotions, so it must be
        # forwarded, and an emotion changes the audio, so it must key the
        # cache. Labels outside the set are dropped here (they wouldn't
        # change the wrapped provider's output, so keying on them would
        # only fragment the cache).
        self.available_emotions = frozenset(
            getattr(provider, "available_emotions", ()) or ()
        )

    def refresh_signature(self) -> None:
        """Re-read the wrapped provider's cache_signature and emotion set.
        Required after an in-place voice rebind (qwen3 rebind_voice): both
        are captured at construction, and a stale signature would replay the
        previous voice's cached audio for the same text."""
        self._signature = getattr(
            self.provider, "cache_signature", type(self.provider).__name__
        )
        self.available_emotions = frozenset(
            getattr(self.provider, "available_emotions", ()) or ()
        )

    def _resolve_emotion(self, emotion: str | None) -> str | None:
        if emotion and emotion in self.available_emotions:
            return emotion
        return None

    def _cache_key(
        self,
        text: str,
        speaking_rate: float,
        pitch_shift: float,
        emotion: str | None = None,
    ) -> str:
        parts = [
            str(self._signature),
            text,
            repr(speaking_rate),
            repr(pitch_shift),
        ]
        # Appended only when set, so every pre-emotion cache entry keeps
        # its key. NOTE: the label, not the clip content, is what's keyed —
        # re-recording an emotion ref means clearing cache/tts by hand.
        if emotion is not None:
            parts.append(f"emotion={emotion}")
        payload = "\x1f".join(parts)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.wav"

    def _read_cache(self, path: Path) -> bytes | None:
        if not path.exists():
            return None
        try:
            data = path.read_bytes()
        except OSError as exc:
            LOGGER.warning("TTS cache read failed for %s: %s", path, exc)
            return None
        if not data:
            LOGGER.warning("TTS cache file %s is empty; will resynthesize", path)
            return None
        return data

    def _write_cache(self, path: Path, data: bytes) -> None:
        # Ask the wrapped provider to judge the exact bytes about to become
        # permanent (feature-detected, same pattern as synthesize_async).
        # A drift-guarded provider can return audio that FAILED its guard
        # (best_of exhausted = "best of three bad rolls"; streamed shorts
        # below the prefix-check length) — caching that pins a wrong-voice
        # render forever, which the 2026-07-28 cache audit found had
        # actually happened (a 6.7s clip at 0.60 similarity replaying on
        # every hit). Judging only at write time keeps the cost off the
        # cache-hit path, and judging the bytes themselves (not a
        # last-call flag) stays correct under the pipeline's concurrent
        # sentence synthesis.
        judge = getattr(self.provider, "judge_speaker", None)
        if judge is not None:
            try:
                verdict = judge(data)
            except Exception:  # pragma: no cover - judge must not break caching
                verdict = None
            if verdict is False:
                telemetry.incr("tts.speaker_guard_uncached")
                LOGGER.warning(
                    "TTS cache write skipped for %s: clip failed the speaker "
                    "guard (wrong-voice render); it will play once but not stick",
                    path.name,
                )
                return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        except OSError as exc:
            LOGGER.warning("TTS cache write failed for %s: %s", path, exc)

    def synthesize(
        self,
        text: str,
        *,
        speaking_rate: float = 1.0,
        pitch_shift: float = 0.0,
        emotion: str | None = None,
    ) -> bytes:
        emotion = self._resolve_emotion(emotion)
        kwargs = {"speaking_rate": speaking_rate, "pitch_shift": pitch_shift}
        if emotion is not None:
            kwargs["emotion"] = emotion
        if len(text) > self.max_chars:
            return self.provider.synthesize(text, **kwargs)

        key = self._cache_key(text, speaking_rate, pitch_shift, emotion)
        path = self._cache_path(key)

        cached = self._read_cache(path)
        if cached is not None:
            return cached

        audio = self.provider.synthesize(text, **kwargs)
        self._write_cache(path, audio)
        return audio

    def _make_synthesize_async(self, inner_async):
        async def synthesize_async(
            text: str,
            *,
            speaking_rate: float = 1.0,
            pitch_shift: float = 0.0,
            emotion: str | None = None,
        ) -> bytes:
            emotion = self._resolve_emotion(emotion)
            kwargs = {"speaking_rate": speaking_rate, "pitch_shift": pitch_shift}
            if emotion is not None:
                kwargs["emotion"] = emotion
            if len(text) > self.max_chars:
                return await inner_async(text, **kwargs)

            key = self._cache_key(text, speaking_rate, pitch_shift, emotion)
            path = self._cache_path(key)

            cached = self._read_cache(path)
            if cached is not None:
                return cached

            audio = await inner_async(text, **kwargs)
            self._write_cache(path, audio)
            return audio

        return synthesize_async

    def _make_synthesize_stream(self, inner_stream):
        def synthesize_stream(
            text: str,
            *,
            speaking_rate: float = 1.0,
            pitch_shift: float = 0.0,
            emotion: str | None = None,
        ) -> Iterator[bytes]:
            """Cache-aware chunked synthesis.

            Cache hit: yield the whole cached WAV as a single chunk (no
            reason to re-chunk audio that's already synthesized). Cache
            miss: stream chunks straight from the wrapped provider as they
            arrive, and — for texts short enough to be cache-eligible —
            also collect+concatenate them into one WAV and write it to disk
            so the *next* request for this line gets a cache hit instead of
            re-running generation.
            """
            emotion = self._resolve_emotion(emotion)
            kwargs = {"speaking_rate": speaking_rate, "pitch_shift": pitch_shift}
            if emotion is not None:
                kwargs["emotion"] = emotion
            cacheable = len(text) <= self.max_chars
            if cacheable:
                key = self._cache_key(text, speaking_rate, pitch_shift, emotion)
                path = self._cache_path(key)
                cached = self._read_cache(path)
                if cached is not None:
                    yield cached
                    return

            chunks: list[bytes] = []
            for chunk in inner_stream(text, **kwargs):
                if cacheable:
                    chunks.append(chunk)
                yield chunk

            if cacheable and chunks:
                assembled = _concat_wav_chunks(chunks)
                if assembled is not None:
                    self._write_cache(path, assembled)

        return synthesize_stream

    def close(self) -> None:
        close = getattr(self.provider, "close", None)
        if close is not None:
            close()


def _concat_wav_chunks(chunks: list[bytes]) -> bytes | None:
    """Concatenate consecutive mono PCM WAV chunks (same rate/width) into one
    WAV file, for writing a streamed synthesis to the disk cache.

    Returns ``None`` if the chunks aren't uniform mono PCM WAVs (defensive;
    should not happen with our own providers) rather than writing a corrupt
    cache entry.
    """
    frames: list[bytes] = []
    params = None
    try:
        for chunk in chunks:
            with wave.open(io.BytesIO(chunk), "rb") as wav_file:
                chunk_params = (
                    wav_file.getnchannels(),
                    wav_file.getsampwidth(),
                    wav_file.getframerate(),
                )
                if params is None:
                    params = chunk_params
                elif chunk_params != params:
                    LOGGER.warning(
                        "TTS stream chunks have mismatched WAV params; "
                        "skipping cache write for this line"
                    )
                    return None
                frames.append(wav_file.readframes(wav_file.getnframes()))
    except (wave.Error, EOFError) as exc:
        LOGGER.warning("Failed to assemble streamed WAV chunks for cache: %s", exc)
        return None

    if params is None:
        return None

    channels, sampwidth, framerate = params
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out_file:
        out_file.setnchannels(channels)
        out_file.setsampwidth(sampwidth)
        out_file.setframerate(framerate)
        out_file.writeframes(b"".join(frames))
    return buffer.getvalue()
