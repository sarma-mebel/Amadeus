"""Tests for the disk TTS cache wrapper (amadeus/providers/tts/cache.py)."""

from __future__ import annotations

import asyncio

from amadeus.providers.tts.cache import CachedTTSProvider


class FakeTTSProvider:
    """Records calls and returns deterministic "audio" bytes."""

    def __init__(self, cache_signature: str = "fake-provider"):
        self.cache_signature = cache_signature
        self.calls: list[tuple[str, float, float]] = []
        self.closed = False

    def synthesize(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ) -> bytes:
        self.calls.append((text, speaking_rate, pitch_shift))
        return f"audio:{text}:{speaking_rate}:{pitch_shift}".encode("utf-8")

    def close(self) -> None:
        self.closed = True


class FakeAsyncTTSProvider(FakeTTSProvider):
    """Fake provider that also exposes synthesize_async."""

    def __init__(self, cache_signature: str = "fake-async-provider"):
        super().__init__(cache_signature)
        self.async_calls: list[tuple[str, float, float]] = []

    async def synthesize_async(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ) -> bytes:
        self.async_calls.append((text, speaking_rate, pitch_shift))
        return f"audio-async:{text}:{speaking_rate}:{pitch_shift}".encode("utf-8")


def test_cache_hit_avoids_second_synthesize_call(tmp_path):
    inner = FakeTTSProvider()
    cached = CachedTTSProvider(inner, cache_dir=str(tmp_path))

    first = cached.synthesize("hello")
    second = cached.synthesize("hello")

    assert first == second
    assert len(inner.calls) == 1


def test_different_text_or_rate_produces_different_keys(tmp_path):
    inner = FakeTTSProvider()
    cached = CachedTTSProvider(inner, cache_dir=str(tmp_path))

    cached.synthesize("hello")
    cached.synthesize("goodbye")
    cached.synthesize("hello", speaking_rate=1.2)
    cached.synthesize("hello", pitch_shift=0.5)

    assert len(inner.calls) == 4
    cache_files = list(tmp_path.glob("*.wav"))
    assert len(cache_files) == 4


def test_max_chars_bypasses_cache(tmp_path):
    inner = FakeTTSProvider()
    cached = CachedTTSProvider(inner, cache_dir=str(tmp_path), max_chars=5)

    long_text = "this text is definitely longer than five characters"
    cached.synthesize(long_text)
    cached.synthesize(long_text)

    assert len(inner.calls) == 2
    assert list(tmp_path.glob("*.wav")) == []


def test_corrupt_cache_file_triggers_resynthesize(tmp_path):
    inner = FakeTTSProvider()
    cached = CachedTTSProvider(inner, cache_dir=str(tmp_path))

    cached.synthesize("hello")
    assert len(inner.calls) == 1

    # Corrupt the cache file on disk (simulate a truncated/empty write).
    cache_files = list(tmp_path.glob("*.wav"))
    assert len(cache_files) == 1
    cache_files[0].write_bytes(b"")

    result = cached.synthesize("hello")
    assert len(inner.calls) == 2
    assert result == b"audio:hello:1.0:0.0"
    # Overwritten with valid content, so a third call hits cache again.
    cached.synthesize("hello")
    assert len(inner.calls) == 2


def test_signature_uses_provider_cache_signature_when_present(tmp_path):
    provider_a = FakeTTSProvider(cache_signature="provider-a")
    provider_b = FakeTTSProvider(cache_signature="provider-b")
    cache_a = CachedTTSProvider(provider_a, cache_dir=str(tmp_path))
    cache_b = CachedTTSProvider(provider_b, cache_dir=str(tmp_path))

    cache_a.synthesize("hello")
    cache_b.synthesize("hello")

    # Different provider identity -> different cache key -> both synthesized.
    assert len(provider_a.calls) == 1
    assert len(provider_b.calls) == 1
    assert len(list(tmp_path.glob("*.wav"))) == 2


def test_falls_back_to_class_name_without_cache_signature(tmp_path):
    class NoSignatureProvider(FakeTTSProvider):
        def __init__(self):
            self.calls = []
            self.closed = False

    inner = NoSignatureProvider()
    cached = CachedTTSProvider(inner, cache_dir=str(tmp_path))

    cached.synthesize("hello")
    cached.synthesize("hello")

    assert len(inner.calls) == 1


def test_close_delegates_to_inner_provider(tmp_path):
    inner = FakeTTSProvider()
    cached = CachedTTSProvider(inner, cache_dir=str(tmp_path))

    cached.close()

    assert inner.closed is True


def test_close_is_safe_without_inner_close(tmp_path):
    class NoCloseProvider:
        def __init__(self):
            self.calls = []

        def synthesize(self, text, *, speaking_rate=1.0, pitch_shift=0.0):
            self.calls.append(text)
            return b"audio"

    inner = NoCloseProvider()
    cached = CachedTTSProvider(inner, cache_dir=str(tmp_path))

    cached.close()  # must not raise


def test_synthesize_async_present_when_inner_supports_it(tmp_path):
    inner = FakeAsyncTTSProvider()
    cached = CachedTTSProvider(inner, cache_dir=str(tmp_path))

    assert hasattr(cached, "synthesize_async")

    result1 = asyncio.run(cached.synthesize_async("hello"))
    result2 = asyncio.run(cached.synthesize_async("hello"))

    assert result1 == result2
    assert len(inner.async_calls) == 1
    # Sync path stays untouched by the async cache.
    assert len(inner.calls) == 0


def test_synthesize_async_absent_when_inner_lacks_it(tmp_path):
    inner = FakeTTSProvider()
    cached = CachedTTSProvider(inner, cache_dir=str(tmp_path))

    # The web layer decides sync-vs-async via getattr(tts, "synthesize_async", None);
    # the attribute must be truly absent, not merely None, when unsupported.
    assert getattr(cached, "synthesize_async", None) is None
    assert not hasattr(cached, "synthesize_async")


def test_load_tts_wraps_when_enabled(tmp_path, monkeypatch):
    import amadeus.provider_loader as provider_loader
    from amadeus.config import Config

    fake_inner = FakeTTSProvider()
    monkeypatch.setattr(
        provider_loader, "_load_tts_provider", lambda config: fake_inner
    )
    config = Config.from_dict(
        {
            "providers": {
                "tts_cache_enabled": True,
                "tts_cache_dir": str(tmp_path / "cache"),
            }
        }
    )
    provider = provider_loader.load_tts(config)

    assert isinstance(provider, CachedTTSProvider)
    assert provider.provider is fake_inner


def test_load_tts_does_not_wrap_when_disabled(tmp_path, monkeypatch):
    import amadeus.provider_loader as provider_loader
    from amadeus.config import Config

    fake_inner = FakeTTSProvider()
    monkeypatch.setattr(
        provider_loader, "_load_tts_provider", lambda config: fake_inner
    )
    config = Config.from_dict(
        {
            "providers": {
                "tts_cache_enabled": False,
            }
        }
    )
    provider = provider_loader.load_tts(config)

    assert not isinstance(provider, CachedTTSProvider)
    assert provider is fake_inner


# ---- Emotion-aware caching --------------------------------------------------


class FakeEmotionTTSProvider(FakeTTSProvider):
    """Fake provider advertising emotion refs, recording the emotion kwarg."""

    def __init__(self):
        super().__init__("fake-emotion-provider")
        self.available_emotions = frozenset({"happy", "sad"})
        self.emotion_calls: list[str | None] = []

    def synthesize(
        self,
        text: str,
        *,
        speaking_rate: float = 1.0,
        pitch_shift: float = 0.0,
        emotion: str | None = None,
    ) -> bytes:
        self.emotion_calls.append(emotion)
        self.calls.append((text, speaking_rate, pitch_shift))
        return f"audio:{text}:{emotion}".encode("utf-8")


def test_emotion_keys_cache_separately_and_is_forwarded(tmp_path):
    inner = FakeEmotionTTSProvider()
    cached = CachedTTSProvider(inner, cache_dir=str(tmp_path))

    assert cached.available_emotions == frozenset({"happy", "sad"})
    base = cached.synthesize("hello")
    happy = cached.synthesize("hello", emotion="happy")
    happy_again = cached.synthesize("hello", emotion="happy")

    assert base != happy
    assert happy == happy_again
    assert inner.emotion_calls == [None, "happy"]  # third call was a hit
    assert len(list(tmp_path.glob("*.wav"))) == 2


def test_unknown_emotion_is_dropped_not_fragmenting_cache(tmp_path):
    inner = FakeEmotionTTSProvider()
    cached = CachedTTSProvider(inner, cache_dir=str(tmp_path))

    cached.synthesize("hello")
    cached.synthesize("hello", emotion="confused")  # not in available -> hit

    assert inner.emotion_calls == [None]
    assert len(list(tmp_path.glob("*.wav"))) == 1


def test_provider_without_emotions_never_receives_kwarg(tmp_path):
    inner = FakeTTSProvider()
    cached = CachedTTSProvider(inner, cache_dir=str(tmp_path))

    assert cached.available_emotions == frozenset()
    # Passing a label anyway must not crash the inner provider (which has no
    # emotion kwarg) nor create a second cache entry.
    cached.synthesize("hello", emotion="happy")
    cached.synthesize("hello")

    assert len(inner.calls) == 1
    assert len(list(tmp_path.glob("*.wav"))) == 1


class FakeJudgingProvider(FakeTTSProvider):
    """Fake provider exposing the judge_speaker hook (drift-guard providers)."""

    def __init__(self, verdict: bool | None):
        super().__init__("fake-judging-provider")
        self.verdict = verdict
        self.judged: list[bytes] = []

    def judge_speaker(self, wav_data: bytes) -> bool | None:
        self.judged.append(wav_data)
        return self.verdict


def test_guard_failed_clip_is_returned_but_never_cached(tmp_path):
    """A wrong-voice render (best_of exhausted) may play once, but caching it
    would pin the wrong voice to that line forever — the 2026-07-28 cache
    audit found exactly that (a 6.7s clip at 0.60 similarity on every hit)."""
    inner = FakeJudgingProvider(verdict=False)
    cached = CachedTTSProvider(inner, cache_dir=str(tmp_path))

    first = cached.synthesize("hello")
    second = cached.synthesize("hello")

    assert first == second  # audio still served
    assert len(inner.calls) == 2  # no cache entry -> re-synthesized
    assert list(tmp_path.glob("*.wav")) == []
    assert inner.judged  # the judge actually saw the bytes


def test_guard_passing_clip_caches_normally(tmp_path):
    inner = FakeJudgingProvider(verdict=True)
    cached = CachedTTSProvider(inner, cache_dir=str(tmp_path))

    cached.synthesize("hello")
    cached.synthesize("hello")

    assert len(inner.calls) == 1
    assert len(list(tmp_path.glob("*.wav"))) == 1


def test_unjudgeable_clip_still_caches(tmp_path):
    """verdict None = guard off/decode failure -> no objection, keep the old
    caching behavior rather than disabling the cache wholesale."""
    inner = FakeJudgingProvider(verdict=None)
    cached = CachedTTSProvider(inner, cache_dir=str(tmp_path))

    cached.synthesize("hello")
    cached.synthesize("hello")

    assert len(inner.calls) == 1
    assert len(list(tmp_path.glob("*.wav"))) == 1
