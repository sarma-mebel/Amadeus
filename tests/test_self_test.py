"""Self-test legs (amadeus/web/self_test.py).

The feature exists to make silent degradation visible, so the tests are
mostly about the FAILURE shapes: a dummy LLM must be reported as broken
rather than working, a provider that raises must become a fail result
rather than a 500, and the memory leg must never touch the user's store.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from amadeus.config import Config
from amadeus.memory.vector_store import VectorMatch
from amadeus.providers.llm.base import DummyProvider
from amadeus.web import self_test


class _FakeLLM:
    def __init__(self, reply: str = "OK") -> None:
        self._reply = reply

    def generate(self, messages, **kwargs):  # noqa: ANN001 - test double
        return self._reply


class _RaisingLLM:
    def generate(self, messages, **kwargs):  # noqa: ANN001 - test double
        raise RuntimeError("429 rate limited")


class _FakeTTS:
    def __init__(self, audio: bytes = b"RIFFfake") -> None:
        self._audio = audio

    def synthesize(self, text, **kwargs):  # noqa: ANN001 - test double
        return self._audio


class _FakeSTT:
    pass


def _store(tmp_path: Path, **overrides):
    config = Config()
    config.memory.enable_long_term = False
    config.providers.tts = "dummy"
    config.providers.stt = "dummy"
    for key, value in overrides.pop("providers", {}).items():
        setattr(config.providers, key, value)
    for key, value in overrides.pop("memory", {}).items():
        setattr(config.memory, key, value)
    return SimpleNamespace(
        config=config,
        llm=overrides.get("llm", _FakeLLM()),
        tts=overrides.get("tts"),
        embedder=overrides.get("embedder"),
        get_stt=lambda: overrides.get("stt"),
    )


# --- LLM leg ----------------------------------------------------------------


def test_dummy_llm_is_reported_as_a_failure(tmp_path: Path) -> None:
    """The headline failure this feature exists for.

    A keyless install answers every message with canned text and looks like
    a working assistant; reporting it as "pass" would defeat the point.
    """
    store = _store(tmp_path, llm=DummyProvider())
    result = self_test.check_llm(store)
    assert result.status == self_test.FAIL
    assert "dummy" in result.detail.lower()
    assert result.fix and "API key" in result.fix


def test_working_llm_passes(tmp_path: Path) -> None:
    result = self_test.check_llm(_store(tmp_path, llm=_FakeLLM("OK")))
    assert result.status == self_test.PASS


def test_empty_llm_reply_fails(tmp_path: Path) -> None:
    result = self_test.check_llm(_store(tmp_path, llm=_FakeLLM("   ")))
    assert result.status == self_test.FAIL
    assert "empty" in result.detail.lower()


def test_raising_provider_becomes_a_failed_leg_not_an_exception(
    tmp_path: Path,
) -> None:
    """A self-test that cannot report is worse than no self-test."""
    store = _store(tmp_path, llm=_RaisingLLM())
    report = self_test.run_self_test(store)
    llm_leg = next(leg for leg in report["legs"] if leg["key"] == "llm")
    assert llm_leg["status"] == self_test.FAIL
    assert "429" in llm_leg["detail"]


# --- TTS / STT legs ---------------------------------------------------------


def test_tts_disabled_is_skip_not_fail(tmp_path: Path) -> None:
    """Turned off is a configuration choice, not a broken install."""
    result = self_test.check_tts(_store(tmp_path))
    assert result.status == self_test.SKIP


def test_tts_configured_but_unloaded_fails_with_install_hint(tmp_path: Path) -> None:
    store = _store(tmp_path, providers={"tts": "qwen3_tts"}, tts=None)
    result = self_test.check_tts(store)
    assert result.status == self_test.FAIL
    assert result.fix and "qwen3_tts" in result.fix


def test_tts_returning_no_audio_fails(tmp_path: Path) -> None:
    store = _store(tmp_path, providers={"tts": "pyttsx3"}, tts=_FakeTTS(b""))
    result = self_test.check_tts(store)
    assert result.status == self_test.FAIL
    assert "no audio" in result.detail.lower()


def test_tts_success_reports_byte_count(tmp_path: Path) -> None:
    store = _store(tmp_path, providers={"tts": "pyttsx3"}, tts=_FakeTTS(b"12345"))
    result = self_test.check_tts(store)
    assert result.status == self_test.PASS
    assert result.extra["audio_bytes"] == 5


def test_stt_configured_but_unloaded_fails(tmp_path: Path) -> None:
    store = _store(tmp_path, providers={"stt": "faster_whisper"}, stt=None)
    result = self_test.check_stt(store)
    assert result.status == self_test.FAIL
    assert result.fix and "voice" in result.fix


def test_stt_loaded_passes(tmp_path: Path) -> None:
    store = _store(tmp_path, providers={"stt": "faster_whisper"}, stt=_FakeSTT())
    assert self_test.check_stt(store).status == self_test.PASS


def test_stt_reports_the_wrapped_provider_not_the_speculative_wrapper(
    tmp_path: Path,
) -> None:
    """SessionStore wraps STT in _SpeculativeSTT; the report must name the
    provider the user configured, not our internal decorator.

    The wrapper proxies unknown attributes through __getattr__, so probing
    the wrong attribute name resolves to something else rather than failing
    — the bug would be a plausible-looking wrong name, not a crash.
    """
    from amadeus.web.sessions import _SpeculativeSTT

    wrapped = _SpeculativeSTT.__new__(_SpeculativeSTT)
    wrapped.provider = _FakeSTT()
    store = _store(tmp_path, providers={"stt": "faster_whisper"}, stt=wrapped)
    result = self_test.check_stt(store)
    assert result.status == self_test.PASS
    assert result.extra["provider"] == "_FakeSTT"


# --- Memory leg -------------------------------------------------------------


class _VectorEmbedder:
    """Deterministic 2-d embedder: 'aardvark' text lands on one axis."""

    relevance_floor = 0.0

    def embed_text(self, text: str):
        return [1.0, 0.0] if "aardvark" in text.lower() else [0.0, 1.0]

    def similarity(self, a, b) -> float:
        return sum(x * y for x, y in zip(a, b))


def test_memory_leg_never_writes_to_the_users_store(tmp_path: Path) -> None:
    """The canary fact must land in a temp store, never the real one.

    A self-test that pollutes what she remembers would be a worse bug than
    anything it detects.
    """
    real_store = tmp_path / "vectors.json"
    real_store.write_text(json.dumps({"records": []}), encoding="utf-8")
    config_memory = {
        "enable_long_term": True,
        "embeddings_path": str(real_store),
        "profile_path": str(tmp_path / "profile.json"),
    }
    store = _store(tmp_path, memory=config_memory, embedder=_VectorEmbedder())

    result = self_test.check_memory(store)
    assert result.status == self_test.PASS

    payload = json.loads(real_store.read_text(encoding="utf-8"))
    records = payload.get("records", payload)
    assert records == [], "self-test wrote into the user's real memory store"


def test_memory_probe_query_clears_the_trivial_query_gate() -> None:
    """The probe query must be long enough to survive recall's own gate.

    recall_min_query_chars (default 10) skips the embed for short queries
    and returns nothing. A shorter probe made the memory leg report a false
    failure on every healthy install — pointing users at a migration they
    did not need — which is exactly the kind of wrong-but-confident output
    this whole feature exists to prevent.
    """
    from amadeus.config import MemorySettings

    # Read the query the leg actually issues rather than parsing the source:
    # inspect.getsource follows a monkeypatched recall_detailed, so a
    # source-text assertion silently depends on test ordering.
    seen: list[str] = []

    class _RecordingEmbedder(_VectorEmbedder):
        def embed_text(self, text: str):
            seen.append(text)
            return super().embed_text(text)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = _store(
            Path(tmp),
            memory={
                "enable_long_term": True,
                "embeddings_path": str(Path(tmp) / "v.json"),
                "profile_path": str(Path(tmp) / "p.json"),
            },
            embedder=_RecordingEmbedder(),
        )
        self_test.check_memory(store)

    minimum = MemorySettings().recall_min_query_chars
    assert any(len(text) >= minimum for text in seen), (
        f"the memory leg's recall query is shorter than "
        f"recall_min_query_chars ({minimum}); recall will skip the embed "
        "and the leg will report a false failure on healthy installs"
    )


def test_memory_leg_skipped_when_long_term_off(tmp_path: Path) -> None:
    assert self_test.check_memory(_store(tmp_path)).status == self_test.SKIP


def test_memory_leg_fails_without_an_embedder(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        memory={
            "enable_long_term": True,
            "embeddings_path": str(tmp_path / "v.json"),
            "profile_path": str(tmp_path / "p.json"),
        },
        embedder=None,
    )
    result = self_test.check_memory(store)
    assert result.status == self_test.FAIL
    assert "embedder" in result.detail.lower()


def test_memory_leg_flags_unrecallable_facts_with_the_migration_hint(
    tmp_path: Path, monkeypatch
) -> None:
    """The embedder-dimension mismatch is the silent recall killer this
    leg exists to name — the fix must point at scripts/migrate_memory.py."""
    monkeypatch.setattr(
        "amadeus.memory.manager.MemoryManager.recall_detailed",
        lambda self, query, limit=3, **kw: [],
    )
    store = _store(
        tmp_path,
        memory={
            "enable_long_term": True,
            "embeddings_path": str(tmp_path / "v.json"),
            "profile_path": str(tmp_path / "p.json"),
        },
        embedder=_VectorEmbedder(),
    )
    result = self_test.check_memory(store)
    assert result.status == self_test.FAIL
    assert result.fix and "migrate_memory" in result.fix


# --- Report shape -----------------------------------------------------------


def test_summary_points_at_the_most_fundamental_failure(tmp_path: Path) -> None:
    """Leg order is capability order, so the first failure is the one to
    fix first — fixing chat often clears the rest."""
    store = _store(
        tmp_path,
        llm=DummyProvider(),
        providers={"tts": "qwen3_tts"},
        tts=None,
    )
    report = self_test.run_self_test(store)
    summary = report["summary"]
    assert summary["status"] == self_test.FAIL
    assert summary["headline"].startswith("Chat model")
    assert "API key" in summary["next_step"]
    assert summary["failed"] == 2


def test_all_clear_reports_pass(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        llm=_FakeLLM(),
        providers={"tts": "pyttsx3", "stt": "faster_whisper"},
        tts=_FakeTTS(),
        stt=_FakeSTT(),
    )
    report = self_test.run_self_test(store)
    assert report["summary"]["status"] == self_test.PASS
    assert report["summary"]["next_step"] is None
    assert report["summary"]["failed"] == 0


def test_every_leg_reports_a_duration(tmp_path: Path) -> None:
    report = self_test.run_self_test(_store(tmp_path))
    assert all(leg["duration_ms"] is not None for leg in report["legs"])
    assert report["duration_ms"] >= 0


def test_failing_legs_always_carry_an_actionable_fix(tmp_path: Path) -> None:
    """A leg that fails without a next step is just a counter with nicer
    formatting — the whole premise of the feature is actionability."""
    store = _store(
        tmp_path,
        llm=DummyProvider(),
        providers={"tts": "qwen3_tts", "stt": "faster_whisper"},
        tts=None,
        stt=None,
    )
    report = self_test.run_self_test(store)
    for leg in report["legs"]:
        if leg["status"] in (self_test.FAIL, self_test.SKIP):
            assert leg.get("fix"), f"leg {leg['key']} failed with no fix text"
