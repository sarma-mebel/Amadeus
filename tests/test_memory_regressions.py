"""Regression pins for the 2026-08 memory review pass.

Each test targets one verified bug: the in-flight summary drain losing
mid-job messages, rollback leaving retracted text in the extraction buffer,
stale in-process tone re-applied at later boundaries, presence deliveries
minting phantom session boundaries, the silent profile-save failure, and
forget's lookup being eaten by recall's fixation guard.

All paths isolated to tmp_path; no live provider calls — rides the local
fakes from test_memory.py.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from amadeus import telemetry
from amadeus.config import Config
from amadeus.memory.manager import MemoryManager
from amadeus.memory.profile import updated_trust

from test_memory import (
    VectorEmbedder,
    _extraction_config,
    _iso_days_ago,
    _wait_for,
)


@pytest.fixture(autouse=True)
def _clean_telemetry():
    telemetry.reset()
    yield
    telemetry.reset()


class _BlockingLLMProvider:
    """Summary provider that blocks in generate() until released, so a test
    can append messages WHILE a summary job is in flight."""

    name = "fake-blocking"

    def __init__(self, release_event: threading.Event, result_text: str) -> None:
        self._release = release_event
        self._result_text = result_text
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        self._release.wait(timeout=5)
        return self._result_text

    def last_usage(self):
        return None


def _profile_gap_config(tmp_path: Path) -> Config:
    """Profile features on (via episodes) with no LLM/embedder needs —
    the shape the session-boundary tests in test_profile_state.py use."""
    config = Config()
    config.memory.persist_path = None
    config.memory.embeddings_path = None
    config.memory.enable_long_term = False
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = True
    config.memory.session_gap_minutes = 30
    config.memory.summary_trigger_messages = 50
    config.memory.profile_path = str(tmp_path / "profile.json")
    return config


# ---------------------------------------------------------------------------
# Bug 1: summary drain must not swallow messages appended mid-job
# ---------------------------------------------------------------------------


def test_summary_drain_keeps_messages_added_during_in_flight_job(
    tmp_path: Path,
) -> None:
    """_on_summary_ready used _unsummarized.clear(), but summarize()
    snapshotted the buffer at submit time — a message appended while the job
    ran was silently lost from summarization/extraction. The drain must
    remove only the submitted prefix."""
    release = threading.Event()
    summarizer = _BlockingLLMProvider(release, "a summary")
    config = _extraction_config(tmp_path, enable_fact_extraction=False)
    # High trigger so the late arrival can't queue a re-run that would
    # drain the buffer again before we can observe it.
    config.memory.summary_trigger_messages = 50
    manager = MemoryManager(config, summary_provider=summarizer)

    try:
        manager.add_message("user", "message one")
        manager.add_message("assistant", "reply one")
        manager.summarize()
        _wait_for(lambda: summarizer.calls >= 1)
        assert (
            manager._summary_future is not None and not manager._summary_future.done()
        )

        # Arrives while the job is blocked — the job never saw it.
        manager.add_message("user", "late arrival")
    finally:
        release.set()
    _wait_for(lambda: manager._summary_future is None)
    manager.flush()

    # The first two (submitted) messages drained; the late one is still
    # pending for the next summarization/extraction pass.
    assert [msg.content for msg in manager._unsummarized] == ["late arrival"]


# ---------------------------------------------------------------------------
# Bug 2: rollback must retract the message from the extraction buffer too
# ---------------------------------------------------------------------------


def test_rollback_user_message_clears_unsummarized_and_counter(
    tmp_path: Path,
) -> None:
    """Popping short_term alone left the retracted text queued in
    _unsummarized — a cancelled turn's words could still mint durable facts
    through the next extraction pass."""
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None
    config.memory.summary_trigger_messages = 50
    # No profile features: keeps the default repo-root profile_path from
    # being touched (conftest's leak guard) — rollback doesn't need it.
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    config.memory.profile_path = None
    manager = MemoryManager(config)

    manager.add_message("user", "hello")
    manager.add_message("assistant", "hi")
    counter_before = manager._messages_since_summary

    manager.add_message("user", "retract me")
    assert manager.rollback_user_message("retract me") is True

    assert [msg.content for msg in manager.short_term] == ["hello", "hi"]
    assert [msg.content for msg in manager._unsummarized] == ["hello", "hi"]
    assert manager._messages_since_summary == counter_before

    # The pop persisted: a fresh manager over the same store must not
    # resurrect the retracted message.
    restored = MemoryManager(config)
    assert "retract me" not in [msg.content for msg in restored.short_term]

    # Non-matching tail is a no-op.
    assert manager.rollback_user_message("never said this") is False
    assert len(manager.short_term) == 2


# ---------------------------------------------------------------------------
# Bug 3: stale in-process tone must not re-apply at later boundaries
# ---------------------------------------------------------------------------


def test_stale_in_process_tone_not_reapplied_at_later_boundary(
    tmp_path: Path,
) -> None:
    """_apply_session_boundary_state preferred the in-process _last_tone but
    never cleared it, so in a long-lived process every later boundary where
    extraction produced no fresh tone re-applied the stale tone (duplicate
    trust/arc drift + a tone_history entry stamped today)."""
    config = _profile_gap_config(tmp_path)
    manager = MemoryManager(config)

    manager.add_message("user", "hello")
    manager.flush()
    manager._last_tone = 0.6
    manager.short_term[-1].timestamp = _iso_days_ago(1)
    manager.add_message("user", "back after a day")  # boundary 1: consumes tone
    manager.flush()

    assert len(manager.profile.tone_history) == 1
    assert manager.profile.tone_history[0]["tone"] == 0.6
    assert manager._last_tone is None  # consumed one-shot
    trust_after_first = manager.profile.trust_level
    sessions_after_first = manager.profile.session_count

    # Boundary 2, no fresh tone from extraction: must apply the tone-less
    # baseline only — no second tone_history entry, no warm-tone bonus.
    manager.short_term[-1].timestamp = _iso_days_ago(1)
    manager.add_message("user", "back again")
    manager.flush()

    assert manager.profile.session_count == sessions_after_first + 1
    assert len(manager.profile.tone_history) == 1
    assert manager.profile.trust_level == updated_trust(
        trust_after_first, sessions_after_first + 1, None
    )


# ---------------------------------------------------------------------------
# Bug 5: presence deliveries must not mint a phantom session boundary
# ---------------------------------------------------------------------------


def test_presence_delivery_commit_skips_boundary_and_last_seen(
    tmp_path: Path,
) -> None:
    """A reach-out/idle-chatter commit is guaranteed to follow a long idle
    gap, so a plain add_message fired the gap-finalize boundary (phantom
    session_count + trust/arc/tone drift for a "session" of her own
    unprompted line) and stamped last_seen while the user was absent."""
    config = _profile_gap_config(tmp_path)
    manager = MemoryManager(config)

    manager.add_message("user", "hi")
    manager.add_message("assistant", "hey")
    manager.flush()
    sessions_before = manager.profile.session_count
    last_seen_before = manager.profile.last_seen
    manager._last_tone = 0.4  # would be misapplied by a phantom boundary

    manager.short_term[-1].timestamp = _iso_days_ago(1)
    manager.add_message(
        "assistant", "hey, did that deadline work out?", presence_delivery=True
    )
    manager.flush()

    # No phantom session; the user never showed up.
    assert manager.profile.session_count == sessions_before
    assert manager.profile.last_seen == last_seen_before
    assert manager.profile.tone_history == []
    # The commit itself still happened — it's intentional context.
    assert manager.short_term[-1].content == "hey, did that deadline work out?"
    assert manager._unsummarized[-1].content == "hey, did that deadline work out?"

    # A real user return still counts a session as before.
    manager.short_term[-1].timestamp = _iso_days_ago(1)
    manager.add_message("user", "it did!")
    manager.flush()
    assert manager.profile.session_count == sessions_before + 1
    assert manager.profile.last_seen != last_seen_before


# ---------------------------------------------------------------------------
# Bug 6: profile-save failure must count, not just log
# ---------------------------------------------------------------------------


def test_profile_save_failure_counts(tmp_path: Path, monkeypatch) -> None:
    config = _profile_gap_config(tmp_path)
    manager = MemoryManager(config)

    def _boom(path):
        raise OSError("disk full")

    monkeypatch.setattr(manager.profile, "save_atomic", _boom)
    manager._save_profile()
    assert telemetry.snapshot().get("memory.profile_save_failed") == 1


# ---------------------------------------------------------------------------
# Bug 7: forget must survive recall's fixation guard
# ---------------------------------------------------------------------------


def test_forget_finds_fact_despite_recent_conversation_about_it(
    tmp_path: Path,
) -> None:
    """ "Forget what I said about X" almost always follows talk about X, and
    conversational recall's fixation guard (recall_recent_dedupe) drops
    facts whose content words appear in the recent window — so forget's
    best match was filtered away. The lookup now rides the unfiltered
    find_facts path."""
    embedder = VectorEmbedder({"cilantro": (1.0, 0.0)}, default=(0.0, 1.0))
    manager = MemoryManager(_extraction_config(tmp_path), embedder=embedder)
    manager.remember("User hates cilantro")

    # The fact's content words are all over the recent short-term window —
    # exactly the state the fixation guard punishes.
    manager.add_message("user", "so about that cilantro thing")
    manager.add_message("assistant", "you hates cilantro, I remember")
    manager.flush()

    forgotten = manager.forget("what I said about cilantro")
    assert forgotten == "User hates cilantro"
    assert list(manager.vector_store.dump()) == []


def test_forget_still_declines_unrelated_queries(tmp_path: Path) -> None:
    """The relevance floor survives the find_facts routing: a query that
    lands below it must not delete the nearest unrelated fact."""
    embedder = VectorEmbedder({"cilantro": (1.0, 0.0)}, default=(0.0, 1.0))
    config = _extraction_config(tmp_path)
    config.memory.recall_min_score = 0.5
    manager = MemoryManager(config, embedder=embedder)
    manager.remember("User hates cilantro")

    # Unrelated query embeds orthogonally (score 0.0 < floor 0.5).
    assert manager.forget("quantum computing") is None
    assert [r.text for r in manager.vector_store.dump()] == ["User hates cilantro"]
