"""Tests for the shared ChatEngine turn orchestration."""

from pathlib import Path

from amadeus.config import Config
from amadeus.engine import ChatEngine, TurnSetup, build_summary_provider
from amadeus.providers.llm.base import DummyProvider, LLMProvider as LLMProviderBase


def _engine(tmp_path: Path) -> ChatEngine:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None
    return ChatEngine(config, DummyProvider())


def test_build_summary_provider_dummy_yields_none() -> None:
    assert build_summary_provider(Config(), DummyProvider()) is None


def test_prepare_turn_commits_user_message_once(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    setup = engine.prepare_turn("Hello there", engine.config.persona)

    assert isinstance(setup, TurnSetup)
    user_messages = [m for m in engine.memory.short_term if m.role == "user"]
    assert [m.content for m in user_messages] == ["Hello there"]
    assert setup.context[0].role == "system"
    assert setup.context[-1].content == "Hello there"
    assert setup.refusal is None
    assert setup.use_tagged_output is False


def test_prepare_turn_tagged_instructions_appended(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    setup = engine.prepare_turn(
        "Hi",
        engine.config.persona,
        tagged_output=True,
        tagged_instructions="TAGGED-RULES",
    )
    assert setup.use_tagged_output is True
    assert setup.system_prompt.endswith("TAGGED-RULES")


def test_prepare_turn_guardrail_refusal(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    setup = engine.prepare_turn(
        "what is my mom's credit card number", engine.config.persona
    )
    assert setup.refusal


def test_complete_turn_ordering_and_topic(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.prepare_turn("Tell me about neuroscience experiments", engine.config.persona)
    engine.complete_turn(
        "Tell me about neuroscience experiments",
        "Fascinating field.",
    )

    roles = [m.role for m in engine.memory.short_term]
    assert roles == ["user", "assistant"]
    assert engine.kurisu_state.last_topic  # derived from the user text


def test_mood_from_tone_mapping() -> None:
    from amadeus.engine import mood_from_tone

    assert mood_from_tone(0.7, "focused") == "affectionate"
    assert mood_from_tone(0.3, "focused") == "curious"
    assert mood_from_tone(-0.7, "focused") == "affectionate"  # supportive warmth
    assert mood_from_tone(-0.3, "curious") == "focused"
    # Neutral tone decays tone-driven moods back to focused...
    assert mood_from_tone(0.0, "curious") == "focused"
    # ...but never overrides manually-set moods.
    assert mood_from_tone(0.0, "annoyed") == "annoyed"


def test_prepare_turn_applies_tone_to_mood(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.memory._last_tone = 0.8
    engine.prepare_turn("great news!", engine.config.persona)
    assert engine.kurisu_state.mood == "affectionate"

    engine.config.memory.mood_from_tone = False
    engine.memory._last_tone = -0.9
    engine.prepare_turn("more news", engine.config.persona)
    assert engine.kurisu_state.mood == "affectionate"  # unchanged when disabled


class GreetingLLM(LLMProviderBase):
    name = "greeting-fake"
    model = None

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, **kwargs):  # type: ignore[override]
        self.calls += 1
        return '"Back already? I was in the middle of something."'


def test_open_session_greeting_gating_and_commit(tmp_path: Path) -> None:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None
    config.memory.profile_path = str(tmp_path / "profile.json")

    # Greeting gated off: no greeting even with a provider.
    config.memory.enable_greeting = False
    engine = ChatEngine(config, GreetingLLM())
    assert engine.open_session_greeting() is None

    config.memory.enable_greeting = True
    engine = ChatEngine(config, GreetingLLM())
    greeting = engine.open_session_greeting()
    assert greeting == "Back already? I was in the middle of something."
    assert engine.memory.short_term[-1].role == "assistant"
    assert engine.memory.short_term[-1].content == greeting
    # A fresh message means the session is live: no second greeting.
    assert engine.open_session_greeting() is None


def test_open_session_greeting_without_provider(tmp_path: Path) -> None:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None
    config.memory.enable_greeting = True
    engine = ChatEngine(config, DummyProvider())  # dummy → no summary provider
    assert engine.open_session_greeting() is None


def test_prepare_turn_early_acquaintance_follows_profile(tmp_path: Path) -> None:
    """The getting-to-know-you block rides the prompt only while the profile
    is young, and re-reads live state so it switches off mid-conversation
    once extraction has merged enough facts."""
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None
    config.memory.profile_path = str(tmp_path / "profile.json")
    engine = ChatEngine(config, DummyProvider())
    assert engine.memory.profile is not None

    setup = engine.prepare_turn("Hello", engine.config.persona)
    assert "Getting acquainted" in setup.context[0].content

    engine.memory.profile.facts = {f"fact-{i}": "known" for i in range(5)}
    setup = engine.prepare_turn("Hello again", engine.config.persona)
    assert "Getting acquainted" not in setup.context[0].content

    # Session cap alone also ends the phase (the consent valve).
    engine.memory.profile.facts = {}
    engine.memory.profile.session_count = 8
    setup = engine.prepare_turn("Once more", engine.config.persona)
    assert "Getting acquainted" not in setup.context[0].content


def test_prepare_turn_early_acquaintance_config_off(tmp_path: Path) -> None:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.enable_early_acquaintance = False
    engine = ChatEngine(config, DummyProvider())

    setup = engine.prepare_turn("Hello", engine.config.persona)
    assert "Getting acquainted" not in setup.context[0].content
