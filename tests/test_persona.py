import datetime as dt

from amadeus.config import Config, PersonaSettings
from amadeus.persona import (
    KurisuArc,
    KurisuState,
    build_system_prompt,
    is_late_night,
    time_of_day_bucket,
)


def test_system_prompt_mentions_safety() -> None:
    persona = PersonaSettings(tsundere_level=0.8, scientific_formality=0.9)
    prompt = build_system_prompt(persona)
    assert "Safety:" in prompt
    assert "Dials:" in prompt


def test_system_prompt_discourages_forced_lore_and_fake_activity() -> None:
    persona = PersonaSettings(tsundere_level=0.5, scientific_formality=0.6)
    prompt = build_system_prompt(persona, user_text="What are you doing?")
    assert (
        "Never describe yourself with machine or corporate assistant language" in prompt
    )
    assert "Do not say you are processing data streams" in prompt
    assert "answer from the current chat context" in prompt
    assert "Waiting for you to show up" in prompt
    assert "not to this conversation or stored user memories" in prompt
    assert "listen first" in prompt
    assert "Never guilt-trip them for being away" in prompt
    assert "drop banter entirely" in prompt


def test_system_prompt_keeps_kurisu_voice_when_comforting() -> None:
    """The comfort register must stay in-character: therapy-speak banned, the
    listen-or-ideas menu capped, personality explicitly retained on sad turns
    (the eval's emotional_register_tracking scenario scored 4/10 without this)."""
    persona = PersonaSettings()
    prompt = build_system_prompt(persona, user_text="I failed my exam.")
    assert "comfort in YOUR voice, not a therapist's" in prompt
    assert "I'm so sorry to hear that" in prompt  # listed as forbidden
    assert "Vulnerability turns off teasing, not personality" in prompt
    assert "at most once per conversation" in prompt
    assert "noise, not a trend line" in prompt  # the edge-retained example


def test_system_prompt_does_not_carry_previous_topic_into_greeting() -> None:
    persona = PersonaSettings()
    state = KurisuState(last_topic="neurodegenerative research")
    prompt = build_system_prompt(persona, state=state, user_text="Hi")
    assert "Likely context: neurodegenerative research" not in prompt


def test_system_prompt_renders_nonzero_arc_values() -> None:
    persona = PersonaSettings()
    arc = KurisuArc(
        curiosity_vs_annoyance=0.5,
        logic_vs_emotion=-0.3,
        pride_vs_vulnerability=-0.2,
    )
    prompt = build_system_prompt(persona, arc=arc)
    assert "Curiosity vs annoyance leans curiosity (+0.50)." in prompt
    assert "Logic vs emotion leans emotion (-0.30)." in prompt
    assert "Confidence vs vulnerability leans vulnerability (-0.20)." in prompt


def test_system_prompt_arc_none_renders_hidden_arc_default() -> None:
    persona = PersonaSettings()
    prompt = build_system_prompt(persona, arc=None)
    assert (
        "Hidden arc: she enjoys the exchange but hides it behind intellectual "
        "restraint." in prompt
    )


def test_system_prompt_includes_per_turn_clock() -> None:
    """Time awareness must live in every turn, not only the greeting."""
    persona = PersonaSettings()
    moment = dt.datetime(2026, 7, 17, 23, 41)  # a Friday, late night
    prompt = build_system_prompt(persona, now=moment)
    assert "Now: Friday 23:41 (late night)." in prompt
    assert "Never recite the time mechanically" in prompt


def test_system_prompt_clock_defaults_to_wall_clock() -> None:
    prompt = build_system_prompt(PersonaSettings())
    assert "Now: " in prompt


def test_time_of_day_buckets_and_late_night_window() -> None:
    assert time_of_day_bucket(3) == "the middle of the night"
    assert time_of_day_bucket(7) == "early morning"
    assert time_of_day_bucket(10) == "morning"
    assert time_of_day_bucket(14) == "afternoon"
    assert time_of_day_bucket(20) == "evening"
    assert time_of_day_bucket(23) == "late night"
    assert is_late_night(dt.datetime(2026, 7, 17, 2, 0))
    assert not is_late_night(dt.datetime(2026, 7, 17, 5, 0))
    assert not is_late_night(dt.datetime(2026, 7, 17, 12, 0))


def test_engine_hydrated_arc_renders_in_prepare_turn_prompt(tmp_path) -> None:
    """Engine-level test placed here (per plan) to avoid file contention with
    the parallel agent's memory/manager.py work: a ChatEngine hydrated from a
    profile with nonzero arc floats produces a prepare_turn(...) system
    prompt containing the arc lean text."""
    from amadeus.engine import ChatEngine
    from amadeus.memory.profile import UserProfile
    from amadeus.providers.llm.base import DummyProvider

    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    # Profile hydration is gated on these (see MemoryManager._profile_features_enabled).
    config.memory.enable_fact_extraction = True

    profile = UserProfile(
        arc_curiosity_vs_annoyance=0.5,
        arc_logic_vs_emotion=-0.3,
        arc_pride_vs_vulnerability=-0.2,
    )
    profile.save_atomic(tmp_path / "profile.json")

    engine = ChatEngine(config, DummyProvider())
    persona = PersonaSettings()
    setup = engine.prepare_turn("Hello", persona)
    assert "Curiosity vs annoyance leans curiosity (+0.50)." in setup.system_prompt
    assert "Logic vs emotion leans emotion (-0.30)." in setup.system_prompt
    assert (
        "Confidence vs vulnerability leans vulnerability (-0.20)."
        in setup.system_prompt
    )


def test_system_prompt_explains_interrupted_marker() -> None:
    """A cancelled streamed turn commits '... [interrupted]' to memory; the
    prompt must tell her what that means or she treats it as literal text."""
    prompt = build_system_prompt(PersonaSettings())
    assert "[interrupted]" in prompt
    assert "never mention the marker itself" in prompt
    assert "resume naturally" in prompt


def test_system_prompt_has_stt_garble_repair_rule() -> None:
    """Voice mode hands her word salad sometimes; she must ask to repeat
    rather than earnestly answer nonsense."""
    prompt = build_system_prompt(PersonaSettings())
    assert "speech-to-text" in prompt
    assert "ask them to say it again" in prompt


def test_system_prompt_stays_within_its_size_budget() -> None:
    """The persona rides a small model, and every added rule dilutes the
    others' pull on it (the 2026-07-17 fact-recall flake followed a week of
    prompt growth). Current size is ~13.3k chars; the ceiling gives ~2%
    headroom. If this fails, the intended fix is rule-out-before-rule-in —
    merge or delete an existing rule — not bumping the number. Grow the
    ceiling only as a conscious decision that the prompt deserves to be
    bigger, ideally with eval evidence. Ceiling history: 13000 -> 13600 on
    2026-07-31 for the memory-honesty/abstention rule (LongMemEval's fifth
    ability, best-practices-audit gap; already compressed to ~490 chars —
    eval scenario memory_abstention pins the behavior, judge evidence
    pending the next baseline re-record).

    Measured against the WORST-CASE length beat: _length_policy randomly
    samples one of three beat texts of different lengths, which made this
    test a coin flip once the ceiling got tight (2026-08-03: it failed only
    when the longest beat happened to be drawn — a full-suite "flake" that
    cost a bisection to diagnose). Pinning random.choices to the longest
    candidate makes the ceiling deterministic."""
    from unittest import mock

    from amadeus import persona as persona_module

    with mock.patch.object(
        persona_module.random,
        "choices",
        lambda options, weights, k: [max(options, key=len)],
    ):
        prompt = build_system_prompt(
            PersonaSettings(), user_text="How does memory consolidation work?"
        )
    assert len(prompt) <= 13600, (
        f"persona prompt is {len(prompt)} chars (> 13600 budget, worst-case "
        "length beat); remove or merge a rule instead of raising the ceiling"
    )


def test_examples_match_teasing_budget() -> None:
    """At the trust-gated budget "0" the few-shots must not demonstrate
    teasing — the model follows examples over rules, so exemplars like
    'Obviously I knew you could do it' silently overrode the no-teasing
    dial for every first-session user (default trust 0.5 < 0.60)."""
    low_trust = build_system_prompt(
        PersonaSettings(), state=KurisuState(trust_level=0.5)
    )
    assert "Do not tease at all" in low_trust
    assert "Obviously I knew you could do it" not in low_trust
    assert "Don't let it go to your head" not in low_trust

    high_trust = build_system_prompt(
        PersonaSettings(), state=KurisuState(trust_level=0.85)
    )
    assert "Do not tease at all" not in high_trust
    assert "Obviously I knew you could do it" in high_trust


def test_guardedness_and_safe_mode_stay_out_of_static_prefix() -> None:
    """Guardedness (warmth clause) and safe_mode (tone) are per-request
    mutable on the web path; they must live in the dynamic tail so flipping
    them doesn't invalidate the provider's cached static prefix."""
    open_persona = PersonaSettings()
    open_persona.emotional_guardedness = 0.5
    guarded_persona = PersonaSettings()
    guarded_persona.emotional_guardedness = 0.8
    prompt_open = build_system_prompt(open_persona, state=KurisuState())
    prompt_guarded = build_system_prompt(guarded_persona, state=KurisuState())
    # First line = persona_traits, the bulk of the cacheable prefix.
    assert prompt_open.split("\n")[0] == prompt_guarded.split("\n")[0]
    # The warmth dial still reaches the prompt, in the tail.
    assert "Warmth:" in prompt_open and "Warmth:" in prompt_guarded

    unsafe = PersonaSettings()
    unsafe.safe_mode = False
    prompt_unsafe = build_system_prompt(unsafe, state=KurisuState())
    prompt_safe = build_system_prompt(PersonaSettings(), state=KurisuState())
    assert prompt_safe.split("\n")[0] == prompt_unsafe.split("\n")[0]


def test_persona_capsule_builtin_and_pack() -> None:
    from amadeus.config import CharacterSettings
    from amadeus.persona import persona_capsule

    builtin = persona_capsule(None)
    assert "tsundere" in builtin.lower()
    assert "therapy" in builtin.lower()

    pack = CharacterSettings(
        name="Rin",
        full_name="Rin Asakura",
        system_prompt="You are Rin Asakura, a soft-spoken archivist. "
        "You speak in calm, complete sentences and never raise your voice. "
        + "Filler. " * 80,
    )
    derived = persona_capsule(pack)
    assert "Kurisu" not in derived
    assert "Rin Asakura" in derived

    pack.voice_style = "gentle, bookish, endlessly patient; never sarcastic"
    styled = persona_capsule(pack)
    assert styled == "Voice: gentle, bookish, endlessly patient; never sarcastic"


def test_system_prompt_has_memory_abstention_rule() -> None:
    """LongMemEval's fifth ability: when the user asserts a shared memory
    that was never said, she must decline rather than confabulate — and the
    rule must ride grounding_policy so custom packs get it too."""
    prompt = build_system_prompt(PersonaSettings())
    assert "Memory honesty" in prompt
    assert "brother in Osaka" in prompt  # models follow examples over rules

    from amadeus.config import CharacterSettings

    pack = CharacterSettings(name="Rin", system_prompt="You are Rin, a calm archivist.")
    pack_prompt = build_system_prompt(PersonaSettings(), character=pack)
    assert "Memory honesty" in pack_prompt


def test_recall_framing_scopes_memory_claims() -> None:
    """The recall block must declare itself the ONLY long-term memory
    surfaced this turn, so partial recall can't license invented extras."""
    from amadeus.memory.context import RECALL_FRAMING

    assert "ONLY long-term memories" in RECALL_FRAMING
    assert "do not claim recollections beyond" in RECALL_FRAMING


def test_early_acquaintance_block_is_caller_gated() -> None:
    """The getting-to-know-you nudge appears only when the caller says the
    profile is still young (engine._early_acquaintance_active) — the default
    prompt stays byte-identical for established users."""
    on = build_system_prompt(PersonaSettings(), early_acquaintance=True)
    off = build_system_prompt(PersonaSettings())
    assert "Getting acquainted" in on
    assert "at most one such question per reply" in on
    # Consent posture rides in the block itself: deflection ends a topic.
    assert "drop that subject entirely" in on
    assert "Getting acquainted" not in off
