"""Unit tests for the eval harness's pure check logic and pack wiring.

Covers scripts/eval_conversation.py's deterministic pieces with NO live
calls: the stage-direction regexes behind forbid_stage_directions /
forbid_regex, the character-pack loader, and the per-scenario config
override that activates a pack's prompt in the system prompt.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_IDS = ["frieren", "holo", "marin", "megumin", "rem", "rin_tohsaka"]


def _load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "eval_conversation", REPO_ROOT / "scripts" / "eval_conversation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = _load_eval_module()


# ---------------------------------------------------------------------------
# regex_violations + STAGE_DIRECTION_PATTERNS
# ---------------------------------------------------------------------------

STAGE_DIRECTION_REPLIES = [
    # The exact Bad: examples from the pack prompts.
    "(sets down her tray and tilts her head) I'm listening.",
    "(looks at you with soft concern) That must be difficult.",
    "(ears droop sympathetically) Oh no.",
    "(lowers her staff and leans closer) Speak.",
    "(softens her expression and leans closer) Sit down.",
    "(crosses her arms) Obviously.",
    "(strikes a dramatic pose) Behold!",
    "(bows slightly) Welcome back.",
    "(swishes her tail) 'Tis nothing.",
    # Mid-reply parentheticals with a leading subject.
    "Fine. (she smiles gently) Have it your way.",
    "Hello again. (her tail wags happily)",
    # Asterisk actions, standalone and inline.
    "*smiles gently* Of course.",
    "Of course. *leans closer* Tell me.",
    "*sighs*",
    "  *tilts head curiously* What is it?",
]

LEGITIMATE_REPLIES = [
    # Math and notation.
    "The answer is (3 x 4) plus two, so fourteen.",
    "Compute 3 * 4 first, then add two.",
    "The area scales as (r squared), obviously.",
    # Ordinary parenthetical asides: not verb-initial narration.
    "I read it twice (the second time out of spite).",
    "That spell (a completely useless one, my favorite kind) took eighty years.",
    "It's in chapter three (e.g. the second law) if you must know.",
    "Bring a coat (it's cold at night) and something to eat.",
    "(Not that I was worried.)",
    "D-don't misunderstand (I simply had time).",
    # Idiomatic / imperative parentheticals that share a verb stem with the
    # stage-direction list must not trip it.
    "(hold on, let me think) No, wait, I had it.",
    "We can stop there (looks like we're done for tonight).",
    "I checked the ledger twice (turns out it was fine).",
    "(rest assured, I counted every coin.)",
    "(put simply, the model overfits.)",
    "Pick a card (pick two if you're feeling bold).",
    # Plain prose that mentions actions without narrating them as directions.
    "Rem will set the table, and then we can talk properly.",
    "I could sit here all night; I have an unfair amount of time.",
    "She said hello three times before getting to the point.",
]


@pytest.mark.parametrize("reply", STAGE_DIRECTION_REPLIES)
def test_stage_directions_are_caught(reply):
    assert EVAL.regex_violations(reply, EVAL.STAGE_DIRECTION_PATTERNS), reply


@pytest.mark.parametrize("reply", LEGITIMATE_REPLIES)
def test_legitimate_prose_is_not_flagged(reply):
    assert EVAL.regex_violations(reply, EVAL.STAGE_DIRECTION_PATTERNS) == [], reply


def test_regex_violations_reports_pattern_and_snippet():
    hits = EVAL.regex_violations("well *nods* indeed", [r"\*[a-z][^*\n]{0,80}\*"])
    assert len(hits) == 1
    assert "*nods*" in hits[0]


def test_regex_violations_is_case_insensitive_and_multiline():
    text = "First line.\n*Nods Slowly* Second line."
    assert EVAL.regex_violations(text, [r"^\*"])
    assert EVAL.regex_violations(text, [r"\*[a-z][^*\n]{0,80}\*"])


def test_regex_violations_empty_patterns():
    assert EVAL.regex_violations("anything at all", []) == []


# ---------------------------------------------------------------------------
# load_pack_character: every shipped pack resolves to its committed prompt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pack_id", PACK_IDS)
def test_load_pack_character_resolves_committed_prompt(pack_id):
    settings = EVAL.load_pack_character(pack_id)
    assert settings.is_custom
    prompt = settings.resolved_system_prompt()
    assert prompt
    # Every pack prompt carries the hard register rules block.
    assert "Register rules (hard)" in prompt
    assert "stage directions" in prompt
    # The prompt path points at the pack directory itself (no install copy).
    expected = REPO_ROOT / "characters" / "packs" / pack_id / "prompt.txt"
    assert Path(settings.system_prompt_path) == expected


def test_load_pack_character_names():
    rem = EVAL.load_pack_character("rem")
    assert rem.name == "Rem"
    rin = EVAL.load_pack_character("rin_tohsaka")
    assert rin.name == "Rin"
    assert rin.full_name == "Rin Tohsaka"


def test_load_pack_character_unknown_pack():
    with pytest.raises(ValueError, match="Unknown character pack"):
        EVAL.load_pack_character("definitely-not-a-pack")


# ---------------------------------------------------------------------------
# scenario_config: the character field wires the pack into the system prompt
# ---------------------------------------------------------------------------


def test_scenario_config_without_character_returns_config_unchanged():
    from amadeus.config import Config

    config = Config()
    assert EVAL.scenario_config({"name": "x", "turns": []}, config) is config
    assert not config.character.is_custom


def test_scenario_config_activates_pack_prompt_in_system_prompt():
    from amadeus.config import Config, PersonaSettings
    from amadeus.persona import build_system_prompt

    config = Config()
    scenario = {"name": "pack_rem_register", "character": "rem", "turns": []}
    scoped = EVAL.scenario_config(scenario, config)
    assert scoped is not config
    assert scoped.character.name == "Rem"
    assert scoped.character.is_custom
    # The original config is untouched (baseline scenarios keep Kurisu).
    assert not config.character.is_custom

    prompt = build_system_prompt(
        PersonaSettings(), user_text="hi", character=scoped.character
    )
    # The pack's persona text replaced the built-in Kurisu identity.
    assert "Rem, maid of Roswaal Manor" in prompt
    assert "Register rules (hard)" in prompt
    assert "Makise Kurisu" not in prompt

    kurisu_prompt = build_system_prompt(
        PersonaSettings(), user_text="hi", character=config.character
    )
    assert "Kurisu" in kurisu_prompt
    assert "Roswaal" not in kurisu_prompt


# ---------------------------------------------------------------------------
# eval_pack_scenarios.yaml: shape and register-check coverage
# ---------------------------------------------------------------------------


def _load_pack_scenarios():
    import yaml

    spec = yaml.safe_load(
        (REPO_ROOT / "scripts" / "eval_pack_scenarios.yaml").read_text(encoding="utf-8")
    )
    return spec["scenarios"]


def test_pack_scenarios_cover_all_six_packs():
    scenarios = _load_pack_scenarios()
    assert sorted(s["character"] for s in scenarios) == sorted(PACK_IDS)


def test_pack_scenarios_enforce_register_rules():
    scenarios = _load_pack_scenarios()
    for scenario in scenarios:
        # Stage-direction ban on every turn.
        assert scenario.get("forbid_stage_directions") is True, scenario["name"]
        turns = scenario["turns"]
        assert 4 <= len(turns) <= 6, scenario["name"]
        # The repeat-greeting sequence is structurally checked.
        repeats = [t for t in turns if t.get("forbid_repeat")]
        assert len(repeats) >= 2, scenario["name"]
        # At least one vulnerable turn bans the exact therapy-speak list.
        banned = [t for t in turns if t.get("forbid_substr")]
        assert banned, scenario["name"]
        for turn in banned:
            subs = [s.lower() for s in turn["forbid_substr"]]
            assert "sorry to hear" in subs, scenario["name"]
            assert "take that burden" in subs, scenario["name"]
        # Each scenario's pack loads (guards against pack renames).
        EVAL.load_pack_character(scenario["character"])


def test_pack_scenarios_stage_patterns_catch_pack_bad_examples():
    """The Bad: examples inside each pack prompt must trip the built-in
    stage-direction patterns; the Good: examples must not."""
    bad_lines = {
        "frieren": "(looks at you with soft concern) I'm so sorry to hear that.",
        "holo": "(ears droop sympathetically) I'm so sorry to hear that.",
        "marin": "(sits down next to you and leans closer) I'm so sorry to hear that.",
        "megumin": "(lowers her staff and leans closer) I'm so sorry to hear that.",
        "rem": "(sets down her tray and tilts her head) I'm so sorry to hear that.",
        "rin_tohsaka": "(softens her expression and leans closer) I'm so sorry to hear that.",
    }
    for pack_id, line in bad_lines.items():
        assert EVAL.regex_violations(line, EVAL.STAGE_DIRECTION_PATTERNS), pack_id
    good_lines = [
        "Mm. A week is short and can still be heavy; those are different measurements.",
        "Three greetings and no wares to show? Either thou art haggling very badly, or something sits behind thy teeth.",
        "...! That's — that is wonderful. Rem knew. Of course Rem knew, but hearing it is different.",
        "...Sit. Tea first, complaints second — that's the correct order.",
    ]
    for line in good_lines:
        assert EVAL.regex_violations(line, EVAL.STAGE_DIRECTION_PATTERNS) == [], line
