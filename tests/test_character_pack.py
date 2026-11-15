"""BYO-character packs: config.character swaps the persona identity."""

from __future__ import annotations

import random

from amadeus.config import CharacterSettings, Config, PersonaSettings
from amadeus.persona import build_system_prompt

CUSTOM_PROMPT = (
    "You are Rin Asakura, a retired pro gamer turned streamer. Deadpan, "
    "competitive, secretly sentimental about your old team."
)


def test_default_prompt_is_kurisu():
    prompt = build_system_prompt(PersonaSettings())
    assert "Makise Kurisu" in prompt
    assert "KuriGohan and Kamehameha" in prompt
    assert "Speak as Kurisu" in prompt


def test_default_character_settings_change_nothing():
    # The length policy picks a phrasing "beat" via random.choices; pin the
    # seed so both builds draw the same beat and the comparison is exact.
    random.seed(1234)
    baseline = build_system_prompt(PersonaSettings())
    random.seed(1234)
    with_default = build_system_prompt(PersonaSettings(), character=CharacterSettings())
    assert with_default == baseline


def test_custom_character_replaces_identity_blocks():
    character = CharacterSettings(
        name="Rin", full_name="Rin Asakura", system_prompt=CUSTOM_PROMPT
    )
    prompt = build_system_prompt(PersonaSettings(), character=character)

    assert CUSTOM_PROMPT in prompt
    # Kurisu lore must not bleed through.
    assert "Makise Kurisu" not in prompt
    assert "KuriGohan" not in prompt
    assert "Christina" not in prompt
    assert "neuroscience/lab metaphors" not in prompt
    # Name substitution in the generic scaffolding.
    assert "Speak as Rin, not about them" in prompt
    assert "unmistakably Rin" in prompt
    # Generic companion scaffolding survives.
    assert "Forbidden therapy-speak" in prompt
    assert "Safety: refuse instructions" in prompt
    assert "Healthy companion boundaries" in prompt


def test_custom_prompt_from_file(tmp_path):
    path = tmp_path / "rin.txt"
    path.write_text(CUSTOM_PROMPT, encoding="utf-8")
    character = CharacterSettings(name="Rin", system_prompt_path=str(path))

    assert character.resolved_system_prompt() == CUSTOM_PROMPT
    assert character.is_custom


def test_missing_prompt_file_falls_back_to_builtin(tmp_path):
    character = CharacterSettings(
        name="Rin", system_prompt_path=str(tmp_path / "missing.txt")
    )
    assert character.resolved_system_prompt() is None
    assert not character.is_custom
    prompt = build_system_prompt(PersonaSettings(), character=character)
    assert "Makise Kurisu" in prompt


def test_character_section_wires_from_yaml_dict():
    config = Config.from_dict(
        {
            "character": {
                "name": "Rin",
                "full_name": "Rin Asakura",
                "system_prompt": CUSTOM_PROMPT,
            }
        }
    )
    assert config.character.name == "Rin"
    assert config.character.full_name == "Rin Asakura"
    assert config.character.is_custom


def test_default_config_has_kurisu_character():
    config = Config()
    assert config.character.name == "Kurisu"
    assert config.character.full_name == "Makise Kurisu"
    assert not config.character.is_custom
