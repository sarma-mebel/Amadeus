from pathlib import Path

from amadeus.config import Config


def test_default_config_load(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("llm_model: test-model", encoding="utf-8")
    config = Config.load(cfg_path)
    assert config.llm_model == "test-model"
    assert config.persona.tsundere_level == 0.5


def test_validate_clean_default_config() -> None:
    assert Config().validate() == []


def test_validate_flags_suspect_values() -> None:
    cases = [
        (lambda c: setattr(c.providers, "llm", "chatgpt"), "providers.llm"),
        (lambda c: setattr(c.providers, "tts", "espeak"), "providers.tts"),
        (lambda c: setattr(c.providers, "stt", "whisperx"), "providers.stt"),
        (lambda c: setattr(c.persona, "tsundere_level", 1.5), "tsundere_level"),
        (
            lambda c: setattr(c.persona, "emotional_guardedness", -0.2),
            "emotional_guardedness",
        ),
        (lambda c: setattr(c.memory, "recall_min_score", -1.0), "recall_min_score"),
        (lambda c: setattr(c.memory, "dedupe_threshold", 3.0), "dedupe_threshold"),
        # Fact extraction with long-term storage explicitly disabled: facts
        # have nowhere to go (both flags default on now, so force the clash).
        (
            lambda c: (
                setattr(c.memory, "enable_fact_extraction", True),
                setattr(c.memory, "enable_long_term", False),
            ),
            "enable_long_term",
        ),
    ]
    for mutate, expected_fragment in cases:
        config = Config()
        mutate(config)
        warnings = config.validate()
        assert any(
            expected_fragment in w for w in warnings
        ), f"expected a warning mentioning {expected_fragment!r}, got {warnings}"


def test_streaming_enabled_by_default() -> None:
    """WS streaming is on by default; explicit false still disables it."""
    assert Config().web.streaming_enabled is True
    assert Config.from_dict({}).web.streaming_enabled is True
    loaded = Config.from_dict({"web": {"streaming_enabled": False}})
    assert loaded.web.streaming_enabled is False


def test_validate_flags_bad_tools_gate_and_search_provider() -> None:
    config = Config()
    config.tools.gate = "sometimes"
    assert any("tools.gate" in w for w in config.validate())

    config = Config()
    config.web_search.provider = "bing"
    assert any("web_search.provider" in w for w in config.validate())
