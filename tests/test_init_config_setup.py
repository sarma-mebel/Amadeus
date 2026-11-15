"""Tests for the interactive provider setup that follows --init-config."""

from __future__ import annotations

import getpass
from pathlib import Path

import yaml

from amadeus.cli import _prompt_provider_setup
from amadeus.config import save_default_config


def _write_default(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    save_default_config(path)
    return path


def _tty(monkeypatch, is_tty: bool) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: is_tty)


def test_non_tty_leaves_config_untouched(tmp_path, monkeypatch):
    path = _write_default(tmp_path)
    before = path.read_text(encoding="utf-8")
    _tty(monkeypatch, False)

    _prompt_provider_setup(path)

    assert path.read_text(encoding="utf-8") == before


def test_gemini_key_is_written(tmp_path, monkeypatch):
    path = _write_default(tmp_path)
    _tty(monkeypatch, True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "")  # default: gemini
    monkeypatch.setattr(
        getpass, "getpass", lambda _prompt: "fake-gemini-key-for-testing-123456"
    )

    _prompt_provider_setup(path)

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["providers"]["llm"] == "gemini"
    assert data["gemini_api_key"] == "fake-gemini-key-for-testing-123456"


def test_openai_choice_writes_openai_key(tmp_path, monkeypatch):
    path = _write_default(tmp_path)
    _tty(monkeypatch, True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "openai")
    monkeypatch.setattr(
        getpass, "getpass", lambda _prompt: "sk-fake-key-for-testing-123456"
    )

    _prompt_provider_setup(path)

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["providers"]["llm"] == "openai"
    assert data["openai_api_key"] == "sk-fake-key-for-testing-123456"
    assert "gemini_api_key" not in {k: v for k, v in data.items() if v}


def test_short_key_is_rejected(tmp_path, monkeypatch):
    path = _write_default(tmp_path)
    before = path.read_text(encoding="utf-8")
    _tty(monkeypatch, True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "gemini")
    monkeypatch.setattr(getpass, "getpass", lambda _prompt: "too-short")

    _prompt_provider_setup(path)

    assert path.read_text(encoding="utf-8") == before


def test_key_with_whitespace_is_rejected(tmp_path, monkeypatch):
    path = _write_default(tmp_path)
    before = path.read_text(encoding="utf-8")
    _tty(monkeypatch, True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "gemini")
    monkeypatch.setattr(
        getpass,
        "getpass",
        lambda _prompt: "pasted chat fragment that is definitely not a key",
    )

    _prompt_provider_setup(path)

    assert path.read_text(encoding="utf-8") == before


def test_skip_leaves_config_untouched(tmp_path, monkeypatch):
    path = _write_default(tmp_path)
    before = path.read_text(encoding="utf-8")
    _tty(monkeypatch, True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "skip")

    _prompt_provider_setup(path)

    assert path.read_text(encoding="utf-8") == before


def test_eof_during_prompt_is_graceful(tmp_path, monkeypatch):
    path = _write_default(tmp_path)
    before = path.read_text(encoding="utf-8")
    _tty(monkeypatch, True)

    def _raise_eof(_prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise_eof)

    _prompt_provider_setup(path)

    assert path.read_text(encoding="utf-8") == before
