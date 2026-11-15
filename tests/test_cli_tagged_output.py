"""Tagged-output ([emotion] English || Japanese) wiring in the CLI chat loop.

Mirrors the web path's tagged-grammar parsing (see test_streaming_ws.py's
FakeTTS pattern) but drives amadeus.cli.ChatSession directly: a scripted
DummyProvider stands in for the LLM, a FakeTTS records what text it was
asked to speak. Fully offline — no live provider calls.
"""

from __future__ import annotations

import builtins

from amadeus.cli import ChatSession
from amadeus.config import Config
from amadeus.providers.llm.base import DummyProvider
from amadeus.providers.tts.base import TTSProvider


class FakeTTS(TTSProvider):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(
        self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0
    ) -> bytes:
        self.calls.append(text)
        return f"WAV:{text}".encode()


def _config(tmp_path) -> Config:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = None
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.audio.enable_voice = True
    config.tools.enabled = False
    return config


def _session(config: Config, llm, tts) -> ChatSession:
    return ChatSession(config, llm, tts, stt=None, embedder=None)


def test_cli_tagged_reply_speaks_japanese_and_prints_clean_english(
    tmp_path, monkeypatch, capsys
):
    config = _config(tmp_path)
    config.web.streaming_tagged_output = True
    scripted = DummyProvider(
        scripted_reply=(
            "[happy] Dopamine reinforces reward learning. || ドーパミンは報酬学習を強化する。\n"
            "[relaxed] It also supports memory consolidation. || 記憶の定着も支援する。"
        )
    )
    fake_tts = FakeTTS()
    session = _session(config, scripted, fake_tts)

    inputs = iter(["Tell me about dopamine.", "/quit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))

    session.run()
    out = capsys.readouterr().out

    # Clean joined English is displayed — no raw "[happy]"/"||" grammar leaks.
    assert "Dopamine reinforces reward learning." in out
    assert "It also supports memory consolidation." in out
    assert "||" not in out
    # Dimmed emotion tags are shown alongside the sentences.
    assert "[happy]" in out
    assert "[relaxed]" in out

    # TTS spoke the Japanese half (default voice_language), not raw tagged text.
    assert len(fake_tts.calls) == 1
    spoken = fake_tts.calls[0]
    assert "ドーパミン" in spoken
    assert "[happy]" not in spoken
    assert "||" not in spoken

    # Memory holds the clean English display text, not the tagged grammar.
    assistant_messages = [m for m in session.memory.short_term if m.role == "assistant"]
    assert assistant_messages, "expected the reply to be committed to memory"
    assert "[happy]" not in assistant_messages[0].content
    assert "||" not in assistant_messages[0].content


def test_cli_untagged_reply_falls_back_to_raw_text(tmp_path, monkeypatch, capsys):
    """When the model ignores the tagged grammar, behavior is unchanged:
    the raw text is printed and spoken verbatim."""
    config = _config(tmp_path)
    config.web.streaming_tagged_output = True
    plain_reply = "This is a completely untagged plain reply from the model."
    scripted = DummyProvider(scripted_reply=plain_reply)
    fake_tts = FakeTTS()
    session = _session(config, scripted, fake_tts)

    inputs = iter(["Say something plain.", "/quit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))

    session.run()
    out = capsys.readouterr().out

    assert plain_reply in out
    assert len(fake_tts.calls) == 1
    assert fake_tts.calls[0] == plain_reply


def test_cli_tagged_output_disabled_matches_legacy_behavior(
    tmp_path, monkeypatch, capsys
):
    """With web.streaming_tagged_output off, a tagged-looking reply is
    treated as plain text — the CLI never parses it (dummy/--safe parity)."""
    config = _config(tmp_path)
    config.web.streaming_tagged_output = False
    raw = "[happy] Dopamine reinforces reward learning. || ドーパミンは報酬学習を強化する。"
    scripted = DummyProvider(scripted_reply=raw)
    fake_tts = FakeTTS()
    session = _session(config, scripted, fake_tts)

    inputs = iter(["Tell me about dopamine.", "/quit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))

    session.run()
    out = capsys.readouterr().out

    # Raw text passed straight through: streaming path echoes it verbatim.
    assert raw in out
    assert len(fake_tts.calls) == 1
    assert fake_tts.calls[0] == raw


def test_cli_announces_due_reminders_via_shared_engine_method(
    tmp_path, monkeypatch, capsys
):
    """cli.py's _announce_due_reminders must delegate to the shared
    ChatEngine.due_reminder_lines rather than re-implementing the due-check
    and phrasing (the duplication this task removes)."""
    import datetime as dt

    from amadeus.reminders import ReminderStore

    config = _config(tmp_path)
    config.reminders.enabled = True
    config.reminders.store_path = str(tmp_path / "reminders.json")
    store = ReminderStore(config.reminders.store_path)
    store.add("check the scanner booking", dt.datetime.now() - dt.timedelta(seconds=30))

    scripted = DummyProvider(scripted_reply="[neutral] Fine. || わかったわ。")
    fake_tts = FakeTTS()
    session = _session(config, scripted, fake_tts)

    calls: list[tuple[bool]] = []
    original = session.due_reminder_lines

    def spy(catch_up: bool = False):
        calls.append((catch_up,))
        return original(catch_up=catch_up)

    monkeypatch.setattr(session, "due_reminder_lines", spy)

    inputs = iter(["/quit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))
    session.run()
    out = capsys.readouterr().out

    # Session-start catch-up call happened, routed through the engine method.
    assert any(call == (True,) for call in calls)
    assert "scanner booking" in out
    # A second poll for the same reminder is silent (pop_due marks it announced).
    assert len(fake_tts.calls) == 1
