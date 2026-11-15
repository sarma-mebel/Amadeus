"""Discord text-chat bridge: config parsing, pure decision logic
(should_respond/split_reply/strip_mention), and turn-flow behavior.

None of this imports discord.py — the test environment doesn't have it
installed, and the module's design keeps every decision in plain functions
that take plain data (see amadeus/discord_bridge.py's module docstring).
"""

from pathlib import Path

from amadeus.config import Config, DiscordSettings
from amadeus.discord_bridge import (
    DiscordUserSession,
    handle_incoming_message,
    should_respond,
    split_reply,
    strip_mention,
)
from amadeus.providers.llm.base import ChatMessage

# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_discord_settings_defaults():
    config = Config()
    assert config.discord.enabled is False
    assert config.discord.token is None
    assert config.discord.allowed_user_ids == []
    assert config.discord.reply_to_mentions is True
    assert config.discord.reply_to_dms is True


def test_discord_settings_from_dict():
    config = Config.from_dict(
        {
            "discord": {
                "enabled": True,
                "token": "from-config",
                "allowed_user_ids": [111, 222],
                "reply_to_mentions": False,
                "reply_to_dms": False,
            }
        }
    )
    assert config.discord.enabled is True
    assert config.discord.token == "from-config"
    assert config.discord.allowed_user_ids == [111, 222]
    assert config.discord.reply_to_mentions is False
    assert config.discord.reply_to_dms is False


def test_discord_token_env_fallback(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "from-env")
    config = Config.from_dict({})
    assert config.discord.token == "from-env"

    # Explicit config value wins over the env var.
    config = Config.from_dict({"discord": {"token": "from-config"}})
    assert config.discord.token == "from-config"


def test_discord_token_none_without_env(monkeypatch):
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    assert Config.from_dict({}).discord.token is None


# ---------------------------------------------------------------------------
# should_respond matrix
# ---------------------------------------------------------------------------


def _settings(**overrides) -> DiscordSettings:
    settings = DiscordSettings()
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


def test_should_respond_ignores_bots():
    assert not should_respond(
        author_id=1,
        is_bot=True,
        is_dm=True,
        mentions_bot=False,
        settings=_settings(),
    )


def test_should_respond_ignores_ambient_channel_messages():
    # Not a DM, no mention: never respond (no channel reading at all).
    assert not should_respond(
        author_id=1,
        is_bot=False,
        is_dm=False,
        mentions_bot=False,
        settings=_settings(),
    )


def test_should_respond_dm_allowed_by_default():
    assert should_respond(
        author_id=1,
        is_bot=False,
        is_dm=True,
        mentions_bot=False,
        settings=_settings(),
    )


def test_should_respond_dm_denied_when_flag_off():
    assert not should_respond(
        author_id=1,
        is_bot=False,
        is_dm=True,
        mentions_bot=False,
        settings=_settings(reply_to_dms=False),
    )


def test_should_respond_mention_allowed_by_default():
    assert should_respond(
        author_id=1,
        is_bot=False,
        is_dm=False,
        mentions_bot=True,
        settings=_settings(),
    )


def test_should_respond_mention_denied_when_flag_off():
    assert not should_respond(
        author_id=1,
        is_bot=False,
        is_dm=False,
        mentions_bot=True,
        settings=_settings(reply_to_mentions=False),
    )


def test_should_respond_allowlist_denies_unlisted_dm():
    settings = _settings(allowed_user_ids=[42])
    assert not should_respond(
        author_id=1, is_bot=False, is_dm=True, mentions_bot=False, settings=settings
    )
    assert should_respond(
        author_id=42, is_bot=False, is_dm=True, mentions_bot=False, settings=settings
    )


def test_should_respond_allowlist_denies_unlisted_mention():
    settings = _settings(allowed_user_ids=[42])
    assert not should_respond(
        author_id=7, is_bot=False, is_dm=False, mentions_bot=True, settings=settings
    )
    assert should_respond(
        author_id=42, is_bot=False, is_dm=False, mentions_bot=True, settings=settings
    )


def test_should_respond_empty_allowlist_means_anyone():
    settings = _settings(allowed_user_ids=[])
    assert should_respond(
        author_id=999999,
        is_bot=False,
        is_dm=True,
        mentions_bot=False,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# split_reply
# ---------------------------------------------------------------------------


def test_split_reply_short_passthrough():
    assert split_reply("Hello there.") == ["Hello there."]


def test_split_reply_empty():
    assert split_reply("") == []


def test_split_reply_splits_long_text_at_sentence_boundaries():
    sentence = "This is a sentence that repeats itself over and over. "
    text = sentence * 100  # well over 2000 chars
    chunks = split_reply(text, limit=200)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 200
    # Every chunk (but possibly the last) should end on a sentence boundary.
    for chunk in chunks[:-1]:
        assert chunk.rstrip().endswith(".")
    # Rejoining recovers all the sentences (whitespace-normalized).
    rejoined = " ".join(chunks)
    assert rejoined.count("This is a sentence") == text.count("This is a sentence")


def test_split_reply_hard_splits_a_single_giant_sentence():
    # One "sentence" (no periods) far longer than the limit: must still be
    # split into pieces that each fit, with nothing lost.
    giant = "a" * 5000
    chunks = split_reply(giant, limit=2000)
    assert len(chunks) == 3
    assert all(len(c) <= 2000 for c in chunks)
    assert "".join(chunks) == giant


def test_split_reply_respects_custom_limit():
    text = "Short sentence one. Short sentence two. Short sentence three."
    chunks = split_reply(text, limit=30)
    assert all(len(c) <= 30 for c in chunks)
    assert len(chunks) > 1


# ---------------------------------------------------------------------------
# strip_mention
# ---------------------------------------------------------------------------


def test_strip_mention_removes_plain_mention():
    assert strip_mention("<@123> hello there", 123) == "hello there"


def test_strip_mention_removes_nickname_mention():
    assert strip_mention("<@!123> hello there", 123) == "hello there"


def test_strip_mention_leaves_other_ids_alone():
    assert strip_mention("<@999> hello <@123>", 123) == "<@999> hello"


def test_strip_mention_normalizes_whitespace():
    assert strip_mention("<@123>   hello   there  ", 123) == "hello there"


def test_strip_mention_empty_content():
    assert strip_mention("", 123) == ""


def test_strip_mention_no_mention_present():
    assert strip_mention("just a message", 123) == "just a message"


# ---------------------------------------------------------------------------
# Turn flow with a scripted engine (no discord.py, no live API calls)
# ---------------------------------------------------------------------------


class ScriptedProvider:
    """Minimal fake LLM: returns a fixed reply, records every call."""

    name = "scripted"
    model = None

    def __init__(self, reply: str = "Fine. Ask your question already.") -> None:
        self.reply = reply
        self.calls: list[list[ChatMessage]] = []
        self.last_usage = None

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        return self.reply

    def supports_tools(self):
        return False


def _isolated_config(tmp_path: Path) -> Config:
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory_store.json")
    config.memory.embeddings_path = str(tmp_path / "memory_embeddings.json")
    config.memory.profile_path = str(tmp_path / "memory_profile.json")
    config.memory.enable_long_term = False
    config.tools.enabled = False
    return config


def test_message_produces_a_reply(tmp_path: Path):
    config = _isolated_config(tmp_path)
    llm = ScriptedProvider(reply="Tch, fine. Here's your answer.")
    sessions: dict[int, DiscordUserSession] = {}

    reply = handle_incoming_message(
        sessions=sessions,
        config=config,
        llm=llm,
        embedder=None,
        discord_user_id=555,
        text="What is the boiling point of water?",
    )

    assert reply == "Tch, fine. Here's your answer."
    assert 555 in sessions
    session = sessions[555]
    assert session.user_id == "discord-555"
    roles = [m.role for m in session.memory.short_term]
    assert roles == ["user", "assistant"]


def test_message_reuses_the_same_session_per_user(tmp_path: Path):
    config = _isolated_config(tmp_path)
    llm = ScriptedProvider()
    sessions: dict[int, DiscordUserSession] = {}

    handle_incoming_message(
        sessions=sessions,
        config=config,
        llm=llm,
        embedder=None,
        discord_user_id=1,
        text="hi",
    )
    handle_incoming_message(
        sessions=sessions,
        config=config,
        llm=llm,
        embedder=None,
        discord_user_id=1,
        text="hi again",
    )

    assert len(sessions) == 1
    roles = [m.role for m in sessions[1].memory.short_term]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_message_scopes_memory_per_discord_user(tmp_path: Path):
    config = _isolated_config(tmp_path)
    llm = ScriptedProvider()
    sessions: dict[int, DiscordUserSession] = {}

    handle_incoming_message(
        sessions=sessions,
        config=config,
        llm=llm,
        embedder=None,
        discord_user_id=1,
        text="hi",
    )
    handle_incoming_message(
        sessions=sessions,
        config=config,
        llm=llm,
        embedder=None,
        discord_user_id=2,
        text="hi",
    )

    assert sessions[1].user_id == "discord-1"
    assert sessions[2].user_id == "discord-2"
    assert sessions[1].memory.persist_path != sessions[2].memory.persist_path


def test_goodbye_finalizes_the_session(tmp_path: Path):
    config = _isolated_config(tmp_path)
    config.memory.enable_farewell = True
    llm = ScriptedProvider(reply="Should never be called for a goodbye turn")
    sessions: dict[int, DiscordUserSession] = {}

    # Establish a conversation first (should_bid_farewell requires one).
    handle_incoming_message(
        sessions=sessions,
        config=config,
        llm=llm,
        embedder=None,
        discord_user_id=9,
        text="hello there",
    )
    session = sessions[9]
    # The farewell line comes from summary_provider.generate(); point it at
    # a distinct scripted reply so we can tell it apart from a normal turn.
    farewell_provider = ScriptedProvider(reply="Go on, then. Don't be a stranger.")
    session.summary_provider = farewell_provider

    reply = handle_incoming_message(
        sessions=sessions,
        config=config,
        llm=llm,
        embedder=None,
        discord_user_id=9,
        text="gotta go, bye!",
    )

    assert reply == "Go on, then. Don't be a stranger."
    # The goodbye + farewell were committed as a real turn.
    roles = [m.role for m in session.memory.short_term]
    assert roles[-2:] == ["user", "assistant"]
    assert session.memory.short_term[-2].content == "gotta go, bye!"
    assert session.memory.short_term[-1].content == reply


def test_goodbye_without_farewell_falls_back_to_normal_turn(tmp_path: Path):
    # enable_farewell defaults False -> close_session_farewell always
    # returns None -> the goodbye-shaped message is treated as a normal turn
    # instead of silently producing nothing.
    config = _isolated_config(tmp_path)
    llm = ScriptedProvider(reply="A normal reply.")
    sessions: dict[int, DiscordUserSession] = {}

    reply = handle_incoming_message(
        sessions=sessions,
        config=config,
        llm=llm,
        embedder=None,
        discord_user_id=3,
        text="bye!",
    )

    assert reply == "A normal reply."


class FailingProvider:
    name = "failing"
    model = None
    last_usage = None

    def generate(self, messages, **kwargs):
        raise RuntimeError("boom")

    def supports_tools(self):
        return False


def test_failed_turn_returns_canned_reply_and_counts(tmp_path: Path):
    from amadeus import telemetry

    telemetry.reset()
    config = _isolated_config(tmp_path)
    llm = FailingProvider()
    sessions: dict[int, DiscordUserSession] = {}

    reply = handle_incoming_message(
        sessions=sessions,
        config=config,
        llm=llm,
        embedder=None,
        discord_user_id=77,
        text="hello?",
    )

    assert reply  # some canned in-character failure line, never a crash
    assert telemetry.snapshot().get("discord.turn_failures") == 1
    # The failed user message must have been rolled back, not left orphaned.
    session = sessions[77]
    assert [m.role for m in session.memory.short_term] == []


def test_reply_and_denied_counters(tmp_path: Path):
    from amadeus import telemetry

    telemetry.reset()
    config = _isolated_config(tmp_path)
    llm = ScriptedProvider()
    sessions: dict[int, DiscordUserSession] = {}

    handle_incoming_message(
        sessions=sessions,
        config=config,
        llm=llm,
        embedder=None,
        discord_user_id=1,
        text="hello",
    )

    assert telemetry.snapshot().get("discord.replies") == 1
