"""Optional Discord text-chat bridge (DMs + @mentions), no voice, no slash
commands. Shares the same ChatEngine/memory tiers as the CLI: one
:class:`~amadeus.engine.ChatEngine` per Discord user id, built lazily on
that user's first message and kept for the life of the process.

Design mirrors the CLI (``amadeus/cli.py``), not the web app: no FastAPI,
no streaming, no tagged bilingual output — just a plain English reply per
turn (``ChatEngine.prepare_turn`` -> ``llm.generate`` ->
``ChatEngine.complete_turn``), same as ``ChatSession._run_chat_loop``'s
non-streaming branch.

All decision logic (``should_respond``, ``split_reply``, ``strip_mention``)
is pure and takes plain data, so it is fully testable without the
``discord.py`` package installed — only :func:`run` and
:class:`DiscordBridgeClient` touch the real library, imported lazily.

Run: ``python -m amadeus.discord_bridge --config config.local.yaml``.
See docs/discord.md for bot setup (token, intents, invite URL).
"""

from __future__ import annotations

import argparse
import dataclasses
import re
from pathlib import Path
from typing import Iterable, Optional

from . import telemetry
from .config import Config, DiscordSettings
from .engine import ChatEngine
from .farewell import is_goodbye
from .logging_utils import get_logger, setup_logging
from .memory.backup import user_scoped_path
from .paths import migrate_legacy_cwd_files
from .persona import build_system_prompt
from .provider_loader import load_embedder, load_llm

LOGGER = get_logger(__name__)

DISCORD_REPLY_LIMIT = 2000


# ---------------------------------------------------------------------------
# Pure decision logic — no discord.py import required to exercise any of this.
# ---------------------------------------------------------------------------


def should_respond(
    *,
    author_id: int,
    is_bot: bool,
    is_dm: bool,
    mentions_bot: bool,
    settings: DiscordSettings,
) -> bool:
    """Whether the bridge should generate a reply for this incoming message.

    Bots are always ignored (never reply to another bot, avoids loops).
    Outside a DM or an explicit @mention, the bridge never responds at all
    (no ambient channel reading). ``allowed_user_ids`` empty means anyone
    may talk to it in the enabled channels; non-empty restricts BOTH DMs
    and mentions to that allowlist.
    """
    if is_bot:
        return False
    if not is_dm and not mentions_bot:
        return False
    if is_dm and not settings.reply_to_dms:
        return False
    if mentions_bot and not is_dm and not settings.reply_to_mentions:
        return False
    if settings.allowed_user_ids and author_id not in settings.allowed_user_ids:
        return False
    return True


def strip_mention(content: str, bot_user_id: int) -> str:
    """Remove this bot's mention token(s) from a message, tidying whitespace.

    ``@Kurisu what's the weather`` (mentions render as ``<@123.../>`` in the
    raw content) becomes ``what's the weather`` — the model should never see
    the raw mention token in the user turn.
    """
    pattern = re.compile(rf"<@!?{bot_user_id}>")
    stripped = pattern.sub("", content or "")
    return " ".join(stripped.split()).strip()


def _sentence_spans(text: str) -> list[str]:
    """Split into sentence-ish chunks, keeping the trailing punctuation."""
    # Split after ., !, ?, or a newline, keeping the delimiter with the
    # preceding chunk. Falls back to the whole text as one "sentence" when
    # no boundary is found.
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p for p in pieces if p.strip()]


def split_reply(text: str, limit: int = DISCORD_REPLY_LIMIT) -> list[str]:
    """Split ``text`` into chunks that each fit under Discord's message limit.

    Prefers sentence boundaries (so a message never cuts off mid-thought);
    a single sentence longer than ``limit`` on its own is hard-split at the
    limit as a last resort so the bridge can never crash trying to send an
    oversized message.
    """
    text = text or ""
    if len(text) <= limit:
        return [text] if text else []

    chunks: list[str] = []
    current = ""
    for sentence in _sentence_spans(text):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(sentence) <= limit:
            current = sentence
        else:
            # A single sentence longer than the limit: hard-split it.
            for start in range(0, len(sentence), limit):
                piece = sentence[start : start + limit]
                if start + limit >= len(sentence):
                    current = piece
                else:
                    chunks.append(piece)
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Engine wiring — one ChatEngine per Discord user id.
# ---------------------------------------------------------------------------


class DiscordUserSession(ChatEngine):
    """Per-Discord-user chat engine, mirroring the CLI's ``ChatSession``
    minus the terminal/audio bits: plain persona system prompt built once,
    no TTS/STT, no push-to-talk."""

    def __init__(self, config: Config, llm, embedder, *, discord_user_id: int) -> None:
        user_id = f"discord-{discord_user_id}"
        persist_path = user_scoped_path(
            _optional_path(config.memory.persist_path), user_id
        )
        embeddings_path = user_scoped_path(
            _optional_path(config.memory.embeddings_path), user_id
        )
        profile_path = user_scoped_path(
            _optional_path(config.memory.profile_path), user_id
        )
        super().__init__(
            config,
            llm,
            embedder=embedder,
            persist_path=persist_path,
            embeddings_path=embeddings_path,
            profile_path=profile_path,
        )
        self.user_id = user_id
        self.persona = dataclasses.replace(config.persona)
        self.system_prompt = build_system_prompt(
            self.persona,
            state=self.kurisu_state,
            profile_block=self.profile_block,
            character=getattr(config, "character", None),
        )


def _optional_path(value: Optional[str]) -> Optional[Path]:
    return Path(value) if value else None


def run_turn(session: DiscordUserSession, user_text: str) -> str:
    """Run one non-streaming chat turn and return the plain English reply.

    Mirrors ``WebChatSession._handle``'s plain-generate branch (no tools —
    kept deliberately simple for a first cut) with tagged/bilingual output
    turned off, since Discord only ever wants plain text. Never raises: a
    failed provider call replies with the engine's canned failure line and
    counts ``discord.turn_failures``, same "degrade loudly, never crash the
    bot loop" contract as everywhere else in this codebase.
    """
    setup = session.prepare_turn(user_text, session.persona, tagged_output=False)
    if setup.refusal:
        reply = setup.refusal
    else:
        try:
            reply = session._generate_with_health(setup.context)
        except Exception as exc:  # noqa: BLE001 - must never crash the bot loop
            LOGGER.exception("Discord turn failed for %s", session.user_id)
            telemetry.incr("discord.turn_failures")
            from . import provider_health

            kind = provider_health.classify_and_record(exc)
            session.rollback_user_message(user_text)
            if kind == "rate_limited":
                return session._canned_failure_reply("rate_limited")
            if kind == "invalid_key":
                return session._canned_failure_reply("invalid_key")
            return session._canned_failure_reply("lost")
    session.complete_turn(user_text, reply)
    return reply


def handle_incoming_message(
    *,
    sessions: dict[int, DiscordUserSession],
    config: Config,
    llm,
    embedder,
    discord_user_id: int,
    text: str,
) -> Optional[str]:
    """Route one already-allowlist-checked message to the right session and
    return the reply text (goodbye finalizes the session instead of a normal
    turn). Pure aside from session construction/turn execution — no
    discord.py types touch this function, so it's fully unit-testable."""
    session = sessions.get(discord_user_id)
    if session is None:
        session = DiscordUserSession(
            config, llm, embedder, discord_user_id=discord_user_id
        )
        sessions[discord_user_id] = session

    if is_goodbye(text):
        # Mirrors POST /api/farewell's goodbye branch: generate the send-off
        # BEFORE committing anything, so a line that never materializes
        # (cooldown, disabled, no provider) leaves memory untouched and this
        # message can fall through to a normal chat turn instead.
        farewell = session.close_session_farewell(text)
        if farewell:
            session.memory.add_message("user", text)
            session.complete_turn(text, farewell)
            session.memory.finalize_session(wait=False)
            telemetry.incr("discord.farewells")
            return farewell
        # Read as a goodbye but produced no line: treat as a normal turn.

    reply = run_turn(session, text)
    telemetry.incr("discord.replies")
    return reply


# ---------------------------------------------------------------------------
# discord.py wiring — thin layer over the pure functions above.
# ---------------------------------------------------------------------------


def _import_discord():
    try:
        import discord  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise RuntimeError(
            "The Discord bridge needs the optional discord.py package. Install it "
            'with: pip install "amadeus-kurisu[discord]" (or: pip install '
            '"discord.py>=2.3"). See docs/discord.md.'
        ) from exc
    return discord


class DiscordBridgeClient:
    """Thin wrapper around a ``discord.Client`` that delegates every
    decision to the pure functions in this module. Constructed only after
    ``discord`` has been imported successfully."""

    def __init__(self, config: Config, llm, embedder) -> None:
        discord = _import_discord()
        self.config = config
        self.llm = llm
        self.embedder = embedder
        self.sessions: dict[int, DiscordUserSession] = {}

        intents = discord.Intents.default()
        intents.message_content = True
        intents.dm_messages = True
        self.client = discord.Client(intents=intents)

        @self.client.event
        async def on_ready() -> None:  # pragma: no cover - live network glue
            LOGGER.info("Discord bridge connected as %s", self.client.user)

        @self.client.event
        async def on_message(message) -> None:  # pragma: no cover - live network glue
            await self._on_message(message)

    async def _on_message(self, message) -> None:  # pragma: no cover - live glue
        discord = _import_discord()
        if message.author == self.client.user:
            return
        is_dm = isinstance(message.channel, discord.DMChannel)
        mentions_bot = self.client.user in getattr(message, "mentions", [])
        settings = self.config.discord
        if not should_respond(
            author_id=message.author.id,
            is_bot=bool(message.author.bot),
            is_dm=is_dm,
            mentions_bot=mentions_bot,
            settings=settings,
        ):
            telemetry.incr("discord.denied")
            return
        bot_user_id = self.client.user.id
        text = strip_mention(message.content, bot_user_id)
        if not text:
            return
        async with message.channel.typing():
            reply = await _run_blocking(
                handle_incoming_message,
                sessions=self.sessions,
                config=self.config,
                llm=self.llm,
                embedder=self.embedder,
                discord_user_id=message.author.id,
                text=text,
            )
        if not reply:
            return
        for chunk in split_reply(reply):
            await message.channel.send(chunk)

    def run(self, token: str) -> None:  # pragma: no cover - live network glue
        self.client.run(token)


async def _run_blocking(func, *args, **kwargs):  # pragma: no cover - live glue
    import asyncio
    import functools

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discord text-chat bridge for the Amadeus companion"
    )
    parser.add_argument("--config", type=str, help="Path to config.yaml", default=None)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = Config.load(Path(args.config) if args.config else None)
    setup_logging(config)
    migrate_legacy_cwd_files(config)

    if not config.discord.enabled:
        LOGGER.error("discord.enabled is false in config; nothing to do.")
        print(
            "The Discord bridge is disabled. Set discord.enabled: true (and a "
            "token) in your config first — see docs/discord.md."
        )
        return 1
    if not config.discord.token:
        LOGGER.error("discord.token is not set (config or DISCORD_BOT_TOKEN env var).")
        print(
            "No Discord bot token configured. Set discord.token in your config or "
            "the DISCORD_BOT_TOKEN environment variable — see docs/discord.md."
        )
        return 1

    llm = load_llm(config)
    embedder = load_embedder(config)

    client = DiscordBridgeClient(config, llm, embedder)
    print("Discord bridge starting. El Psy Kongroo.")
    client.run(config.discord.token)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
