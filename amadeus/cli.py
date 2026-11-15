"""Command line interface for the Amadeus assistant."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import select
import sys
import textwrap
from pathlib import Path
from typing import List

from colorama import Fore, Style, init as colorama_init

from . import telemetry
from .audio.ptt import PushToTalkController
from .config import Config, save_default_config
from .logging_utils import get_logger, setup_logging
from .engine import ChatEngine
from .memory.backup import maybe_daily_backup
from .memory.manager import PROTECTED, EmbeddingProvider
from .paths import migrate_legacy_cwd_files
from .memory.profile import familiarity_guardedness
from .persona import build_system_prompt
from .provider_loader import load_embedder, load_llm, load_stt, load_tts
from .providers.llm.base import ChatMessage, DummyProvider, LLMProvider
from .providers.stt.base import STTProvider
from .providers.tts.base import TTSProvider
from .web.streaming import (
    build_tagged_instructions,
    parse_complete_reply_events,
    speech_text,
)

LOGGER = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kurisu-inspired voice assistant")
    parser.add_argument("--config", type=str, help="Path to config.yaml", default=None)
    parser.add_argument(
        "--init-config", action="store_true", help="Write default config and exit"
    )
    parser.add_argument("--text", action="store_true", help="Force text-only mode")
    parser.add_argument(
        "--safe", action="store_true", help="Enable safe/classroom mode"
    )
    return parser


def ensure_transcript_path(path: str) -> Path:
    dest = Path(path)
    dest.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return dest / f"session-{timestamp}.jsonl"


def log_transcript(path: Path, entry: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


class ChatSession(ChatEngine):
    """Manage the interactive chat loop.

    Turn orchestration lives in :class:`amadeus.engine.ChatEngine`; this
    adapter adds the terminal-only parts: stdout streaming, transcript
    logging, push-to-talk input, and local audio playback.
    """

    def __init__(
        self,
        config: Config,
        llm: LLMProvider,
        tts: TTSProvider | None,
        stt: STTProvider | None,
        embedder: EmbeddingProvider | None,
    ) -> None:
        # Rolling daily backup of the memory files before this session
        # touches them (same hook the web SessionStore runs per user).
        maybe_daily_backup(config, None)
        super().__init__(config, llm, embedder=embedder)
        self.tts = tts
        self.stt = stt
        self.persona = dataclasses.replace(config.persona)
        if (
            self.persona.emotional_guardedness is None
            and self.memory.profile is not None
        ):
            self.persona.emotional_guardedness = familiarity_guardedness(
                self.persona.tsundere_level, self.memory.profile.session_count
            )
        self.system_prompt = build_system_prompt(
            self.persona,
            state=self.kurisu_state,
            profile_block=self.profile_block,
            character=getattr(config, "character", None),
        )
        self.transcript_path = ensure_transcript_path(config.transcript_path)
        colorama_init()
        self.ptt = (
            PushToTalkController(config.audio, stt)
            if stt and config.audio.enable_voice
            else None
        )
        self.muted = False
        # Local reminders (poll-driven; no scheduler thread). Due-reminder
        # reads/phrasing live on the shared ChatEngine.due_reminder_lines
        # (its own lazily-constructed ReminderStore) so the CLI and web
        # front-ends announce identically; this flag just gates the PTT idle
        # poll's throttle below.
        self.reminders_enabled = bool(
            getattr(config, "reminders", None) and config.reminders.enabled
        )
        self._last_reminder_check = 0.0

    def run(self) -> None:
        now = dt.datetime.utcnow().strftime("%H:%M:%S")
        print(
            Fore.CYAN
            + f"[{now}] Future Gadget Lab systems online. Ask Kurisu anything."
            + Style.RESET_ALL
        )
        print(
            Fore.MAGENTA
            + "Commands: /quit, /save, /mute, /status, /remember <text>, /forget <text>, /memory"
            + Style.RESET_ALL
        )

        greeting = self.open_session_greeting()
        if greeting:
            self._print_reply(greeting)
            if self.tts and not self.muted:
                try:
                    audio = self.tts.synthesize(
                        greeting,
                        speaking_rate=self.config.persona.speaking_rate,
                        pitch_shift=self.config.persona.pitch_shift,
                    )
                    from .audio.voice import play_audio

                    play_audio(audio)
                except Exception as exc:  # pragma: no cover - runtime audio errors
                    LOGGER.error("Failed to play greeting audio: %s", exc)

        # Session-start catch-up: announce reminders that came due while closed.
        self._announce_due_reminders(catch_up=True)

        # NOTE: the system prompt is inserted at context position 0 for every
        # request; do not also add it to memory or it gets sent twice and
        # pollutes summaries.
        if self.ptt:
            self.ptt.start()
        try:
            self._run_chat_loop()
        finally:
            # Always finalize — not just when episodes/reflection are on. The
            # session-final pass also owns fact extraction for short sessions,
            # persisting retrieval-reinforcement bumps, and stashing the
            # session tone for the next boundary; skipping it silently lost
            # all three whenever episodes and reflection were both off.
            memory_cfg = self.config.memory
            if (
                getattr(memory_cfg, "enable_fact_extraction", False)
                or memory_cfg.enable_episodes
                or getattr(memory_cfg, "enable_reflection", False)
            ):
                print(Fore.YELLOW + "Consolidating memory…" + Style.RESET_ALL)
            self.memory.finalize_session(wait=True)
            print(
                Fore.MAGENTA
                + "Shutting down Future Gadget Lab systems. El Psy Kongroo."
                + Style.RESET_ALL
            )
            if self.ptt:
                self.ptt.stop()

    def _run_chat_loop(self) -> None:
        while True:
            # Blocking-input mode fires reminders at the next turn boundary;
            # PTT mode fires them from the idle poll loop (see _get_user_input).
            self._announce_due_reminders()
            try:
                user_input = self._get_user_input()
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if not user_input.strip():
                continue
            if user_input.startswith("/"):
                if user_input == "/quit":
                    # One warm send-off (anti-dark-pattern: she lets you go
                    # gladly). Deliberately not fired on Ctrl-C — an abrupt
                    # interrupt shouldn't cost an LLM call on the way out.
                    farewell = self.close_session_farewell()
                    if farewell:
                        self.memory.add_message("assistant", farewell)
                        self._speak_line(farewell)
                    break
                if user_input == "/save":
                    print(
                        Fore.YELLOW
                        + f"Transcript saved to {self.transcript_path}"
                        + Style.RESET_ALL
                    )
                    continue
                if user_input == "/mute":
                    self.muted = not self.muted
                    state = "muted" if self.muted else "unmuted"
                    print(Fore.YELLOW + f"Speech playback {state}." + Style.RESET_ALL)
                    continue
                if user_input == "/status":
                    self._show_status()
                    continue
                if user_input.startswith("/remember"):
                    text = user_input[len("/remember") :].strip()
                    if not text:
                        print(
                            Fore.YELLOW
                            + "Usage: /remember <something to remember>"
                            + Style.RESET_ALL
                        )
                    elif self.memory.remember(text):
                        print(
                            Fore.YELLOW
                            + f"Committed to memory: {text}"
                            + Style.RESET_ALL
                        )
                    else:
                        print(
                            Fore.YELLOW
                            + "Long-term memory is disabled (enable memory.enable_long_term "
                            + "and install the embedding extra)."
                            + Style.RESET_ALL
                        )
                    continue
                if user_input.startswith("/forget"):
                    text = user_input[len("/forget") :].strip()
                    if not text:
                        print(
                            Fore.YELLOW
                            + "Usage: /forget <what to forget>"
                            + Style.RESET_ALL
                        )
                    else:
                        forgotten = self.memory.forget(text)
                        if forgotten is PROTECTED:
                            print(
                                Fore.YELLOW
                                + "That memory is pinned. Unpin it before you ask me to forget it."
                                + Style.RESET_ALL
                            )
                        elif forgotten:
                            print(
                                Fore.YELLOW
                                + f"Forgotten: {forgotten}"
                                + Style.RESET_ALL
                            )
                        else:
                            print(
                                Fore.YELLOW
                                + "No matching memory found."
                                + Style.RESET_ALL
                            )
                    continue
                if user_input == "/memory":
                    self._show_memory()
                    continue
                print(Fore.YELLOW + "Unknown command" + Style.RESET_ALL)
                continue

            timestamp = dt.datetime.utcnow().strftime("%H:%M:%S")
            print(Fore.GREEN + f"[{timestamp}] You » {user_input}" + Style.RESET_ALL)
            use_tagged = bool(getattr(self.config.web, "streaming_tagged_output", True))
            setup = self.prepare_turn(
                user_input,
                self.persona,
                tagged_output=use_tagged,
                tagged_instructions=(
                    build_tagged_instructions(
                        bool(
                            getattr(self.config.web, "streaming_inner_thoughts", False)
                        ),
                        char_name=getattr(
                            getattr(self.config, "character", None), "name", None
                        )
                        or "Kurisu",
                    )
                    if use_tagged
                    else None
                ),
            )
            self.system_prompt = setup.system_prompt
            context = setup.context

            LOGGER.info("Dispatching request to %s", self.llm.name)
            refusal = setup.refusal
            cancelled = False
            # speech_line is what gets spoken; defaults to the displayed
            # response text and is only overridden when the tagged
            # "[emotion] en || ja" grammar parses cleanly below.
            speech_line: str | None = None
            if refusal:
                response = refusal
                self._print_reply(response)
            else:
                if self._tools_active() and self._should_use_tools(user_input):
                    self._print_thinking()
                    response = self._run_tool_loop(context)
                    self._print_reply(response)
                else:
                    self._print_thinking()
                    # Tagged mode can't be echoed token-by-token: a live
                    # partial "[emotion] en || ja" line would leak the
                    # grammar onto the terminal. Only plain (untagged) chat
                    # keeps the character-by-character stream; tagged replies
                    # are collected in full, then parsed and printed clean —
                    # same one-shot shape as the non-streaming branch below.
                    if self.llm.supports_stream() and not use_tagged:
                        try:
                            response, cancelled = self._collect_stream(context)
                        except Exception as exc:  # pragma: no cover - provider errors
                            LOGGER.error(
                                "Streaming failed, falling back to blocking generate(): %s",
                                exc,
                            )
                            response = self.llm.generate(context)
                            telemetry.record_llm_usage(self.llm, "chat")
                            self._print_reply(response)
                    else:
                        raw_response = self.llm.generate(context)
                        telemetry.record_llm_usage(self.llm, "chat")
                        response, speech_line = self._present_tagged_reply(
                            raw_response, use_tagged
                        )
            self.complete_turn(user_input, response)
            log_transcript(
                self.transcript_path,
                {
                    "timestamp": dt.datetime.utcnow().isoformat(),
                    "user": user_input,
                    "assistant": response,
                    "cancelled": cancelled,
                },
            )
            if self.tts and not self.muted and response and not cancelled:
                try:
                    audio = self.tts.synthesize(
                        speech_line if speech_line is not None else response,
                        speaking_rate=self.config.persona.speaking_rate,
                        pitch_shift=self.config.persona.pitch_shift,
                    )
                    from .audio.voice import play_audio

                    play_audio(audio)
                except Exception as exc:  # pragma: no cover - runtime audio errors
                    LOGGER.error("Failed to play synthesized speech: %s", exc)

    # Tool-calling methods (_tools_active / _should_use_tools / _run_tool_loop)
    # live on the base ChatEngine so the web frontend shares them.

    def _speak_line(self, line: str) -> None:
        """Print + speak a single line, reusing the standard TTS path."""
        self._print_reply(line)
        if self.tts and not self.muted:
            try:
                audio = self.tts.synthesize(
                    line,
                    speaking_rate=self.config.persona.speaking_rate,
                    pitch_shift=self.config.persona.pitch_shift,
                )
                from .audio.voice import play_audio

                play_audio(audio)
            except Exception as exc:  # pragma: no cover - runtime audio errors
                LOGGER.error("Failed to speak reminder: %s", exc)

    def _announce_due_reminders(self, catch_up: bool = False) -> None:
        """Speak any reminders that are due. `catch_up` phrases missed ones as
        'while you were away'. Safe to call frequently — no-op if none due.

        Delegates the actual due-check + phrasing to the shared
        ``ChatEngine.due_reminder_lines`` so the CLI and web front-ends
        announce reminders identically; this method only adds the CLI's
        print+speak side effects.
        """
        for line in self.due_reminder_lines(catch_up=catch_up):
            self._speak_line(line)

    def _print_thinking(self) -> None:
        now = dt.datetime.utcnow().strftime("%H:%M:%S")
        print(Fore.YELLOW + f"[{now}] Kurisu is thinking…" + Style.RESET_ALL)

    def _print_reply(self, response: str) -> None:
        reply_time = dt.datetime.utcnow().strftime("%H:%M:%S")
        print(Fore.CYAN + f"[{reply_time}] Kurisu » {response}" + Style.RESET_ALL)

    def _present_tagged_reply(
        self, raw_response: str, use_tagged: bool
    ) -> tuple[str, str | None]:
        """Parse+print a plain-chat reply through the tagged grammar.

        Mirrors the web path's ``[emotion] English || Japanese`` parsing
        (``amadeus.web.streaming.parse_complete_reply_events``): on a clean
        parse, prints the joined English text (each sentence's emotion tag
        dimmed ahead of it) and returns the Japanese half for TTS. When the
        model didn't follow the format (or the feature is off), this is a
        no-op — the raw text is printed and returned exactly as before, so
        the untagged behavior is unchanged byte-for-byte.

        Returns ``(display_text, speech_text_or_None)``; ``None`` speech
        text tells the caller to speak ``display_text`` as-is (the untagged
        fallback path).
        """
        if use_tagged and raw_response:
            sentences, _thoughts = parse_complete_reply_events(raw_response)
            if sentences:
                reply_time = dt.datetime.utcnow().strftime("%H:%M:%S")
                sys.stdout.write(Fore.CYAN + f"[{reply_time}] Kurisu » ")
                for i, sentence in enumerate(sentences):
                    if i > 0:
                        sys.stdout.write(" ")
                    if sentence.en:
                        sys.stdout.write(
                            Style.DIM
                            + f"[{sentence.emotion_raw or sentence.emotion}] "
                            + Style.RESET_ALL
                        )
                        sys.stdout.write(Fore.CYAN + sentence.en)
                sys.stdout.write(Style.RESET_ALL + "\n")
                sys.stdout.flush()
                display_text = " ".join(s.en for s in sentences if s.en)
                japanese_text = " ".join(s.ja for s in sentences if s.ja)
                voice_language = getattr(
                    self.config.audio, "voice_language", "japanese"
                )
                speech = speech_text(display_text, japanese_text, voice_language)
                return display_text, speech
        # Fallback: the model didn't follow the tagged grammar (or tagging is
        # off) — keep exactly the pre-existing raw-text behavior.
        self._print_reply(raw_response)
        return raw_response, None

    def _collect_stream(self, context: List[ChatMessage]) -> tuple[str, bool]:
        timestamp = dt.datetime.utcnow().strftime("%H:%M:%S")
        sys.stdout.write(Fore.CYAN + f"[{timestamp}] Kurisu » ")
        sys.stdout.flush()
        pieces: List[str] = []
        cancelled = False
        try:
            for chunk in self.llm.stream(context):
                if not chunk:
                    continue
                pieces.append(chunk)
                sys.stdout.write(chunk)
                sys.stdout.flush()
        except KeyboardInterrupt:
            cancelled = True
        finally:
            sys.stdout.write(Style.RESET_ALL + "\n")
            sys.stdout.flush()
        # After the stream generator finished (usage rides its final chunk);
        # an interrupted stream that never saw usage is a no-op.
        telemetry.record_llm_usage(self.llm, "chat")
        if cancelled:
            print(Fore.YELLOW + "[cancelled] Response interrupted." + Style.RESET_ALL)
        return "".join(pieces), cancelled

    def _show_status(self) -> None:
        persona = self.config.persona
        summary = self.memory.summary or "No summary yet."
        summary_snippet = (
            textwrap.shorten(summary, width=120, placeholder="…")
            if summary
            else summary
        )
        short_term_count = len(self.memory.short_term)
        long_term_count = (
            self.memory.long_term_count if self.config.memory.enable_long_term else 0
        )
        llm_name = getattr(self.llm, "name", self.llm.__class__.__name__)
        tts_name = self.tts.__class__.__name__ if self.tts else "disabled"
        # Same slow-engine surfacing as /api/status: the qwen3 provider's
        # upstream fallback is ~10x slower and otherwise invisible here.
        _tts_inner = getattr(self.tts, "provider", self.tts)
        if getattr(_tts_inner, "_engine", None) == "upstream":
            tts_name += " (SLOW fallback — pip install faster-qwen3-tts)"
        stt_name = self.stt.__class__.__name__ if self.stt else "disabled"
        summary_name = (
            getattr(self.summary_provider, "name", "truncate")
            if self.summary_provider
            else "truncate"
        )
        summary_strategy = self.memory.summary_info.get("strategy", "truncate")

        print(Fore.MAGENTA + "--- Kurisu Status ---" + Style.RESET_ALL)
        print(
            f"LLM: {llm_name} | Summary: {summary_name} | TTS: {tts_name} | STT: {stt_name} | Muted: {'yes' if self.muted else 'no'}"
        )
        print(
            "Persona: rate={:.2f}, pitch={:.2f}, tsundere={:.2f}, scientific={:.2f}, safe_mode={}".format(
                persona.speaking_rate,
                persona.pitch_shift,
                persona.tsundere_level,
                persona.scientific_formality,
                "on" if persona.safe_mode else "off",
            )
        )
        print(f"Memory: short_term={short_term_count}, long_term={long_term_count}")
        print(f"Summary ({summary_strategy}): {summary_snippet}")

    def _show_memory(self) -> None:
        overview = self.memory.memory_overview()
        profile = overview.get("profile") or {}
        counts = overview.get("counts") or {}
        print(Fore.MAGENTA + "--- What Kurisu Remembers ---" + Style.RESET_ALL)
        facts = profile.get("facts") or {}
        if facts:
            print("About you:")
            for key, value in facts.items():
                print(f"  - {key}: {value}")
        if profile.get("session_count"):
            since = f" since {profile['first_met']}" if profile.get("first_met") else ""
            print(f"Sessions: {profile['session_count']}{since}")
        if profile.get("last_episode"):
            print(f"Last time: {profile['last_episode']}")
        if profile.get("open_threads"):
            print("Open threads: " + "; ".join(profile["open_threads"]))
        stored = overview.get("facts") or []
        if stored:
            print("Stored facts:")
            for text in stored:
                print(f"  - {text}")
        print(
            "Totals: {} facts, {} episodes, {} reflections".format(
                counts.get("facts", 0),
                counts.get("episodes", 0),
                counts.get("reflections", 0),
            )
        )
        if not (facts or stored or profile.get("last_episode")):
            print("Nothing yet — enable memory features or use /remember.")

    def _get_user_input(self) -> str:
        prompt = Fore.GREEN + "You » " + Style.RESET_ALL
        if not self.ptt:
            return input(prompt)
        sys.stdout.write(prompt)
        sys.stdout.flush()
        buffer = ""
        while True:
            if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                buffer = sys.stdin.readline()
                return buffer.rstrip("\n")
            transcript = self.ptt.poll()
            if transcript:
                print(Fore.YELLOW + f"[voice] {transcript}" + Style.RESET_ALL)
                return transcript
            # Fire due reminders while idle, throttled to ~5s (loop wakes 10x/s).
            if self.reminders_enabled:
                now_ts = dt.datetime.now().timestamp()
                if now_ts - self._last_reminder_check >= 5.0:
                    self._last_reminder_check = now_ts
                    self._announce_due_reminders()
                    # Re-print the prompt consumed by any reminder output.
                    sys.stdout.write(prompt)
                    sys.stdout.flush()


def _prompt_provider_setup(path: Path) -> None:
    """Interactive follow-up to --init-config: offer to wire a real LLM key.

    Without this the freshly written config boots straight into the silent
    dummy provider. Only runs on a TTY; any refusal/EOF leaves the file as
    save_default_config wrote it.
    """

    if not sys.stdin.isatty():
        return
    print("Amadeus replies with canned dummy text until an LLM API key is configured.")
    try:
        choice = (
            input("Set up a provider now? [gemini/openai/skip] (gemini): ")
            .strip()
            .lower()
            or "gemini"
        )
        if choice not in ("gemini", "openai"):
            print(
                "Skipped — add a key later in the web UI settings or by editing the config."
            )
            return
        key_url = (
            "https://aistudio.google.com/app/apikey (free)"
            if choice == "gemini"
            else "https://platform.openai.com/api-keys"
        )
        print(f"Get a key at: {key_url}")
        import getpass

        key = getpass.getpass(f"{choice.capitalize()} API key (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        print(
            "\nSkipped — add a key later in the web UI settings or by editing the config."
        )
        return
    # Same loose shape check as the web settings API: real keys are long and
    # have no internal whitespace (catches pasted chat fragments).
    if len(key) < 20 or any(ch.isspace() for ch in key):
        print("That doesn't look like an API key — leaving the config unchanged.")
        return

    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("providers", {})["llm"] = choice
    data[f"{choice}_api_key"] = key
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    print(f"Saved: providers.llm={choice} with your key in {path}. She's awake now.")


def main(argv: List[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.init_config:
        path = Path(args.config or "config.yaml")
        existed = path.exists()
        save_default_config(path)
        if existed:
            print(f"Config already exists at {path}; leaving it untouched.")
            return 0
        print(f"Wrote default config to {path}")
        _prompt_provider_setup(path)
        return 0

    config = Config.load(Path(args.config) if args.config else None)
    if args.safe:
        config.persona.safe_mode = True

    setup_logging(config)
    migrate_legacy_cwd_files(config)

    llm = load_llm(config)
    if config.providers.llm != "dummy" and isinstance(llm, DummyProvider):
        print(
            f"{Fore.YELLOW}!! providers.llm='{config.providers.llm}' could not be "
            "loaded — running on the canned DUMMY provider. Check the API key / "
            f"extras, then restart.{Style.RESET_ALL}"
        )
    stt = None if args.text else load_stt(config)
    tts = None if args.text or not config.audio.enable_voice else load_tts(config)
    embedder = load_embedder(config)

    session = ChatSession(config, llm, tts, stt, embedder)
    session.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
