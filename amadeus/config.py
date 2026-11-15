"""Configuration loading utilities for the Amadeus assistant."""

from __future__ import annotations

import dataclasses
import logging as logging_module
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - fallback to bundled parser
    from .utils import simple_yaml as yaml


DEFAULT_CONFIG_LOCATIONS = (
    Path.cwd() / "config.local.yaml",
    Path.cwd() / "config.yaml",
    Path.home() / ".config" / "amadeus" / "config.yaml",
    Path(__file__).resolve().parent.parent / "config" / "config.yaml",
    Path(__file__).resolve().parent.parent / "config" / "config.example.yaml",
)


@dataclasses.dataclass
class CharacterSettings:
    """Bring-your-own-character pack. Defaults reproduce the built-in Kurisu.

    A custom character = set ``name``/``full_name`` and supply
    ``system_prompt`` (inline) or ``system_prompt_path`` (file). The custom
    prompt replaces the built-in Kurisu identity/lore/example blocks in
    ``persona.build_system_prompt``; the generic companion scaffolding
    (listening behavior, boundaries, safety, length policy, dials) is kept
    around it. Pair with a VRM entry in static/models/models.json and the
    provider voice fields (qwen3_tts_ref_audio/ref_text or melotts paths)
    for a full character swap. See docs/characters.md.
    """

    # Short in-prompt name ("Speak as {name}...") and the longer form used
    # by the greeting/farewell/brief/diary prompts.
    name: str = "Kurisu"
    full_name: str = "Makise Kurisu"
    # Custom persona prose. Inline text wins over the file path.
    system_prompt: Optional[str] = None
    system_prompt_path: Optional[str] = None
    # Optional id of this character's avatar in static/models/models.json;
    # the UI auto-selects it when the active character changes.
    model_id: Optional[str] = None
    # Optional birthday ("MM-DD", "July 25", or an ISO date — parsed by
    # memory.profile.parse_birthday). Deliberately None by default even
    # though these defaults otherwise reproduce built-in Kurisu: the engine
    # gives the BUILT-IN character July 25 on its own, and a custom pack
    # must opt in explicitly — otherwise every guest character would
    # silently inherit Kurisu's birthday.
    birthday: Optional[str] = None
    # Optional 1–3 sentence voice spec for the satellite prompts (presence
    # lines, greeting/farewell, brief/diary/reach-out) via
    # persona.persona_capsule. Distinct from the manifest's ``voice:`` block
    # (TTS reference audio). When unset, packs fall back to the opening
    # sentences of their system prompt; the built-in Kurisu has her own.
    voice_style: Optional[str] = None

    def resolved_system_prompt(self) -> Optional[str]:
        """The custom persona text, or None to use the built-in Kurisu."""

        if self.system_prompt and self.system_prompt.strip():
            return self.system_prompt.strip()
        if self.system_prompt_path:
            try:
                text = Path(self.system_prompt_path).read_text(encoding="utf-8").strip()
                return text or None
            except OSError:
                logging_module.getLogger(__name__).warning(
                    "character.system_prompt_path %s could not be read; "
                    "falling back to the built-in persona",
                    self.system_prompt_path,
                )
        return None

    @property
    def is_custom(self) -> bool:
        return self.resolved_system_prompt() is not None


@dataclasses.dataclass
class PersonaSettings:
    """Persona knobs that control how Kurisu speaks."""

    speaking_rate: float = 1.0
    pitch_shift: float = 0.0
    tsundere_level: float = 0.5
    scientific_formality: float = 0.6
    safe_mode: bool = False
    # Preferred over tsundere_level when set; derived from relationship
    # familiarity at runtime unless pinned here.
    emotional_guardedness: Optional[float] = None


@dataclasses.dataclass
class AudioSettings:
    """Audio input/output knobs."""

    input_device: Optional[str] = None
    output_device: Optional[str] = None
    sample_rate: int = 16000
    vad_aggressiveness: int = 2
    push_to_talk_key: str = "space"
    enable_voice: bool = True
    # Which half of the bilingual reply drives TTS: "japanese" (default -
    # the tagged-output Japanese line, or a translation on the fallback
    # path) or "english" (speak the display text directly, no translation).
    voice_language: str = "japanese"


@dataclasses.dataclass
class ProviderSettings:
    """Providers selected by the user for LLM/STT/TTS."""

    llm: str = "openai"
    stt: str = "faster_whisper"
    # Whisper model size for faster_whisper ("tiny"/"base"/"small"/
    # "medium"/"distil-small.en"...). Default "small": "base" mis-heard
    # often enough to break trust in voice input; bigger is slower on CPU.
    stt_model: str = "small"
    tts: str = "pyttsx3"
    embedding: str = "sentence_transformers"
    # Custom OpenAI-compatible endpoint for providers.llm: openai — points the
    # same adapter at Ollama (http://localhost:11434/v1), LM Studio, vLLM,
    # OpenRouter, and friends. Keyless local servers work without
    # openai_api_key (a placeholder key is sent). None = api.openai.com (the
    # SDK also honours the OPENAI_BASE_URL env var on its own).
    openai_base_url: Optional[str] = None
    # Gemini API transport: None = SDK default (grpc). Set "rest" when the
    # gRPC endpoint is having an outage (504s) while REST still answers.
    gemini_transport: Optional[str] = None
    # Gemini embedder knobs (providers.embedding: gemini). The dimension
    # truncates the model's native MRL output; changing either after use
    # requires scripts/migrate_memory.py (persisted vectors keep the old dim).
    gemini_embedding_model: str = "models/gemini-embedding-2"
    gemini_embedding_dim: int = 768
    huggingface_tts_path: str = "mio/amadeus"
    huggingface_tts_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    huggingface_inference_model: Optional[str] = None
    huggingface_inference_voice: Optional[str] = None
    huggingface_inference_api_name: str = "/predict"
    huggingface_space_id: Optional[str] = None
    # MeloTTS custom model settings
    melotts_config_path: str = "models/kurisu_voice/config.json"
    melotts_checkpoint_path: str = "models/kurisu_voice/G_latest.pth"
    melotts_language: str = "JP"
    melotts_device: str = "cpu"
    melotts_speaker_id: Optional[int] = None
    # MeloTTS ONNX model settings (optimized CPU inference)
    melotts_onnx_model_path: str = "models/kurisu_voice_onnx/model.onnx"
    melotts_onnx_tokens_path: str = "models/kurisu_voice_onnx/tokens.txt"
    melotts_onnx_lexicon_path: str = "models/kurisu_voice_onnx/lexicon.txt"
    melotts_onnx_speaker_id: int = 0
    # Qwen3-TTS zero-shot voice clone settings (GPU)
    qwen3_tts_model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    qwen3_tts_ref_audio: str = "models/kurisu_voice_qwen/kurisu_ref.wav"
    qwen3_tts_ref_text: str = (
        "ビクトルコンドリア大学の精神生理学研究所がビジュアルリビルディング"
    )
    qwen3_tts_language: str = "Japanese"
    qwen3_tts_device: str = "cuda:0"
    qwen3_tts_dtype: str = "bfloat16"
    # Sampling for the autoregressive TTS decode. Lower = more stable clone
    # timbre (the engine defaults, 0.9/50, occasionally drift to a different
    # speaker on short sentences); higher = livelier prosody.
    qwen3_tts_temperature: float = 0.7
    qwen3_tts_top_k: int = 20
    # Speaker-consistency guard: score each generated clip against the
    # reference with a d-vector encoder and regenerate on drift (the voice
    # flipping to a different speaker). Needs the optional resemblyzer
    # package; silently disabled when missing.
    # Threshold re-calibrated 2026-07-28 against the 16 other character
    # reference voices: the closest same-gender impostors score 0.80-0.82
    # against the Kurisu ref, so the original 0.78 passed near-voice drift
    # outright; 0.84 splits impostor-max from the 0.88-0.94 true-voice band.
    qwen3_tts_speaker_guard: bool = True
    qwen3_tts_speaker_guard_threshold: float = 0.84
    # Streaming latency knobs. Chunk size is in 12Hz codec frames (4 ≈ 320ms
    # of audio per chunk, matching Qwen3-TTS's native packet size); the guard
    # prefix is how much audio the drift guard buffers and scores before
    # releasing ANY of a streamed reply — a direct tax on time-to-first-
    # audio, but measured 2026-07-28 the speaker embedding cannot separate
    # true voice from a similar impostor below ~1.5s of audio (ranges
    # overlap at 0.5-1.0s), so a shorter prefix only pretends to verify.
    qwen3_tts_stream_chunk_size: int = 4
    qwen3_tts_guard_prefix_seconds: float = 1.5
    # Disk cache for synthesized audio (canned/recurring lines: greetings,
    # presence lines, fillers) so slow providers like Qwen3-TTS don't
    # re-pay synthesis cost for text they've already spoken.
    tts_cache_enabled: bool = True
    tts_cache_dir: str = "cache/tts"
    tts_cache_max_chars: int = 80


@dataclasses.dataclass
class WebSearchSettings:
    """Configuration for optional web search augmentation."""

    enabled: bool = False
    # "auto" picks Tavily when api_key is set, else keyless DuckDuckGo (needs
    # the `ddgs` package — pip install "amadeus-kurisu[search]"). Explicit
    # values: "tavily" | "duckduckgo".
    provider: str = "auto"
    api_key: Optional[str] = None
    max_results: int = 3
    timeout: float = 15.0


@dataclasses.dataclass
class DiscordSettings:
    """Optional Discord text-chat bridge (DMs + @mentions, no voice).

    Shares the same ChatEngine/memory as the CLI: one chat session per
    Discord user id (see amadeus/discord_bridge.py). Default off — the
    `discord` extra (discord.py) isn't installed unless requested.
    """

    enabled: bool = False
    # Bot token from the Discord developer portal. Env fallback below,
    # same pattern as the provider API keys.
    token: Optional[str] = None
    # Empty (default) = anyone may DM/mention the bot. Non-empty = only
    # these Discord user ids may talk to it at all (DMs and mentions alike).
    allowed_user_ids: List[int] = dataclasses.field(default_factory=list)
    reply_to_mentions: bool = True
    reply_to_dms: bool = True


@dataclasses.dataclass
class DesktopFilesSettings:
    """Read-only access to the desktop file tree via the desktop-ops bridge."""

    enabled: bool = False
    root: Optional[str] = None  # defaults to the bridge's DESKTOP_OPS_ROOT


@dataclasses.dataclass
class ToolsSettings:
    """Text ReAct tool-calling for the single-model chat path."""

    enabled: bool = False
    max_iterations: int = 4
    # When does a turn enter the tool loop?
    #   "auto"    (default) — providers with native function calling (Gemini)
    #             get tools on every turn and the model decides whether to call
    #             one (a no-tool turn stays a single LLM call); providers on
    #             the text-ReAct path keep the cheap keyword gate, since ReAct
    #             prompting every turn is costly and derails the persona.
    #   "keyword" — legacy keyword-hint gate for every provider.
    #   "always"  — tool loop on every turn regardless of provider.
    gate: str = "auto"
    # Injection defense: only these tools may execute inside the ReAct loop.
    # code_interpreter runs sandboxed (subprocess: python -I, temp cwd, trimmed
    # env, rlimits, hard timeout), so it is allowed by default.
    # "memory" is the model-facing remember/correct/forget tool — write-only
    # by design (no read/search action: recall already injects memories, and
    # a read action would hand the tool loop a private-data source to pair
    # with network tools). Archives, never hard-deletes.
    # "mcp" is a meta-entry admitting every bridged MCP tool (their names are
    # config-derived: mcp_<server>_<tool>); remove it to require listing each
    # bridged tool name individually. See agents/tools.py::tool_allowed.
    allowed: List[str] = dataclasses.field(
        default_factory=lambda: [
            "desktop_files",
            "web_search",
            "calculator",
            "code_interpreter",
            "pubmed_search",
            "arxiv_search",
            "reminders",
            "play_animation",
            "memory",
            "mcp",
        ]
    )
    # Avatar animations the play_animation tool may perform ("dance for me").
    # Names must match the animation manifest's display names and should stay
    # vetted (static/animations/SOURCES.md); empty list disables the tool.
    animation_clips: List[str] = dataclasses.field(
        default_factory=lambda: [
            "Greeting",
            "Peace Sign",
            "Jump",
            "Spin",
            "Model Pose",
            "Squat",
            "Show Full Body",
            "Goodbye Wave",
            "UAL Dance Loop",
        ]
    )
    # Hard wall-clock timeout for one code_interpreter execution (seconds).
    code_timeout_seconds: float = 10.0
    # Exfiltration guard: after ANY successful desktop_files call in a turn
    # (read, find, list — paths and filenames are sensitive too), refuse
    # network tools (web_search) for the rest of that turn so local data
    # can't leak out through a search query.
    block_network_after_file_read: bool = True
    # Defense-in-depth: refuse a web_search call whose query copies long spans
    # from untrusted tool output produced earlier this turn.
    network_arg_guard: bool = True


@dataclasses.dataclass
class LiteratureSettings:
    """Free, keyless scientific-literature search tools (PubMed + arXiv)."""

    enabled: bool = False


@dataclasses.dataclass
class McpSettings:
    """MCP (Model Context Protocol) client bridge: tools from user-configured
    MCP servers surface in the chat tool registry as mcp_<server>_<tool>.
    Needs the [mcp] extra (pip install "amadeus-kurisu[mcp]", Python 3.10+).

    Each ``servers`` entry is a dict:
      name: short identifier (becomes part of the tool names)  [required]
      command + args + env: stdio server launch spec, OR
      url: streamable-HTTP server endpoint
      allowed_tools: optional list restricting which of the server's tools
        are bridged (default: all)
      network: whether the server can reach the network (default true).
        Network-capable bridged tools are refused after a desktop_files
        call the same turn — the exact same exfiltration guard as
        web_search. Set false ONLY for genuinely local-only servers.

    Connections are cached process-wide (stdio servers are real child
    processes; /api/status probes must not respawn them per poll)."""

    enabled: bool = False
    servers: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    connect_timeout_seconds: float = 15.0
    call_timeout_seconds: float = 30.0


@dataclasses.dataclass
class DailyBriefSettings:
    """Pulse-style morning brief, generated by an in-process scheduler and
    delivered through the first greeting of the day (plus GET /api/brief).
    Costs one cheap-model call per user per day (+1 optional arXiv search).
    Only runs while the server is up — catch-up happens at next startup."""

    enabled: bool = True
    # Local hour (0-23) after which the day's brief becomes due.
    hour: int = 6
    # Also search arXiv for the first open thread's topic. Additionally
    # requires literature.enabled.
    include_literature: bool = False
    max_arxiv_results: int = 2
    store_path: str = "daily_brief.json"
    # Token budget for the brief-flavored greeting generation.
    max_tokens: int = 220


@dataclasses.dataclass
class DiarySettings:
    """Kurisu's own daily in-character diary note, generated by the same
    in-process scheduler as the daily brief."""

    enabled: bool = True
    # Local hour (0-23) after which the day's entry becomes due.
    hour: int = 7
    store_path: str = "kurisu_diary.json"
    # Token budget for the diary generation call.
    max_tokens: int = 180


@dataclasses.dataclass
class InsightsSettings:
    """Nightly memory consolidation (amadeus/insights.py): one cheap-model
    call per user per day distills the profile (facts, tone history, open
    threads) into a handful of standing behavioral insights — how to talk
    to this person, not what to remember about them — injected into every
    turn's system prompt. Generated by the same in-process scheduler as the
    brief/diary; the model may (and often should) answer SKIP."""

    enabled: bool = True
    # Local hour (0-23) after which the day's consolidation becomes due.
    # Early morning: it reads yesterday's accumulated state ("sleep").
    hour: int = 5
    store_path: str = "insights.json"
    # Profiles younger than this many sessions skip the call entirely —
    # there is nothing well-supported to distill yet.
    min_sessions: int = 3
    # Token budget for the consolidation call.
    max_tokens: int = 260


@dataclasses.dataclass
class AmbientSettings:
    """Ambient desktop awareness (desktop app only): the client polls
    zero-cost signals — focused-window title, battery, now-playing — and
    reports only noteworthy TRANSITIONS (same app focused for hours, battery
    low, track change); she answers with at most one short line per
    cooldown, SKIP allowed. Text-only (no screenshots — that's the separate
    screen-companion opt-in) and never committed to memory."""

    enabled: bool = True
    # Shorter than a typical song, so with the client's per-track music
    # events every track is a remark OPPORTUNITY (her SKIP judgment is the
    # real rate limiter). Raise this if she feels chatty.
    cooldown_seconds: float = 180.0
    max_tokens: int = 60


@dataclasses.dataclass
class SleepScheduleSettings:
    """Her own sleep schedule (local hours). While it's her sleep window she
    is visibly off-duty: unprompted idle chatter is suppressed, a pending
    reach-out is held (not consumed) until she's awake, pokes get a groggy
    half-awake register, and a session opened mid-window greets as someone
    who was just woken. A normal chat message always works — the user
    outranks the schedule; she's sleepy, not unavailable. The strongest
    cheap "she has a life independent of me" signal (Gatebox lineage).
    start == end disables the window."""

    enabled: bool = True
    # She's asleep from start_hour (inclusive) to end_hour (exclusive),
    # wrapping midnight when start > end.
    start_hour: int = 1
    end_hour: int = 8


@dataclasses.dataclass
class VocalEmotionSettings:
    """Perception of HOW the user's voice sounded (prosody), independent of
    what they said. The STT endpoints (plain /api/stt and the duplex loop)
    analyze the same audio clip they transcribe — heuristic by default
    (energy/pitch/rate over numpy, dependency-free), with an optional ONNX
    speech-emotion model auto-discovered from models/vocal_emotion/ (see
    scripts/fetch_vocal_emotion.py) when mode allows it. The result is
    stashed per-session and consumed by exactly the NEXT chat turn within
    ttl_seconds, rendering one line in the dynamic prompt tail
    (persona.py) — never stored to memory, never in transcripts."""

    enabled: bool = True
    # "auto" (default) and "heuristic" both run the speaker-relative prosody
    # heuristic and never touch onnxruntime. "model" opts into the ONNX
    # classifier exclusively (no heuristic fallback, silent if missing).
    # "auto" deliberately does NOT prefer the model: measured 2026-08-05,
    # emotion2vec+ base read ordinary spoken lines as six different emotions
    # with zero "neutral" at a median 0.997 confidence, so min_confidence
    # cannot screen its mistakes. See analyze() in vocal_emotion.py.
    mode: str = "auto"
    ttl_seconds: float = 25.0
    # Below this confidence the analyzer abstains (heuristic) or the model
    # verdict is discarded — a low-confidence guess is worse than no hint.
    min_confidence: float = 0.5
    # Explicit ONNX file path; unset auto-discovers the first *.onnx under
    # models/vocal_emotion/ (or $AMADEUS_VOCAL_EMOTION_DIR).
    model_path: Optional[str] = None


@dataclasses.dataclass
class FocusSettings:
    """Focus / co-working mode ("body doubling"): the user tells her they're
    going heads-down for N minutes, she goes quiet for the duration (same
    suppression shape as her sleep window — idle chatter off, ambient
    comments off, a pending reach-out held not consumed), and gives exactly
    one warm check-in line when the session ends. Chat always works; only
    unprompted lines are muted. See amadeus/focus_mode.py."""

    enabled: bool = True
    checkin_enabled: bool = True
    default_minutes: int = 50
    checkin_max_tokens: int = 60


@dataclasses.dataclass
class ReachOutSettings:
    """Proactive reach-out: one short unprompted in-character line per day
    about an OPEN THREAD the user left hanging, generated by the same
    in-process scheduler as the brief/diary and delivered through the
    presence poll (voice when visible, opt-in OS notification when hidden).
    Content-bearing only — no open thread, or the user hasn't actually been
    away, means no line is generated at all (companion-ethics.md)."""

    enabled: bool = False
    # Local hour (0-23) after which the day's reach-out becomes due.
    hour: int = 17
    # The user must have been away at least this long before she reaches
    # out — reaching out mid-conversation would be absurd, and reaching out
    # an hour after they left reads as clingy.
    min_idle_hours: float = 8.0
    store_path: str = "reachout.json"
    # Token budget for the generation call.
    max_tokens: int = 90


@dataclasses.dataclass
class AsyncNotifySettings:
    """Async delivery of due reach-outs/reminders when no client is around
    to receive the normal 30s poll (amadeus/async_notify.py). Opt-in and
    off by default: the frontend poll stays the primary delivery path, this
    is only the fallback for "the app is closed for the day". Reuses the
    Discord bot token from ``discord.token``/``DISCORD_BOT_TOKEN`` — a
    second token field would just be the same credential typed twice."""

    enabled: bool = False
    # Only "discord" exists today; kept as a string so a future channel
    # (e.g. email) is additive, not a breaking rename.
    channel: str = "discord"
    # The Discord user id to DM. Required for delivery to do anything.
    discord_user_id: str = ""
    # A client must be silent for at least this long before delivery is
    # even considered — an open app/pet window already handles delivery
    # via the poll, so this is the "nobody is home" threshold.
    min_client_stale_minutes: int = 30
    # User quiet hours (distinct from HER sleep_schedule): no DM in this
    # local-hour window, wrapping midnight when start > end. start == end
    # disables (never quiet).
    quiet_hours_start: int = 22
    quiet_hours_end: int = 8
    # Hard cap: at most this many async DMs per day, total (reach-out +
    # reminders combined) — never a second channel for the same content.
    max_per_day: int = 1


@dataclasses.dataclass
class FollowThroughSettings:
    """Overnight research on the user's next-time thread (amadeus/followup.py),
    run by the same in-process scheduler as the brief/diary/reach-out and
    delivered through the next greeting ("I read up on it — ..."). At most
    ONE research pass per thread ever — a delivered note or a SKIP verdict
    is final, never retried nightly. Needs ``web_search.enabled`` for the
    actual searches (a missing backend counts ``followup.no_search_backend``
    instead of silently disabling). Costs up to two cheap-model calls plus
    ``max_searches`` web searches per new thread."""

    enabled: bool = True
    # Local hour (0-23) after which a new thread's research becomes due.
    # Overnight by design: the note should feel like "while you were away".
    hour: int = 3
    # Hard cap on search queries per thread (the planning call may choose
    # fewer, or SKIP entirely).
    max_searches: int = 2
    max_results_per_search: int = 3
    store_path: str = "followup.json"
    # Token budget for the note generation call.
    max_tokens: int = 160


@dataclasses.dataclass
class LifeThreadsSettings:
    """Slow-moving pursuits of the character's OWN (amadeus/life_threads.py)
    that progress every few days via the in-process scheduler and surface
    through the diary and idle chatter — her life develops between
    conversations instead of resetting daily. Hard-capped at ONE cheap-model
    call per day total (a seed or one progression), so cost is negligible.
    Threads are never about the user (companion-ethics.md)."""

    enabled: bool = True
    # Local hour (0-23) after which the day's (at most one) step becomes due.
    hour: int = 7
    # Concurrent active pursuits. Two keeps her life varied without turning
    # the diary into a status report.
    max_active: int = 2
    # A thread progresses at most every this-many days — weeks-long arcs by
    # construction.
    progress_interval_days: int = 3
    # After this many steps the progression prompt steers toward an ending.
    max_progressions: int = 6
    store_path: str = "life_threads.json"
    # Token budget for the seed/progress calls.
    max_tokens: int = 140
    # Opinion-asks: she may occasionally ask the user's take on one of her
    # own pursuits, and their answer feeds the NEXT progression as context.
    #
    # This deliberately widens the "never about, for, or involving the
    # person you talk to" rule, so it ships default-off with the narrowest
    # honest shape: the answer influences what she thinks, the progression
    # prompt still forbids mentioning them, and a progression whose stored
    # text leaks the user is REJECTED (life.progress_user_leak_rejected)
    # rather than stored. A thread is asked about at most once ever, no
    # field records whether the user answered — so an ignored ask is
    # terminal and the system structurally cannot sulk. Costs no extra LLM
    # call: the ask rides an already-happening idle-chatter generation.
    enable_opinion_asks: bool = False
    min_days_between_asks: int = 21


@dataclasses.dataclass
class SelfEvalSettings:
    """Nightly conversation-quality self-check, default OFF. Runs a small
    subset of scripts/eval_conversation.py's scenarios against an ISOLATED
    tmp memory store (never the live user store — same isolation
    scripts/eval_conversation.py itself uses) once per day after `hour`,
    appends one JSON line to `trend_path`, and flags a regression when the
    latest run drops more than 2.0 below the rolling 7-entry median for a
    scenario. Makes real LLM calls (chat + judge) — costs money/quota, which
    is why it defaults off and carries its own daily call budget separate
    from the brief/diary/reach-out jobs."""

    enabled: bool = False
    # Local hour (0-23) after which the day's self-eval becomes due.
    hour: int = 4
    # Small, cheap default subset — NOT the full scenario file (that's a
    # judge-sampled, multi-scenario live-LLM run; a nightly job needs to stay
    # cheap enough to actually run unattended every day).
    scenarios: List[str] = dataclasses.field(
        default_factory=lambda: [
            "greeting_loop_no_repeat",
            "in_character_consistency",
            "interruption_grace",
        ]
    )
    # Hard ceiling on scenario runs per day, independent of `scenarios`'
    # length — a guard against a misconfigured scenario list or a retry bug
    # burning quota unattended overnight.
    max_calls_per_day: int = 10
    # Appended to (one JSON line per run), never overwritten. Relative paths
    # anchor under the platform user-data dir like other user-data files
    # (amadeus/paths.py resolve_user_data_path).
    trend_path: str = "self_eval_trend.jsonl"
    # Regression threshold: latest scenario score vs its rolling median over
    # the last 7 trend entries. Same magnitude as eval_conversation.py's
    # --tolerance default (absorbs LLM judge score wobble).
    regression_tolerance: float = 2.0


@dataclasses.dataclass
class VisionSettings:
    """Per-request screenshot/image vision (Gemini/OpenAI multimodal).

    Capture is always user-initiated (a click, a paste, a drop) and the
    image lives only in that one request — never persisted, never stored in
    memory (a text marker is remembered instead). Requires a vision-capable
    LLM provider; on anything else attachments are silently dropped.
    """

    enabled: bool = True
    # Longest-edge downscale target applied client-side before upload.
    max_image_px: int = 1568
    jpeg_quality: int = 85
    # Screen-companion mode (pet window, opt-in): minimum seconds between
    # unprompted screen-peek comments, enforced server-side like the poke
    # cooldown so a client bug can't turn the peek timer into an LLM burst.
    screen_peek_cooldown_seconds: float = 120.0
    # Generation budget for the one-line comment. Generous because thinking
    # models (gemini-3-flash) spend reasoning tokens inside the same cap —
    # 60 tokens produced truncated one-word replies in live testing.
    screen_peek_max_tokens: int = 800
    # Server-side hard cap on the decoded upload size.
    max_upload_bytes: int = 5_000_000


@dataclasses.dataclass
class ReadAlongSettings:
    """ "Read together" mode: the desktop pet reads a book/manga alongside the
    user. The client detects page turns and posts one screenshot per page;
    she keeps ephemeral in-RAM notes for continuity and speaks only
    occasionally. Additionally gated on vision.enabled + a vision-capable
    provider, same as screen_comment — this is a vision feature, not a
    separate one. Notes and the page count live only in process memory for
    the reading session (ChatEngine._ReadAlongState) and are never persisted
    or committed to the conversation store; see docs/companion-ethics.md.
    """

    # Floor between SPOKEN lines (the "SAY:" half of a page reply). Notes are
    # never throttled — every page still gets a silent continuity note even
    # while a SAY is being swallowed by the cooldown, so the running summary
    # never falls behind just because she isn't allowed to talk yet.
    speak_cooldown_seconds: float = 90.0
    # Hard cap on vision LLM calls per reading session (a "session" = from
    # the first page posted until read_along_end() clears the state). Protects
    # free-tier daily quotas from a long manga chapter or a client bug that
    # posts pages faster than the user actually turns them.
    max_pages_per_session: int = 300
    # Rolling continuity buffer: how many of the most recent per-page notes
    # stay available to condition the next page's prompt. Old notes fall off
    # the deque as new ones are added — she isn't meant to recite the whole
    # book, just remember "a few pages back."
    max_notes: int = 10
    # Generation budget for the per-page NOTE+SAY call. Generous for the same
    # reason as screen_peek_max_tokens: thinking models spend reasoning
    # tokens inside the same cap, and a tight budget produces truncated
    # one-word replies in live testing.
    comment_max_tokens: int = 800


@dataclasses.dataclass
class WatchTogetherSettings:
    """ "Watch together" mode: she keeps the user company during an activity
    (a show, a game, a long work session) via periodic screen frames the
    client posts on its own timer. Structurally read_along's sibling: same
    vision.enabled + vision-capable-provider gates, same ephemeral in-RAM
    NOTE/SAY continuity (ChatEngine._watch_together), nothing ever persisted
    or committed to the conversation store; see docs/companion-ethics.md.
    Most frames deserve silence — the prompt and the cooldown both push
    toward rare, earned comments rather than narration.
    """

    # Floor between SPOKEN lines. Notes still advance on every frame while
    # a SAY is being swallowed, exactly like read_along.
    speak_cooldown_seconds: float = 75.0
    # Hard cap on vision LLM calls per watch session (first frame posted
    # until watch_end() clears the state). At the client's default ~60s
    # sampling this is several hours of watching; mostly a guard against a
    # client bug posting frames in a tight loop.
    max_frames_per_session: int = 240
    # Rolling continuity buffer of per-frame notes ("a few minutes back"),
    # not a transcript of the whole session.
    max_notes: int = 8
    # Generation budget for the per-frame NOTE+SAY call (thinking models
    # spend reasoning tokens inside the same cap).
    comment_max_tokens: int = 800


@dataclasses.dataclass
class MomentsSettings:
    """Opt-in shared-activity memories (amadeus/moments.py): at the end of a
    read-along or watch-together session, one cheap-model call may distill
    the session's ephemeral notes into a single durable, recallable memory
    ("we watched the Evangelion finale together — user loved the ending").
    Everything else about read-along/watch-together stays exactly as
    forgetful as before (frames and per-page/per-frame notes are still
    never stored, see docs/companion-ethics.md) — this adds only the one
    distilled closing line, and only when explicitly enabled. Off by
    default: this is the one path in co_activity.py that writes to durable
    memory, so it needs its own opt-in rather than riding read_along's or
    watch_together's."""

    enabled: bool = False
    # Hard ceiling on the distilled line's length (truncated by store_fact's
    # existing fact_max_chars machinery too, but this keeps the model's own
    # target tight).
    max_chars: int = 200
    # At most this many moments stored per day, across both read-along and
    # watch-together — the privacy/growth valve for a feature that, unlike
    # the rest of co_activity, actually persists something.
    max_per_day: int = 3


@dataclasses.dataclass
class RhythmSettings:
    """Opt-in rhythm memory (amadeus/rhythm.py): a coarse 14-day tally of
    the shape of the user's week — how many late nights, how many long
    focus stretches, roughly which hours, and which COARSE CATEGORY of app
    (editor/browser/terminal/chat/media/other). Never window titles, never
    document or track names, never per-event timestamps; the client buckets
    before posting, so a raw title never crosses the network.

    Off by default and gated TWICE — this flag plus a per-browser
    localStorage opt-in — because it is the only feature that keeps a
    durable record of the user's computer use. Turning it off ERASES the
    stored tally rather than merely halting writes. Deliberately NOT in the
    presence presets, same category as screen watching.

    Surfacing is at most once a week, as greeting color chosen by a pure
    LLM-free predicate. She notices; she does not measure (see
    docs/companion-ethics.md)."""

    enabled: bool = False
    store_path: str = "rhythm.json"
    retention_days: int = 14
    # Floor between rhythm mentions. A weekly cadence is what keeps this
    # "she noticed" rather than a standing report.
    min_days_between_mentions: int = 7


@dataclasses.dataclass
class RelationalMemorySettings:
    """Opt-in relational memory (amadeus/memory/relational.py): at session
    end, one cheap-model call may record a single line about something that
    changed BETWEEN her and the user ("we finally talked about why you left
    the lab"). The bar is a change in the relationship, not a summary of
    what happened — most sessions correctly produce nothing.

    Off by default, and the caps below are the reason. This is the second
    path that writes durable first-person memory, and a first-person
    session record is the exact shape that made her persona compound across
    sessions until she fixated (see extraction_jobs.run_episode_job, which
    is deliberately neutral third-person because of it). Episodes fixated
    because EVERY session wrote one; the weekly cap here is the structural
    guard, and the prompt's SKIP-default is only the second line of
    defense. If she starts fixating again, ``max_per_days`` is the dial."""

    enabled: bool = False
    # Below this many messages there is nothing a relationship could have
    # changed in. Cheaper than a wasted LLM call.
    min_messages: int = 6
    # The period cap, deliberately NOT per-day. A relational event is rare
    # by definition, and per-session volume is what made episodes fixate.
    max_per_days: int = 7
    max_per_period: int = 1
    max_chars: int = 200
    max_tokens: int = 120
    # Below a co-activity moment's 0.6: relational memory should surface
    # when genuinely relevant, never route as eagerly as a stated fact.
    salience: float = 0.55


@dataclasses.dataclass
class RemindersSettings:
    """Local spoken reminders/timers (no network, no OS notifications)."""

    enabled: bool = False
    store_path: str = "reminders.json"


@dataclasses.dataclass
class CalendarSettings:
    """Read-only ICS calendar awareness for the daily brief
    (amadeus/calendar_feed.py). No OAuth, no writes, stdlib-only parsing.
    Active iff ``ics_url`` or ``ics_path`` is set — there is deliberately no
    separate ``enabled`` flag, mirroring how vision/watch_together key off a
    real prerequisite instead of a redundant switch. Fetching happens only
    from the scheduler tick (never a chat turn); results are cached to
    ``cache_path`` with a ``refresh_ttl_hours`` TTL, so a fetch failure
    serves the last-good snapshot instead of going empty."""

    ics_url: str = ""
    ics_path: str = ""
    # Today's remaining events + this many hours further out (so an early
    # brief still sees tomorrow morning's first meeting).
    lookahead_hours: int = 36
    refresh_ttl_hours: float = 6.0
    max_events: int = 8
    cache_path: str = "calendar_cache.json"


@dataclasses.dataclass
class MemorySettings:
    """Configuration for short and long term memory."""

    max_context_messages: int = 12
    summary_trigger_messages: int = 8
    # Estimated-token ceiling on the assembled per-turn request (system
    # prompt + short-term context + recall block; chars/4 heuristic). Every
    # other bound here is counts/chars, so the actual prompt size was
    # unmeasured and unbounded — a window of long pasted messages could blow
    # a small model's context with no warning. Over budget, the turn trims
    # recall items first (lowest-ranked), then oldest short-term messages
    # (never below 4, never the current user message or the system prompt).
    # Generous by design: catches pathology, not normal turns. 0 disables
    # trimming (the estimate is still recorded). Counters:
    # memory.context_trimmed_recall / memory.context_trimmed_messages; the
    # per-turn estimate rides llm.prompt_est_tokens on /api/status.
    context_token_budget: int = 12000
    # When True, the short-term context window + rolling summary are treated
    # as per-conversation state instead of per-user state: starting a new
    # conversation never carries the previous conversation's tail into
    # context. Long-term memory (facts/episodes/profile/relationship) is
    # unaffected either way — it's still per-user and always reloaded.
    scope_context_to_session: bool = False
    # None = reuse the chat model (build_summary_provider); set a name to
    # summarize on a cheaper model than the chat one.
    summary_model: Optional[str] = None
    summary_max_chars: int = 500
    persist_path: Optional[str] = "memory_store.json"
    # Persistence backend for the message/summary store and the JSON vector
    # store: "json" (default — the legacy rewrite-whole-file JSON stores) or
    # "sqlite" (stdlib sqlite3, WAL mode so the scheduler's disk-snapshot
    # readers never block on the app's writes). The same configured paths are
    # used either way, with the suffix swapped to .db — per-user and
    # per-character scoping are unaffected. Convert an existing JSON store
    # with scripts/convert_memory_sqlite.py before flipping this.
    storage_backend: str = "json"
    enable_long_term: bool = True
    embeddings_path: Optional[str] = "memory_embeddings.json"
    vector_backend: str = "json"
    dedupe_threshold: float = 0.96
    vector_path: Optional[str] = "memory_vectors"
    # Retrieval quality (active whenever enable_long_term is on).
    # recall_min_score is a lower bound only: the recall path takes
    # max(recall_min_score, embedder.relevance_floor) because embedding
    # families put unrelated text at wildly different score baselines —
    # gemini-embedding-2 scores pure noise at ~0.45-0.59, so 0.35 alone
    # filtered nothing there and every turn got top-k memories injected
    # regardless of relevance. See EmbeddingProvider.relevance_floor
    # (memory/embedders.py) for the measured per-embedder floors; the
    # memory.recall_floor_filtered counter shows when the floor is biting.
    recall_min_score: float = 0.35
    recency_half_life_days: float = 30.0
    # Per-turn recall should surface durable *facts*, not re-narrate whole
    # sessions: episode/reflection records duplicate the rolling summary and
    # make Kurisu recite the same session back every turn. They stay available
    # for the "what do you remember" overview either way. Default on.
    recall_facts_only: bool = True
    # Drop a recalled item whose cosine similarity to the current rolling
    # summary exceeds this — the summary is already in context, so a near-copy
    # in the recall block is pure repetition fuel. 0 disables the check.
    recall_summary_dedupe: float = 0.82
    # Drop a recalled item when this fraction of its content words already
    # appear in the recent conversation window (the current user message is
    # never counted — it's what recall answers). A fact that was just
    # discussed is already in short-term context; re-injecting it every turn
    # is what makes her hang onto old topics instead of following the user
    # to the present one. Lexical, so it costs nothing. 0 disables.
    recall_recent_dedupe: float = 0.6
    # Latency guards on the per-turn recall (measured: the Gemini query
    # embed costs ~0.5s BEFORE the LLM can start). Queries shorter than
    # recall_min_query_chars ("hi", "ok", "thanks") skip recall entirely —
    # they carry no searchable meaning and are exactly the turns where
    # snappiness matters most. recall_embed_timeout_seconds bounds the
    # network embed; on timeout the turn proceeds without long-term recall
    # (short-term context and the rolling summary still apply). 0 disables
    # either guard. The deadline must sit ABOVE the typical round-trip:
    # warm Gemini embeds measure ~0.45s, so the original 0.35s cap timed
    # out on every real query — recall was silently empty on all network-
    # embedder turns (and each turn still burned the full 0.35s waiting).
    # 0.8s covers warm+cold real calls and only bounds true outliers.
    recall_min_query_chars: int = 10
    recall_embed_timeout_seconds: float = 0.8
    # Speculative recall: when the web STT endpoint finishes transcribing a
    # voice clip, the per-turn long-term recall for that transcript is
    # pre-started in the background, so the query-embed round trip (~0.45s
    # warm on a network embedder) overlaps the client's transcript->chat
    # round trip instead of sitting serially inside the turn. The
    # pre-started future is consumed only when the arriving chat message
    # matches the transcript (whitespace-normalized); any mismatch falls
    # back to the normal serial recall, so a wrong guess can never change
    # what a turn recalls. Health: memory.speculative_recall_hits/_misses
    # counters in /api/status.
    speculative_recall: bool = True
    # Entity-graph memory layer (amadeus/memory/graph.py): typed entities
    # (person/project/place/date/preference) with relations and supersede
    # provenance, mined from the SAME extraction LLM call (no extra calls).
    # When enabled, entity mentions in the user message pull that entity's
    # current facts + one-hop relations ahead of vector recall. Stored in the
    # SQLite state db when storage_backend is "sqlite", else a JSON sidecar
    # (<store>_graph.json) next to the message store. Default off.
    graph_enabled: bool = False
    # How long a pre-started recall future stays consumable before it is
    # treated as stale; a newer STT transcript always replaces the previous
    # future regardless.
    speculative_recall_ttl_seconds: float = 10.0
    # Fact extraction + user profile (rides the existing summary LLM call)
    enable_fact_extraction: bool = True
    # Friction memory: the extraction job may note a TOPIC that went badly
    # between them, so a misunderstanding persists across sessions and then
    # heals. Rides the existing extraction call — no extra LLM call, and no
    # prompt characters at all when off. Default off: this is the one
    # relational signal that can move in an unpleasant direction, and its
    # anti-guilt rails are prompt-enforced. It never touches trust, arc,
    # tone or warmth (pinned in tests/test_friction_ethics.py) — friction is
    # information she has about a subject, never a mood she is in about the
    # user. Repair is automatic at friction_max_age_days regardless of
    # anything the user does; a grudge only they can clear is a debt.
    enable_friction_memory: bool = False
    friction_max_age_days: int = 21
    fact_supersede_threshold: float = 0.86
    # Hint-driven supersede (the extractor's free-text "replaces" field) must
    # clear a HIGHER bar than the automatic near-duplicate scan above: the
    # hint is the model's paraphrase from its own recollection of the
    # conversation, and at a loose threshold a hallucinated hint matches a
    # semantically-adjacent but DIFFERENT fact and takes it out. Superseded
    # facts are archived to the fact-archive sidecar, never hard-deleted.
    fact_supersede_hint_threshold: float = 0.75
    # Hard cap on stored fact records per user, mirroring max_episodes —
    # without it fact records grow monotonically forever and every recall's
    # O(n) scan slows with them. Past the cap the weakest unpinned facts
    # (decayed strength, then oldest) are archived on write. 0 disables.
    # Counter: memory.facts_evicted.
    max_facts: int = 2000
    # Stored fact text is truncated to this many characters (a runaway
    # extractor "fact" would otherwise be injected verbatim into future
    # prompts). Counter: memory.fact_truncated. 0 disables.
    fact_max_chars: int = 600
    extraction_max_tokens: int = 400
    profile_path: Optional[str] = "memory_profile.json"
    profile_max_facts: int = 12
    max_open_threads: int = 3
    # Open threads older than this are silently dropped — asking about a
    # weeks-dead topic reads as unsettling rather than attentive. 0 disables.
    open_thread_max_age_days: int = 21
    # Episodic session summaries
    enable_episodes: bool = True
    session_gap_minutes: int = 30
    episode_max_tokens: int = 200
    # Store-side hygiene: a new episode this similar to an existing one
    # supersedes it (delete the older, keep the newer) instead of accreting a
    # near-duplicate. Low-content sessions (repeated greetings) otherwise pile
    # up "the user greeted and mentioned X" records forever. 0 disables.
    episode_supersede_threshold: float = 0.9
    # Hard cap on stored episodes per user; the oldest, lowest-salience
    # episodes beyond this are pruned on write. 0 disables.
    max_episodes: int = 200
    # Reflection + organic forgetting
    enable_reflection: bool = False
    reflection_max_tokens: int = 200
    # On by default: unretrieved facts decay toward strength_floor and drop
    # out of recall (see strength_half_life_days/strength_floor below) —
    # without this, stale facts never get forgotten (Memora: 64% of
    # memory-agent errors trace to stale facts). Pinned records are exempt.
    enable_forgetting: bool = True
    strength_half_life_days: float = 90.0
    strength_floor: float = 0.25
    retrieval_reinforcement: float = 0.15
    # Consolidation: at session end, at most once per consolidation_min_days,
    # merge clusters of near-duplicate unpinned facts into one stronger fact
    # via the cheap summary model (memory/consolidation.py). Archive-not-
    # delete (reason "consolidated"); merged strength carries the cluster's
    # corroboration; the OLDEST timestamp is kept so forgetting/confidence
    # stay honest. Default off.
    # Counters: memory.consolidation_merged/_absorbed/_skipped/_rejected/
    # _failures.
    enable_consolidation: bool = False
    consolidation_min_days: float = 7.0
    # Cluster threshold sits BELOW fact_supersede_threshold (0.86 — pairs at
    # or above it already merged at store time) and above the measured
    # 0.6-0.8 band where distinct-but-adjacent records live.
    consolidation_similarity: float = 0.83
    consolidation_max_merges: int = 3
    consolidation_max_cluster: int = 4
    consolidation_min_facts: int = 12
    consolidation_max_tokens: int = 160
    # Calibrated recall: annotate recalled facts with how sure she should
    # sound, so a stale or never-corroborated memory gets hedged instead of
    # asserted. Purely MECHANICAL (retrieval count x age x volatility class,
    # memory/text_utils.py) — no extra LLM call, no stored field, and a
    # confident fact renders exactly as it did before, so this is default-on
    # rather than a ledger-tracked trial. Off restores flat rendering.
    recall_confidence: bool = True
    # Presence: carry conversational tone into Kurisu's mood across turns
    # (only active when fact extraction supplies a tone), and open new
    # sessions with one short in-character greeting (1 cheap LLM call).
    mood_from_tone: bool = True
    enable_greeting: bool = True
    greeting_max_tokens: int = 60
    # Getting acquainted: while the profile holds fewer than max_facts real
    # facts AND fewer than max_sessions conversations have happened, the
    # system prompt nudges her to ask one light getting-to-know-you question
    # when it fits the flow (name, what fills their days, interests). The
    # session cap is the consent valve: a user who never shares stops being
    # asked once it's crossed, even with an empty profile. Needs the profile
    # ("core memory") features on — without a profile there is nowhere for
    # answers to land and the curiosity phase could never end.
    enable_early_acquaintance: bool = True
    early_acquaintance_max_facts: int = 5
    early_acquaintance_max_sessions: int = 8
    # Presence: reactions to being "poked" (clicking the avatar) and
    # unprompted chatter after a quiet stretch — both reuse the greeting's
    # cheap-model, server-gated pattern. Cooldowns are enforced server-side
    # (not just client debouncing) so a scripted client can't spam either.
    enable_poke_reactions: bool = False
    poke_cooldown_seconds: float = 8.0
    poke_max_tokens: int = 40
    enable_idle_chatter: bool = False
    idle_cooldown_seconds: float = 240.0
    idle_max_tokens: int = 60
    # Presence: one warm in-character send-off when the user says goodbye.
    # Deliberately anti-dark-pattern: the prompt forbids guilt/retention
    # tactics, and window-close never generates anything (finalize only).
    enable_farewell: bool = True
    farewell_max_tokens: int = 40
    farewell_cooldown_seconds: float = 120.0
    # Per-session chat transcripts for the web sessions API. Resolved per
    # user the same way as persist_path (memory_transcripts_users/<id>.json).
    # Unlike in-memory sessions (TTL-evicted), transcripts persist on disk
    # until explicitly deleted via DELETE /api/sessions/{id}.
    transcripts_path: Optional[str] = "memory_transcripts.json"
    # Persona/voice identity snapshots (versioning + rollback + the
    # "you changed my settings" greeting announcement). None disables.
    persona_versions_path: Optional[str] = "persona_versions.json"
    # Per-character long-term memory isolation. When a custom character pack
    # is active, the conversation store, vector store, user profile, session
    # transcripts, and the brief/diary/reach-out stores live under
    # <memory_dir>/characters/<character_id>/ so a guest character never
    # greets the user with Kurisu's accumulated knowledge (or lists her
    # conversations). Reminders stay shared — real-world timers, not persona
    # memory. The builtin Kurisu ALWAYS keeps the legacy paths above,
    # unscoped — an existing store is never moved or re-parented. Off =
    # every character shares one store (the pre-0.13 behavior). Scoping is
    # applied where the paths are consumed (MemoryManager/SessionStore/
    # scheduler/backup), never written back into this config — the values
    # above always hold the base paths.
    per_character_memory: bool = True


@dataclasses.dataclass
class WakeWordSettings:
    """Local wake-word listener: WS /api/wake runs an openwakeword ONNX model
    over streamed mic audio and emits a {"type": "wake"} frame on detection —
    the client then starts the normal duplex call. Needs the [wake] extra and
    a model in models/wake_word/ (scripts/fetch_wake_word.py). Detection is
    entirely local; audio is never stored and never leaves the machine.
    Default OFF: an always-on microphone is a privacy opt-in (same category
    as screen watching — deliberately excluded from presence presets), and
    the per-browser toggle (localStorage kurisu_wake_word) must ALSO be on."""

    enabled: bool = False
    # Explicit path to a wake model .onnx; None = first *.onnx found in
    # $AMADEUS_WAKE_WORD_DIR or models/wake_word/ (feature models excluded).
    model: Optional[str] = None
    # Detection threshold on the model's per-chunk score (0-1).
    threshold: float = 0.5
    # Ignore repeat detections within this many seconds of AUDIO time after
    # a hit (the wake phrase itself echoes across several 80 ms chunks).
    refractory_seconds: float = 2.0


@dataclasses.dataclass
class LoggingSettings:
    """Configuration for logging output."""

    level: str = "INFO"
    logfile: Optional[str] = "logs/amadeus.log"


@dataclasses.dataclass
class WebSettings:
    """Configuration for the FastAPI web app's chat pipeline."""

    # Ask the LLM to emit "[emotion] English || Japanese" per sentence instead
    # of relying on VADER + deep-translator. Falls back automatically when the
    # model ignores the format.
    streaming_tagged_output: bool = True
    # WebSocket streaming endpoint (/api/chat/stream), on by default; the POST
    # endpoint always remains available regardless of this flag. Set false to
    # force clients onto the POST path (WS returns fallback_to_post).
    streaming_enabled: bool = True
    # Optional unspoken "[thought]" line in replies: displayed as a faint
    # ghost line, never voiced, never stored in memory. Off by default. The
    # parser tolerantly strips [thought] lines even when this is off (a
    # stubborn model can't force one to be spoken).
    streaming_inner_thoughts: bool = False
    # Non-verbal vocalizations: a standalone "[laugh]"/"[sigh]" line in the
    # tagged grammar splices a pre-recorded clip (in her voice) into the
    # ordered audio stream. Needs clips generated by
    # scripts/generate_nonverbals.py under static/nonverbals/ — without
    # them this flag is inert (tags are stripped, nothing is advertised to
    # the model, no audio plays).
    streaming_nonverbals: bool = True
    # NOTE: raising this does not speed up MeloTTSONNXProvider (the packaged
    # Kurisu voice). Its synthesize_async runs synchronously on the event
    # loop and ONNX Runtime claims all physical cores per call, so
    # workers>1 cannot run inferences in parallel for that provider — it
    # only helps providers with real async/thread-safe synthesis. Evaluated
    # as a TTFA fix and rejected; do not bump the default without first
    # changing the ONNX provider's threading model.
    tts_stream_workers: int = 1
    # Split the first sentence of a turn into a short first clause + the
    # remainder, so the first TTS chunk synthesizes (and starts playing)
    # much faster than the whole sentence would — cuts time-to-first-audio
    # when the active TTS provider runs slower than realtime. WS streaming
    # path only; the POST endpoint is unaffected.
    streaming_first_clause_split: bool = True
    # Use a TTS provider's sub-sentence streaming synthesis API
    # (``synthesize_stream``), when the active provider exposes one, instead
    # of synthesizing each sentence as a single blocking call. Cuts
    # time-to-first-audio further than streaming_first_clause_split alone,
    # since audio starts playing after the first codec chunk (~1s) rather
    # than the whole first clause/sentence. Providers without a
    # synthesize_stream method are unaffected regardless of this flag. WS
    # streaming path only; the POST endpoint is unaffected.
    streaming_chunked_tts: bool = True
    # Coalesce sentences shorter than this many characters into the next
    # sentence before TTS submission. Each submitted text is an independent
    # zero-shot generation, and very short sentences are the most likely to
    # drift to a different-sounding voice — merging keeps the voice
    # consistent across the reply. Sentence 1 is never held (first-audio
    # latency). 0 disables. WS streaming path only.
    streaming_tts_min_unit_chars: int = 24
    # Synthesize the whole reply as a single TTS generation instead of one
    # generation per sentence. Every unit is an independent zero-shot clone
    # generation, and every unit boundary is a chance for the voice to land
    # on a different speaker — one pass removes the boundaries entirely.
    # Audio can only start once the LLM has finished the reply, so this
    # trades a small amount of first-audio latency (roughly the LLM tail;
    # chunked synthesis still streams the audio itself) for a voice that
    # stays consistent across the reply. Supersedes
    # streaming_tts_min_unit_chars and streaming_first_clause_split while
    # enabled. WS streaming path only; the POST endpoint already
    # synthesizes the full reply in one call.
    streaming_tts_full_reply: bool = True
    # Safety cap for full-reply mode: accumulated speech text longer than
    # this many characters is submitted as its own unit, so a very long
    # reply (essays) splits into a few large generations
    # instead of one huge one — Qwen3's generation slows disproportionately
    # on very long inputs. Typical replies stay a single unit. 0 = no cap.
    streaming_tts_max_unit_chars: int = 360
    # Variable "think delay": pause a beat BEFORE the first sentence/audio of a
    # streamed turn so time-to-first-audio isn't robotically constant. The
    # whole rest of the pipeline optimizes toward minimum TTFA, which reads as
    # inhuman — a real person takes a moment on a heavy/emotional question and
    # snaps back on a quip. The delay is sampled from a distribution keyed off
    # signals that already exist (length-analysis hint, user-input-length
    # bucket, sentence emotion): heavy/long/emotional -> longer beat; short
    # quip -> near zero. Bounded and cancel-safe (a barge-in during the pause
    # aborts cleanly). Disabled -> behavior is exactly as before. WS streaming
    # path only; the POST endpoint is unaffected.
    think_delay_enabled: bool = True
    # Lower/upper bounds (seconds) for the sampled think delay. The computed
    # value is always clamped into [min, max]; a trivial quip lands near min,
    # a long emotional turn near max. Keep the range small — this deliberately
    # spends latency, so the ceiling caps how much. min>=max collapses to a
    # fixed delay of that value.
    think_delay_min_seconds: float = 0.0
    think_delay_max_seconds: float = 1.2
    # Evict web chat sessions idle longer than this many hours (swept on
    # access). Each session pins provider refs and a background summary
    # executor thread; unbounded growth was a slow leak on long-lived
    # multi-user servers. 0 disables eviction. Users with a stable user_id
    # lose nothing: their memory persists on disk and reloads next visit.
    session_ttl_hours: float = 6.0
    # Optional bearer-token auth for every /api/* route (plus ?token=... on
    # the /api/chat/stream WebSocket). None (default) = open, exactly the
    # local-use behavior before this setting existed. Falls back to the
    # AMADEUS_API_TOKEN env var, following the same secret-env pattern as
    # the provider API keys. /healthz and static/SPA routes stay open.
    api_token: Optional[str] = None
    # Origins allowed by the CORS middleware. Default keeps the historical
    # wildcard; lock down to explicit origins for shared deployments.
    cors_origins: List[str] = dataclasses.field(default_factory=lambda: ["*"])
    # Public-demo hardening. demo_mode locks every settings/persona mutation
    # endpoint behind a 403 (anonymous visitors must not rewrite the server's
    # providers or keys) and reports itself on /api/status so the UI can hide
    # those controls. Local installs never need this.
    demo_mode: bool = False
    # Per-client-IP cap on the expensive turn endpoints (POST /api/chat,
    # /api/tts, /api/stt, and each /api/chat/stream turn), sliding one-minute
    # window. None disables the limiter — except in demo_mode, which defaults
    # it to 10/min so a public instance can't be drained through one IP.
    rate_limit_per_minute: Optional[int] = None
    # Full-duplex voice conversation loop (2026-07-26): a second WebSocket
    # endpoint (/api/chat/duplex) that accepts a continuous mic stream, runs
    # energy-based endpoint detection + barge-in server-side, and drives the
    # same turn orchestrator as /api/chat/stream. Strictly additive —
    # /api/chat and /api/chat/stream are byte-identical regardless of this
    # flag. Server-side default ON since 2026-07-27 (promotion from
    # prototype); the CLIENT still opts in per browser via the localStorage
    # `kurisu_duplex` toggle in Settings → Features, so flipping this on
    # changes nothing until a user turns duplex on there. See
    # amadeus/web/streaming/duplex.py for the protocol and state machine.
    duplex_enabled: bool = True
    # Semantic end-of-turn prediction (smart-turn-v3 ONNX, CPU) layered on
    # the duplex energy VAD: an "incomplete utterance" verdict at the
    # silence hangover holds the endpoint open up to
    # duplex_smart_turn_hold_ms longer, so a mid-thought pause isn't cut
    # off. Fully optional — without onnxruntime + a fetched model
    # (scripts/fetch_smart_turn.py) endpointing stays energy-only and
    # web.duplex_smart_turn_unavailable counts once. duplex_smart_turn_model
    # pins an exact .onnx path (default: auto-discover models/smart_turn/).
    duplex_smart_turn: bool = True
    duplex_smart_turn_model: Optional[str] = None
    duplex_smart_turn_hold_ms: float = 2400.0
    # Verbal backchannels during duplex: at micro-pauses in sustained user
    # speech the server sends an advisory {type:"backchannel"} frame and the
    # client plays a short pre-rendered acknowledgement clip ("un", "hm") in
    # her voice — the turn-taking rhythm cue. No clips generated
    # (scripts/generate_fillers.py) means the frame is a silent no-op.
    duplex_backchannels: bool = True


@dataclasses.dataclass
class Config:
    """Full application configuration."""

    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    huggingface_api_token: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    local_model: Optional[str] = None
    # Penalize repeated tokens in chat replies (0.0-1.0-ish) to break the
    # model's opener ruts ("Oh, really…" every turn). Applied only to the
    # user-facing reply, not summaries/JSON extraction. 0 disables.
    llm_frequency_penalty: float = 0.4
    # OpenRouter rate-limit failover (OpenAI-compatible). When the primary LLM
    # hits a 429/quota error, the same request retries on this :free model.
    llm_fallback_enabled: bool = False
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    character: CharacterSettings = dataclasses.field(default_factory=CharacterSettings)
    persona: PersonaSettings = dataclasses.field(default_factory=PersonaSettings)
    audio: AudioSettings = dataclasses.field(default_factory=AudioSettings)
    providers: ProviderSettings = dataclasses.field(default_factory=ProviderSettings)
    memory: MemorySettings = dataclasses.field(default_factory=MemorySettings)
    web_search: WebSearchSettings = dataclasses.field(default_factory=WebSearchSettings)
    discord: DiscordSettings = dataclasses.field(default_factory=DiscordSettings)
    desktop_files: DesktopFilesSettings = dataclasses.field(
        default_factory=DesktopFilesSettings
    )
    tools: ToolsSettings = dataclasses.field(default_factory=ToolsSettings)
    mcp: McpSettings = dataclasses.field(default_factory=McpSettings)
    literature: LiteratureSettings = dataclasses.field(
        default_factory=LiteratureSettings
    )
    reminders: RemindersSettings = dataclasses.field(default_factory=RemindersSettings)
    calendar: CalendarSettings = dataclasses.field(default_factory=CalendarSettings)
    vision: VisionSettings = dataclasses.field(default_factory=VisionSettings)
    read_along: ReadAlongSettings = dataclasses.field(default_factory=ReadAlongSettings)
    watch_together: WatchTogetherSettings = dataclasses.field(
        default_factory=WatchTogetherSettings
    )
    moments: MomentsSettings = dataclasses.field(default_factory=MomentsSettings)
    relational_memory: RelationalMemorySettings = dataclasses.field(
        default_factory=RelationalMemorySettings
    )
    rhythm: RhythmSettings = dataclasses.field(default_factory=RhythmSettings)
    daily_brief: DailyBriefSettings = dataclasses.field(
        default_factory=DailyBriefSettings
    )
    diary: DiarySettings = dataclasses.field(default_factory=DiarySettings)
    insights: InsightsSettings = dataclasses.field(default_factory=InsightsSettings)
    reach_out: ReachOutSettings = dataclasses.field(default_factory=ReachOutSettings)
    async_notify: AsyncNotifySettings = dataclasses.field(
        default_factory=AsyncNotifySettings
    )
    sleep_schedule: SleepScheduleSettings = dataclasses.field(
        default_factory=SleepScheduleSettings
    )
    focus: FocusSettings = dataclasses.field(default_factory=FocusSettings)
    ambient: AmbientSettings = dataclasses.field(default_factory=AmbientSettings)
    vocal_emotion: VocalEmotionSettings = dataclasses.field(
        default_factory=VocalEmotionSettings
    )
    follow_through: FollowThroughSettings = dataclasses.field(
        default_factory=FollowThroughSettings
    )
    life_threads: LifeThreadsSettings = dataclasses.field(
        default_factory=LifeThreadsSettings
    )
    self_eval: SelfEvalSettings = dataclasses.field(default_factory=SelfEvalSettings)
    logging: LoggingSettings = dataclasses.field(default_factory=LoggingSettings)
    web: WebSettings = dataclasses.field(default_factory=WebSettings)
    wake_word: WakeWordSettings = dataclasses.field(default_factory=WakeWordSettings)
    transcript_path: str = "logs/transcripts"
    kurisu_isms_path: str = "assets/kurisu_isms.txt"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create a configuration object from a dictionary."""

        def build(field_cls: Any, value: Optional[Dict[str, Any]]):
            """Construct a settings dataclass, ignoring (and warning about) unknown keys."""
            value = value or {}
            known = {field.name for field in dataclasses.fields(field_cls)}
            unknown = sorted(set(value) - known)
            if unknown:
                logging_module.getLogger(__name__).warning(
                    "Ignoring unknown %s config keys: %s",
                    field_cls.__name__,
                    ", ".join(unknown),
                )
            return field_cls(**{key: val for key, val in value.items() if key in known})

        # Every settings-dataclass field wires itself: the YAML section name
        # is the field name, and the field's default_factory is the settings
        # class. Adding a new section is therefore just adding the dataclass
        # field on Config — no from_dict edit required.
        sections = {
            field.name: build(field.default_factory, data.get(field.name))
            for field in dataclasses.fields(cls)
            if field.default_factory is not dataclasses.MISSING
            and dataclasses.is_dataclass(field.default_factory)
        }
        # Secret-env fallback, same pattern as the provider API keys below.
        if sections["web"].api_token is None:
            sections["web"].api_token = os.getenv("AMADEUS_API_TOKEN") or None
        if sections["discord"].token is None:
            sections["discord"].token = os.getenv("DISCORD_BOT_TOKEN") or None
        return cls(
            openai_api_key=data.get("openai_api_key") or os.getenv("OPENAI_API_KEY"),
            gemini_api_key=data.get("gemini_api_key")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY"),
            huggingface_api_token=data.get("huggingface_api_token")
            or os.getenv("HUGGINGFACE_API_TOKEN"),
            llm_model=data.get("llm_model", "gpt-4o-mini"),
            llm_frequency_penalty=float(data.get("llm_frequency_penalty", 0.4)),
            local_model=data.get("local_model"),
            llm_fallback_enabled=bool(data.get("llm_fallback_enabled", False)),
            openrouter_api_key=data.get("openrouter_api_key")
            or os.getenv("OPENROUTER_API_KEY"),
            openrouter_model=data.get(
                "openrouter_model", "meta-llama/llama-3.3-70b-instruct:free"
            ),
            transcript_path=data.get("transcript_path", "logs/transcripts"),
            kurisu_isms_path=data.get("kurisu_isms_path", "assets/kurisu_isms.txt"),
            **sections,
        )

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        """Load configuration from YAML file."""

        config_path = path
        if config_path is None:
            for candidate in DEFAULT_CONFIG_LOCATIONS:
                if candidate.exists():
                    config_path = candidate
                    break

        data: Dict[str, Any] = {}
        if config_path and config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        config = cls.from_dict(data)
        config.resolve_user_data_paths()
        config.validate()
        return config

    def resolve_user_data_paths(self) -> None:
        """Anchor relative user-data paths to the platform data dir.

        Bare-filename defaults (memory_store.json etc.) used to resolve
        against the CWD, so running from a different directory silently split
        a user's memory. Absolute paths — e.g. the desktop build's explicit
        overrides — pass through untouched.
        """

        from amadeus.paths import resolve_user_data_path

        self.memory.persist_path = resolve_user_data_path(self.memory.persist_path)
        self.memory.embeddings_path = resolve_user_data_path(
            self.memory.embeddings_path
        )
        self.memory.profile_path = resolve_user_data_path(self.memory.profile_path)
        self.memory.transcripts_path = resolve_user_data_path(
            self.memory.transcripts_path
        )
        self.memory.vector_path = resolve_user_data_path(self.memory.vector_path)
        self.memory.persona_versions_path = resolve_user_data_path(
            self.memory.persona_versions_path
        )
        self.reminders.store_path = resolve_user_data_path(self.reminders.store_path)
        self.calendar.cache_path = resolve_user_data_path(self.calendar.cache_path)
        self.daily_brief.store_path = resolve_user_data_path(
            self.daily_brief.store_path
        )
        self.diary.store_path = resolve_user_data_path(self.diary.store_path)
        self.insights.store_path = resolve_user_data_path(self.insights.store_path)
        self.reach_out.store_path = resolve_user_data_path(self.reach_out.store_path)
        self.follow_through.store_path = resolve_user_data_path(
            self.follow_through.store_path
        )
        self.life_threads.store_path = resolve_user_data_path(
            self.life_threads.store_path
        )
        self.self_eval.trend_path = resolve_user_data_path(self.self_eval.trend_path)

    def memory_character_id(self) -> Optional[str]:
        """Stable id keying per-character memory isolation, or None.

        None means "use the legacy shared paths": the builtin Kurisu (no
        custom character prompt resolves), or ``memory.per_character_memory``
        off. Custom characters key off the slugified full_name (falling back
        to name) — the same normalization character packs use for their
        install id — so the id survives restarts and config round-trips.
        Two packs with an identical full_name would share a store; they also
        share a persona name, so that is the honest behavior.

        Deliberately a method on Config (not a mutation of the memory paths):
        settings saves and character switches rebuild configs by deep-merging
        ``dataclasses.asdict(runtime_config)``, so a scoped path written into
        ``memory.persist_path`` would survive the character section being
        removed and leak the guest's store to Kurisu. Consumers scope at the
        point of use via :meth:`character_memory_path` instead.
        """

        if not getattr(self.memory, "per_character_memory", True):
            return None
        character = getattr(self, "character", None)
        if character is None or not character.is_custom:
            return None
        label = (getattr(character, "full_name", "") or "").strip() or (
            getattr(character, "name", "") or ""
        ).strip()
        if not label:
            return None
        from amadeus.characters import slugify

        try:
            return slugify(label)
        except ValueError:
            # Non-latin name (e.g. a Japanese-only pack name): fall back to a
            # deterministic digest so isolation still holds.
            import hashlib

            return hashlib.md5(label.encode("utf-8")).hexdigest()[:12]

    def character_memory_path(self, value: Optional[str]) -> Optional[str]:
        """Apply per-character memory scoping to one base path (pure)."""

        from amadeus.paths import character_scoped_path

        return character_scoped_path(value, self.memory_character_id())

    KNOWN_LLM_PROVIDERS = {"openai", "gemini", "local", "dummy"}
    KNOWN_TTS_PROVIDERS = {
        "openai",
        "pyttsx3",
        "huggingface",
        "huggingface_inference",
        "melotts",
        "melotts_onnx",
        "qwen3_tts",
    }
    KNOWN_STT_PROVIDERS = {"faster_whisper", "dummy"}

    def validate(self) -> List[str]:
        """Sanity-check the loaded config; warn (never raise) on suspect values.

        Misconfiguration otherwise surfaces as a silent dummy-provider
        fallback — warnings make the cause visible at startup.
        """
        warnings: List[str] = []

        if self.providers.llm not in self.KNOWN_LLM_PROVIDERS:
            warnings.append(
                f"providers.llm '{self.providers.llm}' is not one of "
                f"{sorted(self.KNOWN_LLM_PROVIDERS)}; the dummy provider will be used"
            )
        if self.providers.tts not in self.KNOWN_TTS_PROVIDERS:
            warnings.append(
                f"providers.tts '{self.providers.tts}' is not one of "
                f"{sorted(self.KNOWN_TTS_PROVIDERS)}; voice will fail to load"
            )
        if self.providers.stt not in self.KNOWN_STT_PROVIDERS:
            warnings.append(
                f"providers.stt '{self.providers.stt}' is not one of "
                f"{sorted(self.KNOWN_STT_PROVIDERS)}; voice input will be disabled"
            )

        for dial in ("tsundere_level", "scientific_formality"):
            value = getattr(self.persona, dial, None)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                warnings.append(f"persona.{dial}={value} is outside [0, 1]")
        guardedness = getattr(self.persona, "emotional_guardedness", None)
        if guardedness is not None and not 0.0 <= float(guardedness) <= 1.0:
            warnings.append(
                f"persona.emotional_guardedness={guardedness} is outside [0, 1]"
            )

        non_negative_memory_fields = (
            "max_context_messages",
            "summary_trigger_messages",
            "summary_max_chars",
            "recall_min_score",
            "recency_half_life_days",
            "session_gap_minutes",
            "strength_half_life_days",
            "strength_floor",
            "retrieval_reinforcement",
            "extraction_max_tokens",
            "episode_max_tokens",
            "reflection_max_tokens",
        )
        for field_name in non_negative_memory_fields:
            value = getattr(self.memory, field_name, None)
            if value is not None and float(value) < 0:
                warnings.append(f"memory.{field_name}={value} must not be negative")
        if not 0.0 <= float(self.memory.dedupe_threshold) <= 1.0:
            warnings.append(
                f"memory.dedupe_threshold={self.memory.dedupe_threshold} is outside [0, 1]"
            )
        if (
            getattr(self.memory, "enable_fact_extraction", False)
            and not self.memory.enable_long_term
        ):
            warnings.append(
                "memory.enable_fact_extraction is on but memory.enable_long_term is off — "
                "facts have nowhere to be stored (profile updates still work)"
            )

        gate = (getattr(self.tools, "gate", "auto") or "auto").lower()
        if gate not in {"auto", "keyword", "always"}:
            warnings.append(
                f"tools.gate '{self.tools.gate}' is not one of ['auto', 'keyword', "
                "'always']; the keyword gate will be used"
            )
        ws_provider = (getattr(self.web_search, "provider", "auto") or "auto").lower()
        if ws_provider not in {"auto", "tavily", "duckduckgo"}:
            warnings.append(
                f"web_search.provider '{self.web_search.provider}' is not one of "
                "['auto', 'tavily', 'duckduckgo']; web search will be unavailable"
            )

        for message in warnings:
            logging_module.getLogger(__name__).warning("Config: %s", message)
        return warnings


def save_default_config(path: Path) -> None:
    """Write the default configuration to ``path`` if it does not exist."""

    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    config = Config()
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataclasses.asdict(config), handle, sort_keys=False)
