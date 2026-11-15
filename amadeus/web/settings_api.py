"""In-app settings: read/write the user's config YAML, and test candidate
providers before committing to them.

Games don't ask you to hand-edit a config file — this is the backend half
of that first-run/settings panel: a safe subset of Config that the frontend
can read and edit, persisted to the user's own config file (never the
bundled example), with a connection test that actually exercises the
provider instead of just validating the shape of the input.
"""

from __future__ import annotations

import dataclasses
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .. import cli, telemetry
from ..config import Config
from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - fallback to bundled parser
    from ..utils import simple_yaml as yaml  # type: ignore


def user_config_path() -> Path:
    """Where in-app settings changes are persisted.

    Distinct from Config.load()'s search order: that also considers
    config.local.yaml/config.yaml in the CWD and the bundled example, but
    writes must never land on the example, and CWD-relative files are a poor
    target for a packaged desktop app launched from an arbitrary directory.
    """
    override = os.getenv("AMADEUS_USER_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "amadeus" / "config.yaml"


def load_raw(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_raw(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    tmp_path.replace(path)  # atomic on POSIX and Windows alike


def effective_raw(config: Config) -> Dict[str, Any]:
    """The subset of the *running* config an update must not silently drop.

    A key can be live via an env var (OPENAI_API_KEY) or a config file this
    process started with that differs from the in-app settings path (e.g.
    config.local.yaml) — neither shows up in `load_raw(user_config_path())`.
    Deliberately narrow: only secrets, not the whole config tree, so a save
    can't accidentally snapshot file-only settings
    into the user config and freeze them there.
    """
    raw: Dict[str, Any] = {}
    if config.openai_api_key:
        raw["openai_api_key"] = config.openai_api_key
    if config.gemini_api_key:
        raw["gemini_api_key"] = config.gemini_api_key
    if config.huggingface_api_token:
        raw["huggingface_api_token"] = config.huggingface_api_token
    return raw


def _is_dummy(config: Config) -> bool:
    """Whether the effective LLM would resolve to the offline dummy fallback."""
    provider = config.providers.llm
    if provider == "openai":
        return not config.openai_api_key
    if provider == "gemini":
        return not config.gemini_api_key
    if provider == "local":
        return not config.local_model
    return provider not in Config.KNOWN_LLM_PROVIDERS or provider == "dummy"


class ProviderStatus(BaseModel):
    llm: str
    llm_is_dummy: bool
    tts: str
    stt: str


class SettingsResponse(BaseModel):
    """Safe-subset view of the running config. API keys are redacted:
    present-but-masked when set, null when not, so the UI can show
    "already configured" without ever round-tripping the secret."""

    is_first_run: bool
    providers: ProviderStatus
    openai_api_key_set: bool
    gemini_api_key_set: bool
    huggingface_api_token_set: bool
    llm_model: str
    persona: Dict[str, Any]
    streaming_enabled: bool
    audio_enable_voice: bool
    voice_language: str
    # Feature toggles that used to require hand-editing the YAML.
    memory_enable_long_term: bool
    web_search_enabled: bool
    tools_enabled: bool
    reminders_enabled: bool


class SettingsUpdate(BaseModel):
    """Partial update; unset fields are left untouched. Empty string for a
    key field is ignored (not treated as "clear the key") — clearing a key
    is a deliberate enough action that it should require config-file editing
    rather than an accidental blank-submit in the UI."""

    llm_provider: Optional[str] = Field(default=None)
    tts_provider: Optional[str] = Field(default=None)
    stt_provider: Optional[str] = Field(default=None)
    llm_model: Optional[str] = Field(default=None)
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    huggingface_api_token: Optional[str] = None
    huggingface_inference_model: Optional[str] = None
    tsundere_level: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    scientific_formality: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    safe_mode: Optional[bool] = None
    streaming_enabled: Optional[bool] = None
    audio_enable_voice: Optional[bool] = None
    voice_language: Optional[str] = Field(default=None, pattern="^(japanese|english)$")
    memory_enable_long_term: Optional[bool] = None
    web_search_enabled: Optional[bool] = None
    tools_enabled: Optional[bool] = None
    reminders_enabled: Optional[bool] = None

    @field_validator("openai_api_key", "gemini_api_key", "huggingface_api_token")
    @classmethod
    def _sane_api_key(cls, value: Optional[str]) -> Optional[str]:
        """Reject values that can't be an API key before they get persisted.

        A pasted chat fragment once landed in the key field and got saved,
        silently breaking every provider call until someone read the server
        logs. Keys never contain internal whitespace and are never this
        short — a deliberately loose shape check so provider format changes
        don't lock anyone out. Whitespace-only collapses to None (= "leave
        unchanged"), matching the empty-string semantics above.
        """
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        if any(ch.isspace() for ch in stripped) or len(stripped) < 20:
            raise ValueError(
                "that doesn't look like an API key (keys have no spaces and "
                "are at least 20 characters); the previous key was kept"
            )
        return stripped


class SettingsUpdateResponse(BaseModel):
    settings: SettingsResponse
    needs_restart: bool
    restart_reason: Optional[str] = None


class ProviderTestRequest(BaseModel):
    kind: str = Field(..., pattern="^(llm|tts)$")
    provider: str
    api_key: Optional[str] = None
    model: Optional[str] = None


class ProviderTestResponse(BaseModel):
    ok: bool
    message: str


def build_settings_response(
    config: Config, *, llm: Optional[Any] = None
) -> SettingsResponse:
    # llm_is_dummy prefers the live provider instance when the caller has one:
    # a correctly-configured provider that failed to construct at runtime
    # falls back to the dummy, and the config-shape check can't see that.
    # is_first_run stays config-derived — first-run is about configuration.
    if llm is not None:
        from ..providers.llm.base import DummyProvider

        llm_is_dummy = isinstance(llm, DummyProvider)
    else:
        llm_is_dummy = _is_dummy(config)
    return SettingsResponse(
        is_first_run=_is_dummy(config),
        providers=ProviderStatus(
            llm=config.providers.llm,
            llm_is_dummy=llm_is_dummy,
            tts=config.providers.tts,
            stt=config.providers.stt,
        ),
        openai_api_key_set=bool(config.openai_api_key),
        gemini_api_key_set=bool(config.gemini_api_key),
        huggingface_api_token_set=bool(config.huggingface_api_token),
        llm_model=config.llm_model,
        persona={
            "tsundere_level": config.persona.tsundere_level,
            "scientific_formality": config.persona.scientific_formality,
            "safe_mode": config.persona.safe_mode,
        },
        streaming_enabled=config.web.streaming_enabled,
        audio_enable_voice=config.audio.enable_voice,
        voice_language=str(
            getattr(config.audio, "voice_language", "japanese") or "japanese"
        ),
        memory_enable_long_term=config.memory.enable_long_term,
        web_search_enabled=config.web_search.enabled,
        tools_enabled=config.tools.enabled,
        reminders_enabled=config.reminders.enabled,
    )


def apply_update(
    raw: Dict[str, Any], update: SettingsUpdate
) -> tuple[Dict[str, Any], bool, Optional[str]]:
    """Merge ``update`` into the raw YAML dict. Returns (new_raw, needs_restart, reason).

    Never accepts an empty string as "set this secret" — that would silently
    wipe a configured key on a UI bug or accidental blank submit.
    """
    raw = {**raw}
    providers = {**raw.get("providers", {})}
    persona = {**raw.get("persona", {})}
    web = {**raw.get("web", {})}
    audio = {**raw.get("audio", {})}
    memory = {**raw.get("memory", {})}
    web_search = {**raw.get("web_search", {})}
    tools = {**raw.get("tools", {})}
    reminders = {**raw.get("reminders", {})}

    needs_restart = False
    reasons: List[str] = []

    def note_restart(reason: str) -> None:
        nonlocal needs_restart
        needs_restart = True
        if reason not in reasons:
            reasons.append(reason)

    if update.llm_provider is not None:
        providers["llm"] = update.llm_provider
        note_restart("LLM provider changed")
    if update.tts_provider is not None:
        providers["tts"] = update.tts_provider
        note_restart("Voice provider changed")
    if update.stt_provider is not None:
        providers["stt"] = update.stt_provider
        note_restart("Voice input provider changed")
    if update.llm_model is not None:
        raw["llm_model"] = update.llm_model
        note_restart("Model changed")
    if update.openai_api_key:
        raw["openai_api_key"] = update.openai_api_key
        note_restart("API key changed")
    if update.gemini_api_key:
        raw["gemini_api_key"] = update.gemini_api_key
        note_restart("API key changed")
    if update.huggingface_api_token:
        raw["huggingface_api_token"] = update.huggingface_api_token
        note_restart("API key changed")
    if update.huggingface_inference_model is not None:
        providers["huggingface_inference_model"] = update.huggingface_inference_model
        note_restart("Voice model changed")

    if update.tsundere_level is not None:
        persona["tsundere_level"] = update.tsundere_level
    if update.scientific_formality is not None:
        persona["scientific_formality"] = update.scientific_formality
    if update.safe_mode is not None:
        persona["safe_mode"] = update.safe_mode
    if update.streaming_enabled is not None:
        web["streaming_enabled"] = update.streaming_enabled
        note_restart("Streaming mode changed")
    if update.audio_enable_voice is not None:
        audio["enable_voice"] = update.audio_enable_voice
    if update.voice_language is not None:
        audio["voice_language"] = update.voice_language
    # Feature toggles: registries/managers are built per session or at
    # startup, so existing sessions keep the old behavior — flag the restart.
    if update.memory_enable_long_term is not None:
        memory["enable_long_term"] = update.memory_enable_long_term
        note_restart("Memory settings changed")
    if update.web_search_enabled is not None:
        web_search["enabled"] = update.web_search_enabled
        note_restart("Tool settings changed")
    if update.tools_enabled is not None:
        tools["enabled"] = update.tools_enabled
        note_restart("Tool settings changed")
    if update.reminders_enabled is not None:
        reminders["enabled"] = update.reminders_enabled
        note_restart("Tool settings changed")

    if providers:
        raw["providers"] = providers
    if persona:
        raw["persona"] = persona
    if web:
        raw["web"] = web
    if audio:
        raw["audio"] = audio
    if memory:
        raw["memory"] = memory
    if web_search:
        raw["web_search"] = web_search
    if tools:
        raw["tools"] = tools
    if reminders:
        raw["reminders"] = reminders

    return raw, needs_restart, ("; ".join(reasons) if reasons else None)


def deep_merge_config(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def persist_and_reload(store: Any, new_raw: Dict[str, Any]) -> Config:
    """Shared persist tail for settings writes (PUT /api/settings and
    persona-version rollback): write the user config file, build the
    effective runtime config, and hot-apply it via ``reload_providers``.

    ``store`` is the SessionStore (typed Any to avoid a circular import).
    """
    settings_path = user_config_path()
    write_raw(settings_path, new_raw)

    runtime_raw = dataclasses.asdict(store.config)
    new_config = Config.from_dict(deep_merge_config(runtime_raw, new_raw))
    new_config.resolve_user_data_paths()
    new_config.validate()
    # Persona/audio dials apply immediately; provider/model/streaming
    # changes are persisted but only take full effect for sessions created
    # after this call (see reload_providers docstring).
    store.reload_providers(new_config)
    _maybe_start_scheduler(store, new_config)
    return new_config


def _maybe_start_scheduler(store: Any, config: Config) -> None:
    """Start the background scheduler when a settings save newly needs one.

    The lifespan gate (web/app.py) is evaluated exactly once at boot, so
    enabling a scheduler feature via settings used to silently never run it
    until restart. The lifespan stashes the running event loop and the
    (possibly None) scheduler on the store; when the new config satisfies
    the gate and nothing is running, construct and start one exactly the way
    the lifespan does — the scheduler reads its live config through
    ``store.config`` every tick, so subsequent saves need no restart. A
    scheduler is never STOPPED here: with every feature disabled each job
    no-ops per tick, which the lifespan gate already accepts as harmless.

    Best-effort by contract (a settings save must never fail on this); the
    no-loop path counts loudly instead of silently not running.
    """
    try:
        from .scheduler import BackgroundScheduler, scheduler_features_enabled

        if not scheduler_features_enabled(config):
            return
        if getattr(store, "scheduler", None) is not None:
            return
        loop = getattr(store, "scheduler_loop", None)
        if loop is None or loop.is_closed():
            # No lifespan ran (embedded/test usage) so there is no loop to
            # start the task on — loud, per the degrade-loudly convention.
            telemetry.incr("scheduler.not_running_but_needed")
            LOGGER.warning(
                "A scheduler-driven feature is enabled but no background "
                "scheduler is running (no event loop available); restart "
                "the server to start it."
            )
            return
        import asyncio

        scheduler = BackgroundScheduler(store, config)
        store.scheduler = scheduler
        asyncio.run_coroutine_threadsafe(scheduler.start(), loop)
        telemetry.incr("scheduler.started_on_reload")
        LOGGER.info("Background scheduler started after settings change")
    except Exception as exc:  # never let scheduler start-up break a save
        telemetry.incr("scheduler.not_running_but_needed")
        LOGGER.warning("Could not start background scheduler on reload: %s", exc)


def _candidate_config(base: Config, request: ProviderTestRequest) -> Config:
    """A scratch Config with the candidate provider/key merged in — never
    persisted, used only to construct a throwaway provider for testing."""
    updated = dataclasses.replace(base)
    if request.kind == "llm":
        updated.providers = dataclasses.replace(updated.providers, llm=request.provider)
        if request.model:
            updated.llm_model = request.model
        if request.api_key:
            if request.provider == "openai":
                updated.openai_api_key = request.api_key
            elif request.provider == "gemini":
                updated.gemini_api_key = request.api_key
    else:
        updated.providers = dataclasses.replace(updated.providers, tts=request.provider)
        if request.api_key:
            updated.huggingface_api_token = request.api_key
        if request.model:
            updated.providers = dataclasses.replace(
                updated.providers,
                huggingface_space_id=request.model,
                huggingface_inference_model=request.model,
            )
    return updated


# Provider test calls are user-initiated and rare; a lock keeps two
# concurrent tests from racing on the same shared HTTP client libraries'
# global state (mostly a defensive measure for exotic provider SDKs).
_test_lock = threading.Lock()


def run_provider_test(
    base_config: Config, request: ProviderTestRequest
) -> ProviderTestResponse:
    candidate = _candidate_config(base_config, request)
    with _test_lock:
        try:
            if request.kind == "llm":
                if request.provider == "dummy":
                    return ProviderTestResponse(
                        ok=True, message="Dummy provider needs no connection."
                    )
                provider = cli.load_llm(candidate)
                if getattr(provider, "__class__", None).__name__ == "DummyProvider":
                    return ProviderTestResponse(
                        ok=False,
                        message="No API key configured for this provider — falling back to dummy.",
                    )
                reply = provider.generate(
                    [cli.ChatMessage(role="user", content="Say OK.")]
                )
                if not reply or not str(reply).strip():
                    return ProviderTestResponse(
                        ok=False, message="Provider returned an empty response."
                    )
                return ProviderTestResponse(ok=True, message="Connected successfully.")

            provider = cli.load_tts(candidate)
            audio = provider.synthesize("Test.", speaking_rate=1.0, pitch_shift=0.0)
            if not audio:
                return ProviderTestResponse(
                    ok=False, message="Provider returned no audio."
                )
            return ProviderTestResponse(ok=True, message="Connected successfully.")
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            return ProviderTestResponse(ok=False, message=str(exc)[:300])
