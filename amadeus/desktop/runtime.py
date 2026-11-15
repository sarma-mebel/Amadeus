"""Runtime configuration helpers for packaged desktop builds."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - fallback to bundled parser
    from ..utils import simple_yaml as yaml

from .. import gpu_runtime
from ..config import Config

_LOGGER = logging.getLogger(__name__)


REQUIRED_ASSET_PATHS = (
    "models/kurisu_voice_onnx/model.onnx",
    "models/kurisu_voice_onnx/tokens.txt",
    "models/kurisu_voice_onnx/lexicon.txt",
    "static/models/Kurisu.vrm",
    "static/environment/steinsgate_-_future_gadget_lab.glb",
    "static/environment/steinsgate_time_machine.glb",
    "assets/kurisu_isms.txt",
)
# Additional outfit VRMs (frontend/src/models.ts) and the optional
# static/models/models.json manifest that registers them are never
# packaging-gated: only the default Kurisu.vrm above is required.

# The Qwen3 voice-clone reference clip. Bundled only in the GPU desktop
# variant and never required for the default CPU/ONNX build, so it is kept out
# of REQUIRED_ASSET_PATHS (which gates packaging) and validated on its own.
QWEN_REF_ASSET_PATH = "models/kurisu_voice_qwen/kurisu_ref.wav"


def cuda_available() -> bool:
    """Return True only when a usable CUDA GPU is present.

    Guarded so it is safe to call from the default CPU sidecar, where torch is
    excluded entirely: a missing torch (ImportError) or any probe failure is
    treated as "no GPU" so the caller falls back to the ONNX voice.

    Logs the reason at INFO/WARNING so a fallback to ONNX in the GPU build is
    diagnosable — the most common cause is a CPU-only torch wheel (its
    ``torch.version.cuda`` is ``None``), which reports no GPU even on an NVIDIA
    machine.
    """

    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - torch excluded in CPU build
        _LOGGER.info("cuda_available: torch import failed (%s); assuming no GPU", exc)
        return False

    try:
        built_with_cuda = torch.version.cuda
        available = bool(torch.cuda.is_available())
        if available:
            _LOGGER.info(
                "cuda_available: True (torch %s, cuda %s)",
                torch.__version__,
                built_with_cuda,
            )
        elif not built_with_cuda:
            _LOGGER.warning(
                "cuda_available: False — torch %s is a CPU-only build "
                "(torch.version.cuda is None); the CUDA wheel was not installed.",
                torch.__version__,
            )
        else:
            _LOGGER.warning(
                "cuda_available: False — torch %s is a CUDA build but no GPU is "
                "visible (driver/hardware issue?).",
                torch.__version__,
            )
        return available
    except Exception as exc:  # pragma: no cover - depends on runtime hardware
        _LOGGER.warning("cuda_available: probe failed (%s); assuming no GPU", exc)
        return False


@dataclass(frozen=True)
class AssetValidationResult:
    """Required packaged asset validation summary."""

    root: Path
    present: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing


def _as_resolved_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def validate_required_assets(resources_dir: Path | str) -> AssetValidationResult:
    """Check that the desktop package includes the runtime assets it needs."""

    root = _as_resolved_path(resources_dir)
    present: list[str] = []
    missing: list[str] = []
    for relative in REQUIRED_ASSET_PATHS:
        if (root / relative).exists():
            present.append(relative)
        else:
            missing.append(relative)
    return AssetValidationResult(
        root=root, present=tuple(present), missing=tuple(missing)
    )


def build_desktop_config(
    *,
    app_data_dir: Path | str,
    resources_dir: Path | str,
    prefer_qwen: bool = False,
) -> Config:
    """Build a standalone-safe default config.

    Secrets intentionally stay out of this YAML. The Tauri shell injects the
    Gemini key into the backend process environment after reading OS keychain.

    ``prefer_qwen`` is set by the GPU installer variant. The Qwen3 cloned voice
    is selected only when the bundled reference clip exists AND a GPU is
    plausible; otherwise this silently falls back to the ONNX voice, so a GPU
    build launched on a machine without an NVIDIA GPU still speaks (just in
    the ONNX voice). The default CPU build passes ``prefer_qwen=False`` and
    always uses ONNX.

    GPU-plausibility is ``cuda_available() or gpu_runtime.gpu_likely_present()``
    — not ``cuda_available()`` alone. The slim GPU installer (2026-07-31,
    see ``amadeus/gpu_runtime.py``) ships without CUDA torch bundled, so
    ``cuda_available()`` (which imports torch) is unconditionally False on
    first launch even on a real GPU machine; baking ONNX in that case would
    be permanent, since this config is written ONCE. ``gpu_likely_present()``
    is a torch-free ``nvidia-smi`` probe that stays True once the driver is
    visible, so a slim build optimistically bakes qwen3_tts — the existing
    session-level ONNX fallback in ``web/sessions.py`` covers every launch
    until the runtime pack finishes downloading and activates. A full
    (non-slim) build behaves exactly as before: torch is bundled, so
    ``cuda_available()`` alone already answers correctly and
    ``gpu_likely_present()`` only ever adds a True where it was already True.
    A machine with a driver but genuinely broken CUDA (e.g. a CPU-only torch
    wheel) still degrades to the ONNX fallback at provider-load time, same as
    today — this function only decides what gets baked into the config, not
    what actually loads.
    """

    app_data = _as_resolved_path(app_data_dir)
    resources = _as_resolved_path(resources_dir)
    memory_dir = app_data / "memory"
    logs_dir = app_data / "logs"

    config = Config()
    config.gemini_api_key = None
    config.openai_api_key = None
    config.huggingface_api_token = None
    config.llm_model = "gemini-flash-lite-latest"
    config.kurisu_isms_path = str(resources / "assets" / "kurisu_isms.txt")
    config.transcript_path = str(app_data / "transcripts")

    config.audio.enable_voice = True
    config.providers.llm = "gemini"
    # Push-to-talk voice input. Harmless when the sidecar was packaged
    # without -IncludeSTT: load_stt logs the missing dependency,
    # /api/stt/status reports unavailable, and the frontend hides the mic.
    # When bundled, faster-whisper pulls its ~145MB "base" model on first
    # use (same runtime-download pattern as the Qwen3 voice weights).
    config.providers.stt = "faster_whisper"

    ref_clip = resources / "models" / "kurisu_voice_qwen" / "kurisu_ref.wav"
    if prefer_qwen:
        clip_present = ref_clip.exists()
        has_cuda_torch = cuda_available()
        driver_visible = gpu_runtime.gpu_likely_present()
        has_gpu = has_cuda_torch or driver_visible
        use_qwen = clip_present and has_gpu
        if use_qwen:
            _LOGGER.info(
                "Voice: selecting Qwen3 cloned voice (ref clip present; "
                "cuda_available=%s, nvidia-smi visible=%s).",
                has_cuda_torch,
                driver_visible,
            )
        else:
            _LOGGER.warning(
                "Voice: prefer_qwen set but falling back to ONNX "
                "(ref clip present=%s, cuda_available=%s, nvidia-smi visible=%s).",
                clip_present,
                has_cuda_torch,
                driver_visible,
            )
    else:
        use_qwen = False

    if use_qwen:
        config.providers.tts = "qwen3_tts"
        config.providers.qwen3_tts_ref_audio = str(ref_clip)
    else:
        config.providers.tts = "melotts_onnx"
    # The ONNX voice paths are always populated so it can serve as the runtime
    # fallback even in the GPU build (see load-time fallback in server.py).
    config.providers.melotts_onnx_model_path = str(
        resources / "models" / "kurisu_voice_onnx" / "model.onnx"
    )
    config.providers.melotts_onnx_tokens_path = str(
        resources / "models" / "kurisu_voice_onnx" / "tokens.txt"
    )
    config.providers.melotts_onnx_lexicon_path = str(
        resources / "models" / "kurisu_voice_onnx" / "lexicon.txt"
    )

    config.memory.persist_path = str(memory_dir / "memory_store.json")
    config.memory.embeddings_path = str(memory_dir / "memory_embeddings.json")
    config.memory.profile_path = str(memory_dir / "memory_profile.json")

    # Memory-as-personality. Long-term memory is ON in the desktop build using
    # the dependency-free HashingEmbedding + JSON vector store (config below):
    # the packaged sidecar can't ship sentence-transformers/torch, but the
    # hashing embedder needs neither, so facts, episodes, reflections, user
    # `remember`, and cross-session recall all actually persist and retrieve
    # (lexical/keyword match rather than semantic — an acceptable trade for
    # zero bundle cost). Without this, _store_fact / _run_episode_job /
    # _run_reflection_job / recall all silently no-op (they hard-require an
    # embedder + vector store).
    # NOTE: long_term/fact_extraction/episodes/greeting and
    # summary_model=None ("reuse chat model") are now also the library
    # defaults; the explicit overrides here are kept as the desktop contract
    # and still cover reflection/poke/idle, which library defaults leave off.
    config.memory.enable_long_term = True
    config.providers.embedding = "hashing"
    config.memory.vector_backend = "json"
    config.memory.summary_model = config.llm_model
    config.memory.enable_fact_extraction = True
    config.memory.enable_episodes = True
    config.memory.enable_greeting = True
    config.memory.enable_reflection = True
    # Presence features: the frontend is already fully wired for both
    # (ChatInterface POSTs /api/poke on avatar clicks and /api/idle on a
    # timer) — they just default off. A companion app should react.
    config.memory.enable_poke_reactions = True
    config.memory.enable_idle_chatter = True

    # Tool calling: the safe, keyless tools ship enabled — paper search
    # (PubMed/arXiv) and local reminders both cost nothing and need no
    # setup, and the frontend polls /api/reminders/due for announcements.
    # desktop_files stays OFF (reading the user's file tree is an explicit
    # opt-in, and its root needs configuring) as does web_search (needs a
    # Tavily key).
    config.tools.enabled = True
    config.literature.enabled = True
    config.reminders.enabled = True
    config.reminders.store_path = str(memory_dir / "reminders.json")

    config.logging.logfile = str(logs_dir / "amadeus.log")
    # Absolute, app-data-anchored TTS cache. The library default is the
    # relative "cache/tts", which resolves against CWD — for an installed app
    # that's Program Files, where every cache write fails with Access denied.
    config.providers.tts_cache_dir = str(app_data / "cache" / "tts")
    config.web.streaming_enabled = True
    config.web.streaming_tagged_output = True
    if use_qwen:
        # One resident Qwen3 model leaves ~3.5GB VRAM headroom on the 8GB
        # baseline card, so two sentence-synthesis workers can overlap and
        # cut long-reply latency without spilling.
        config.web.tts_stream_workers = 2
    return config


def _remove_secret_fields(payload: dict) -> dict:
    sanitized = dict(payload)
    for key in ("openai_api_key", "gemini_api_key", "huggingface_api_token"):
        sanitized.pop(key, None)
    web_search = sanitized.get("web_search")
    if isinstance(web_search, dict):
        web_search = dict(web_search)
        web_search.pop("api_key", None)
        sanitized["web_search"] = web_search
    return sanitized


def _create_parent_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def write_desktop_config(path: Path | str, config: Config) -> None:
    """Write desktop config YAML without persisting provider secrets."""

    config_path = _as_resolved_path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    _create_parent_dirs(
        Path(value)
        for value in (
            config.memory.persist_path,
            config.memory.embeddings_path,
            config.memory.profile_path,
            config.logging.logfile,
        )
        if value
    )
    Path(config.transcript_path).mkdir(parents=True, exist_ok=True)

    payload = _remove_secret_fields(dataclasses.asdict(config))
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


# Resource-anchored config keys and their locations relative to the
# resources dir: (yaml section or None for top-level, key, relative path).
_RESOURCE_PATH_KEYS = (
    (None, "kurisu_isms_path", "assets/kurisu_isms.txt"),
    ("providers", "melotts_onnx_model_path", "models/kurisu_voice_onnx/model.onnx"),
    ("providers", "melotts_onnx_tokens_path", "models/kurisu_voice_onnx/tokens.txt"),
    ("providers", "melotts_onnx_lexicon_path", "models/kurisu_voice_onnx/lexicon.txt"),
    ("providers", "qwen3_tts_ref_audio", QWEN_REF_ASSET_PATH),
)


def reconcile_desktop_config(
    path: Path | str,
    *,
    app_data_dir: Path | str,
    resources_dir: Path | str,
) -> bool:
    """Refresh stale machine paths in an existing desktop config.

    The desktop config is written once, ever (user edits must survive app
    updates) — but that means a config inherited from a previous install
    keeps ABSOLUTE paths into that install's tree. Seen live (v0.15.0): a
    config written by a portable build under Downloads pointed the ONNX
    model, qwen ref clip, and kurisu_isms at the portable dir; deleting that
    dir would have silently killed the voice entirely.

    Rules, deliberately conservative:
    - A resource-anchored path that no longer EXISTS is rewritten to this
      install's copy (only when this install actually has the file). A path
      that still exists is left alone — it may be a deliberate custom ref.
    - A RELATIVE ``tts_cache_dir`` is anchored under app data: relative
      resolves against CWD, which is Program Files for an installed app, and
      every cache write fails with Access denied.

    Returns True when the file was modified. Raises on unreadable/corrupt
    YAML — the caller decides how loudly to degrade.
    """

    config_path = _as_resolved_path(path)
    resources = _as_resolved_path(resources_dir)
    app_data = _as_resolved_path(app_data_dir)

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Desktop config is not a mapping: {config_path}")

    changed = False

    def _section(name: str | None) -> dict | None:
        if name is None:
            return payload
        section = payload.get(name)
        return section if isinstance(section, dict) else None

    for section_name, key, relative in _RESOURCE_PATH_KEYS:
        section = _section(section_name)
        if section is None or key not in section:
            continue
        value = section.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        if Path(value).exists():
            continue
        replacement = resources / relative
        if not replacement.exists():
            continue  # this install doesn't ship it either; leave as-is
        section[key] = str(replacement)
        changed = True
        _LOGGER.warning(
            "Desktop config: %s pointed at a missing file (%s); rewrote to %s",
            key,
            value,
            replacement,
        )

    providers = _section("providers")
    if providers is not None:
        cache_dir = providers.get("tts_cache_dir")
        if (
            isinstance(cache_dir, str)
            and cache_dir.strip()
            and not Path(cache_dir).is_absolute()
        ):
            providers["tts_cache_dir"] = str(app_data / "cache" / "tts")
            changed = True
            _LOGGER.warning(
                "Desktop config: relative tts_cache_dir (%s) anchored to app data",
                cache_dir,
            )

    if changed:
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
    return changed
