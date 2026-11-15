"""Lazy provider construction for CLI, web, and desktop entrypoints."""

from __future__ import annotations

from .config import Config
from .logging_utils import get_logger
from .memory.manager import EmbeddingProvider
from .providers.llm.base import DummyProvider, LLMProvider
from .providers.stt.base import STTProvider
from .providers.tts.base import TTSProvider

LOGGER = get_logger(__name__)


def _load_primary_llm(config: Config) -> LLMProvider:
    provider = config.providers.llm
    base_url = config.providers.openai_base_url
    if provider == "openai" and (config.openai_api_key or base_url):
        from .providers.llm.openai_provider import OpenAIProvider

        # Keyless OpenAI-compatible servers (Ollama, LM Studio) still require
        # the SDK to receive *some* api_key string.
        return OpenAIProvider(
            model=config.llm_model,
            api_key=config.openai_api_key or "not-needed",
            base_url=base_url,
        )
    if provider == "gemini" and config.gemini_api_key:
        from .providers.llm.gemini_provider import GeminiProvider

        return GeminiProvider(
            model=config.llm_model,
            api_key=config.gemini_api_key,
            transport=config.providers.gemini_transport,
        )
    if provider == "local" and config.local_model:
        from .providers.llm.local_provider import LocalTransformersProvider

        return LocalTransformersProvider(model_name=config.local_model)
    LOGGER.warning("Falling back to dummy LLM provider")
    return DummyProvider()


def load_llm(config: Config) -> LLMProvider:
    primary = _load_primary_llm(config)
    # Wrap with OpenRouter rate-limit failover if configured (never wrap the
    # dummy — nothing to fail over from).
    if (
        config.llm_fallback_enabled
        and config.openrouter_api_key
        and not isinstance(primary, DummyProvider)
    ):
        try:
            from .providers.llm.openai_provider import OpenAIProvider
            from .providers.llm.fallback import FallbackLLMProvider

            fallback = OpenAIProvider(
                model=config.openrouter_model,
                api_key=config.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
            )
            LOGGER.info(
                "LLM rate-limit failover enabled: %s -> %s",
                getattr(primary, "name", "?"),
                config.openrouter_model,
            )
            return FallbackLLMProvider(primary, fallback)
        except Exception as exc:  # pragma: no cover - keep primary if wrap fails
            LOGGER.warning("Could not enable LLM failover: %s", exc)
    return primary


def load_tts(config: Config) -> TTSProvider:
    provider = _load_tts_provider(config)
    if config.providers.tts_cache_enabled:
        from .providers.tts.cache import CachedTTSProvider

        return CachedTTSProvider(
            provider,
            cache_dir=config.providers.tts_cache_dir,
            max_chars=config.providers.tts_cache_max_chars,
        )
    return provider


def _load_tts_provider(config: Config) -> TTSProvider:
    provider = config.providers.tts
    if provider == "openai" and config.openai_api_key:
        from .providers.tts.openai_tts import OpenAITTSProvider

        return OpenAITTSProvider(api_key=config.openai_api_key)
    if provider == "pyttsx3":
        from .providers.tts.pyttsx3_provider import Pyttsx3Provider

        return Pyttsx3Provider()
    if provider == "huggingface":
        from .providers.tts.huggingface_tts import HuggingFaceTTSProvider

        return HuggingFaceTTSProvider(
            model_path=config.providers.huggingface_tts_path,
            model_kwargs=config.providers.huggingface_tts_kwargs,
        )
    if provider == "huggingface_inference":
        from .providers.tts.huggingface_inference import HuggingFaceInferenceTTSProvider

        space_id = (
            config.providers.huggingface_space_id
            or config.providers.huggingface_inference_model
        )
        if not space_id:
            raise RuntimeError(
                "Hugging Face Space ID is required when providers.tts is set to 'huggingface_inference'. "
                "Set providers.huggingface_space_id or providers.huggingface_inference_model in your config."
            )
        return HuggingFaceInferenceTTSProvider(
            model=space_id,
            voice=config.providers.huggingface_inference_voice,
            api_token=config.huggingface_api_token,
            api_name=config.providers.huggingface_inference_api_name,
            request_kwargs=config.providers.huggingface_tts_kwargs,
        )
    if provider == "melotts":
        from .providers.tts.melotts_provider import MeloTTSProvider

        return MeloTTSProvider(
            config_path=config.providers.melotts_config_path,
            checkpoint_path=config.providers.melotts_checkpoint_path,
            language=config.providers.melotts_language,
            device=config.providers.melotts_device,
            speaker_id=config.providers.melotts_speaker_id,
        )
    if provider == "melotts_onnx":
        from .providers.tts.melotts_onnx_provider import MeloTTSONNXProvider

        return MeloTTSONNXProvider(
            model_path=config.providers.melotts_onnx_model_path,
            tokens_path=config.providers.melotts_onnx_tokens_path,
            lexicon_path=config.providers.melotts_onnx_lexicon_path,
            speaker_id=config.providers.melotts_onnx_speaker_id,
        )
    if provider == "qwen3_tts":
        from .providers.tts.qwen3_tts_provider import Qwen3TTSProvider

        return Qwen3TTSProvider(
            model_id=config.providers.qwen3_tts_model_id,
            ref_audio=config.providers.qwen3_tts_ref_audio,
            ref_text=config.providers.qwen3_tts_ref_text,
            language=config.providers.qwen3_tts_language,
            device=config.providers.qwen3_tts_device,
            dtype=config.providers.qwen3_tts_dtype,
            temperature=config.providers.qwen3_tts_temperature,
            top_k=config.providers.qwen3_tts_top_k,
            speaker_guard=config.providers.qwen3_tts_speaker_guard,
            speaker_guard_threshold=config.providers.qwen3_tts_speaker_guard_threshold,
            stream_chunk_size=config.providers.qwen3_tts_stream_chunk_size,
            guard_prefix_seconds=config.providers.qwen3_tts_guard_prefix_seconds,
        )
    raise RuntimeError(f"Unsupported TTS provider: {provider}")


def load_stt(config: Config) -> STTProvider | None:
    provider = config.providers.stt
    if provider == "faster_whisper":
        try:
            from .providers.stt.faster_whisper import FasterWhisperProvider

            model_size = str(getattr(config.providers, "stt_model", "small") or "small")
            return FasterWhisperProvider(model_size=model_size)
        except RuntimeError as exc:
            LOGGER.error("Failed to initialize faster-whisper: %s", exc)
            return None
    if provider == "dummy":
        LOGGER.info("Dummy STT selected; voice capture disabled")
        return None
    LOGGER.warning("Unsupported STT provider '%s'", provider)
    return None


def load_embedder(config: Config) -> EmbeddingProvider | None:
    if not config.memory.enable_long_term:
        return None
    if config.providers.embedding == "sentence_transformers":
        try:
            from .memory.manager import SentenceTransformerEmbedding

            return SentenceTransformerEmbedding()
        except Exception as exc:
            # Long-term memory is on by default, but the [memory] extra is
            # opt-in — fall back to the dependency-free keyword embedder
            # (lexical recall only) instead of crashing startup.
            LOGGER.warning(
                "sentence-transformers unavailable (%s); "
                "falling back to hashing embedder (lexical recall only).",
                exc,
            )
            from .memory.manager import HashingEmbedding

            return HashingEmbedding()
    if config.providers.embedding == "gemini":
        if config.gemini_api_key:
            try:
                from .memory.manager import GeminiEmbedding

                return GeminiEmbedding(
                    api_key=config.gemini_api_key,
                    model=config.providers.gemini_embedding_model,
                    output_dim=config.providers.gemini_embedding_dim,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Gemini embedder unavailable (%s); "
                    "falling back to hashing embedder (lexical recall only).",
                    exc,
                )
        else:
            LOGGER.warning(
                "providers.embedding is 'gemini' but no Gemini API key is set; "
                "falling back to hashing embedder (lexical recall only)."
            )
        from .memory.manager import HashingEmbedding

        return HashingEmbedding()
    if config.providers.embedding == "hashing":
        # Dependency-free keyword embedder — used by the packaged desktop build,
        # which can't ship sentence-transformers/torch. Lexical recall only.
        from .memory.manager import HashingEmbedding

        return HashingEmbedding()
    return None
