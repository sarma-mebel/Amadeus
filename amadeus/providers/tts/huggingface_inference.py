"""Text-to-speech provider that calls a Hugging Face Space via the Gradio API."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional
import base64

from .base import TTSProvider


class HuggingFaceInferenceTTSProvider(TTSProvider):
    """Synthesize audio using a hosted Hugging Face Space (Gradio interface)."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        api_token: Optional[str] = None,
        voice: Optional[str] = None,
        api_name: str = "/predict",
        request_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if not model:
            raise ValueError(
                "A Hugging Face Space ID must be provided for the TTS provider"
            )

        self._space = model
        self._api_token = api_token
        self._voice = voice
        self._api_name = api_name or "/predict"
        self._request_kwargs = dict(request_kwargs or {})
        # The Gradio Client performs a network handshake with the Space at
        # construction time. Provider objects are built eagerly at process
        # startup (see SessionStore.__init__), so connecting here would make
        # every boot — including the packaged desktop app, offline — block
        # on that handshake. Defer it to the first real synthesis call.
        self._client: Any = None
        self._client_lock = threading.Lock()

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                try:
                    from gradio_client import Client
                except ImportError as exc:  # pragma: no cover - optional dependency
                    raise RuntimeError(
                        "The gradio-client package is required for the Hugging Face Space TTS provider"
                    ) from exc

                # gradio-client renamed hf_token= to token= in newer releases;
                # feature-detect so both generations work.
                import inspect

                client_params = inspect.signature(Client.__init__).parameters
                token_kwarg = "hf_token" if "hf_token" in client_params else "token"
                self._client = Client(self._space, **{token_kwarg: self._api_token})
        return self._client

    def synthesize(
        self,
        text: str,
        *,
        speaking_rate: float = 1.0,
        pitch_shift: float = 0.0,
    ) -> bytes:
        """Generate speech from ``text`` using the configured Hugging Face model."""

        # The Hugging Face Inference API currently does not expose knobs for
        # speaking rate or pitch on the text-to-speech endpoint. We accept the
        # parameters to satisfy the :class:`TTSProvider` protocol but only pass
        # along supported options.
        payload: dict[str, Any] = dict(self._request_kwargs)
        api_name = payload.pop("api_name", self._api_name)

        try:
            client = self._ensure_client()
            # Gradio Interface signature: fn(english_text, speaking_rate)
            result = client.predict(text, speaking_rate, api_name=api_name, **payload)
        except Exception as exc:  # pragma: no cover - network failure
            raise RuntimeError(
                f"Failed to synthesize speech with Space '{self._space}': {exc}"
            ) from exc

        return self._coerce_audio_bytes(result)

    @staticmethod
    def _coerce_audio_bytes(result: Any) -> bytes:
        """Normalize Gradio outputs to a WAV byte payload."""

        # Gradio returns tuples/lists of outputs; the first item is the audio widget
        if isinstance(result, (list, tuple)) and result:
            return HuggingFaceInferenceTTSProvider._coerce_audio_bytes(result[0])

        if isinstance(result, (bytes, bytearray)):
            return bytes(result)
        if isinstance(result, str):
            if result.startswith("data:audio"):
                try:
                    header, encoded = result.split(",", 1)
                except ValueError as exc:
                    raise TypeError(
                        f"Malformed audio data URI from Hugging Face Space: {result[:80]!r}"
                    ) from exc
                return base64.b64decode(encoded)
            path = Path(result)
            if path.exists():
                return path.read_bytes()
            # A string that is neither a data URI nor an existing file is an
            # error message or stale temp path — surface it instead of
            # returning the text as if it were audio.
            raise TypeError(
                "Hugging Face Space returned a string that is neither an audio data URI "
                f"nor an existing file path: {result[:120]!r}"
            )
        if hasattr(result, "read"):
            data = result.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
            return bytes(data)
        if hasattr(result, "content"):
            content = result.content
            if isinstance(content, str):
                content = content.encode("utf-8")
            return bytes(content)
        if isinstance(result, Iterable):
            return b"".join(
                HuggingFaceInferenceTTSProvider._coerce_audio_bytes(chunk)
                for chunk in result
            )
        if isinstance(result, dict):
            if "data" in result and isinstance(result["data"], str):
                encoded = result["data"]
                if encoded.startswith("data:audio"):
                    _, payload = encoded.split(",", 1)
                    return base64.b64decode(payload)
            if "path" in result and result["path"]:
                path = Path(str(result["path"]))
                if path.exists():
                    return path.read_bytes()
        if hasattr(result, "path"):
            try:
                path = Path(result.path)  # type: ignore[attr-defined]
                if path.exists():
                    return path.read_bytes()
            except Exception:
                pass
        raise TypeError(
            "Unexpected response type from Hugging Face inference client: "
            f"{type(result)!r}"
        )
