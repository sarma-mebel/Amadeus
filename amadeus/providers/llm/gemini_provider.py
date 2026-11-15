"""Google Gemini backend."""

from __future__ import annotations

import base64
import inspect
import logging
from typing import Iterable, List, Dict, Any, Tuple

from ... import telemetry
from .base import ChatMessage, LLMProvider, ToolTurn

LOGGER = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT_SECONDS = 12.0
_NO_CHUNK = object()  # sentinel: stream yielded nothing before erroring


class GeminiProvider(LLMProvider):
    """Adapter around Google's Gemini Generative AI API."""

    name = "gemini"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        transport: str | None = None,
    ) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "google-generativeai package is required for the Gemini provider"
            ) from exc

        if not api_key:
            raise RuntimeError("Gemini provider requires an API key")

        self._genai = genai
        # transport: None = SDK default (grpc). "rest" is the escape hatch
        # for gRPC endpoint outages — observed live 2026-07-16: every gRPC
        # call 504'd while the REST endpoint answered the same model in
        # under a second. Config: providers.gemini_transport.
        if transport:
            self._genai.configure(api_key=api_key, transport=transport)
        else:
            self._genai.configure(api_key=api_key)
        self._model = self._genai.GenerativeModel(model)
        # Older releases of google-generativeai don't support system_instruction; feature‐detect once.
        signature = inspect.signature(self._model.generate_content)
        self._supports_system_instruction = "system_instruction" in signature.parameters
        # frequency_penalty was added to GenerationConfig in later SDK
        # releases; probe once so older installs degrade gracefully.
        try:
            self._genai.types.GenerationConfig(frequency_penalty=0.1)
            self._supports_frequency_penalty = True
        except Exception:  # pragma: no cover - depends on SDK version
            self._supports_frequency_penalty = False

    def _capture_usage(self, response: object) -> None:
        """Record token usage from a response/chunk onto ``last_usage``.

        Best-effort: ``usage_metadata`` may be absent (older SDKs, some
        transports) — then ``last_usage`` stays as-is (callers reset it to
        None at call start), and this never raises.
        """
        try:
            meta = getattr(response, "usage_metadata", None)
            if meta is None:
                return
            prompt = getattr(meta, "prompt_token_count", None)
            completion = getattr(meta, "candidates_token_count", None)
            if prompt is None and completion is None:
                return
            self.last_usage = {
                "prompt": int(prompt or 0),
                "completion": int(completion or 0),
            }
        except Exception:  # pragma: no cover - observability must never raise
            pass

    def _generate_content(self, payload, request_kwargs, generation_config):
        """generate_content with a one-time retry that drops frequency_penalty.

        The SDK accepts the field, but many Gemini models reject it at the
        API ('Penalty is not enabled for this model'). We can't know that
        upfront, so on the first such rejection we disable it for this
        provider and retry without — every later call skips it cleanly.
        """
        try:
            return self._model.generate_content(payload, **request_kwargs)
        except Exception as exc:
            if "frequency_penalty" not in generation_config or (
                "penalty" not in str(exc).lower()
            ):
                raise
            self._supports_frequency_penalty = False
            generation_config.pop("frequency_penalty", None)
            return self._model.generate_content(payload, **request_kwargs)

    def _apply_frequency_penalty(
        self, generation_config: Dict[str, Any], kwargs: Dict[str, object]
    ) -> None:
        """Add frequency_penalty when the caller asked for one (>0).

        Only the chat path passes it — summary/extraction/tool calls emit
        short factual text or JSON where penalizing repeated tokens (keys,
        structure) is undesirable, so they leave it at 0. Feature-guarded so
        an SDK too old to know the field silently no-ops instead of erroring.
        """
        penalty = kwargs.get("frequency_penalty")
        if not penalty:
            return
        if not getattr(self, "_supports_frequency_penalty", True):
            return
        generation_config["frequency_penalty"] = float(penalty)

    def _build_payload(
        self, messages: Iterable[ChatMessage]
    ) -> Tuple[str | None, List[Dict[str, Any]]]:
        system_instruction: str | None = None
        history: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.role
            if role == "system":
                if system_instruction is None:
                    system_instruction = msg.content
                    continue
                # Additional system messages (memory summaries, etc.) are folded into the user stream.
                role = "user"
            if role == "assistant":
                role = "model"
            parts: List[Dict[str, Any]] = [{"text": msg.content}] if msg.content else []
            # Vision: inline image attachments ride as native inline_data
            # parts alongside the text (this one seam covers both generate()
            # and stream(), which share _build_payload).
            for image in getattr(msg, "images", None) or []:
                try:
                    raw = base64.b64decode(image.data)
                except Exception:
                    continue  # malformed attachment: drop it, keep the text
                parts.append({"inline_data": {"mime_type": image.mime, "data": raw}})
            if not parts:
                parts = [{"text": ""}]
            history.append({"role": role, "parts": parts})
        return system_instruction, history

    def generate(self, messages: Iterable[ChatMessage], **kwargs: object) -> str:
        system_instruction, history = self._build_payload(messages)
        # Temperature defaults to 0.8 if not provided by caller.
        temperature = kwargs.get("temperature", 0.8)
        payload = list(history)
        generation_config: Dict[str, Any] = {"temperature": temperature}
        self._apply_frequency_penalty(generation_config, kwargs)
        max_tokens = kwargs.get("max_tokens")
        if max_tokens:
            generation_config["max_output_tokens"] = int(max_tokens)
        request_kwargs: Dict[str, Any] = {"generation_config": generation_config}
        request_kwargs["request_options"] = {
            "timeout": float(kwargs.get("timeout", DEFAULT_REQUEST_TIMEOUT_SECONDS))
        }
        if system_instruction:
            if self._supports_system_instruction:
                request_kwargs["system_instruction"] = system_instruction
            else:
                payload.insert(
                    0, {"role": "user", "parts": [{"text": system_instruction}]}
                )

        self.last_usage = None
        response = self._generate_content(payload, request_kwargs, generation_config)
        self._capture_usage(response)
        try:
            text = getattr(response, "text", None)
        except Exception:
            text = None
        if text:
            return text
        # Fallback: concatenate returned parts if .text is not populated.
        for candidate in getattr(response, "candidates", []) or []:
            parts = getattr(candidate, "content", None)
            if not parts:
                continue
            texts = []
            for part in getattr(parts, "parts", []):
                piece = getattr(part, "text", None)
                if piece:
                    texts.append(piece)
            if texts:
                return "".join(texts)
        return ""

    def _extract_chunk_text(self, chunk: object) -> str:
        try:
            text = getattr(chunk, "text", None)
        except Exception:
            text = None
        if text:
            return text
        pieces = []
        for candidate in getattr(chunk, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []):
                piece = getattr(part, "text", None)
                if piece:
                    pieces.append(piece)
        return "".join(pieces)

    def stream(self, messages: Iterable[ChatMessage], **kwargs: object):
        system_instruction, history = self._build_payload(messages)
        temperature = kwargs.get("temperature", 0.8)
        payload = list(history)
        generation_config: Dict[str, Any] = {"temperature": temperature}
        self._apply_frequency_penalty(generation_config, kwargs)
        max_tokens = kwargs.get("max_tokens")
        if max_tokens:
            generation_config["max_output_tokens"] = int(max_tokens)
        request_kwargs: Dict[str, Any] = {"generation_config": generation_config}
        request_kwargs["request_options"] = {
            "timeout": float(kwargs.get("timeout", DEFAULT_REQUEST_TIMEOUT_SECONDS))
        }
        if system_instruction:
            if self._supports_system_instruction:
                request_kwargs["system_instruction"] = system_instruction
            else:
                payload.insert(
                    0, {"role": "user", "parts": [{"text": system_instruction}]}
                )

        # Pull the first chunk to surface an eager penalty rejection, then
        # continue the SAME iterator (Gemini's streaming response replays if
        # re-iterated, which would double every chunk).
        self.last_usage = None
        try:
            iterator = iter(
                self._model.generate_content(payload, stream=True, **request_kwargs)
            )
            first = next(iterator, _NO_CHUNK)
        except Exception as exc:
            if "frequency_penalty" not in generation_config or (
                "penalty" not in str(exc).lower()
            ):
                raise
            self._supports_frequency_penalty = False
            generation_config.pop("frequency_penalty", None)
            iterator = iter(
                self._model.generate_content(payload, stream=True, **request_kwargs)
            )
            first = next(iterator, _NO_CHUNK)
        if first is not _NO_CHUNK:
            # Usage rides the FINAL chunk; capturing every chunk (later ones
            # overwrite earlier) means last_usage is populated by the time the
            # generator is exhausted, without needing to know which is last.
            self._capture_usage(first)
            piece = self._extract_chunk_text(first)
            if piece:
                yield piece
            for chunk in iterator:
                self._capture_usage(chunk)
                piece = self._extract_chunk_text(chunk)
                if piece:
                    yield piece

    def supports_stream(self) -> bool:
        return True

    def supports_vision(self) -> bool:
        return True

    # --- Native function calling --------------------------------------------

    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(
        self, messages: Iterable[ChatMessage], tools: list, **kwargs: object
    ) -> ToolTurn:
        """Native Gemini function calling. On the FIRST step pass ChatMessages;
        on continuation pass history=<prior ToolTurn.history> plus the executed
        tool's (name, result) via the `tool_result` kwarg — either one dict
        ({"name", "result"}) or a list of them when the prior turn was a
        parallel-call batch. The Gemini API expects all function_response
        parts answering ONE model turn to ride back in a SINGLE Content, so a
        list becomes multiple Parts on one user Content, not one Content per
        call."""
        from google.generativeai import protos

        gem_tools = [
            {
                "function_declarations": [
                    {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    }
                    for t in tools
                ]
            }
        ]

        history = kwargs.get("history")
        if history is None:
            # First step: build native history from the ChatMessages. Gemini
            # takes the system prompt via system_instruction, not a turn.
            system_instruction, raw_history = self._build_payload(messages)
            history = [
                protos.Content(
                    role=("model" if h["role"] == "model" else "user"),
                    parts=[protos.Part(text=h["parts"][0]["text"])],
                )
                for h in raw_history
            ]
            self._tool_system = system_instruction  # remember for continuation
        else:
            # Continuation: append the executed tool result(s) as one Content
            # carrying one function_response Part per call.
            tr = kwargs.get("tool_result") or {}
            tool_results = tr if isinstance(tr, list) else [tr]
            response_parts = [
                protos.Part(
                    function_response=protos.FunctionResponse(
                        name=t.get("name", ""), response={"result": t.get("result", "")}
                    )
                )
                for t in tool_results
            ]
            history = list(history)
            history.append(protos.Content(role="user", parts=response_parts))

        gen_config = {"temperature": kwargs.get("temperature", 0.4)}
        req: Dict[str, Any] = {
            "tools": gem_tools,
            "generation_config": gen_config,
            "request_options": {
                "timeout": float(kwargs.get("timeout", DEFAULT_REQUEST_TIMEOUT_SECONDS))
            },
        }
        sysinstr = getattr(self, "_tool_system", None)
        if sysinstr and self._supports_system_instruction:
            req["system_instruction"] = sysinstr

        self.last_usage = None
        response = self._model.generate_content(list(history), **req)
        self._capture_usage(response)
        cand = response.candidates[0]
        model_content = cand.content
        # Collect EVERY function_call part — Gemini can emit several in one
        # response (parallel function calling). Returning only the first and
        # silently dropping the rest used to force the model to re-request
        # calls 2..n on a later turn, burning iterations against MAX_ITERS for
        # no reason. `raw` carries the full ordered batch (list of
        # (name, args, native_fc) tuples) when there's more than one call;
        # tool_name/args still mirror the FIRST call for any caller that only
        # looks at those two fields (backward compatible).
        calls = []
        for part in model_content.parts:
            fc = getattr(part, "function_call", None)
            if fc and fc.name:
                calls.append((fc.name, dict(fc.args), fc))
        if calls:
            new_history = list(history) + [
                model_content
            ]  # echo model turn back verbatim
            name, args, fc = calls[0]
            if len(calls) > 1:
                telemetry.incr("tools.parallel_calls_batch")
                LOGGER.debug(
                    "gemini returned %d parallel function calls: %s",
                    len(calls),
                    [c[0] for c in calls],
                )
                return ToolTurn(
                    tool_name=name, args=args, raw=calls, history=new_history
                )
            return ToolTurn(tool_name=name, args=args, raw=fc, history=new_history)
        # No call -> final text.
        text = ""
        for part in model_content.parts:
            if getattr(part, "text", None):
                text += part.text
        return ToolTurn(text=text, history=list(history) + [model_content])
