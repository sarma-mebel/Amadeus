"""GeminiProvider streaming + frequency_penalty fallback (no live API)."""

import sys
import types

import pytest


class _FakeGenerationConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeChunk:
    def __init__(self, text, usage_metadata=None):
        self.text = text
        self.candidates = []
        self.usage_metadata = usage_metadata


class _FakeModel:
    """Records generation configs; can be told to reject frequency_penalty."""

    def __init__(self, chunks, *, reject_penalty=False, usage=None):
        self._chunks = chunks
        self.reject_penalty = reject_penalty
        self.usage = usage  # attached to the response / FINAL stream chunk
        self.configs = []

    def generate_content(self, payload, **kwargs):
        gc = kwargs.get("generation_config", {})
        self.configs.append(dict(gc))
        if self.reject_penalty and "frequency_penalty" in gc:
            raise RuntimeError("400 Penalty is not enabled for this model")
        if kwargs.get("stream"):
            # A fresh iterator each call — mirrors Gemini's re-iterable
            # response, which is exactly what the double-yield bug tripped on.
            chunks = [_FakeChunk(c) for c in self._chunks]
            if chunks and self.usage is not None:
                chunks[-1].usage_metadata = self.usage
            return iter(chunks)
        return _FakeChunk("".join(self._chunks), usage_metadata=self.usage)


@pytest.fixture()
def provider(monkeypatch):
    fake_genai = types.ModuleType("google.generativeai")
    fake_genai.configure = lambda **_: None
    fake_genai.GenerativeModel = lambda *_a, **_k: None
    fake_genai.types = types.SimpleNamespace(GenerationConfig=_FakeGenerationConfig)
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)

    from amadeus.providers.llm.gemini_provider import GeminiProvider

    p = GeminiProvider.__new__(GeminiProvider)
    p._genai = fake_genai
    p._supports_system_instruction = True
    p._supports_frequency_penalty = True
    return p


def _msgs(text="hi"):
    from amadeus.providers.llm.base import ChatMessage

    return [ChatMessage(role="user", content=text)]


def test_stream_yields_each_chunk_once(provider):
    provider._model = _FakeModel(["ALPHA ", "BRAVO ", "CHARLIE"])
    out = "".join(provider.stream(_msgs()))
    assert out == "ALPHA BRAVO CHARLIE"
    assert out.count("ALPHA") == 1  # the re-iteration double-yield regression


def test_stream_penalty_rejection_falls_back(provider):
    model = _FakeModel(["DELTA ", "ECHO"], reject_penalty=True)
    provider._model = model
    out = "".join(provider.stream(_msgs(), frequency_penalty=0.4))
    assert out == "DELTA ECHO"
    assert provider._supports_frequency_penalty is False  # disabled after reject
    # First (rejected) config had the penalty; the retry config dropped it.
    assert "frequency_penalty" in model.configs[0]
    assert "frequency_penalty" not in model.configs[-1]


def test_generate_penalty_rejection_falls_back(provider):
    model = _FakeModel(["hello"], reject_penalty=True)
    provider._model = model
    assert provider.generate(_msgs(), frequency_penalty=0.4) == "hello"
    assert provider._supports_frequency_penalty is False


def test_penalty_omitted_when_zero(provider):
    model = _FakeModel(["ok"])
    provider._model = model
    provider.generate(_msgs(), frequency_penalty=0.0)
    assert "frequency_penalty" not in model.configs[0]


def test_build_payload_text_only_unchanged(provider):
    """Text-only messages keep the exact historical payload shape."""
    _system, history = provider._build_payload(_msgs("hello"))
    assert history == [{"role": "user", "parts": [{"text": "hello"}]}]


def test_build_payload_appends_inline_image_parts(provider):
    import base64

    from amadeus.providers.llm.base import ChatMessage, ImageContent

    raw = b"\x89PNG-fake-bytes"
    message = ChatMessage(
        role="user",
        content="what's on my screen?",
        images=[ImageContent(mime="image/png", data=base64.b64encode(raw).decode())],
    )
    _system, history = provider._build_payload([message])
    parts = history[0]["parts"]
    assert parts[0] == {"text": "what's on my screen?"}
    assert parts[1] == {"inline_data": {"mime_type": "image/png", "data": raw}}


def test_build_payload_drops_malformed_image(provider):
    from amadeus.providers.llm.base import ChatMessage, ImageContent

    message = ChatMessage(
        role="user",
        content="hi",
        images=[ImageContent(mime="image/png", data="!!!not-base64!!!")],
    )
    _system, history = provider._build_payload([message])
    assert history[0]["parts"] == [{"text": "hi"}]


def test_supports_vision(provider):
    assert provider.supports_vision() is True


# --- token usage capture ----------------------------------------------------


def _usage(prompt=12, completion=5):
    return types.SimpleNamespace(
        prompt_token_count=prompt, candidates_token_count=completion
    )


def test_generate_populates_last_usage(provider):
    provider._model = _FakeModel(["hello"], usage=_usage(12, 5))
    provider.generate(_msgs())
    assert provider.last_usage == {"prompt": 12, "completion": 5}


def test_generate_without_usage_resets_last_usage(provider):
    provider._model = _FakeModel(["hello"])
    provider.last_usage = {"prompt": 1, "completion": 1}  # stale from a prior call
    provider.generate(_msgs())
    assert provider.last_usage is None


def test_stream_populates_last_usage_from_final_chunk(provider):
    provider._model = _FakeModel(["ALPHA ", "BRAVO"], usage=_usage(30, 9))
    out = "".join(provider.stream(_msgs()))
    assert out == "ALPHA BRAVO"
    assert provider.last_usage == {"prompt": 30, "completion": 9}


def test_stream_without_usage_leaves_none(provider):
    provider._model = _FakeModel(["ALPHA ", "BRAVO"])
    provider.last_usage = {"prompt": 1, "completion": 1}  # stale from a prior call
    "".join(provider.stream(_msgs()))
    assert provider.last_usage is None


# --- native tool calling: parallel function_call parts ----------------------
#
# Gemini can return multiple function_call parts in ONE response ("parallel
# function calling"). The provider used to `return` on the first part found,
# silently dropping the rest with no log/counter. It must now collect ALL
# function_call parts and surface them via ToolTurn.raw as an ordered list of
# (name, args, raw_fc) tuples, so the tool loop can execute the whole batch in
# one iteration instead of losing calls 2..n.


class _FakeFunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class _FakePart:
    def __init__(self, function_call=None, text=None):
        self.function_call = function_call
        self.text = text


class _FakeContent:
    def __init__(self, parts):
        self.parts = parts


class _FakeCandidate:
    def __init__(self, content):
        self.content = content


class _FakeToolResponse:
    def __init__(self, parts, usage_metadata=None):
        self.candidates = [_FakeCandidate(_FakeContent(parts))]
        self.usage_metadata = usage_metadata


class _FakeToolModel:
    """Minimal stand-in for GenerativeModel.generate_content in the native
    tool-calling path — returns a scripted response regardless of payload."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_content(self, payload, **kwargs):
        self.calls.append({"payload": payload, "kwargs": kwargs})
        return self.response


def _fake_protos_module():
    """A tiny fake of google.generativeai.protos with just enough surface
    (Content, Part, FunctionResponse) for generate_with_tools's continuation
    branch to build native history objects."""
    import types as _types

    class _Content:
        def __init__(self, role=None, parts=None):
            self.role = role
            self.parts = parts or []

    class _Part:
        def __init__(self, text=None, function_response=None):
            self.text = text
            self.function_response = function_response

    class _FunctionResponse:
        def __init__(self, name=None, response=None):
            self.name = name
            self.response = response

    mod = _types.ModuleType("google.generativeai.protos")
    mod.Content = _Content
    mod.Part = _Part
    mod.FunctionResponse = _FunctionResponse
    return mod


@pytest.fixture()
def tool_provider(monkeypatch, provider):
    """Extends the `provider` fixture with the fake `protos` submodule that
    generate_with_tools imports lazily."""
    import sys

    monkeypatch.setitem(
        sys.modules, "google.generativeai.protos", _fake_protos_module()
    )
    return provider


def test_generate_with_tools_surfaces_all_parallel_function_calls(tool_provider):
    from amadeus import telemetry

    telemetry.reset()
    parts = [
        _FakePart(function_call=_FakeFunctionCall("web_search", {"query": "a"})),
        _FakePart(function_call=_FakeFunctionCall("calculator", {"expression": "1+1"})),
        _FakePart(function_call=_FakeFunctionCall("desktop_files", {"action": "find"})),
    ]
    tool_provider._model = _FakeToolModel(_FakeToolResponse(parts))

    turn = tool_provider.generate_with_tools(
        _msgs("do three things"),
        [{"name": "web_search"}, {"name": "calculator"}, {"name": "desktop_files"}],
    )

    # First call still mirrors tool_name/args (backward compatible).
    assert turn.tool_name == "web_search"
    assert turn.args == {"query": "a"}
    # ALL three calls must be recoverable from raw — nothing silently dropped.
    assert isinstance(turn.raw, list)
    assert len(turn.raw) == 3
    names = [c[0] for c in turn.raw]
    assert names == ["web_search", "calculator", "desktop_files"]
    assert turn.raw[1][1] == {"expression": "1+1"}
    assert turn.raw[2][1] == {"action": "find"}
    # Batch counter fires for a >1-call response.
    assert telemetry.snapshot().get("tools.parallel_calls_batch") == 1


def test_generate_with_tools_single_call_raw_is_not_a_list(tool_provider):
    """Regression: a single function_call must keep raw as the bare native fc
    object (old shape), not a one-element list — so any caller that only
    understands the pre-batch interface is unaffected."""
    parts = [_FakePart(function_call=_FakeFunctionCall("web_search", {"query": "a"}))]
    tool_provider._model = _FakeToolModel(_FakeToolResponse(parts))

    turn = tool_provider.generate_with_tools(_msgs("search"), [{"name": "web_search"}])

    assert turn.tool_name == "web_search"
    assert turn.args == {"query": "a"}
    assert not isinstance(turn.raw, list)
    assert turn.raw.name == "web_search"


def test_generate_with_tools_no_call_returns_text(tool_provider):
    parts = [_FakePart(text="the final answer")]
    tool_provider._model = _FakeToolModel(_FakeToolResponse(parts))

    turn = tool_provider.generate_with_tools(_msgs("hi"), [{"name": "web_search"}])

    assert turn.tool_name is None
    assert turn.text == "the final answer"


def test_generate_with_tools_continuation_sends_one_part_per_batched_result(
    tool_provider,
):
    """The continuation call (tool_result as a list) must build ONE Content
    carrying one function_response Part per call — not one Content per call —
    matching what the Gemini API expects for a batch of parallel responses."""
    first_parts = [
        _FakePart(function_call=_FakeFunctionCall("web_search", {"query": "a"})),
        _FakePart(function_call=_FakeFunctionCall("calculator", {"expression": "1+1"})),
    ]
    model = _FakeToolModel(_FakeToolResponse(first_parts))
    tool_provider._model = model

    first_turn = tool_provider.generate_with_tools(
        _msgs("do two things"), [{"name": "web_search"}, {"name": "calculator"}]
    )

    # Second call in the script: the model gives a final answer.
    model.response = _FakeToolResponse([_FakePart(text="done")])

    tool_provider.generate_with_tools(
        _msgs("do two things"),
        [{"name": "web_search"}, {"name": "calculator"}],
        history=first_turn.history,
        tool_result=[
            {"name": "web_search", "result": "result A"},
            {"name": "calculator", "result": "result B"},
        ],
    )

    # Inspect the payload the SECOND generate_content call received.
    continuation_payload = model.calls[-1]["payload"]
    last_content = continuation_payload[-1]
    assert last_content.role == "user"
    assert len(last_content.parts) == 2
    assert last_content.parts[0].function_response.name == "web_search"
    assert last_content.parts[0].function_response.response == {"result": "result A"}
    assert last_content.parts[1].function_response.name == "calculator"
    assert last_content.parts[1].function_response.response == {"result": "result B"}


def test_transport_forwarded_to_configure(monkeypatch):
    """providers.gemini_transport: "rest" must reach genai.configure —
    the escape hatch for gRPC endpoint outages (504s while REST answers)."""
    import sys
    import types
    from unittest import mock

    genai = types.ModuleType("google.generativeai")
    genai.configure = mock.MagicMock()
    model = mock.MagicMock()
    model.generate_content = lambda payload: None
    genai.GenerativeModel = mock.MagicMock(return_value=model)
    genai.types = mock.MagicMock()
    google_pkg = types.ModuleType("google")
    google_pkg.generativeai = genai
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.generativeai", genai)

    from amadeus.providers.llm.gemini_provider import GeminiProvider

    GeminiProvider(model="gemini-flash-lite-latest", api_key="k", transport="rest")
    genai.configure.assert_called_with(api_key="k", transport="rest")

    GeminiProvider(model="gemini-flash-lite-latest", api_key="k")
    genai.configure.assert_called_with(api_key="k")
