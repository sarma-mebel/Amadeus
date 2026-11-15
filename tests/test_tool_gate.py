"""tools.gate selection (auto/keyword/always) and native no-tool-call cost."""

from amadeus.config import Config
from amadeus.engine import ChatEngine
from amadeus.providers.llm.base import ChatMessage, DummyProvider, ToolTurn


class _NativeLLM(DummyProvider):
    """Dummy provider that claims native function calling and scripts turns."""

    def __init__(self, turns=None):
        super().__init__()
        self.turns = list(turns or [])
        self.tool_calls = 0  # generate_with_tools invocations
        self.generate_calls = 0

    def supports_tools(self) -> bool:
        return True

    def generate(self, messages, **kwargs):
        self.generate_calls += 1
        return super().generate(messages, **kwargs)

    def generate_with_tools(self, messages, tools, **kwargs):
        self.tool_calls += 1
        return self.turns.pop(0)


def _engine(tmp_path, llm, gate="auto"):
    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.tools.enabled = True
    config.tools.gate = gate
    return ChatEngine(config, llm)


# --- gate selection -----------------------------------------------------


def test_keyword_gate_matches_hints_only(tmp_path):
    engine = _engine(tmp_path, _NativeLLM(), gate="keyword")
    # Even a native provider stays keyword-gated in "keyword" mode.
    assert engine._should_use_tools("find my manuscript file") is True
    assert engine._should_use_tools("hello there, how are you") is False


def test_auto_gate_native_provider_every_turn(tmp_path):
    engine = _engine(tmp_path, _NativeLLM(), gate="auto")
    assert engine._should_use_tools("hello there, how are you") is True
    assert engine._should_use_tools("hi") is True  # even short casual chat


def test_auto_gate_react_provider_keeps_keyword_gate(tmp_path):
    engine = _engine(tmp_path, DummyProvider(), gate="auto")
    assert engine._should_use_tools("hello there, how are you") is False
    assert engine._should_use_tools("find my manuscript file") is True


def test_always_gate_any_provider(tmp_path):
    engine = _engine(tmp_path, DummyProvider(), gate="always")
    assert engine._should_use_tools("hello there, how are you") is True


def test_unknown_gate_falls_back_to_keyword(tmp_path):
    engine = _engine(tmp_path, DummyProvider(), gate="bogus")
    assert engine._should_use_tools("hello there") is False
    assert engine._should_use_tools("search for cats") is True


def test_gate_default_is_auto():
    assert Config().tools.gate == "auto"
    assert Config.from_dict({"tools": {"gate": "always"}}).tools.gate == "always"


# --- native no-tool-call turn costs exactly one LLM call -----------------


def _context():
    return [
        ChatMessage(role="system", content="You are Kurisu."),
        ChatMessage(role="user", content="hello there"),
    ]


def test_native_no_tool_turn_is_single_llm_call(tmp_path):
    llm = _NativeLLM(turns=[ToolTurn(text="Hello. What do you want?")])
    engine = _engine(tmp_path, llm, gate="auto")
    reply = engine._run_tool_loop(_context())
    assert reply == "Hello. What do you want?"
    assert llm.tool_calls == 1
    assert llm.generate_calls == 0


def test_native_tool_call_turn_executes_and_continues(tmp_path):
    llm = _NativeLLM(
        turns=[
            ToolTurn(tool_name="calculator", args={"expression": "2 + 2"}),
            ToolTurn(text="It's 4, obviously."),
        ]
    )
    engine = _engine(tmp_path, llm, gate="auto")
    reply = engine._run_tool_loop(_context())
    assert reply == "It's 4, obviously."
    assert llm.tool_calls == 2
    assert llm.generate_calls == 0


def test_native_tool_turn_reports_progress(tmp_path):
    llm = _NativeLLM(
        turns=[
            ToolTurn(tool_name="calculator", args={"expression": "2 + 2"}),
            ToolTurn(text="It's 4, obviously."),
        ]
    )
    engine = _engine(tmp_path, llm, gate="auto")
    events = []
    reply = engine._run_tool_loop(_context(), progress_callback=events.append)
    assert reply == "It's 4, obviously."
    assert events == [
        {"tool": "calculator", "status": "running"},
        {"tool": "calculator", "status": "done"},
    ]


def test_progress_callback_errors_never_break_the_turn(tmp_path):
    llm = _NativeLLM(
        turns=[
            ToolTurn(tool_name="calculator", args={"expression": "1 + 1"}),
            ToolTurn(text="Two."),
        ]
    )
    engine = _engine(tmp_path, llm, gate="auto")

    def explode(event):
        raise RuntimeError("frontend went away")

    assert engine._run_tool_loop(_context(), progress_callback=explode) == "Two."


def test_native_loop_cap_forces_plain_answer(tmp_path):
    """Hitting the iteration cap must not surface an empty reply."""
    llm = _NativeLLM(
        turns=(
            # max_iterations=4 tool calls, a 5th pending call at the cap, then
            # the forced plain answer after the refusal observation.
            [ToolTurn(tool_name="calculator", args={"expression": "1"})] * 5
            + [ToolTurn(text="Fine. Here's what I found.")]
        )
    )
    engine = _engine(tmp_path, llm, gate="auto")
    reply = engine._run_tool_loop(_context())
    assert reply == "Fine. Here's what I found."


def test_empty_tool_reply_gets_fallback_line(tmp_path):
    class SilentLLM(_NativeLLM):
        def generate_with_tools(self, messages, tools, **kwargs):
            return ToolTurn(text="")

    engine = _engine(tmp_path, SilentLLM(), gate="auto")
    reply = engine._run_tool_loop(_context())
    assert reply.strip()  # never an empty bubble


# --- invalid-key diagnosis ------------------------------------------------


class _AuthFailingLLM(_NativeLLM):
    """Rejects every call the way a provider with a bad key does."""

    def generate(self, messages, **kwargs):
        raise RuntimeError(
            "400 API key not valid. Please pass a valid API key. "
            "[reason: API_KEY_INVALID]"
        )

    def generate_with_tools(self, messages, tools, **kwargs):
        self.tool_calls += 1
        raise RuntimeError(
            "400 API key not valid. Please pass a valid API key. "
            "[reason: API_KEY_INVALID]"
        )


def test_invalid_key_gets_diagnosis_not_lost_in_steps(tmp_path):
    """A rejected key must answer with the check-your-key line, not the
    generic ran-out-of-steps recovery text (which is how the bug hid in the
    wild), and must not retry plain generate()."""
    from amadeus.engine import INVALID_KEY_REPLY

    llm = _AuthFailingLLM()
    engine = _engine(tmp_path, llm)
    reply = engine._run_tool_loop([ChatMessage(role="user", content="hello there")])
    assert reply == INVALID_KEY_REPLY
    assert "ran out of steps" not in reply
    # Native first step only — no text-ReAct fallback, no plain retry.
    assert llm.tool_calls == 1
    assert llm.generate_calls == 0
