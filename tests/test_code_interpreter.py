"""Sandboxed CodeInterpreterTool: subprocess isolation, timeout, env hygiene."""

import os

from amadeus.agents.tools import CodeInterpreterTool
from amadeus.config import Config


def test_happy_path_prints_result():
    tool = CodeInterpreterTool()
    result = tool.execute(code="print(6 * 7)")
    assert result.success is True
    assert result.result["stdout"].strip() == "42"
    assert result.result["exit_code"] == 0


def test_timeout_kills_runaway_code():
    tool = CodeInterpreterTool(timeout_seconds=1.0)
    result = tool.execute(code="while True:\n    pass")
    assert result.success is False
    assert "timed out" in result.error


def test_model_requested_timeout_cannot_exceed_configured():
    tool = CodeInterpreterTool(timeout_seconds=1.0)
    result = tool.execute(code="import time\ntime.sleep(30)", timeout=60)
    assert result.success is False
    assert "timed out after 1s" in result.error


def test_exception_in_user_code_reports_stderr_and_exit_status():
    tool = CodeInterpreterTool()
    result = tool.execute(code="raise ValueError('boom')")
    assert result.success is False
    assert result.result["exit_code"] != 0
    assert "ValueError" in result.result["stderr"]
    assert "boom" in result.error


def test_output_truncation():
    tool = CodeInterpreterTool()
    result = tool.execute(code="print('x' * 100000)")
    assert result.success is True
    assert len(result.result["stdout"]) < 10000
    assert result.result["stdout"].endswith("…[truncated]")


def test_parent_env_does_not_leak(monkeypatch):
    monkeypatch.setenv("AMADEUS_SECRET_SENTINEL", "leak-me")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-nope")
    tool = CodeInterpreterTool()
    result = tool.execute(
        code=(
            "import os\n"
            "print(os.environ.get('AMADEUS_SECRET_SENTINEL'))\n"
            "print(os.environ.get('OPENAI_API_KEY'))"
        )
    )
    assert result.success is True
    assert result.result["stdout"].split("\n")[:2] == ["None", "None"]


def test_runs_in_fresh_temp_cwd_not_repo():
    tool = CodeInterpreterTool()
    result = tool.execute(code="import os\nprint(os.getcwd())")
    assert result.success is True
    cwd = result.result["stdout"].strip()
    assert "amadeus_code_" in cwd
    assert not os.path.exists(cwd)  # cleaned up afterwards


def test_code_interpreter_allowed_by_default_and_timeout_configurable():
    config = Config()
    assert "code_interpreter" in config.tools.allowed
    assert config.tools.code_timeout_seconds == 10.0
    loaded = Config.from_dict({"tools": {"code_timeout_seconds": 3}})
    assert loaded.tools.code_timeout_seconds == 3

    from amadeus.agents.factory import create_tool_registry

    registry = create_tool_registry(loaded)
    assert registry.tools["code_interpreter"].timeout_seconds == 3.0
