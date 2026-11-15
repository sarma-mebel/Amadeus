"""Unit tests for the eval harness's turn-plumbing additions (2026-07-26):
say_goodbye / expect_greeting turn flags, --subset scenario filtering, and
the --trend-file JSON-line format. NO live LLM calls — everything drives a
scripted/dummy provider through the real ChatEngine, the same way
tests/test_engine.py exercises open_session_greeting/close_session_farewell
directly. scripts/eval_conversation.py is not an installed package, so it's
loaded the same way tests/test_eval_checks.py does (importlib from its repo
path).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import List

from amadeus.config import Config
from amadeus.providers.llm.base import ChatMessage, LLMProvider

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "eval_conversation_harness_test", REPO_ROOT / "scripts" / "eval_conversation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = _load_eval_module()


def _isolated_config(tmp_path: Path) -> Config:
    config = Config()
    config.providers.embedding = "hashing"  # dependency-free, no network
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    config.memory.enable_greeting = True
    config.memory.enable_farewell = True
    # Keep recall generous enough that short scripted facts aren't filtered
    # out by the min-query-chars gate.
    config.memory.recall_min_query_chars = 1
    return config


class ScriptedLLM(LLMProvider):
    """Content-routed fake: which reply comes back depends on the system
    prompt's shape, not call order — a strict FIFO queue is fragile here
    because MemoryManager's background jobs (summary/episode/fact
    extraction) also call ``generate`` on this same provider, interleaved
    with the farewell/greeting/chat calls the test cares about, in an order
    the test shouldn't need to know.

    ``goodbye``/``greeting``/``next_time`` fire on the distinctive
    system-prompt markers close_session_farewell / open_session_greeting /
    note_next_time_thread always use (see amadeus/engine.py,
    amadeus/next_time.py). Everything else (chat turns, session-logger
    summaries, fact extraction) gets ``default`` — a plain sentence, never
    JSON, so a fact-extraction call that expects JSON just no-ops instead of
    raising.
    """

    name = "scripted"
    model = None

    def __init__(
        self,
        chat_replies: List[str],
        *,
        goodbye: str = "Take care.",
        greeting: str = "Welcome back.",
        next_time: str = "SKIP",
        default: str = "Fine.",
    ) -> None:
        self.chat_replies = list(chat_replies)
        self.goodbye = goodbye
        self.greeting = greeting
        self.next_time = next_time
        self.default = default
        self.calls: List[List[ChatMessage]] = []

    def generate(self, messages, **kwargs):  # type: ignore[override]
        messages = list(messages)
        self.calls.append(messages)
        system = (
            messages[0].content if messages and messages[0].role == "system" else ""
        )
        if "saying goodbye" in system:
            return self.goodbye
        if "opening a conversation" in system:
            return self.greeting
        if "what to pick up next time" in system:
            return self.next_time
        # Normal chat turns (prepare_turn -> persona.build_system_prompt)
        # always open with this exact preamble — distinct from the
        # summary/episode/fact-extraction background jobs' own system
        # prompts ("You maintain Kurisu's memory...", "You are a neutral
        # session logger..."), which must NOT consume chat_replies.
        if (
            "digital recreation built from Makise Kurisu" in system
            and self.chat_replies
        ):
            return self.chat_replies.pop(0)
        return self.default


# ---------------------------------------------------------------------------
# say_goodbye
# ---------------------------------------------------------------------------


def test_say_goodbye_routes_through_farewell_path(tmp_path):
    """say_goodbye must call close_session_farewell (not a normal chat
    generate) and use its line as the checked reply."""
    config = _isolated_config(tmp_path)
    llm = ScriptedLLM(
        chat_replies=["I mentioned my manuscript submission."],
        goodbye="Good luck out there. See you.",
    )
    scenario = {
        "name": "say_goodbye_smoke",
        "turns": [
            {"user": "About to submit the manuscript tonight."},
            {
                "user": "ok bye!",
                "say_goodbye": True,
                "forbid_substr": ["miss you"],
            },
        ],
    }
    result = EVAL.run_scenario(scenario, config, llm, None)
    assert result["check_failures"] == []
    assert result["transcript"][1]["kurisu"] == "Good luck out there. See you."
    # A call with the send-off system prompt actually happened — confirms
    # close_session_farewell's own prompt was used, not prepare_turn's.
    assert any(
        c[0].role == "system" and "saying goodbye" in c[0].content for c in llm.calls
    )


def test_say_goodbye_commits_user_and_assistant_once(tmp_path):
    """Mirrors POST /api/farewell's commit contract: the goodbye user
    message and the send-off reply each land in memory exactly once. A
    double-commit would duplicate the assistant reply in short-term memory,
    which would surface as a spurious forbid_repeat-style near-duplicate on
    a later turn — so chain a second scenario turn through it as the probe."""
    config = _isolated_config(tmp_path)
    scenario2 = {
        "name": "say_goodbye_commit_2",
        "turns": [
            # should_bid_farewell() requires a non-empty short_term (there
            # must BE a conversation to close), hence a first turn.
            {"user": "quick hello"},
            {"user": "bye for now", "say_goodbye": True},
            {"user": "hey", "new_session": True, "expect_greeting": True},
        ],
    }
    llm2 = ScriptedLLM(chat_replies=["Hi there."])
    result = EVAL.run_scenario(scenario2, config, llm2, None)
    assert result["check_failures"] == []
    assert result["transcript"][1]["kurisu"] == "Take care."
    assert result["transcript"][2]["kurisu"] == "Welcome back."


def test_say_goodbye_falls_back_to_normal_turn_when_no_line(tmp_path):
    """When should_bid_farewell() gates it off (farewell disabled), the
    turn must fall back to an ordinary chat turn instead of leaving a gap."""
    config = _isolated_config(tmp_path)
    config.memory.enable_farewell = False
    llm = ScriptedLLM(chat_replies=["Sure, bye then."])
    scenario = {
        "name": "say_goodbye_fallback",
        "turns": [{"user": "bye", "say_goodbye": True}],
    }
    result = EVAL.run_scenario(scenario, config, llm, None)
    assert result["transcript"][0]["kurisu"] == "Sure, bye then."
    # Fallback used prepare_turn/generate — the plain chat path, not the
    # farewell system prompt.
    assert not any(
        c[0].role == "system" and "saying goodbye" in c[0].content for c in llm.calls
    )


# ---------------------------------------------------------------------------
# expect_greeting
# ---------------------------------------------------------------------------


def test_expect_greeting_checks_the_greeting_not_the_user_reply(tmp_path):
    """expect_recall/forbid_substr on an expect_greeting turn must be
    evaluated against open_session_greeting()'s output, not against the
    reply to the turn's own `user` text."""
    config = _isolated_config(tmp_path)
    llm = ScriptedLLM(
        chat_replies=["About to submit the manuscript tonight.", "Oh, hi."],
        greeting="How did the manuscript submission go?",
        next_time="They were about to submit the manuscript.",
    )
    scenario = {
        "name": "expect_greeting_smoke",
        "turns": [
            {"user": "About to submit the manuscript tonight."},
            {"user": "ok bye!", "say_goodbye": True},
            {
                "user": "hey",
                "new_session": True,
                "expect_greeting": True,
                "expect_recall": ["manuscript"],
            },
        ],
    }
    result = EVAL.run_scenario(scenario, config, llm, None)
    assert result["check_failures"] == []
    # Both the greeting AND the user-text reply appear in the transcript,
    # sequenced (greeting first).
    kurisu_lines = [t["kurisu"] for t in result["transcript"]]
    assert "How did the manuscript submission go?" in kurisu_lines
    assert "Oh, hi." in kurisu_lines
    greeting_idx = kurisu_lines.index("How did the manuscript submission go?")
    reply_idx = kurisu_lines.index("Oh, hi.")
    assert greeting_idx < reply_idx


def test_expect_greeting_fails_the_check_when_greeting_is_none(tmp_path):
    """A greeting that doesn't fire (should_greet() False, e.g. greeting
    disabled) must fail expect_recall loudly, not silently pass."""
    config = _isolated_config(tmp_path)
    config.memory.enable_greeting = False
    llm = ScriptedLLM(chat_replies=["hi there"])
    scenario = {
        "name": "expect_greeting_gated_off",
        "turns": [
            {
                "user": "hey",
                "new_session": True,
                "expect_greeting": True,
                "expect_recall": ["manuscript"],
            }
        ],
    }
    result = EVAL.run_scenario(scenario, config, llm, None)
    assert any("expected recall" in f for f in result["check_failures"])


def test_expect_greeting_without_new_session_still_runs(tmp_path):
    """expect_greeting is documented as only meaningful with new_session,
    but must not crash without it — should_greet() on a fresh engine with
    empty short_term is still True (first turn ever), so this exercises the
    plain not-mid-conversation greeting path instead of erroring."""
    config = _isolated_config(tmp_path)
    llm = ScriptedLLM(chat_replies=["Hello."])
    scenario = {
        "name": "expect_greeting_no_new_session",
        "turns": [{"user": "hey", "expect_greeting": True}],
    }
    result = EVAL.run_scenario(scenario, config, llm, None)
    # No exception; a greeting entry was recorded.
    assert result["transcript"][0]["user"] == "(session opened)"


# ---------------------------------------------------------------------------
# --subset scenario filtering (main()'s wanted_names union logic)
# ---------------------------------------------------------------------------


def test_subset_and_scenario_flags_union(monkeypatch, tmp_path):
    """--subset and (repeated) --scenario narrow to the union of both, with
    no crash when only one is given."""
    scenarios_path = tmp_path / "scenarios.yaml"
    scenarios_path.write_text(
        "scenarios:\n"
        "  - name: alpha\n    turns: [{user: hi}]\n"
        "  - name: beta\n    turns: [{user: hi}]\n"
        "  - name: gamma\n    turns: [{user: hi}]\n"
    )
    calls = []

    def fake_run_scenario(scenario, cfg, llm, embedder, **kwargs):
        calls.append(scenario["name"])
        return {
            "name": scenario["name"],
            "description": "",
            "rubric_focus": "",
            "transcript": [],
            "check_failures": [],
        }

    monkeypatch.setattr(EVAL, "run_scenario", fake_run_scenario)
    monkeypatch.setattr(EVAL, "load_llm", lambda cfg: ScriptedLLM(chat_replies=[]))
    monkeypatch.setattr(EVAL, "load_embedder", lambda cfg: None)
    monkeypatch.setattr(
        EVAL.Config,
        "load",
        classmethod(lambda cls, path=None: _isolated_config(tmp_path)),
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_conversation.py",
            "--scenarios",
            str(scenarios_path),
            "--subset",
            "alpha,gamma",
            "--no-judge",
        ],
    )
    rc = EVAL.main()
    assert rc == 0
    assert sorted(calls) == ["alpha", "gamma"]


# ---------------------------------------------------------------------------
# Trend file
# ---------------------------------------------------------------------------


def test_build_trend_entry_shape(tmp_path):
    config = _isolated_config(tmp_path)
    config.providers.llm = "gemini"
    config.llm_model = "gemini-flash-lite-latest"
    results = [
        {"name": "alpha", "judge": {"overall": 8}},
        {"name": "beta", "judge": {"overall": 6}},
    ]
    entry = EVAL.build_trend_entry(
        results,
        judge_model="judge-model-x",
        config=config,
        timestamp="2026-07-26T00:00:00+00:00",
    )
    assert entry["timestamp"] == "2026-07-26T00:00:00+00:00"
    assert entry["scores"] == {"alpha": 8, "beta": 6}
    assert entry["judge_model"] == "judge-model-x"
    assert entry["config_fingerprint"]["llm_provider"] == "gemini"
    assert entry["config_fingerprint"]["llm_model"] == "gemini-flash-lite-latest"
    # Never a secret: no api-key-shaped fields at all.
    fingerprint_str = json.dumps(entry["config_fingerprint"])
    assert "key" not in fingerprint_str.lower()


def test_append_trend_line_writes_one_json_line_per_call(tmp_path):
    path = tmp_path / "nested" / "trend.jsonl"
    EVAL.append_trend_line(path, {"a": 1})
    EVAL.append_trend_line(path, {"a": 2})
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"a": 2}


def test_trend_file_flag_appends_during_main(monkeypatch, tmp_path):
    scenarios_path = tmp_path / "scenarios.yaml"
    scenarios_path.write_text("scenarios:\n  - name: alpha\n    turns: [{user: hi}]\n")
    trend_path = tmp_path / "trend.jsonl"

    def fake_run_scenario(scenario, cfg, llm, embedder, **kwargs):
        return {
            "name": scenario["name"],
            "description": "",
            "rubric_focus": "",
            "transcript": [],
            "check_failures": [],
        }

    monkeypatch.setattr(EVAL, "run_scenario", fake_run_scenario)
    monkeypatch.setattr(EVAL, "load_llm", lambda cfg: ScriptedLLM(chat_replies=[]))
    monkeypatch.setattr(EVAL, "load_embedder", lambda cfg: None)
    monkeypatch.setattr(
        EVAL.Config,
        "load",
        classmethod(lambda cls, path=None: _isolated_config(tmp_path)),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_conversation.py",
            "--scenarios",
            str(scenarios_path),
            "--no-judge",
            "--trend-file",
            str(trend_path),
        ],
    )
    rc = EVAL.main()
    assert rc == 0
    assert trend_path.exists()
    entry = json.loads(trend_path.read_text().strip())
    assert entry["scores"] == {"alpha": 10}  # no-judge default overall


def test_no_trend_file_flag_writes_nothing(monkeypatch, tmp_path):
    """Pure plumbing contract: omitting --trend-file changes nothing."""
    scenarios_path = tmp_path / "scenarios.yaml"
    scenarios_path.write_text("scenarios:\n  - name: alpha\n    turns: [{user: hi}]\n")

    def fake_run_scenario(scenario, cfg, llm, embedder, **kwargs):
        return {
            "name": scenario["name"],
            "description": "",
            "rubric_focus": "",
            "transcript": [],
            "check_failures": [],
        }

    monkeypatch.setattr(EVAL, "run_scenario", fake_run_scenario)
    monkeypatch.setattr(EVAL, "load_llm", lambda cfg: ScriptedLLM(chat_replies=[]))
    monkeypatch.setattr(EVAL, "load_embedder", lambda cfg: None)
    monkeypatch.setattr(
        EVAL.Config,
        "load",
        classmethod(lambda cls, path=None: _isolated_config(tmp_path)),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["eval_conversation.py", "--scenarios", str(scenarios_path), "--no-judge"],
    )
    before = set(tmp_path.iterdir())
    rc = EVAL.main()
    assert rc == 0
    after = set(tmp_path.iterdir())
    assert before == after
