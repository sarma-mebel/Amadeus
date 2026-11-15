# Amadeus developer tasks. Uses the active Python (a venv/conda env with the
# project installed); override with `make PYTHON=/path/to/python <target>`.
PYTHON ?= python
CONFIG ?= config.local.yaml

.PHONY: help test lint fmt eval eval-baseline eval-quick eval-smoke eval-packs eval-trend soak

help:
	@echo "Targets:"
	@echo "  test           Run the backend test suite (pytest -q)"
	@echo "  lint           ruff check on amadeus/"
	@echo "  fmt            black + ruff --fix on amadeus/"
	@echo "  eval           Conversation-quality eval vs the recorded baseline"
	@echo "  eval-baseline  Record the current run AS the baseline (do this at a known-good state)"
	@echo "  eval-quick     Deterministic checks only, no LLM judge (cheap)"
	@echo "  eval-smoke     Fast 2-scenario deterministic smoke test (mid-tier, no judge)"
	@echo "  eval-packs     Guest character-pack register scenarios (no baseline compare)"
	@echo "  eval-trend     Run the self_eval.scenarios subset and append to TREND_FILE"
	@echo "  soak           Long-session soak test against a RUNNING server (SOAK_URL, SOAK_TURNS)"
	@echo ""
	@echo "The eval targets make LIVE LLM calls and need CONFIG=$(CONFIG) with a real"
	@echo "provider + API key. Override: make CONFIG=config.dev-preview.yaml eval"

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check amadeus/

fmt:
	black amadeus/ && ruff check --fix amadeus/

# Record the current behavior as the golden baseline. Run this after a known-good
# state (e.g. right after a release you're happy with), and commit the baseline
# so `make eval` has something to diff against.
eval-baseline:
	$(PYTHON) scripts/eval_conversation.py --config $(CONFIG) --record-baseline

# Pre-release gate: run the scenarios and fail (non-zero exit) if any scenario's
# judge score dropped more than the tolerance below the baseline, or a
# deterministic check that used to pass now fails.
eval:
	$(PYTHON) scripts/eval_conversation.py --config $(CONFIG) --compare

# Cheap sanity pass: deterministic checks only (forbidden phrases, expected
# recall, near-duplicate replies). No judge, so no per-transcript scoring cost.
eval-quick:
	$(PYTHON) scripts/eval_conversation.py --config $(CONFIG) --no-judge

# Cheap middle tier between unit tests and the full eval gate (OpenHands-style
# smoke check): two known-stable scenarios (10/10/10 judge, zero check
# failures in the committed baseline), deterministic checks only. Fast enough
# to run on every PR; avoids fact_recall_across_topic (known flaky).
eval-smoke:
	$(PYTHON) scripts/eval_conversation.py --config $(CONFIG) --no-judge --scenario memory_across_sessions --scenario interruption_grace

# Guest character packs (frieren/holo/marin/megumin/rem/rin_tohsaka): register
# regressions — stage directions, therapy-speak, looping greetings. Absolute
# gate (deterministic checks + --min-score), NOT compared against the Kurisu
# baseline; keep it out of eval-baseline/eval.
eval-packs:
	$(PYTHON) scripts/eval_conversation.py --config $(CONFIG) --packs --turn-delay 10

# Long-session degradation check (latency drift, RSS/VRAM growth). Needs a
# server ALREADY RUNNING at SOAK_URL; see docs/soak-testing.md.
SOAK_URL ?= http://127.0.0.1:8000
SOAK_TURNS ?= 200
soak:
	$(PYTHON) scripts/soak_test.py --url $(SOAK_URL) --turns $(SOAK_TURNS)

# Manual/CI-runnable equivalent of the nightly self-eval scheduler job
# (config.self_eval, default off, amadeus/self_eval.py): runs a small
# scenario subset and appends one JSON line to TREND_FILE (timestamp,
# per-scenario judge scores, judge model, config fingerprint — see
# docs/conversation-eval.md). Deterministic checks only (--no-judge) is NOT
# used here on purpose: the trend line needs the judge's overall score.
TREND_FILE ?= self_eval_trend.jsonl
TREND_SCENARIOS ?= greeting_loop_no_repeat,in_character_consistency,interruption_grace
eval-trend:
	$(PYTHON) scripts/eval_conversation.py --config $(CONFIG) --subset $(TREND_SCENARIOS) --trend-file $(TREND_FILE)
