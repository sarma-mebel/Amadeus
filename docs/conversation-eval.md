# Conversation-quality eval

Amadeus's value is subjective conversational quality, and the unit tests mock the
LLM so they cannot catch behavioral regressions: recall going noisy, the persona
drifting, repetition creeping in, emotion tags being misused. The conversation
eval is the behavioral safety net for exactly those. It drives scripted
multi-turn scenarios against a live `ChatEngine` (real LLM, real memory
pipeline) and scores each transcript.

It is not part of `pytest`. The test suite must never make live API calls; this
harness always does. Run it by hand before a release.

For process health over long sessions (latency drift, RSS and VRAM growth) the
companion harness is the soak test, documented in docs/soak-testing.md.

## What it does

For each scenario in `scripts/eval_scenarios.yaml`:

1. Spins up a `ChatEngine` with an isolated temp memory store (nothing leaks
   into your real store or between scenarios).
2. Feeds the scripted `user` turns and collects Kurisu's replies.
3. Runs cheap deterministic checks with no LLM cost:
   - `forbid_substr` — phrases the reply must not contain (e.g. "as an AI
     language model", "sorry to hear").
   - `forbid_regex` — regex patterns the reply must not match
     (case-insensitive, multiline). Use this where substrings are too blunt,
     like catching parenthetical stage directions without flagging normal
     parentheses.
   - `expect_recall` — substrings that should surface later (memory working).
   - `forbid_repeat` — the reply must not near-duplicate an earlier one
     (`SequenceMatcher` ratio >= 0.72). This is the check that catches the
     "stuck greeting loop" repetition bug directly.
   - `expect_tagged` — the reply must parse as tagged bilingual output.
   - `expect_emotion_in` — at least one sentence must carry one of the listed
     emotion tags (basic six or rich tags like `embarrassed`/`excited`),
     verifying the emotion grammar tracks content.
   - `new_session` — finalize the session the way the real farewell/quit
     paths do (final fact-extraction pass included), flush memory jobs, and
     rebuild the engine over the same store before this turn, simulating a
     fresh conversation; used to test cross-session recall and
     `scope_context_to_session` behavior.
   - `backdate_store_days` — shift every long-term record's timestamp N days
     into the past before this turn (finalizes and flushes first). Pairs with
     `new_session` to simulate the user coming back weeks later: recalled
     facts carry humanized age annotations ("about 3 weeks ago") in the
     prompt, and this is how a scenario asserts she phrases old memories as
     old (`temporal_callback`).
   - `say_goodbye` — route this turn through the farewell path instead of a
     normal chat turn: mirrors `POST /api/farewell`'s `reason="goodbye"`
     contract (`amadeus/web/routes_meta.py::farewell_endpoint`) by calling
     `ChatEngine.close_session_farewell`, then committing the user+assistant
     halves via `complete_turn` and finalizing the session — only once a
     line was actually produced, exactly like the real endpoint. The
     farewell line becomes the reply under check for this turn (checks run
     against it). If no line comes back (cooldown/disabled/no provider),
     the harness falls back to an ordinary chat turn, mirroring the real
     endpoint's `not_goodbye` signal, so the scenario still gets something
     to check instead of silently skipping.
   - `expect_greeting` — only meaningful paired with `new_session: true` on
     the *same* turn. After the session-boundary rebuild, calls
     `ChatEngine.open_session_greeting()` and treats its output as the reply
     under check for this turn (`forbid_substr`/`expect_recall`/
     `forbid_repeat`/`expect_tagged`/`expect_emotion_in` all apply to the
     greeting text, not to a reply to the turn's `user` message) — this
     matches the real app's sequencing, where the greeting is a standalone
     unprompted line delivered before the user types anything (`POST
     /api/greeting`, fired ahead of the user's first `/api/chat` call in a
     fresh session), not a response to what the user said. The turn's own
     `user` text is still delivered afterward as an ordinary first message
     of the new session (so later turns in the scenario see normal
     conversational continuity), but it is not separately re-checked — a
     scenario that wants both the greeting AND the first real reply checked
     needs two turns, greeting on one (`expect_greeting` + `new_session`),
     content checks on the next. Also pushes the rebuilt engine's
     short-term message timestamps back past `session_gap_minutes` so
     `should_greet()` (`memory/manager.py`) actually returns true on a
     same-second harness rebuild — without this a `new_session` this
     instant would never look like a real gap to the greeting gate unless
     `memory.scope_context_to_session` were also on.
4. Hands the full transcript to an LLM judge that scores it 0-10 on recall,
   non-repetition, in-character, and coherence. By default each transcript is
   judged three times and the median-overall verdict kept (`--judge-samples`),
   because single judge scores swing a point or two between identical runs.

## Workflow

The scores from an LLM judge wobble a point or two run to run, so an absolute
threshold alone is a blunt instrument. The intended workflow is baseline +
diff: record a known-good run, then compare future runs against it and only fail
on a meaningful drop.

```bash
# 1. At a state you're happy with (e.g. just after a good release), record the
#    baseline and commit it.
make CONFIG=config.local.yaml eval-baseline
git add scripts/eval_baseline.json && git commit -m "Refresh eval baseline"

# 2. Before each release, compare. Exits non-zero if any scenario's judge score
#    fell more than the tolerance (default 2) below baseline, or a deterministic
#    check that used to pass now fails.
make eval

# Cheap sanity pass with no judge cost (deterministic checks only):
make eval-quick
```

Direct invocation exposes more knobs:

```bash
python scripts/eval_conversation.py --config config.local.yaml \
  --scenario greeting_loop_no_repeat -v         # one scenario, print transcripts
python scripts/eval_conversation.py --compare --tolerance 1.5
python scripts/eval_conversation.py --judge-model gemini-flash-lite-latest
```

The harness refuses to run on the dummy provider (it would score nothing real),
so `CONFIG` must point at a config with a working `providers.llm` + API key.

## Adding scenarios

Append to `scripts/eval_scenarios.yaml`. A scenario is a name, a description
(shown to the judge as intent), an optional `rubric_focus` (extra judge
guidance), and a list of `turns`. Use deterministic checks for hard invariants
and the judge for subjective quality. After adding scenarios, re-record the
baseline so `--compare` has scores for them.

The committed `eval_baseline.json` covers every scenario in
`eval_scenarios.yaml` (all 17, including `next_time_thread_pickup` and
`memory_abstention`, since the 2026-07-31 re-record).

A compared run now enforces that coverage: if any scenario in the run has no
baseline entry, `--compare` exits non-zero before making a single LLM call,
names the missing scenarios, and tells you to re-record with
`make eval-baseline`. This closes the old silent gap where a freshly added
scenario simply skipped comparison and passed. To run a new scenario
uncompared anyway (for example while iterating on it before it deserves a
full re-record), pass `--allow-missing-baseline`; it prints a note and runs
the scenario with checks and judge only. `--subset` and `--scenario` runs are
unaffected as long as every selected scenario has a baseline entry, and
`--record-baseline` runs skip the gate by design.

## Trend tracking (`--subset` / `--trend-file`)

Two flags, independent of everything above, for tracking scores over time
rather than one-shot baseline comparison:

- `--subset name1,name2` — run only the named scenarios (a comma-separated
  alternative to repeated `--scenario`; a run may pass either or both, the
  union of both is used).
- `--trend-file path` — append one JSON line per run to `path`: UTC
  timestamp, each run scenario's judge `overall` score, the judge model, and
  a config fingerprint (`llm_provider`/`llm_model`/`embedding_provider`
  NAMES only — never API keys or other secrets).

```bash
python scripts/eval_conversation.py --config config.local.yaml \
  --subset greeting_loop_no_repeat,in_character_consistency \
  --trend-file self_eval_trend.jsonl
```

Pure plumbing: omitting both flags changes nothing about a normal run.
`make eval-trend` wraps this (see the Makefile) for a quick manual/CI run
against the same scenario subset the nightly job below uses by default.

## Nightly self-eval (`config.self_eval`, default OFF)

`amadeus/self_eval.py` + a `BackgroundScheduler` job
(`amadeus/web/scheduler.py::_self_eval_job`) run this same harness
unattended, once per day, so a quality regression shows up as a trend-file
entry instead of only surfacing the next time someone happens to run
`make eval` by hand. Makes live LLM calls (chat + judge) — costs money/quota,
which is why it defaults off.

Config (`self_eval:` section):

- `enabled` (default `false`)
- `hour` (default `4`) — local hour after which the day's run becomes due;
  same once-per-day-after-hour predicate as the daily brief/diary/reach-out
  jobs (checks the trend file's last entry date, not a separate marker).
- `scenarios` (default: `greeting_loop_no_repeat`, `in_character_consistency`,
  `interruption_grace` — a small, cheap subset, NOT the full scenario file)
- `max_calls_per_day` (default `10`) — hard ceiling on scenario runs per
  day, independent of how long `scenarios` is.
- `trend_path` (default `self_eval_trend.jsonl` under the platform user-data
  dir — same `amadeus/paths.py` resolution as the memory/brief/diary files)
- `regression_tolerance` (default `2.0`, same magnitude as `--tolerance`)

Each run scores against an **isolated tmp memory store**, exactly like every
scenario in a manual run — never the live user's memory. After the run,
`amadeus/self_eval.py::check_regression` compares each scenario's latest
score to the rolling median of its last 7 trend-file entries; a drop past
`regression_tolerance` sets a `self_eval.regression` flag (and
`self_eval.regressions` detail) surfaced on `GET /api/status` under the
`self_eval` key, alongside `self_eval.regressions` incrementing in the
telemetry counters. The Settings -> Diagnostics panel shows the last run
time, per-scenario scores, and highlights a regressed scenario in amber —
tolerant of older backends (the field is simply absent).

The app lifespan starts the background scheduler when `self_eval.enabled` is
set, independently of `daily_brief`/`diary`/`reach_out` (any one of the four
being on is enough to start it).

## Tuning memory against the eval

The recall-quality knobs (`memory.recall_summary_dedupe`,
`memory.episode_supersede_threshold`) were set by reasoning, not measured. The
`fact_recall_across_topic` and `greeting_loop_no_repeat` scenarios are the ones
that exercise them against a real model and embedder; use them to tune the
thresholds if recall feels too aggressive or too leaky.

## Guest character packs

The six guest packs (frieren, holo, marin, megumin, rem, rin_tohsaka) have
their own scenarios file, `scripts/eval_pack_scenarios.yaml`, one scenario per
pack. Each one exercises the hard register rules in
`characters/packs/<id>/prompt.txt`: no stage directions on any turn, no
therapy-speak on vulnerable turns, and no looping replies when the user greets
three times with no content. The stage-direction bug lived for weeks precisely
because the eval only exercised Kurisu, so this gate exists to catch pack
register regressions automatically.

```bash
make eval-packs
# or directly:
python scripts/eval_conversation.py --config config.local.yaml --packs --turn-delay 10
```

Pack scenarios are an absolute gate: deterministic checks must pass and the
judge must clear `--min-score`. They are deliberately not in
`scripts/eval_scenarios.yaml` and have no entries in the committed baseline, so
`make eval` and `make eval-baseline` are unaffected. Do not add them to the
baseline compare; the deterministic register checks are the contract, and the
judge floor absorbs the rest.

Scenario support behind this:

- `character: <pack_id>` (scenario level) activates a guest pack for that
  scenario only. The harness reads `characters/packs/<pack_id>/character.yaml`
  and points `config.character` straight at the pack's committed `prompt.txt`,
  with no install-copy step, so the eval always tests the text in the repo.
  Scenarios without the field keep the config's own character (the built-in
  Kurisu for the baseline scenarios), so existing behavior is unchanged.
- `forbid_stage_directions: true` (scenario level) applies the harness's
  built-in stage-direction regexes (`STAGE_DIRECTION_PATTERNS` in
  `scripts/eval_conversation.py`) to every turn's English display text. The
  patterns catch asterisk actions (`*smiles gently*`) and verb-initial
  parenthetical actions (`(sets down her tray)`, `(ears droop
  sympathetically)`) while leaving legitimate parentheses like `(3 x 4)` or
  `(e.g. the second law)` alone. The pattern logic is unit-tested in
  `tests/test_eval_checks.py` with no live calls.
- `forbid_regex` (turn level) is the general form for custom patterns.

The judge for a character scenario swaps the Kurisu rubric for a generic
character rubric driven by the scenario's `rubric_focus`, and hard-penalizes
stage directions and therapy-speak on the in_character axis.
