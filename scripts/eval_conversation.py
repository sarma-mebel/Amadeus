#!/usr/bin/env python3
"""Conversation-quality eval harness for Amadeus / Kurisu.

Drives scripted multi-turn scenarios against a LIVE ChatEngine (real LLM +
real memory pipeline), then scores each transcript with an LLM judge on a
fixed rubric. This is the behavioral safety net the unit tests can't provide:
it catches recall going noisy, the persona drifting, repetition creeping in,
and format/character regressions that only show up against a real model.

This is NOT a pytest file — it makes live API calls, which the test suite
forbids. Run it by hand or in a nightly job.

    python scripts/eval_conversation.py --config config.local.yaml
    python scripts/eval_conversation.py --scenario greeting_loop_no_repeat -v
    python scripts/eval_conversation.py --judge-model gemini-flash-lite-latest
    python scripts/eval_conversation.py --packs        # guest character packs

Exit code is non-zero if any deterministic check fails or any scenario's
judge score falls below --min-score (default 6/10), so it can gate CI.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import re
import sys
import tempfile
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Repo root on sys.path so `python scripts/eval_conversation.py` works without
# an editable install (mirrors conftest.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from amadeus.config import CharacterSettings, Config, PersonaSettings  # noqa: E402
from amadeus.engine import ChatEngine  # noqa: E402
from amadeus.provider_loader import load_embedder, load_llm  # noqa: E402
from amadeus.web.streaming import parse_complete_reply  # noqa: E402

# --------------------------------------------------------------------------
# Deterministic check helpers (pure — unit-tested in tests/test_eval_checks.py)
# --------------------------------------------------------------------------

# Stage directions are narrated actions that leak into spoken audio — the
# register rules in every character pack ban them outright ("Words only").
# Two shapes exist in the wild:
#   *smiles gently* / *leans closer*   — asterisk-wrapped actions. The rules
#       ban asterisks entirely, so any letter-initial *...* snippet counts
#       (math like "3 * 4" has whitespace/digits after the asterisk and is
#       not matched).
#   (sets down her tray and tilts her head) / (ears droop sympathetically)
#       — parenthetical actions. Legitimate parentheses ("(3 x 4)",
#       "(e.g. the second law)", "(which is rare)") must NOT trip the check,
#       so the pattern requires the parenthetical to open with a narration
#       lead-in: an optional subject (she/he/her tail/his ears/...) followed
#       by a third-person body/action verb from a curated list.
# Base (uninflected) body/action verbs seen in stage directions. Two uses:
#   verb-initial parentheticals need the third-person -s/-es form ("(smiles
#   softly)"), so imperative/idiomatic parentheses like "(hold on)",
#   "(put simply)", "(rest assured)", "(pick one)" never match;
#   subject-led parentheticals ("(she smiles)", "(ears droop)") accept any
#   inflection because the narration subject already disambiguates.
_STAGE_DIRECTION_VERB_STEMS = (
    "lean|tilt|set|lower|raise|lift|cross|fold|clasp|smile|nod|sigh|giggle|"
    "laugh|chuckle|blush|shrug|wink|grin|frown|pout|smirk|bow|stare|glance|"
    "look|gaze|peer|blink|turn|tap|twirl|flip|tuck|adjust|straighten|fidget|"
    "swish|wag|flick|droop|perk|flatten|soften|narrow|widen|roll|avert|"
    "close|clutch|hug|squeeze|reach|step|walk|sit|stand|kneel|hop|bounce|"
    "clap|wave|point|gesture|strike|puff|huff|exhale|inhale|sip|place|put|"
    "rest|brush|sweep|pat|tug|shake|toss|hum|gasp|shift|settle|fiddle|hide|"
    "cover|press|pull|hold|drop|pick"
)
# Third-person singular forms for the verb-initial case ("smiles", "presses",
# "swishes"). Idioms that would otherwise false-positive are excluded by
# lookahead: "(looks like rain)", "(turns out it was fine)".
_STAGE_DIRECTION_VERBS_3SG = (
    rf"(?:{_STAGE_DIRECTION_VERB_STEMS})(?:s|es)(?!\s+(?:like|out)\b)"
)
_STAGE_DIRECTION_SUBJECT = r"(?:she|he|they|ears?|tail|eyes?|(?:her|his|their)\s+\w+)"

STAGE_DIRECTION_PATTERNS = [
    # (leans closer) / (smiles softly) — verb-initial, third-person form only
    rf"\({_STAGE_DIRECTION_VERBS_3SG}\b[^)]*\)",
    # (she tilts her head) / (ears droop sympathetically) / (her tail wags)
    rf"\({_STAGE_DIRECTION_SUBJECT}\s+"
    rf"(?:{_STAGE_DIRECTION_VERB_STEMS})(?:s|es|ing)?\b[^)]*\)",
    # *smiles gently* / *leans closer* — any letter-initial asterisk pair
    # (the register rules ban asterisks outright; math "3 * 4" has no
    # letter after the asterisk and is not matched)
    r"\*[a-z][^*\n]{0,80}\*",
    # a reply that OPENS with an asterisk is action narration even if unpaired
    r"^\s*\*",
]


def regex_violations(text: str, patterns) -> List[str]:
    """Which of ``patterns`` match ``text`` (case-insensitive, multiline).

    Returns one human-readable string per matching pattern, quoting the
    matched span so a failure report shows the offending prose directly.
    Pure function: no I/O, no engine state.
    """
    hits: List[str] = []
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            snippet = match.group(0)[:80]
            hits.append(f"regex {pattern!r} matched {snippet!r}")
    return hits


# --------------------------------------------------------------------------
# Character packs
# --------------------------------------------------------------------------


def load_pack_character(
    pack_id: str, packs_dir: Path | None = None
) -> CharacterSettings:
    """CharacterSettings for a guest pack, read straight from characters/packs.

    Loads the manifest the same way ``amadeus.characters`` does but points
    ``system_prompt_path`` directly at the pack's prompt file — no install
    copy, so the eval always exercises the committed pack prompt text.
    """
    packs_dir = packs_dir or (_REPO_ROOT / "characters" / "packs")
    pack_dir = packs_dir / pack_id
    manifest_path = pack_dir / "character.yaml"
    if not manifest_path.is_file():
        raise ValueError(
            f"Unknown character pack {pack_id!r}: {manifest_path} not found"
        )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    name = str(manifest.get("name") or "").strip()
    full_name = str(manifest.get("full_name") or name).strip()
    if not name:
        raise ValueError(f"Pack {pack_id!r} manifest is missing `name`")
    prompt_path = pack_dir / str(manifest.get("prompt_file") or "prompt.txt")
    if not prompt_path.is_file():
        raise ValueError(f"Pack {pack_id!r} prompt file not found: {prompt_path}")
    return CharacterSettings(
        name=name,
        full_name=full_name,
        system_prompt_path=str(prompt_path),
    )


def scenario_config(scenario: Dict[str, Any], config: Config) -> Config:
    """The per-scenario config: base config, plus the scenario's character.

    ``character: <pack_id>`` activates a guest pack for that scenario only
    (the engine reads ``config.character`` for the system prompt and
    presence identity). Absent = the config's own character — the built-in
    Kurisu for the committed baseline scenarios.
    """
    pack_id = scenario.get("character")
    if not pack_id:
        return config
    return dataclasses.replace(config, character=load_pack_character(str(pack_id)))


# --------------------------------------------------------------------------
# Scenario driving
# --------------------------------------------------------------------------


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "quota" in text or "rate limit" in text


def _generate_with_retry(llm, messages, attempts: int = 4):
    """Call llm.generate, sleeping through free-tier 429s instead of dying.

    Free-tier Gemini quotas are per-minute; an eval run bursts well past them.
    Backoff honors the API's suggested retry_delay when present in the message,
    else sleeps 20s * attempt. Non-quota errors re-raise immediately.
    """
    for attempt in range(1, attempts + 1):
        try:
            return llm.generate(messages)
        except Exception as exc:
            if not _is_rate_limit(exc) or attempt == attempts:
                raise
            delay = 20 * attempt
            match = re.search(r"seconds:\s*(\d+)", str(exc))
            if match:
                delay = max(int(match.group(1)) + 2, 10)
            print(f"    (rate-limited; retrying in {delay}s)", flush=True)
            time.sleep(delay)


def _reply_text(engine: ChatEngine, setup):
    """Run one turn and return (display_text, emotions, tagged_ok).

    ``emotions`` collects each sentence's emotion tag (the rich ``emotion_raw``
    when the model used one, else the basic six-value tag); ``tagged_ok`` is
    whether the reply parsed as tagged output at all — both feed the
    deterministic format-compliance checks.
    """
    if setup.refusal:
        return setup.refusal, [], False
    reply = _generate_with_retry(engine.llm, setup.context)
    emotions: List[str] = []
    tagged_ok = False
    # Tagged output packs "[emotion] English || Japanese" per line; the
    # English half is what drives display + memory. Fall back to raw text.
    if setup.use_tagged_output:
        try:
            sentences = parse_complete_reply(reply)
            display = " ".join(s.en for s in sentences if s.en).strip()
            if display:
                tagged_ok = True
                emotions = [(s.emotion_raw or s.emotion) for s in sentences if s.en]
                reply = display
        except Exception:
            pass
    return reply, emotions, tagged_ok


def _backdate_store(memory, days: float) -> None:
    """Shift every long-term record's timestamp back ``days``, so a scenario
    can simulate "weeks later" without waiting. Vector-store only: recall
    ranking and the recall age annotations both read record metadata
    timestamps; short-term context is handled by ``new_session`` instead."""
    store = getattr(memory, "vector_store", None)
    if store is None or not hasattr(store, "dump"):
        return
    delta = datetime.timedelta(days=days)
    for rec in store.dump():
        meta = dict(rec.metadata or {})
        stamped = meta.get("timestamp")
        try:
            when = datetime.datetime.fromisoformat(stamped) if stamped else None
        except (TypeError, ValueError):
            when = None
        if when is None:
            continue
        meta["timestamp"] = (when - delta).isoformat()
        store.update_metadata(rec.id, meta)
    if hasattr(store, "persist"):
        store.persist()


def _near_duplicate(a: str, b: str, threshold: float = 0.72) -> bool:
    a_n = re.sub(r"\s+", " ", a.lower()).strip()
    b_n = re.sub(r"\s+", " ", b.lower()).strip()
    if not a_n or not b_n:
        return False
    return SequenceMatcher(None, a_n, b_n).ratio() >= threshold


def _run_turn_checks(
    i: int,
    turn: Dict[str, Any],
    reply: str,
    replies: List[str],
    *,
    emotions: Optional[List[str]] = None,
    tagged_ok: bool = False,
    scenario: Optional[Dict[str, Any]] = None,
    prefix: str = "turn",
) -> List[str]:
    """Deterministic checks for one turn's reply against ``replies`` so far.

    Pure-ish (only reads ``turn``/``scenario``/``replies``, returns failure
    strings): shared by the normal per-turn path and the ``expect_greeting``
    path so both get identical forbid_substr/expect_recall/forbid_repeat/
    expect_tagged/expect_emotion_in semantics. ``prefix`` labels failures
    ("turn" vs "greeting") so a report can tell which reply tripped a check.
    """
    emotions = emotions or []
    failures: List[str] = []
    low = reply.lower()
    for forbidden in turn.get("forbid_substr", []):
        if forbidden.lower() in low:
            failures.append(f"{prefix} {i}: forbidden substring {forbidden!r} appeared")
    # forbid_regex: per-turn regex bans; forbid_stage_directions (scenario
    # level) additionally applies the built-in stage-direction patterns to
    # EVERY turn — the register-rule invariant for character packs.
    regex_bans = list(turn.get("forbid_regex", []))
    if scenario and scenario.get("forbid_stage_directions"):
        regex_bans.extend(STAGE_DIRECTION_PATTERNS)
    for hit in regex_violations(reply, regex_bans):
        failures.append(f"{prefix} {i}: {hit}")
    for wanted in turn.get("expect_recall", []):
        if wanted.lower() not in low:
            failures.append(f"{prefix} {i}: expected recall of {wanted!r} not surfaced")
    if turn.get("forbid_repeat"):
        for j, prior in enumerate(replies):
            if _near_duplicate(prior, reply):
                failures.append(f"{prefix} {i}: reply near-duplicates turn {j}")
                break
    if turn.get("expect_tagged") and not tagged_ok:
        failures.append(f"{prefix} {i}: reply did not follow the tagged format")
    expect_emotions = turn.get("expect_emotion_in", [])
    if expect_emotions:
        wanted_set = {e.lower() for e in expect_emotions}
        got = {e.lower() for e in emotions}
        if not (wanted_set & got):
            failures.append(
                f"{prefix} {i}: expected an emotion tag in {sorted(wanted_set)}, "
                f"got {sorted(got) or '[untagged]'}"
            )
    return failures


def run_scenario(
    scenario: Dict[str, Any],
    config: Config,
    llm,
    embedder,
    *,
    verbose: bool = False,
    turn_delay: float = 0.0,
) -> Dict[str, Any]:
    """Drive one scenario end to end. Returns a result dict with transcript
    and any deterministic check failures (judge scoring is a separate pass)."""
    # Per-scenario character: `character: <pack_id>` swaps in a guest pack's
    # prompt/identity for this scenario only. The explicit persist paths below
    # win over per_character_memory scoping, so isolation is unaffected.
    config = scenario_config(scenario, config)
    character_name = None
    if scenario.get("character"):
        character_name = config.character.name
    # Isolated memory per scenario so recall/summary state never leaks between
    # scenarios or into the user's real store.
    tmp = tempfile.mkdtemp(prefix="amadeus-eval-")

    def _build_engine() -> ChatEngine:
        return ChatEngine(
            config,
            llm,
            embedder=embedder,
            persist_path=str(Path(tmp) / "store.json"),
            embeddings_path=str(Path(tmp) / "vectors.json"),
            profile_path=str(Path(tmp) / "profile.json"),
        )

    engine = _build_engine()
    persona = PersonaSettings(**scenario.get("persona_overrides", {}))
    tagged = bool(getattr(config.web, "streaming_tagged_output", False))
    tagged_instructions = None
    if tagged:
        from amadeus.web.streaming import TAGGED_OUTPUT_INSTRUCTIONS

        tagged_instructions = TAGGED_OUTPUT_INSTRUCTIONS

    transcript: List[Dict[str, str]] = []
    failures: List[str] = []
    replies: List[str] = []

    for i, turn in enumerate(scenario["turns"]):
        if turn_delay and i:
            time.sleep(turn_delay)  # pace free-tier per-minute quotas
        shift_days = float(turn.get("backdate_store_days", 0) or 0)
        if shift_days:
            # Age the long-term store before this turn: finalize the session
            # the way the real farewell/quit paths do (this runs the final
            # fact-extraction pass short sessions depend on), then push all
            # record timestamps into the past. Pairs with new_session to
            # simulate the user returning weeks later.
            engine.memory.finalize_session(wait=True)
            engine.memory.flush()
            _backdate_store(engine.memory, shift_days)
        if turn.get("new_session"):
            # Simulate the user closing and reopening the app: finalize the
            # session exactly like the real farewell/quit paths (final fact
            # extraction + episode), flush pending background jobs so
            # long-term memory is durably written, then rebuild the engine
            # over the SAME store paths. With memory.scope_context_to_session
            # on, the new manager must start with a clean short-term
            # window/summary while long-term facts stay recallable.
            engine.memory.finalize_session(wait=True)
            engine.memory.flush()
            engine = _build_engine()
            if turn.get("expect_greeting"):
                # should_greet() (memory/manager.py) is true either when
                # short_term is empty (scope_context_to_session on) or when
                # the LAST message's timestamp is older than
                # session_gap_minutes (30min default). A same-second harness
                # rebuild is neither, so without scope_context_to_session an
                # expect_greeting turn would deterministically see
                # open_session_greeting() return None even though a real
                # user reopening the app after an actual gap would get a
                # greeting. Push the persisted short-term timestamps back
                # far enough to cross the gap — this simulates the real
                # elapsed time a fresh app launch implies, honestly, rather
                # than requiring every expect_greeting scenario's config to
                # carry scope_context_to_session.
                gap_minutes = float(
                    getattr(engine.config.memory, "session_gap_minutes", 30) or 30
                )
                shift = datetime.timedelta(minutes=gap_minutes + 5)
                for msg in engine.memory.short_term:
                    try:
                        when = datetime.datetime.fromisoformat(msg.timestamp)
                    except (TypeError, ValueError):
                        continue
                    msg.timestamp = (when - shift).isoformat()
        user_text = turn["user"]
        emotions: List[str] = []
        tagged_ok = False

        greeting_text: Optional[str] = None
        if turn.get("expect_greeting"):
            # Sequencing note (kept honest against the real app, which is why
            # this is spelled out): the actual client flow is POST
            # /api/greeting BEFORE the user types anything in the new
            # session — open_session_greeting() is a standalone unprompted
            # line, not a reply to this turn's `user` text. Only meaningful
            # right after new_session (should_greet() requires a session
            # gap/fresh short_term); a bare expect_greeting with no
            # new_session on the same turn will almost always see should_greet
            # return False (mid-conversation) and get None back, which is
            # itself a real assertion the checks below will fail loudly on
            # rather than silently skip.
            greeting_text = engine.open_session_greeting()
            checked_reply = greeting_text or ""
            transcript.append(
                {"user": "(session opened)", "kurisu": checked_reply, "emotions": []}
            )
            if verbose:
                print("    [{}] (greeting): {}\n".format(i, checked_reply))
            failures.extend(
                _run_turn_checks(i, turn, checked_reply, replies, prefix="greeting")
            )
            replies.append(checked_reply)
            # The turn's own user text is then delivered as an ordinary
            # first message of the new session — greeting and user turn are
            # sequenced, not merged into one reply, exactly like the real
            # open_session_greeting() (unprompted) followed by the user's
            # actual first /api/chat call.

        if turn.get("say_goodbye"):
            # Mirrors POST /api/farewell's reason="goodbye" contract exactly
            # (see amadeus/web/routes_meta.py farewell_endpoint): validate
            # server-side via close_session_farewell, and only on a produced
            # line commit the user half + assistant half via complete_turn
            # (never double-committed), then finalize the session. Unlike
            # the endpoint's finalize_session(wait=False) (fire-and-forget
            # for a live request), the harness waits so the eval stays
            # deterministic. If no line comes back (cooldown, disabled,
            # provider miss) this falls back to an ordinary turn — the same
            # "not_goodbye" fallback the real endpoint signals to its caller
            # — so the scenario still gets a reply to check instead of a
            # silent gap.
            line = engine.close_session_farewell(user_text)
            if line:
                engine.memory.add_message("user", user_text)
                engine.complete_turn(user_text, line)
                engine.memory.finalize_session(wait=True)
                reply = line
            else:
                setup = engine.prepare_turn(
                    user_text,
                    persona,
                    tagged_output=tagged,
                    tagged_instructions=tagged_instructions,
                )
                reply, emotions, tagged_ok = _reply_text(engine, setup)
                engine.complete_turn(user_text, reply)
        else:
            setup = engine.prepare_turn(
                user_text,
                persona,
                tagged_output=tagged,
                tagged_instructions=tagged_instructions,
            )
            reply, emotions, tagged_ok = _reply_text(engine, setup)
            engine.complete_turn(user_text, reply)
        if turn.get("simulate_interrupt"):
            # Mirror the cancelled-turn commit (_finalize_streamed_turn): a
            # voice barge-in leaves her reply cut mid-thought in memory with
            # a trailing " [interrupted]" marker. Rewrite the committed
            # message the same way so later turns see exactly what a real
            # interruption leaves behind.
            words = reply.split()
            cut = " ".join(words[: max(3, len(words) // 3)])
            reply = f"{cut} [interrupted]"
            for msg in reversed(engine.memory.short_term):
                if msg.role == "assistant":
                    msg.content = reply
                    break
        transcript.append({"user": user_text, "kurisu": reply, "emotions": emotions})
        if verbose:
            print(f"    [{i}] user: {user_text}")
            tags = f"  [tags: {', '.join(emotions)}]" if emotions else ""
            print(f"    [{i}] kurisu: {reply}{tags}\n")

        if turn.get("expect_greeting"):
            # The turn's checks (forbid_substr/expect_recall/forbid_repeat/
            # expect_tagged/expect_emotion_in) were already applied to the
            # GREETING text above — that's the reply under check for this
            # turn, per expect_greeting's contract. The ensuing reply to the
            # turn's own `user` text is still generated (for transcript
            # continuity/realism) but is not separately re-checked, so a
            # scenario doesn't have to satisfy e.g. expect_recall twice.
            pass
        else:
            failures.extend(
                _run_turn_checks(
                    i,
                    turn,
                    reply,
                    replies,
                    emotions=emotions,
                    tagged_ok=tagged_ok,
                    scenario=scenario,
                )
            )
        replies.append(reply)

    result = {
        "name": scenario["name"],
        "description": scenario.get("description", ""),
        "rubric_focus": scenario.get("rubric_focus", ""),
        "transcript": transcript,
        "check_failures": failures,
    }
    if character_name:
        result["character_name"] = character_name
    return result


# --------------------------------------------------------------------------
# LLM judge
# --------------------------------------------------------------------------

JUDGE_SYSTEM = """You are a strict evaluator of a fan-made Kurisu Makise
(Steins;Gate) voice assistant. Kurisu is a brilliant, tsundere neuroscientist:
gruff and easily flustered on the surface, genuinely caring underneath,
scientifically precise. Score the transcript on this rubric, each 0-10:

- recall: relevant earlier facts are used naturally; nothing is forgotten when
  it matters, nothing is dragged in when irrelevant or recited verbatim.
- non_repetition: replies do NOT re-narrate the same facts/summary or reuse the
  same opening structure across turns.
- in_character: stays Kurisu (tsundere, scientific, no generic "as an AI
  language model" disclaimers, no fourth-wall breaks).
- coherence: replies actually respond to the user and flow across turns.

Return ONLY a JSON object:
{"recall": int, "non_repetition": int, "in_character": int, "coherence": int,
 "overall": int, "notes": "one sentence on the biggest issue, if any"}
The `overall` is your holistic 0-10 judgment, not necessarily the average."""

# Character-pack scenarios are judged against the scenario's own register
# description (rubric_focus) rather than the Kurisu persona.
CHARACTER_JUDGE_SYSTEM = """You are a strict evaluator of a fan-made
character companion assistant currently playing {name}. The scenario intent
and focus below describe the character's register. Score the transcript on
this rubric, each 0-10:

- recall: relevant earlier facts are used naturally, nothing shoehorned in.
- non_repetition: replies do NOT reuse the same structure or menu across
  turns (e.g. the same "listen or ideas?" ending twice).
- in_character: stays {name} per the scenario focus. Hard fails: stage
  directions or narrated actions ("(smiles)", "*leans closer*"), generic
  therapy-speak ("I'm so sorry to hear that", "that sounds really heavy"),
  assistant/AI disclaimers, fourth-wall breaks.
- coherence: replies actually respond to the user and flow across turns.

Return ONLY a JSON object:
{{"recall": int, "non_repetition": int, "in_character": int, "coherence": int,
 "overall": int, "notes": "one sentence on the biggest issue, if any"}}
The `overall` is your holistic 0-10 judgment, not necessarily the average."""


def judge_transcript(
    scenario_result: Dict[str, Any], judge_llm, samples: int = 1
) -> Dict[str, Any]:
    """Score the transcript; with samples>1, return the median-overall verdict.

    Judge scores wobble a point or two between identical runs (observed 4<->10
    on one scenario), so a single sample can mislead in both directions. The
    median sample's full verdict is returned, with every sample's overall in
    ``all_overalls`` for transparency.
    """
    from amadeus.providers.llm.base import ChatMessage

    character_name = scenario_result.get("character_name")
    speaker = (character_name or "Kurisu").upper()
    system = (
        CHARACTER_JUDGE_SYSTEM.format(name=character_name)
        if character_name
        else JUDGE_SYSTEM
    )
    lines = [f"Scenario intent: {scenario_result['description']}"]
    if scenario_result["rubric_focus"]:
        lines.append(f"Extra focus: {scenario_result['rubric_focus']}")
    lines.append("\nTranscript:")
    for t in scenario_result["transcript"]:
        lines.append(f"USER: {t['user']}")
        lines.append(f"{speaker}: {t['kurisu']}")
    prompt = "\n".join(lines)
    messages = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=prompt),
    ]

    verdicts = []
    for _ in range(max(1, samples)):
        raw = _generate_with_retry(judge_llm, messages)
        verdicts.append(_parse_judge_json(raw))
    verdicts.sort(key=lambda v: v.get("overall", 0))
    median = verdicts[len(verdicts) // 2]
    if len(verdicts) > 1:
        median = dict(median)
        median["all_overalls"] = [v.get("overall", 0) for v in verdicts]
    return median


def _parse_judge_json(raw: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"overall": 0, "notes": f"unparseable judge output: {raw[:120]}"}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"overall": 0, "notes": f"invalid judge JSON: {raw[:120]}"}
    return data


# --------------------------------------------------------------------------
# Trend file (--trend-file / the nightly self-eval scheduler job)
# --------------------------------------------------------------------------


def build_trend_entry(
    results: List[Dict[str, Any]],
    *,
    judge_model: Optional[str],
    config: Config,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """One JSON-line trend record from a completed eval run.

    Pure function (no I/O) so it's directly unit-testable: given the
    ``results`` list ``main()``/a scheduler job already produced, returns the
    dict that gets appended (as one JSON line) to the trend file. Deliberately
    narrow on config fingerprinting — provider/model NAMES only (never API
    keys or other secrets), matching the "config fingerprint" requirement.
    """
    scores = {r["name"]: r.get("judge", {}).get("overall", 10) for r in results}
    return {
        "timestamp": timestamp or dt_now_utc_iso(),
        "scores": scores,
        "judge_model": judge_model or getattr(config, "llm_model", None),
        "config_fingerprint": {
            "llm_provider": getattr(config.providers, "llm", None),
            "llm_model": getattr(config, "llm_model", None),
            "embedding_provider": getattr(config.providers, "embedding", None),
        },
    }


def dt_now_utc_iso() -> str:
    """UTC timestamp, isolated to one line so tests can monkeypatch it."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def append_trend_line(path: Path, entry: Dict[str, Any]) -> None:
    """Append one JSON line to the trend file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.local.yaml")
    parser.add_argument(
        "--scenarios",
        default=str(Path(__file__).with_name("eval_scenarios.yaml")),
    )
    parser.add_argument(
        "--scenario", action="append", help="run only these scenario name(s)"
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="comma-separated scenario names to run (alternative to repeated "
        "--scenario; e.g. --subset greeting_loop_no_repeat,fact_recall_across_topic)",
    )
    parser.add_argument(
        "--packs",
        action="store_true",
        help="shortcut for --scenarios scripts/eval_pack_scenarios.yaml (guest "
        "character packs; run WITHOUT --compare — packs have no baseline)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="override model for the judge (defaults to the summary/chat model)",
    )
    parser.add_argument("--min-score", type=float, default=6.0)
    parser.add_argument("--no-judge", action="store_true", help="checks only")
    parser.add_argument("--json", action="store_true", help="emit JSON report")
    parser.add_argument(
        "--baseline",
        default=str(Path(__file__).with_name("eval_baseline.json")),
        help="path to the recorded baseline (for --record-baseline / --compare)",
    )
    parser.add_argument(
        "--record-baseline",
        action="store_true",
        help="write this run's scores+transcripts to --baseline and exit 0",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="flag scenarios whose score dropped >--tolerance below baseline, "
        "or whose deterministic checks newly fail",
    )
    parser.add_argument(
        "--allow-missing-baseline",
        action="store_true",
        help="with --compare: tolerate scenarios that have no baseline entry "
        "(they run uncompared) instead of failing before the run starts",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="allowed judge-score drop vs baseline before it counts as a "
        "regression (absorbs LLM judge variance)",
    )
    parser.add_argument(
        "--turn-delay",
        type=float,
        default=0.0,
        help="seconds to sleep between turns (pace free-tier per-minute quotas)",
    )
    parser.add_argument(
        "--judge-samples",
        type=int,
        default=3,
        help="judge each transcript N times and keep the median-overall "
        "verdict (damps judge variance; 1 = single sample)",
    )
    parser.add_argument(
        "--trend-file",
        default=None,
        help="append one JSON line per run to this path: UTC timestamp, "
        "per-scenario overall judge scores, judge model, and a config "
        "fingerprint (provider/model NAMES only, never secrets). Pure "
        "plumbing — omitting the flag changes nothing.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.packs:
        args.scenarios = str(Path(__file__).with_name("eval_pack_scenarios.yaml"))

    config = Config.load(Path(args.config))
    llm = load_llm(config)
    embedder = load_embedder(config)

    if getattr(llm, "name", "") == "dummy":
        print(
            "ERROR: LLM resolved to the dummy provider — the eval needs a real "
            "LLM (set providers.llm + an API key). Check /status logic.",
            file=sys.stderr,
        )
        return 2

    # Judge: reuse the chat LLM by default, or load a (cheaper) override.
    judge_llm = llm
    if args.judge_model and not args.no_judge:
        judge_config = dataclasses.replace(config, llm_model=args.judge_model)
        judge_llm = load_llm(judge_config)

    spec = yaml.safe_load(Path(args.scenarios).read_text())
    scenarios = spec["scenarios"]
    # --subset and --scenario both narrow the run to a name allowlist; a run
    # can pass either (or both — the union is used) with no behavior change
    # when neither is set.
    wanted_names: set = set(args.scenario or [])
    if args.subset:
        wanted_names |= {
            name.strip() for name in args.subset.split(",") if name.strip()
        }
    if wanted_names:
        scenarios = [s for s in scenarios if s["name"] in wanted_names]
        if not scenarios:
            print(f"No scenarios matched {sorted(wanted_names)}", file=sys.stderr)
            return 2

    baseline: Dict[str, Dict[str, Any]] = {}
    if args.compare:
        baseline_path = Path(args.baseline)
        if not baseline_path.exists():
            print(
                f"ERROR: --compare needs a baseline at {baseline_path}; run "
                "--record-baseline first.",
                file=sys.stderr,
            )
            return 2
        baseline = {
            entry["name"]: entry
            for entry in json.loads(baseline_path.read_text())["scenarios"]
        }
        # A scenario absent from the baseline would otherwise skip comparison
        # and silently pass — fail before spending any LLM calls instead.
        missing = sorted(s["name"] for s in scenarios if s["name"] not in baseline)
        if missing and not args.allow_missing_baseline and not args.record_baseline:
            print(
                "ERROR: --compare has no baseline entry for: "
                + ", ".join(missing)
                + f". Re-record the baseline with `make eval-baseline` (writes "
                f"{baseline_path}), or pass --allow-missing-baseline to run "
                "them uncompared.",
                file=sys.stderr,
            )
            return 2
        if missing:
            print(
                f"NOTE: no baseline entry for {', '.join(missing)}; running "
                "uncompared (--allow-missing-baseline)."
            )

    results: List[Dict[str, Any]] = []
    any_failed = False
    for scenario in scenarios:
        print(f"▶ {scenario['name']} …", flush=True)
        try:
            result = run_scenario(
                scenario,
                config,
                llm,
                embedder,
                verbose=args.verbose,
                turn_delay=args.turn_delay,
            )
            if not args.no_judge:
                result["judge"] = judge_transcript(
                    result, judge_llm, samples=args.judge_samples
                )
        except Exception as exc:
            # One scenario dying (exhausted quota retries, provider hiccup)
            # shouldn't discard the rest of the run.
            result = {
                "name": scenario["name"],
                "description": scenario.get("description", ""),
                "rubric_focus": scenario.get("rubric_focus", ""),
                "transcript": [],
                "check_failures": [f"scenario crashed: {exc}"],
            }
        results.append(result)

        checks_ok = not result["check_failures"]
        overall = result.get("judge", {}).get("overall", 10)
        score_ok = overall >= args.min_score
        ok = checks_ok and score_ok

        # Regression vs baseline: a score that fell more than the tolerance, or
        # a deterministic check that used to pass and now fails.
        regressed = False
        regression_notes: List[str] = []
        if args.compare and scenario["name"] in baseline:
            base = baseline[scenario["name"]]
            base_overall = base.get("judge", {}).get("overall", 10)
            if overall < base_overall - args.tolerance:
                regressed = True
                regression_notes.append(
                    f"judge {base_overall}->{overall} (>{args.tolerance} drop)"
                )
            if checks_ok is False and not base.get("check_failures"):
                regressed = True
                regression_notes.append("checks newly failing")

        result["regressed"] = regressed
        any_failed = any_failed or not ok or regressed

        status = "PASS" if (ok and not regressed) else "FAIL"
        judged = "" if args.no_judge else f" | judge overall={overall}/10"
        if args.compare and scenario["name"] in baseline:
            judged += f" (baseline {baseline[scenario['name']].get('judge', {}).get('overall', '?')})"
        print(f"  {status}{judged}")
        for f in result["check_failures"]:
            print(f"    ✗ check: {f}")
        notes = result.get("judge", {}).get("notes")
        if notes and not score_ok:
            print(f"    ✗ judge: {notes}")
        for rn in regression_notes:
            print(f"    ↓ regression: {rn}")

    if args.trend_file:
        entry = build_trend_entry(results, judge_model=args.judge_model, config=config)
        append_trend_line(Path(args.trend_file), entry)
        print(f"Trend line appended to {args.trend_file}")

    if args.record_baseline:
        Path(args.baseline).write_text(json.dumps({"scenarios": results}, indent=2))
        broken = [r["name"] for r in results if r["check_failures"]]
        low = [
            r["name"]
            for r in results
            if r.get("judge", {}).get("overall", 10) < args.min_score
        ]
        print(f"\nBaseline written to {args.baseline} ({len(results)} scenarios).")
        if broken:
            print(
                f"  WARNING: recorded a baseline with failing checks: {broken}. "
                "Fix the underlying behavior before trusting --compare."
            )
        if low:
            print(f"  WARNING: baseline scenarios below --min-score: {low}.")
        return 0

    if args.json:
        print(json.dumps(results, indent=2))

    passed = sum(
        1
        for r in results
        if not r["check_failures"]
        and r.get("judge", {}).get("overall", 10) >= args.min_score
        and not r.get("regressed")
    )
    print(
        f"\n{'FAILED' if any_failed else 'PASSED'}: {passed}/{len(results)} scenarios"
    )
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
