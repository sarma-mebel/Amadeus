#!/usr/bin/env python3
"""Measure whether calibrated recall's confidence actually tracks truth.

Calibrated recall (memory/text_utils.py) hedges a recalled fact based on
retrieval count, age, and a keyword volatility class. Those are plausible
proxies. This script checks whether they are *real* ones, against the only
ground truth available: the store owner saying whether each fact still
holds.

The point is falsifiability. The repo has twice shipped a heuristic that
looked sound and measured degenerate — the absolute vocal-emotion
thresholds (16/17 clips read "bright") and the 0.35 recall floor that
filtered nothing. This is the same measurement applied before, rather than
after, trusting the numbers.

Three phases, resumable:

    # 1. sample facts from the live store into a labelling file
    python scripts/calibrate_recall.py sample --config config.local.yaml

    # 2. label them (edits calibration_labels.json in place; re-runnable)
    python scripts/calibrate_recall.py label

    # 3. score the heuristic against the labels
    python scripts/calibrate_recall.py report

ACCEPTANCE BAR (from the design): facts labelled outdated/wrong must skew
measurably below facts labelled true — concretely, median effective
confidence for "true" at least 0.15 above the median for "outdated". If the
bands do not separate, the mechanical heuristic is noise on this store and
should be reconsidered before any LLM-supplied confidence term is added on
top of it: a second uncalibrated signal stacked on a first is how you get
something that is most certain when it is wrong.

Read-only with respect to the memory store. Nothing here writes vectors,
metadata, or archives; the only file written is the labelling JSON.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amadeus.config import Config  # noqa: E402
from amadeus.memory.text_utils import (  # noqa: E402
    _confidence_clause,
    _days_since,
    _effective_confidence,
    _volatility_class,
)

DEFAULT_LABEL_PATH = REPO_ROOT / "scripts" / "calibration_labels.json"

# Ground-truth labels. "vague" exists so that unjudgeable facts do not get
# silently forced into true/false and pollute the separation measure.
LABELS = {
    "t": ("true", "still true as far as you know"),
    "o": ("outdated", "was true once, is not any more"),
    "w": ("wrong", "never was right / misunderstood"),
    "v": ("vague", "too vague or generic to judge"),
    "s": ("skip", "leave unlabelled for now"),
}


def _iter_store_files(config: Config) -> List[Path]:
    """Every JSON vector-store file the configured memory layout implies.

    Mirrors scripts/migrate_memory.py's discovery: the base store plus the
    sibling ``*_users/`` per-user files, and the per-character subtrees when
    per-character memory is on.
    """
    base = getattr(config.memory, "embeddings_path", None)
    if not base:
        return []
    base_path = Path(base)
    found: List[Path] = []
    if base_path.exists():
        found.append(base_path)
    users_dir = base_path.parent / f"{base_path.stem}_users"
    if users_dir.is_dir():
        found.extend(sorted(users_dir.glob("*.json")))
    if getattr(config.memory, "per_character_memory", False):
        characters = base_path.parent / "characters"
        if characters.is_dir():
            found.extend(sorted(characters.glob("*/*.json")))
            found.extend(sorted(characters.glob("*/*_users/*.json")))
    return found


def _load_records(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  ! skipping {path.name}: {exc}")
        return []
    records = data.get("records") if isinstance(data, dict) else data
    return records if isinstance(records, list) else []


def _score_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Effective confidence for one stored record, or None if not a fact."""
    metadata = record.get("metadata") or {}
    if metadata.get("type") != "fact":
        return None
    text = record.get("text") or ""
    if not text:
        return None
    age_days = _days_since(metadata.get("timestamp"))
    try:
        strength = float(metadata.get("strength", 1.0))
    except (TypeError, ValueError):
        strength = 1.0
    volatility = _volatility_class(text, metadata.get("valid_at"))
    confidence = _effective_confidence(
        strength=strength, age_days=age_days, volatility=volatility
    )
    return {
        "text": text,
        "confidence": round(confidence, 4),
        "clause": _confidence_clause(confidence),
        "strength": round(strength, 3),
        "age_days": None if age_days is None else round(age_days, 1),
        "volatility": volatility,
        "pinned": bool(metadata.get("pinned")),
        "label": None,
    }


def cmd_sample(args: argparse.Namespace) -> int:
    # Config.load takes a Path (or None to run its own search order).
    config = Config.load(Path(args.config) if args.config else None)
    files = _iter_store_files(config)
    if not files:
        print("No vector-store files found. Check memory.embeddings_path.")
        return 1
    print(f"Reading {len(files)} store file(s):")
    scored: List[Dict[str, Any]] = []
    for path in files:
        records = _load_records(path)
        hits = [s for s in (_score_record(r) for r in records) if s]
        print(f"  {path.name}: {len(hits)} facts")
        scored.extend(hits)
    if not scored:
        print("No type=fact records found — nothing to calibrate.")
        return 1

    # Stratify across the confidence range so the labelling set is not all
    # fresh high-confidence facts (which is what a live store is mostly made
    # of, and which would make the separation measure untestable).
    random.seed(args.seed)
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for item in scored:
        edge = min(0.9, round(item["confidence"] * 5) / 5)
        buckets.setdefault(f"{edge:.1f}", []).append(item)
    for pool in buckets.values():
        random.shuffle(pool)
    # Round-robin across buckets rather than taking a fixed slice from each:
    # a fixed per-bucket cap under-fills whenever confidences cluster (a
    # healthy store is mostly fresh high-confidence facts), which silently
    # returned far fewer facts than --count asked for.
    sample: List[Dict[str, Any]] = []
    depth = 0
    while len(sample) < args.count:
        added = False
        for key in sorted(buckets):
            pool = buckets[key]
            if depth < len(pool):
                sample.append(pool[depth])
                added = True
                if len(sample) >= args.count:
                    break
        if not added:
            break  # every bucket exhausted; the store simply has fewer facts
        depth += 1
    random.shuffle(sample)

    existing: Dict[str, str] = {}
    out = Path(args.labels)
    if out.exists():
        for item in json.loads(out.read_text(encoding="utf-8")):
            if item.get("label"):
                existing[item["text"]] = item["label"]
    carried = 0
    for item in sample:
        if item["text"] in existing:
            item["label"] = existing[item["text"]]
            carried += 1

    out.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"\nWrote {len(sample)} facts to {out}"
        + (f" ({carried} labels carried over)" if carried else "")
    )
    print(f"Confidence spread: {_spread(sample)}")
    print("\nNext: python scripts/calibrate_recall.py label")
    return 0


def _spread(items: List[Dict[str, Any]]) -> str:
    values = [i["confidence"] for i in items]
    if not values:
        return "empty"
    hedged = sum(1 for i in items if i["clause"])
    return (
        f"min {min(values):.2f} / median {statistics.median(values):.2f} / "
        f"max {max(values):.2f}; {hedged}/{len(items)} would hedge"
    )


def cmd_label(args: argparse.Namespace) -> int:
    path = Path(args.labels)
    if not path.exists():
        print(f"{path} not found — run `sample` first.")
        return 1
    items = json.loads(path.read_text(encoding="utf-8"))
    todo = [i for i in items if not i.get("label")]
    if not todo:
        print("Everything is labelled. Run `report`.")
        return 0

    print(f"{len(todo)} fact(s) to label. Ctrl-C saves and exits.\n")
    for key, (name, blurb) in LABELS.items():
        print(f"  {key} = {name:9s} {blurb}")
    print()

    try:
        for index, item in enumerate(todo, 1):
            # The heuristic's own verdict is deliberately NOT shown: seeing
            # "she would hedge this" before judging is exactly the anchoring
            # that would manufacture the correlation being measured.
            print(f"[{index}/{len(todo)}] {item['text']}")
            while True:
                choice = input("  label> ").strip().lower()
                if choice in LABELS:
                    break
                print(f"  ? one of: {', '.join(LABELS)}")
            if choice != "s":
                item["label"] = LABELS[choice][0]
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted — saving progress.")
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    done = sum(1 for i in items if i.get("label"))
    print(f"Saved {done}/{len(items)} labelled to {path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.labels)
    if not path.exists():
        print(f"{path} not found — run `sample` and `label` first.")
        return 1
    items = [i for i in json.loads(path.read_text(encoding="utf-8")) if i.get("label")]
    if not items:
        print("No labelled facts yet.")
        return 1

    groups: Dict[str, List[float]] = {}
    for item in items:
        groups.setdefault(item["label"], []).append(item["confidence"])

    print(f"Labelled facts: {len(items)}\n")
    print(f"{'label':10s} {'n':>4s} {'median':>8s} {'mean':>8s} {'hedged':>8s}")
    print("-" * 42)
    for label in ("true", "outdated", "wrong", "vague"):
        values = groups.get(label)
        if not values:
            continue
        hedged = sum(
            1
            for i in items
            if i["label"] == label and _confidence_clause(i["confidence"])
        )
        print(
            f"{label:10s} {len(values):>4d} {statistics.median(values):>8.3f} "
            f"{statistics.mean(values):>8.3f} {hedged:>7d}/{len(values)}"
        )

    true_values = groups.get("true") or []
    stale_values = (groups.get("outdated") or []) + (groups.get("wrong") or [])
    print()
    if len(true_values) < 5 or len(stale_values) < 5:
        print(
            "VERDICT: underpowered — need >=5 labelled facts in both the "
            f"'true' ({len(true_values)}) and outdated+wrong "
            f"({len(stale_values)}) groups before the separation means "
            "anything. Label more, or the store may simply not contain "
            "enough stale facts yet."
        )
        return 0

    separation = statistics.median(true_values) - statistics.median(stale_values)
    print(f"Separation (median true - median stale): {separation:+.3f}")
    if separation >= args.bar:
        print(
            f"VERDICT: PASS — the mechanical heuristic tracks truth on this "
            f"store (bar {args.bar:+.2f}). Hedging is meaningful; an "
            "LLM-supplied confidence term may be worth testing on top."
        )
    else:
        print(
            f"VERDICT: FAIL — separation below the {args.bar:+.2f} bar. The "
            "heuristic is not distinguishing true from stale facts here. Do "
            "NOT stack an LLM confidence term on top; fix or drop the "
            "mechanical signal first (check the volatility half-lives and "
            "whether enable_forgetting is already removing stale facts "
            "before recall ever sees them)."
        )

    # Volatility-class breakdown: a class that never fires, or fires on
    # everything, is a broken keyword table rather than a real signal.
    print("\nVolatility classes in the labelled set:")
    classes: Dict[str, int] = {}
    for item in items:
        classes[item["volatility"]] = classes.get(item["volatility"], 0) + 1
    for name in ("volatile", "stable", "permanent"):
        print(f"  {name:10s} {classes.get(name, 0)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # --labels belongs to every subcommand rather than the parent parser:
    # as a parent argument it would only parse BEFORE the subcommand name,
    # which is the opposite of how everyone types it.
    def _with_labels(sub_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sub_parser.add_argument(
            "--labels",
            default=str(DEFAULT_LABEL_PATH),
            help="labelling file (default: scripts/calibration_labels.json)",
        )
        return sub_parser

    p_sample = _with_labels(
        sub.add_parser("sample", help="sample facts from the live store")
    )
    p_sample.add_argument("--config", default="config.local.yaml")
    p_sample.add_argument("--count", type=int, default=60)
    p_sample.add_argument("--seed", type=int, default=20260806)
    p_sample.set_defaults(func=cmd_sample)

    p_label = _with_labels(
        sub.add_parser("label", help="label sampled facts interactively")
    )
    p_label.set_defaults(func=cmd_label)

    p_report = _with_labels(
        sub.add_parser("report", help="score the heuristic against labels")
    )
    p_report.add_argument(
        "--bar",
        type=float,
        default=0.15,
        help="median separation required to pass (design default: 0.15)",
    )
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
