#!/usr/bin/env python3
"""One-time memory-store migration: episode cleanup + re-embedding.

Two jobs, both idempotent:

1. **Prune near-duplicate episodes.** Before episode_supersede_threshold
   existed, repeated low-content sessions each wrote a near-identical
   "the user greeted..." episode. This applies the same rule retroactively:
   within each store, an episode whose cosine similarity to a NEWER episode
   exceeds the threshold is deleted (pinned episodes exempt). Uses the
   existing stored vectors, so it works before re-embedding.

2. **Re-embed all records with the currently configured embedder.** Switching
   embedders (hashing 512-d -> Gemini 768-d, etc.) leaves persisted vectors
   with the wrong dimension; recall detects the mismatch and disables itself
   until this runs. Re-embedding preserves ids/text/metadata, only vectors
   change. Skipped automatically when dimensions already match (--force
   overrides, e.g. after changing between same-dimension models).

Operates on the JSON vector-store files (the default backend): the base
``memory.embeddings_path`` plus every per-user file in the sibling
``*_users/`` directory — and, when ``memory.per_character_memory`` is on,
the same layout under every ``characters/<id>/`` subdirectory (per-character
guest stores; without this, an embedder switch would leave guests' vectors
at the old dimension and recall's mismatch guard would silently disable
recall for every guest). Each file is backed up to ``<name>.bak-migrate``
before being rewritten atomically.

    python scripts/migrate_memory.py --config config.local.yaml --dry-run
    python scripts/migrate_memory.py --config config.local.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from amadeus.config import Config  # noqa: E402
from amadeus.paths import resolve_user_data_path  # noqa: E402
from amadeus.provider_loader import load_embedder  # noqa: E402


def _cosine(a, b) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def _load_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_records(path: Path, records) -> None:
    backup = path.with_name(path.name + ".bak-migrate")
    if not backup.exists():  # keep the FIRST pre-migration state
        shutil.copy2(path, backup)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(records, handle)
    tmp.replace(path)


def prune_episodes(records, threshold: float):
    """Drop older episodes that near-duplicate a newer one. Returns kept list
    and the number pruned."""
    episodes = [
        (i, rec)
        for i, rec in enumerate(records)
        if (rec.get("metadata") or {}).get("type") == "episode"
        and not (rec.get("metadata") or {}).get("pinned")
    ]
    # Newest last by timestamp (records are append-ordered anyway; timestamp is
    # the tiebreak source of truth).
    episodes.sort(
        key=lambda pair: str((pair[1].get("metadata") or {}).get("timestamp") or "")
    )
    doomed = set()
    for older_pos in range(len(episodes)):
        i, older = episodes[older_pos]
        if i in doomed:
            continue
        for newer_pos in range(older_pos + 1, len(episodes)):
            j, newer = episodes[newer_pos]
            if j in doomed:
                continue
            sim = _cosine(older.get("vector") or [], newer.get("vector") or [])
            if sim >= threshold:
                doomed.add(i)  # older of the pair
                break
    kept = [rec for i, rec in enumerate(records) if i not in doomed]
    return kept, len(doomed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.local.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="re-embed even if dimensions match"
    )
    parser.add_argument(
        "--no-prune", action="store_true", help="skip the episode dedup pass"
    )
    args = parser.parse_args()

    config = Config.load(Path(args.config))
    if getattr(config.memory, "vector_backend", "json") != "json":
        print("Only the JSON vector backend is supported.", file=sys.stderr)
        return 2

    embedder = load_embedder(config)
    if embedder is None:
        print("No embedder configured (enable_long_term off?).", file=sys.stderr)
        return 2
    target_dim = len(embedder.embed_text("dimension probe"))
    threshold = float(getattr(config.memory, "episode_supersede_threshold", 0.9) or 0.9)
    print(
        f"Embedder: {type(embedder).__name__} ({target_dim}-d); "
        f"episode dedup threshold {threshold}"
    )

    base = Path(resolve_user_data_path(config.memory.embeddings_path))

    def _store_files(store_base: Path) -> list[Path]:
        """One store's files: the base file plus its `<stem>_users/*.json`."""
        found = []
        if store_base.exists():
            found.append(store_base)
        users_dir = store_base.with_name(store_base.stem + "_users")
        if users_dir.is_dir():
            found.extend(sorted(users_dir.glob("*.json")))
        return found

    candidates = _store_files(base)
    # Per-character guest stores live under characters/<id>/ next to the base
    # store (amadeus.paths.character_scoped_path). Skipped when the flag is
    # off — that is full legacy behavior, every character shares `base`.
    if getattr(config.memory, "per_character_memory", True):
        characters_root = base.parent / "characters"
        if characters_root.is_dir():
            for char_dir in sorted(characters_root.iterdir()):
                if char_dir.is_dir():
                    candidates.extend(_store_files(char_dir / base.name))
    candidates = [p for p in candidates if not p.name.endswith(".bak-migrate")]
    if not candidates:
        print(f"No vector-store files found under {base.parent}.")
        return 0

    def _label(path: Path) -> str:
        """Log name that keeps the character subdirectory visible."""
        try:
            return str(path.relative_to(base.parent))
        except ValueError:
            return path.name

    for path in candidates:
        try:
            records = _load_records(path)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  SKIP {_label(path)}: unreadable ({exc})")
            continue
        if not isinstance(records, list) or not records:
            print(f"  OK   {_label(path)}: empty, skipped")
            continue

        pruned = 0
        if not args.no_prune:
            records, pruned = prune_episodes(records, threshold)

        current_dim = len(records[0].get("vector") or []) if records else 0
        needs_reembed = bool(records) and (current_dim != target_dim or args.force)
        reembedded = 0
        if needs_reembed:
            for rec in records:
                text = rec.get("text") or ""
                if not text:
                    continue
                rec["vector"] = embedder.embed_text(text)
                reembedded += 1

        action = []
        if pruned:
            action.append(f"pruned {pruned} near-dup episode(s)")
        if reembedded:
            action.append(
                f"re-embedded {reembedded} ({current_dim}-d -> {target_dim}-d)"
            )
        if not action:
            print(f"  OK   {_label(path)}: {len(records)} records, nothing to do")
            continue
        if args.dry_run:
            print(f"  DRY  {_label(path)}: would have {', '.join(action)}")
            continue
        _write_records(path, records)
        print(
            f"  DONE {_label(path)}: {', '.join(action)} ({len(records)} records kept)"
        )

    if args.dry_run:
        print("\nDry run only — nothing written.")
    else:
        print(
            "\nDone. Originals preserved as *.bak-migrate; restart the backend "
            "so per-user managers reload their stores."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
