"""Static pin: manifest entries that feed an AUTOMATIC animation pool must
not be `lazy: true`.

This enforces, as a running test, an invariant that used to live only as
prose in the project notes ("automatic pools ... require EAGER clips — lazy clips
must never be given an automatic role"). The mechanism this pins is derived
directly from the frontend code, not from the doc's paraphrase:

  - `frontend/src/vrm/clipLoader.ts::loadAnimationClips` only fetches/parses
    clips where `!file.lazy` (the `eagerFiles` filter); a lazy clip is
    registered in `clipMeta`/aliases but is NOT added to `loadedClips` (and
    role:"talk" lazy clips are NOT pushed into `talkClipNames`) until
    `ensureClipLoaded` is called on first play.
  - `frontend/src/components/VRMViewer.tsx` builds the scheduler's
    `idleNames`/`actionNames` pools by filtering `idleFiles`/`actionFiles`
    down to `loadedClips[name]` — i.e. eager clips only. A lazy clip
    structurally cannot appear in `idleNames`/`actionNames`.
  - `frontend/src/vrm/animationScheduler.ts`'s `subtleActionNames` (the
    idle-variety one-shot pool) filters `actionNames` (already eager-gated)
    down to `clipMeta[name]?.role === 'subtle'`.
  - The same file's `EMOTION_GESTURE_MAP` reactions
    (`kurisuReact`/`playRandomSubtleGesture`/talk-clip rotation) all gate on
    `loadedClips[...]`, i.e. eager-only, before a pool clip can fire.

None of the above RAISES or warns for an entry that combines an automatic
role (or an emotion-map reference) with `lazy: true` — it just silently
excludes that clip from ever joining its pool, which is exactly the failure
mode this test is meant to catch at manifest-authoring time instead of at
runtime. Pure JSON + regex parsing, no frontend imports, no Node — same
posture as test_contracts.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "static" / "animations" / "manifest.json"
SCHEDULER_PATH = REPO_ROOT / "frontend" / "src" / "vrm" / "animationScheduler.ts"

# Roles that a pool filters on directly by `clipMeta[name]?.role` (or, for
# "idle", by simply being a member of idleFiles that made it into
# loadedClips — see VRMViewer.tsx's idleNames derivation). "gesture" is
# deliberately excluded here: gesture clips only become automatic if their
# NAME is referenced in EMOTION_GESTURE_MAP, which is checked separately
# below — a gesture-role clip nobody references by name is never automatic.
AUTOMATIC_ROLES = {"idle", "talk", "subtle"}


def _derived_default_name(filename: str) -> str:
    """Mirrors clipLoader.ts::toAnimationFile's defaultName derivation
    (used when a manifest entry has no explicit `name`): strip the
    extension, replace underscores with spaces, title-case each word's
    first letter.
    """
    stem = re.sub(r"\.(vrma|fbx|bvh)$", "", filename, flags=re.IGNORECASE)
    stem = stem.replace("_", " ")
    return re.sub(r"\b([a-z])", lambda m: m.group(1).upper(), stem)


def _manifest_entries() -> List[Dict]:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries: List[Dict] = []
    for section in ("idle", "action"):
        for raw in data.get(section, []):
            if isinstance(raw, str):
                entries.append({"file": raw, "section": section})
            elif isinstance(raw, dict):
                merged = dict(raw)
                merged["section"] = section
                entries.append(merged)
    return entries


def _entry_names(entry: Dict) -> Set[str]:
    """Every name this entry could be referenced/resolved by: explicit
    name/label, filename (with directory stripped, matching clipLoader's
    toAnimationFile split), and the derived title-cased default name —
    mirroring clipAliases registration in loadAnimationClips.
    """
    filename = entry.get("file") or entry.get("path") or ""
    base_filename = filename.split("/")[-1]
    names = {base_filename}
    default_name = _derived_default_name(base_filename)
    names.add(default_name)
    explicit = entry.get("name") or entry.get("label")
    if explicit:
        names.add(explicit)
    return names


def _emotion_gesture_map_names() -> Set[str]:
    """Extracts every `name: '...'` reference inside animationScheduler.ts's
    EMOTION_GESTURE_MAP object literal. Regex over the TS source (same
    posture as test_contracts.py's invoke() extraction) — brittle to a
    total rewrite of the map's shape, but that rewrite would need human
    eyes on this test anyway.
    """
    source = SCHEDULER_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"export const EMOTION_GESTURE_MAP:.*?\n=?\s*\{(.*?)\n\}\n", source, re.S
    )
    assert match, (
        "Could not locate EMOTION_GESTURE_MAP in animationScheduler.ts — "
        "the extraction regex needs updating to match the current source "
        "shape before this test can trust its results."
    )
    body = match.group(1)
    return set(re.findall(r"name:\s*'([^']+)'", body))


def test_manifest_has_idle_and_action_entries() -> None:
    """Sanity check the fixture this whole file depends on hasn't gone
    empty/missing — every other test here would vacuously pass against an
    empty manifest and give false confidence.
    """
    assert MANIFEST_PATH.exists(), f"missing {MANIFEST_PATH}"
    entries = _manifest_entries()
    assert len(entries) > 50, (
        f"Expected a substantial committed clip library, found only "
        f"{len(entries)} manifest entries — did the manifest get "
        "truncated or the fixture path change?"
    )


def test_no_automatic_role_entry_is_lazy() -> None:
    """role: idle/talk/subtle clips join their pool purely by being present
    in loadedClips (eager-only) — see clipLoader.ts::loadAnimationClips'
    `eagerFiles` filter and VRMViewer.tsx's idleNames/actionNames
    derivation. A `lazy: true` entry with one of these roles would silently
    never rotate/never play, contradicting the role it was given.
    """
    violations = [
        (entry["section"], entry.get("file") or entry.get("path"), entry.get("role"))
        for entry in _manifest_entries()
        if entry.get("lazy") is True and entry.get("role") in AUTOMATIC_ROLES
    ]
    assert not violations, (
        "The following manifest entries combine an automatic role with "
        "lazy: true — they will never actually join their pool (idle "
        "rotation / talk clip rotation / the subtle idle-variety beat), "
        "because loadAnimationClips only eager-loads non-lazy clips:\n"
        + "\n".join(
            f"  - [{section}] {file!r} role={role!r}"
            for section, file, role in violations
        )
    )


def test_no_emotion_gesture_map_clip_is_lazy() -> None:
    """EMOTION_GESTURE_MAP entries reference clips BY NAME; kurisuReact only
    considers a candidate eligible when `loadedClips[resolveClipName(name)]`
    is set, i.e. the clip already finished eager-loading. A lazy clip
    referenced here would never be eligible until something else happens to
    play it first — same silent-exclusion failure mode as the role-based
    pools, just keyed by name instead of by manifest `role`.
    """
    referenced_names = _emotion_gesture_map_names()
    assert referenced_names, (
        "Extracted zero names from EMOTION_GESTURE_MAP — the extraction "
        "regex likely needs updating; this test would otherwise pass "
        "vacuously."
    )

    violations = []
    for entry in _manifest_entries():
        if entry.get("lazy") is not True:
            continue
        names = _entry_names(entry)
        hit = names & referenced_names
        if hit:
            violations.append(
                (entry["section"], entry.get("file") or entry.get("path"), sorted(hit))
            )

    assert not violations, (
        "The following manifest entries are lazy: true but are referenced "
        "by name in animationScheduler.ts's EMOTION_GESTURE_MAP — an "
        "automatic emotion reaction would never fire for them until "
        "something else happened to trigger a load first:\n"
        + "\n".join(
            f"  - [{section}] {file!r} matches gesture-map name(s) {names!r}"
            for section, file, names in violations
        )
    )
