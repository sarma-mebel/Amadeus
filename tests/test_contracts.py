"""Cross-layer contract tests.

This repo's worst recurring bug class is contract drift between layers that
no single-layer test can see: the frontend and backend (or the frontend and
the Tauri shell) each look correct in isolation, but disagree about a shared
string, and nothing fails until the packaged app or a live user hits it.
Two real incidents motivated this file:

  (a) A Tauri command invoked from JS but missing from build.rs's
      AppManifest export, or missing from the invoking window's capabilities
      allowlist, works fine in `npm run dev` (implicit allow on the bundled
      dev origin) and then fails ONLY in the packaged app with
      "not allowed by ACL" — the dead pet button / vision capture bug.

  (b) Frontend poke regions ('dragged', 'click_through') that the backend's
      PresenceRequest regex did not accept 422'd for a week, silently,
      because the pet UI swallows poke errors by design (a poke reaction is
      a nice-to-have, not something worth surfacing an error toast for).

Neither bug is visible to a backend-only test suite (which never reads
frontend source) or a frontend-only test suite (which never reads the Tauri
manifest or the Pydantic model). This file is pure text/regex parsing over
both trees — no frontend imports, no Node — so it runs as an ordinary
pytest module in a plain backend CI job and still catches drift the moment
either side changes without the other.

Docstrings on each test explain the specific failure archetype it kills;
keep that framing when adding new contracts here so this file continues to
double as documentation of what "drift" has actually looked like in this
codebase.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Set

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"
SRC_TAURI = REPO_ROOT / "frontend" / "src-tauri"


def _frontend_source_files() -> List[Path]:
    """Every .ts/.tsx file under frontend/src, deterministically ordered."""
    files = sorted(FRONTEND_SRC.rglob("*.ts")) + sorted(FRONTEND_SRC.rglob("*.tsx"))
    # De-dupe defensively (rglob patterns don't overlap here, but keep this
    # robust if that ever changes) and re-sort for stable output.
    return sorted(set(files))


# ---------------------------------------------------------------------------
# Contract 1: Tauri command ACL contract
# ---------------------------------------------------------------------------

# Matches invoke('cmd'), invoke("cmd"), invoke<T>('cmd'), invokeFn('cmd') —
# i.e. any identifier ending in "invoke" (optionally with a TS generic arg)
# immediately followed by a string-literal first argument. Tolerant of the
# handful of call-site spellings actually used in this codebase
# (frontend/src/hooks/usePetChat.ts uses `invoke<boolean>(...)`,
# frontend/src/vrm/cursorGaze.ts assigns the tauri invoke fn to a local
# `invokeFn` variable to avoid a static import in a lazy-loaded module).
_INVOKE_CALL_RE = re.compile(
    r"""(?<![.\w])(?:invoke\w*)\s*(?:<[^>(]*>)?\s*\(\s*(['"])([^'"]+)\1"""
)


def _extract_invoke_calls() -> List[tuple[Path, str]]:
    """Returns (file, command_name) for every invoke('literal') call site.

    Plugin-namespaced invokes (e.g. 'plugin:shell|open') are Tauri-plugin
    calls, not application commands — they are governed by plugin
    permissions (e.g. "shell:allow-open"), not the AppManifest command
    list, so they are intentionally excluded here.
    """
    calls: List[tuple[Path, str]] = []
    for path in _frontend_source_files():
        text = path.read_text(encoding="utf-8")
        for match in _INVOKE_CALL_RE.finditer(text):
            command = match.group(2)
            if "|" in command or command.startswith("plugin:"):
                continue
            calls.append((path, command))
    return calls


def _build_rs_commands() -> Set[str]:
    """Command names exported via tauri_build::AppManifest::new().commands(&[...])."""
    build_rs = SRC_TAURI / "build.rs"
    text = build_rs.read_text(encoding="utf-8")
    match = re.search(r"\.commands\(&\[(.*?)\]\)", text, re.DOTALL)
    assert match, (
        f"{build_rs}: could not locate `.commands(&[...])` in the "
        "AppManifest builder — the ACL-export contract test can't verify "
        "anything against this file. If build.rs was restructured, update "
        "_build_rs_commands()'s parser to match."
    )
    body = match.group(1)
    return set(re.findall(r'"([^"]+)"', body))


def _kebab_permission_id(command: str) -> str:
    """Derives the `allow-<kebab-case>` permission identifier Tauri expects
    for a snake_case command name — e.g. `set_pet_click_through` ->
    `allow-set-pet-click-through`. This is exactly the mapping already used
    by every command in capabilities/default.json and capabilities/pet.json;
    verified 1:1 against both files as of writing this test.
    """
    return "allow-" + command.replace("_", "-")


def _capability_permissions(capability_file: Path) -> Set[str]:
    """Flat set of permission identifier strings granted by a capabilities
    JSON file. Permission entries are either bare strings
    ("allow-toggle-pet-window") or objects with an "identifier" key
    (the shell:allow-execute entry) — both forms are handled.
    """
    import json

    data = json.loads(capability_file.read_text(encoding="utf-8"))
    perms: Set[str] = set()
    for entry in data.get("permissions", []):
        if isinstance(entry, str):
            perms.add(entry)
        elif isinstance(entry, dict) and "identifier" in entry:
            perms.add(entry["identifier"])
    return perms


def _capability_windows(capability_file: Path) -> Set[str]:
    import json

    data = json.loads(capability_file.read_text(encoding="utf-8"))
    return set(data.get("windows", []))


# Pragmatic file-path -> window classification. The pet overlay is a
# separate Tauri window (label "pet", see frontend/src-tauri/src/main.rs's
# PET_WINDOW_LABEL and the WebviewWindowBuilder call) that only ever loads
# these modules; everything else loads in the "main" window (the single
# unlabeled window in tauri.conf.json, which Tauri defaults to label
# "main" — confirmed against src-tauri/src/main.rs's own
# `get_webview_window("main")` / `w.label == "main"` usage).
_PET_ONLY_PATH_MARKERS = (
    "components/PetView",
    "hooks/usePet",
    "readTogether.ts",
    "vrm/cursorGaze.ts",
    # Desktop-roaming scheduler (2026-07-26): only the pet window ever
    # starts it (usePetRoam gates on the pet's own state), so its
    # window-motion invokes need granting in capabilities/pet.json only.
    "vrm/petRoam",
    # Tuck-away / hide-and-peek (2026-08-03): pet-only window motion, same
    # rationale as petRoam (its pet_roam_context invoke needs pet.json).
    "vrm/petTuck",
    # Watch-together (2026-07-28) is pet-only too. It currently has no
    # DIRECT invoke (frame capture delegates to readTogether.ts), but
    # without these markers a future direct invoke would be classified as
    # main-window and this suite would then ENFORCE the wrong grant file.
    "watchTogether.ts",
    "hooks/useWatchTogether",
)


def _windows_for_file(path: Path) -> Set[str]:
    rel = path.as_posix()
    if any(marker in rel for marker in _PET_ONLY_PATH_MARKERS):
        return {"pet"}
    return {"main"}


def test_tauri_commands_exported_in_build_rs() -> None:
    """Kills incident (a): a command invoked from JS but never exported via
    tauri_build::AppManifest::new().commands(&[...]) in build.rs is
    implicitly allowed only on the bundled dev origin — and the packaged
    app runs the frontend from the backend-served (remote) origin, where
    every such invoke fails at runtime with "not allowed by ACL". Nothing
    in `npm run dev` or a component test ever exercises that codepath, so
    this shipped broken before (the dead pet button / vision capture bug).
    """
    exported = _build_rs_commands()
    calls = _extract_invoke_calls()
    assert calls, (
        "No invoke('...') call sites found under frontend/src — either the "
        "extraction regex broke, or the frontend stopped calling Tauri "
        "commands entirely. Either way this needs a human look, not a "
        "silent pass."
    )
    missing = sorted({(path, cmd) for path, cmd in calls if cmd not in exported})
    assert not missing, (
        "The following Tauri commands are invoked from the frontend but "
        "are NOT exported in frontend/src-tauri/build.rs's "
        "AppManifest.commands([...]) list, so they will fail with "
        '"not allowed by ACL" in the packaged app even though they work '
        "in `npm run dev`:\n"
        + "\n".join(f"  - {cmd!r} invoked at {path}" for path, cmd in missing)
    )


def test_tauri_commands_covered_by_capabilities() -> None:
    """Kills incident (a)'s other half: a command can be correctly exported
    in build.rs and STILL fail with "not allowed by ACL" if no capabilities
    file grants its `allow-<command>` permission to the window that invokes
    it. The pet overlay and the main window are separate Tauri windows with
    separate capability files (capabilities/pet.json vs
    capabilities/default.json) — a permission granted to the wrong window
    reproduces the exact same silent-in-dev, broken-when-packaged failure.
    """
    default_json = SRC_TAURI / "capabilities" / "default.json"
    pet_json = SRC_TAURI / "capabilities" / "pet.json"
    assert default_json.exists(), f"missing {default_json}"
    assert pet_json.exists(), f"missing {pet_json}"

    # Sanity-check the window mapping itself hasn't silently drifted: if
    # this assumption breaks, every assertion below is checking the wrong
    # thing without any of them failing loudly on their own.
    assert _capability_windows(default_json) == {"main"}, (
        f'{default_json}: expected "windows": ["main"] — the pet/main '
        "file-path classification below assumes this. Update "
        "_windows_for_file() if the window topology changed."
    )
    assert _capability_windows(pet_json) == {"pet"}, (
        f'{pet_json}: expected "windows": ["pet"] — the pet/main '
        "file-path classification below assumes this. Update "
        "_windows_for_file() if the window topology changed."
    )

    perms_by_file = {
        "main": _capability_permissions(default_json),
        "pet": _capability_permissions(pet_json),
    }

    calls = _extract_invoke_calls()
    exported = _build_rs_commands()

    # Group invoking windows per command across every call site, so a
    # command invoked from both windows is required to be granted in both
    # capability files (rather than checking call sites independently and
    # accidentally accepting partial coverage).
    windows_by_command: Dict[str, Set[str]] = {}
    call_sites_by_command: Dict[str, List[Path]] = {}
    for path, cmd in calls:
        if cmd not in exported:
            continue  # already reported by test_tauri_commands_exported_in_build_rs
        windows_by_command.setdefault(cmd, set()).update(_windows_for_file(path))
        call_sites_by_command.setdefault(cmd, []).append(path)

    failures: List[str] = []
    for cmd in sorted(windows_by_command):
        permission_id = _kebab_permission_id(cmd)
        required_windows = windows_by_command[cmd]
        for window in sorted(required_windows):
            if permission_id not in perms_by_file[window]:
                sites = ", ".join(str(p) for p in call_sites_by_command[cmd])
                capability_path = default_json if window == "main" else pet_json
                failures.append(
                    f"  - command {cmd!r} (permission {permission_id!r}) is "
                    f"invoked from the {window!r} window ({sites}) but "
                    f"{permission_id!r} is not granted in {capability_path}"
                )

    assert not failures, (
        "The following commands are exported in build.rs but not granted "
        "to the window that invokes them, so they will fail with "
        '"not allowed by ACL" in the packaged app:\n' + "\n".join(failures)
    )


def test_pet_roam_commands_pinned() -> None:
    """Pins the desktop-roaming command surface (2026-07-26) BY NAME on top
    of the generic invoke-scan tests above. Those tests only see commands
    whose call sites match the invoke-regex — if the roam driver
    (frontend/src/vrm/petRoam.ts) were ever refactored into a call shape the
    regex misses, the generic tests would pass vacuously while the packaged
    app failed with "not allowed by ACL". Roaming is invoked from the pet
    window only, so pet.json (not default.json) must carry the grants.
    """
    roam_commands = {
        "pet_roam_context",
        "animate_pet_window_to",
        "cancel_pet_window_animation",
    }
    exported = _build_rs_commands()
    missing_export = sorted(roam_commands - exported)
    assert not missing_export, (
        "Roam commands missing from frontend/src-tauri/build.rs's "
        f"AppManifest.commands([...]) export: {missing_export}"
    )
    pet_perms = _capability_permissions(SRC_TAURI / "capabilities" / "pet.json")
    missing_perms = sorted(
        _kebab_permission_id(cmd)
        for cmd in roam_commands
        if _kebab_permission_id(cmd) not in pet_perms
    )
    assert not missing_perms, (
        "Roam command permissions missing from "
        f"frontend/src-tauri/capabilities/pet.json: {missing_perms}"
    )


def test_capture_screenshot_pinned_in_both_windows() -> None:
    """capture_screenshot is the one command invoked from BOTH windows (main:
    images.ts vision attach; pet: readTogether/watchTogether frame capture),
    so it needs grants in BOTH capability files. The generic scan only
    demands whichever windows the current call sites classify to — a
    refactor that hides one call site from the regex would silently drop
    that window's requirement. Pin both by name.
    """
    assert (
        "capture_screenshot" in _build_rs_commands()
    ), "capture_screenshot missing from build.rs AppManifest export"
    perm = _kebab_permission_id("capture_screenshot")
    for cap_file in ("default.json", "pet.json"):
        perms = _capability_permissions(SRC_TAURI / "capabilities" / cap_file)
        assert perm in perms, (
            f"frontend/src-tauri/capabilities/{cap_file} must grant {perm!r} "
            "— screenshot capture is invoked from both windows"
        )


def test_invoked_commands_registered_in_generate_handler() -> None:
    """Third leg of the ACL trilogy: a command exported in build.rs and
    granted in capabilities STILL fails at runtime if it's missing from
    main.rs's tauri::generate_handler![...] registration. Neither generic
    test above reads main.rs, so this was an unpinned failure mode.
    """
    main_rs = (SRC_TAURI / "src" / "main.rs").read_text(encoding="utf-8")
    match = re.search(r"generate_handler!\s*\[(.*?)\]", main_rs, flags=re.DOTALL)
    assert match, (
        "frontend/src-tauri/src/main.rs: could not find "
        "tauri::generate_handler![...] — update this test's regex if the "
        "registration moved, don't delete the contract."
    )
    registered = {
        token.strip()
        for token in match.group(1).replace("\n", " ").split(",")
        if token.strip()
    }
    invoked = {cmd for _, cmd in _extract_invoke_calls()}
    missing = sorted(invoked - registered)
    assert not missing, (
        "Commands invoked from the frontend but missing from main.rs's "
        f"generate_handler![...] registration: {missing}"
    )


# ---------------------------------------------------------------------------
# Contract 2: poke region contract
# ---------------------------------------------------------------------------

# Matches firePoke('x'), poke('x'), poke?.('x'), firePokeRef.current('x'),
# etc: any identifier containing "poke"/"Poke" as a substring (so it
# catches both the bare `poke` bridge call and the `firePoke`/
# `firePokeRef.current` call sites — a leading \b would miss "poke" when
# it's not at the start of the identifier, e.g. "firePoke"), optionally
# chained through `.current`/`?.`, followed by a string-literal first
# argument. Deliberately does NOT match identifiers whose next token isn't
# an opening paren with a string literal (e.g. `pokedRegion`, a plain
# variable, or `resolvePokedRegion(vrm, hitPoint, ...)`, whose first arg
# isn't a string) — those aren't call sites this contract cares about.
_POKE_CALL_RE = re.compile(
    r"""\w*[Pp]oke\w*(?:\.current)?\s*\?{0,1}\.?\(\s*(['"])([^'"]+)\1"""
)


def _extract_poke_literals() -> List[tuple[Path, str]]:
    calls: List[tuple[Path, str]] = []
    for path in _frontend_source_files():
        text = path.read_text(encoding="utf-8")
        for match in _POKE_CALL_RE.finditer(text):
            calls.append((path, match.group(2)))
    return calls


def _presence_region_pattern_alternatives() -> List[str]:
    """Parses the `pattern="^(a|b|c)$"` alternation off
    PresenceRequest.region in amadeus/web/models.py.
    """
    models_py = REPO_ROOT / "amadeus" / "web" / "models.py"
    text = models_py.read_text(encoding="utf-8")
    match = re.search(r'pattern\s*=\s*"\^\(([^)]+)\)\$"', text)
    assert match, (
        f"{models_py}: could not find PresenceRequest.region's "
        '`pattern="^(...)$"` — if the field was reshaped (e.g. to an enum), '
        "update _presence_region_pattern_alternatives() to match, don't "
        "just skip this contract."
    )
    return match.group(1).split("|")


def _find_poke_region_context_source() -> Path:
    """_POKE_REGION_CONTEXT is mid-move from amadeus/engine.py into a new
    amadeus/presence.py (concurrent work as of writing this test). Search
    both locations rather than hardcoding one, so this test survives the
    move without edits.
    """
    candidates = [
        REPO_ROOT / "amadeus" / "engine.py",
        REPO_ROOT / "amadeus" / "presence.py",
    ]
    hits = [
        p
        for p in candidates
        if p.exists() and "_POKE_REGION_CONTEXT" in p.read_text(encoding="utf-8")
    ]
    assert hits, (
        "_POKE_REGION_CONTEXT not found in either amadeus/engine.py or "
        "amadeus/presence.py. It may have moved again or been renamed — "
        "update _find_poke_region_context_source()'s candidate list."
    )
    assert len(hits) == 1, (
        "_POKE_REGION_CONTEXT appears in more than one candidate file "
        f"({hits}) — that's its own drift risk (which copy is live?); "
        "resolve the duplication so there is exactly one source of truth."
    )
    return hits[0]


def _poke_region_context_keys() -> Set[str]:
    """Top-level string keys of the _POKE_REGION_CONTEXT dict literal.

    Deliberately indentation-agnostic: as of writing, _POKE_REGION_CONTEXT
    is mid-move from a class attribute in engine.py (body indented 8
    spaces, closing brace at 4) to a module-level dict in the new
    presence.py (body indented 4 spaces, closing brace at 0) — a fixed
    indent assumption broke the moment the sibling agent's move landed.
    Instead this does a balanced-brace scan from the `{` after the name to
    its matching `}`, then regexes top-level `"key":` lines (any leading
    whitespace) out of that span. Nested parens/strings inside each value
    don't contain '{' or '}' anywhere in this dict (plain string
    concatenation), so a naive brace counter is sufficient here.
    """
    source = _find_poke_region_context_source()
    text = source.read_text(encoding="utf-8")
    name_match = re.search(r"_POKE_REGION_CONTEXT\s*=\s*\{", text)
    assert name_match, (
        f"{source}: found the _POKE_REGION_CONTEXT name but could not "
        "locate its opening `= {` — update the parser if the assignment "
        "was reformatted."
    )
    start = name_match.end() - 1  # index of the opening '{'
    depth = 0
    end = None
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    assert end is not None, (
        f"{source}: _POKE_REGION_CONTEXT's opening brace is never closed "
        "(unbalanced '{'/'}') — the balanced-brace scan in "
        "_poke_region_context_keys() found no match."
    )
    body = text[start + 1 : end]
    # Top-level string keys only: "key": (  or  "key": "...  — any leading
    # whitespace, on its own line, so a key mentioned inside a value's
    # prose doesn't get picked up.
    return set(re.findall(r'^\s*"([a-zA-Z_]+)":', body, re.MULTILINE))


def _head_pat_is_specially_handled() -> bool:
    """'head_pat' is deliberately NOT a static _POKE_REGION_CONTEXT key —
    its tone follows the relationship arc and is built per-call. Detect
    that special-casing by grepping for "head_pat" handling in whichever
    file owns the poke-reaction logic, rather than asserting it's a dict
    key (which would be testing the wrong thing).
    """
    source = _find_poke_region_context_source()
    text = source.read_text(encoding="utf-8")
    return bool(re.search(r'region_key\s*==\s*"head_pat"|"head_pat".*context', text))


# The four raycast-derived regions (VRMViewer's hit-testing: head/chest/
# hands map to a mapped bone, body is the "nothing close enough" fallback)
# are allowed to have no bespoke server context — 'body' in particular
# intentionally falls through to the generic "poked/clicked you on screen"
# line in poke_reaction(). Every OTHER region in the pattern is either a
# desktop-pet event region or head_pat, both of which are expected to carry
# a specific, hand-written reaction.
_RAYCAST_REGIONS = {"head", "chest", "hands", "body"}


def test_poke_region_frontend_literals_accepted_by_backend_pattern() -> None:
    """Kills incident (b): every region string literal the frontend can
    actually send via firePoke()/poke?.() must be accepted by
    PresenceRequest.region's regex pattern in amadeus/web/models.py, or the
    request 422s. The pet UI swallows poke failures by design (a poke
    reaction is a nice-to-have; not worth an error toast) so a 422 here is
    invisible to a user or a manual smoke test — 'dragged' and
    'click_through' shipped broken for a week this way.
    """
    literals = _extract_poke_literals()
    assert literals, (
        "No poke('...')-shaped call sites found under frontend/src — "
        "either the extraction regex broke, or poke reactions were "
        "removed entirely. Needs a human look, not a silent pass."
    )
    accepted = set(_presence_region_pattern_alternatives())
    missing = sorted(
        {(path, region) for path, region in literals if region not in accepted}
    )
    assert not missing, (
        "The following poke region literals are sent by the frontend but "
        "are NOT in PresenceRequest.region's pattern in "
        "amadeus/web/models.py, so POST /api/poke 422s (silently, from the "
        "user's perspective — the pet UI swallows poke errors):\n"
        + "\n".join(f"  - {region!r} sent from {path}" for path, region in missing)
    )


def test_poke_region_frontend_literals_have_server_context() -> None:
    """A region the backend's regex accepts but that has no entry in
    _POKE_REGION_CONTEXT (and isn't 'head_pat', which is special-cased) is
    not a 422 — it silently falls through to the generic "poked/clicked you
    on screen, interrupting nothing in particular" line instead of the
    bespoke reaction the frontend event implies. That's a quieter version
    of the same drift: the two sides technically agree the request is
    valid, but disagree about what it means.
    """
    literals = _extract_poke_literals()
    context_keys = _poke_region_context_keys()
    assert _head_pat_is_specially_handled(), (
        "Expected to find head_pat's per-call (relationship-arc-driven) "
        "handling alongside _POKE_REGION_CONTEXT — if head_pat's dispatch "
        "was refactored, update _head_pat_is_specially_handled()'s "
        "detection instead of silently letting this contract go stale."
    )
    missing = sorted(
        {
            (path, region)
            for path, region in literals
            if region not in context_keys
            and region != "head_pat"
            and region not in _RAYCAST_REGIONS
        }
    )
    assert not missing, (
        "The following poke region literals are sent by the frontend but "
        "have no bespoke reaction — neither a _POKE_REGION_CONTEXT entry "
        "nor head_pat's special-cased handling — so they silently fall "
        "through to the generic poke reaction line instead of the specific "
        "one the frontend event implies:\n"
        + "\n".join(f"  - {region!r} sent from {path}" for path, region in missing)
    )


def test_poke_region_pattern_alternatives_have_server_context() -> None:
    """The reverse direction: every region the backend's pattern accepts
    beyond the four raycast regions (head/chest/hands/body, which are
    allowed to fall through to the generic reaction) should have a bespoke
    context. A pattern alternative with no context is a region you CAN
    send that the server can never react to specifically — most likely
    stale left over from a renamed or removed frontend event, or a region
    added to the pattern without ever wiring its reaction up. Either way
    it's drift worth flagging rather than silently carrying forever.
    """
    alternatives = set(_presence_region_pattern_alternatives())
    context_keys = _poke_region_context_keys()
    uncovered = sorted(
        r
        for r in alternatives
        if r not in _RAYCAST_REGIONS and r != "head_pat" and r not in context_keys
    )
    assert not uncovered, (
        "The following regions are accepted by PresenceRequest.region's "
        "pattern but have no bespoke server context (no "
        "_POKE_REGION_CONTEXT entry, and it isn't head_pat, which is "
        "special-cased) — they can be sent but always fall through to the "
        "generic poke reaction line, which is probably stale drift:\n"
        + "\n".join(f"  - {r!r}" for r in uncovered)
    )


# ---------------------------------------------------------------------------
# Contract 3: shared key/event contract
# ---------------------------------------------------------------------------

# Cross-module string contracts: a localStorage key or a window CustomEvent
# name that one file writes/dispatches and a different file reads/listens
# for. Nothing type-checks these — they're just matching string literals —
# so a rename on one side that misses the other compiles fine and fails
# silently at runtime. Keeping this list explicit (rather than trying to
# infer producer/consumer pairs automatically) means adding a new contract
# is a one-line addition here, and each line documents which two concepts
# are actually coupled.
#
# Format: (label, literal). Every literal must appear in >= 2 distinct
# frontend source files; a literal found in exactly one file means one side
# of a rename was missed.
_SHARED_STRING_CONTRACTS: List[tuple[str, str]] = [
    ("volume localStorage key", "kurisu_volume"),
    ("volume-changed event", "kurisu-volume-changed"),
    ("pet-preset localStorage key", "kurisu_pet_preset"),
    ("pet-preset-changed event", "kurisu-pet-preset-changed"),
    ("pet-perch-changed event", "kurisu-pet-perch-changed"),
    ("pet-alive heartbeat localStorage key", "kurisu_pet_alive"),
]


def _files_containing(literal: str) -> List[Path]:
    hits = []
    for path in _frontend_source_files():
        if literal in path.read_text(encoding="utf-8"):
            hits.append(path)
    return hits


@pytest.mark.parametrize(
    "label,literal",
    _SHARED_STRING_CONTRACTS,
    ids=[c[0] for c in _SHARED_STRING_CONTRACTS],
)
def test_shared_string_used_by_producer_and_consumer(label: str, literal: str) -> None:
    """A localStorage key or CustomEvent name is a string contract between
    a producer file (the thing that sets/dispatches it) and a consumer file
    (the thing that reads/listens for it) with nothing in the type system
    holding the two spellings together. If a literal like 'kurisu_volume'
    or 'kurisu-pet-preset-changed' appears in only one file, someone
    renamed one side of the pair and not the other — functionally
    identical to the poke-region drift (incident b), just entirely inside
    the frontend instead of crossing the frontend/backend boundary.
    """
    files = _files_containing(literal)
    assert len(files) >= 2, (
        f"{label} ({literal!r}) is expected to appear in at least two "
        "distinct frontend source files (a producer and a consumer) but "
        f"was found in {len(files)}: {[str(f) for f in files]}. This "
        "usually means one side of a producer/consumer rename was missed."
    )


# ---------------------------------------------------------------------------
# Contract: the three version sources move together
# ---------------------------------------------------------------------------


def test_version_sources_agree() -> None:
    """amadeus.__version__ (healthz/UI), pyproject.toml (package metadata),
    and tauri.conf.json (the version the UPDATER compares against
    latest.json) must be identical. v0.15.1 shipped a smoke run reporting
    version=0.15.0 because __init__.py was bumped nowhere — an installed
    app that self-reports the wrong version makes update verification and
    bug triage lie.
    """
    import json
    import re as _re

    import amadeus

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = _re.search(r'^version = "([^"]+)"', pyproject, flags=_re.MULTILINE)
    assert match, "pyproject.toml: could not find the version line"
    tauri_conf = json.loads(
        (REPO_ROOT / "frontend" / "src-tauri" / "tauri.conf.json").read_text(
            encoding="utf-8"
        )
    )
    assert amadeus.__version__ == match.group(1) == tauri_conf["version"], (
        f"Version sources disagree: amadeus.__version__={amadeus.__version__}, "
        f"pyproject={match.group(1)}, tauri.conf.json={tauri_conf['version']} "
        "— bump all three together."
    )


# ---------------------------------------------------------------------------
# Contract: wake-word wire strings cross the frontend/backend boundary
# ---------------------------------------------------------------------------


def test_wake_word_wire_strings_match_backend() -> None:
    """The /api/wake route path and its error-frame kinds are raw string
    contracts between the backend producers (routes_chat.py sends
    ``wake_disabled``, streaming/wake_word.py sends ``wake_unavailable``)
    and the frontend consumer (audio/wakeWord.ts maps them onto its
    latch-off-until-retoggle behavior). Nothing type-checks across that
    boundary — a rename on one side ships a wake toggle that silently never
    reports why it isn't working.
    """
    frontend = (REPO_ROOT / "frontend" / "src" / "audio" / "wakeWord.ts").read_text(
        encoding="utf-8"
    )
    backend = (REPO_ROOT / "amadeus" / "web" / "routes_chat.py").read_text(
        encoding="utf-8"
    ) + (REPO_ROOT / "amadeus" / "web" / "streaming" / "wake_word.py").read_text(
        encoding="utf-8"
    )
    for literal in ("/api/wake", "wake_disabled", "wake_unavailable"):
        assert literal in frontend, f"frontend wakeWord.ts lost {literal!r}"
        assert literal in backend, f"backend wake path lost {literal!r}"


# ---------------------------------------------------------------------------
# Contract 4: ambient kind contract
# ---------------------------------------------------------------------------


def _ambient_request_pattern_alternatives() -> List[str]:
    """Parses AmbientRequest.kind's `pattern="^(a|b|c)$"` alternation."""
    models_py = REPO_ROOT / "amadeus" / "web" / "models.py"
    text = models_py.read_text(encoding="utf-8")
    match = re.search(
        r"class AmbientRequest\(BaseModel\).*?pattern\s*=\s*\"\^\(([^)]+)\)\$\"",
        text,
        re.DOTALL,
    )
    assert match, (
        f"{models_py}: could not find AmbientRequest.kind's "
        '`pattern="^(...)$"`. If the field was reshaped, update '
        "_ambient_request_pattern_alternatives() rather than skipping this "
        "contract."
    )
    return match.group(1).split("|")


def _presence_ambient_kinds() -> Set[str]:
    """String literals in presence.py's `_AMBIENT_KINDS = {...}` set."""
    presence_py = REPO_ROOT / "amadeus" / "presence.py"
    text = presence_py.read_text(encoding="utf-8")
    match = re.search(r"_AMBIENT_KINDS\s*=\s*\{([^}]+)\}", text)
    assert match, f"{presence_py}: _AMBIENT_KINDS set literal not found"
    return set(re.findall(r"[\"']([a-z_]+)[\"']", match.group(1)))


def _frontend_ambient_kinds() -> Set[str]:
    """Members of ambient.ts's `export type AmbientKind = 'a' | 'b'` union."""
    ambient_ts = FRONTEND_SRC / "ambient.ts"
    text = ambient_ts.read_text(encoding="utf-8")
    match = re.search(r"export type AmbientKind\s*=\s*([^\n]+)", text)
    assert match, f"{ambient_ts}: AmbientKind union not found"
    return set(re.findall(r"'([a-z_]+)'", match.group(1)))


def test_ambient_kinds_agree_across_all_three_layers() -> None:
    """The ambient kind list is hand-copied into THREE places — the frontend
    union (ambient.ts), the Pydantic regex (models.py), and the server-side
    whitelist (presence._AMBIENT_KINDS) — and a partial edit fails SILENTLY:
    the POST 422s at the API boundary and useAmbientSignals swallows the
    error by design (an ambient remark is a nice-to-have, not worth an error
    toast).

    That is precisely the archetype Contract 2 was written for after the
    poke regions 422'd unnoticed for a week; ambient shipped with the same
    three-copy shape and no test. This closes it.
    """
    frontend = _frontend_ambient_kinds()
    pydantic = set(_ambient_request_pattern_alternatives())
    presence = _presence_ambient_kinds()

    assert frontend, "no ambient kinds parsed out of ambient.ts"
    assert frontend == pydantic, (
        "ambient.ts AmbientKind and AmbientRequest.kind's regex disagree: "
        f"frontend-only={sorted(frontend - pydantic)}, "
        f"backend-only={sorted(pydantic - frontend)}. A frontend-only kind "
        "422s silently; a backend-only one is simply dead."
    )
    assert frontend == presence, (
        "ambient.ts AmbientKind and presence._AMBIENT_KINDS disagree: "
        f"frontend-only={sorted(frontend - presence)}, "
        f"backend-only={sorted(presence - frontend)}. A kind that clears the "
        "Pydantic regex but misses _AMBIENT_KINDS is dropped unanswered — "
        "the quietest failure of the three."
    )


def test_rhythm_is_not_an_ambient_kind() -> None:
    """Rhythm (amadeus/rhythm.py) deliberately owns /api/rhythm rather than
    riding /api/ambient: it is a rollup that wants no line back, and routing
    it through ambient_comment would drag it into that path's LLM call and
    cooldown. Pinned so a future refactor doesn't "simplify" it back in.
    """
    assert "rhythm" not in _frontend_ambient_kinds()
    assert "rhythm" not in _presence_ambient_kinds()
    assert "rhythm" not in set(_ambient_request_pattern_alternatives())
