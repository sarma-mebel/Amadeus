"""Character switching endpoints: list packs, switch the active character.

Switching rides the exact same persist path as the settings API — merge the
pack's config patch into the user config, hot-apply via ``reload_providers``.
Combined with the qwen3 in-place voice rebind (sessions._try_voice_rebind),
a pack-to-pack switch takes seconds: persona prompt and identity swap on the
next turn, the voice re-clones on the loaded engine, and the frontend picks
up the new name/avatar from /api/status.

The built-in Kurisu is restorable because the first switch away from her
snapshots the identity-relevant config fields (character section + voice
reference fields + spoken language) to a JSON file; switching to ``kurisu``
restores that snapshot and removes the character section.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..characters import _repo_root as characters_root
from ..characters import install_character_pack
from ..config import Config
from ..logging_utils import get_logger
from . import settings_api
from .sessions import SessionStore

LOGGER = get_logger(__name__)

BUILTIN_ID = "kurisu"
PACK_ID_RE = re.compile(r"^[a-z0-9_]{1,64}$")

# Config leaves that constitute "who she is" for switching purposes. The
# snapshot records their raw-user-config values (or absence) so switching
# back to the built-in restores exactly the pre-pack state.
IDENTITY_LEAVES = (
    ("character",),
    ("providers", "qwen3_tts_ref_audio"),
    ("providers", "qwen3_tts_ref_text"),
    ("providers", "qwen3_tts_language"),
    ("audio", "voice_language"),
)


def _repo_root() -> Path:
    # Delegates to characters.py so the packaged app's AMADEUS_STATIC_DIR
    # override applies identically to listing and installing.
    return characters_root()


def _packs_dir() -> Path:
    return _repo_root() / "characters" / "packs"


def _builtin_snapshot_path() -> Path:
    return _repo_root() / "characters" / "installed" / "_builtin_identity.json"


def _get_leaf(raw: Dict[str, Any], path: tuple) -> tuple[bool, Any]:
    node: Any = raw
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False, None
        node = node[key]
    return True, node


def _delete_leaf(raw: Dict[str, Any], path: tuple) -> None:
    node: Any = raw
    for key in path[:-1]:
        if not isinstance(node, dict) or key not in node:
            return
        node = node[key]
    if isinstance(node, dict):
        node.pop(path[-1], None)


def _set_leaf(raw: Dict[str, Any], path: tuple, value: Any) -> None:
    node = raw
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def _snapshot_builtin_identity(raw: Dict[str, Any]) -> None:
    """Record the identity leaves of the pre-pack user config, once."""
    path = _builtin_snapshot_path()
    if path.exists():
        return
    snapshot = []
    for leaf in IDENTITY_LEAVES:
        present, value = _get_leaf(raw, leaf)
        snapshot.append({"path": list(leaf), "present": present, "value": value})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Built-in identity snapshot written to %s", path)


def _reset_identity_leaves(raw: Dict[str, Any]) -> None:
    """Return the identity leaves to their pre-pack (snapshot) state."""
    snapshot_path = _builtin_snapshot_path()
    entries = []
    if snapshot_path.exists():
        try:
            entries = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            entries = []
    recorded = {tuple(e["path"]): e for e in entries if isinstance(e, dict)}
    for leaf in IDENTITY_LEAVES:
        entry = recorded.get(leaf)
        if entry and entry.get("present"):
            _set_leaf(raw, leaf, entry.get("value"))
        else:
            _delete_leaf(raw, leaf)


def _persist_path() -> Path:
    """The config file the NEXT boot will actually read.

    settings_api.user_config_path() alone is wrong on dev setups: boot takes
    the first existing DEFAULT_CONFIG_LOCATIONS entry, so when a CWD
    config.local.yaml/config.yaml exists, a switch persisted only to
    ~/.config would silently revert on restart. Never targets the bundled
    defaults (the last two search entries)."""
    import os

    override = os.getenv("AMADEUS_USER_CONFIG")
    if override:
        return Path(override).expanduser()
    from ..config import DEFAULT_CONFIG_LOCATIONS

    for candidate in DEFAULT_CONFIG_LOCATIONS[:3]:
        if candidate.exists():
            return candidate
    return settings_api.user_config_path()


def _persist_identity(store: SessionStore, raw: Dict[str, Any]) -> Config:
    """persist_and_reload, but with the identity leaves pinned to exactly
    what ``raw`` says. The settings path merges the user config over the
    runtime config, and a merge cannot express deletion — yet restoring the
    built-in Kurisu IS a deletion (drop the character section and the pack's
    voice overrides so the defaults return)."""
    settings_api.write_raw(_persist_path(), raw)
    merged = settings_api.deep_merge_config(dataclasses.asdict(store.config), raw)
    for leaf in IDENTITY_LEAVES:
        present, value = _get_leaf(raw, leaf)
        if present:
            _set_leaf(merged, leaf, value)
        else:
            _delete_leaf(merged, leaf)
    new_config = Config.from_dict(merged)
    new_config.resolve_user_data_paths()
    new_config.validate()
    store.reload_providers(new_config)
    return new_config


def _pack_manifest(pack_dir: Path) -> Optional[Dict[str, Any]]:
    manifest_path = pack_dir / "character.yaml"
    if not manifest_path.is_file():
        return None
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


class SwitchRequest(BaseModel):
    id: str


def build_router(store: SessionStore) -> APIRouter:
    router = APIRouter()

    def _active_id() -> str:
        character = getattr(store.config, "character", None)
        if not character or not getattr(character, "is_custom", False):
            return BUILTIN_ID
        name = (getattr(character, "name", "") or "").strip().lower()
        for pack_dir in sorted(_packs_dir().iterdir() if _packs_dir().is_dir() else []):
            manifest = _pack_manifest(pack_dir)
            if manifest and str(manifest.get("name", "")).strip().lower() == name:
                return pack_dir.name
        return name or BUILTIN_ID

    @router.get("/api/characters")
    async def list_characters() -> Dict[str, Any]:
        active = _active_id()
        entries: List[Dict[str, Any]] = [
            {
                "id": BUILTIN_ID,
                "name": "Kurisu",
                "full_name": "Makise Kurisu",
                "model_id": None,
                "has_voice": True,
                "builtin": True,
                "active": active == BUILTIN_ID,
            }
        ]
        packs_dir = _packs_dir()
        if packs_dir.is_dir():
            for pack_dir in sorted(packs_dir.iterdir()):
                if not PACK_ID_RE.match(pack_dir.name):
                    continue
                manifest = _pack_manifest(pack_dir)
                if not manifest:
                    continue
                voice = manifest.get("voice") or {}
                has_voice = bool(
                    isinstance(voice, dict)
                    and voice.get("ref_audio")
                    and (pack_dir / str(voice["ref_audio"])).is_file()
                )
                entries.append(
                    {
                        "id": pack_dir.name,
                        "name": str(manifest.get("name", pack_dir.name)),
                        "full_name": str(
                            manifest.get("full_name", manifest.get("name", ""))
                        ),
                        "model_id": manifest.get("model_id"),
                        "has_voice": has_voice,
                        "builtin": False,
                        "active": active == pack_dir.name,
                    }
                )
        return {"characters": entries, "active": active}

    @router.post("/api/characters/switch")
    async def switch_character(request: SwitchRequest) -> Dict[str, Any]:
        target = request.id.strip().lower()
        if not PACK_ID_RE.match(target):
            raise HTTPException(status_code=400, detail="Invalid character id.")

        raw = settings_api.load_raw(_persist_path())

        if _active_id() == BUILTIN_ID:
            _snapshot_builtin_identity(raw)

        if target == BUILTIN_ID:
            _reset_identity_leaves(raw)
        else:
            pack_dir = _packs_dir() / target
            if not _pack_manifest(pack_dir):
                raise HTTPException(
                    status_code=404, detail=f"Unknown character pack: {target}"
                )
            _reset_identity_leaves(raw)
            try:
                result = install_character_pack(pack_dir)
            except (ValueError, FileNotFoundError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            raw = settings_api.deep_merge_config(raw, result.config_patch)

        try:
            new_config = _persist_identity(store, raw)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Switch failed to apply: {exc}"
            ) from exc

        character = getattr(new_config, "character", None)
        LOGGER.info("Character switched to %s", target)
        return {
            "active": target,
            "character": {
                "name": getattr(character, "name", "Kurisu"),
                "full_name": getattr(character, "full_name", "Makise Kurisu"),
                "custom": bool(getattr(character, "is_custom", False)),
                "model_id": getattr(character, "model_id", None),
            },
            "tts_warming": bool(store.tts_warming),
        }

    return router
