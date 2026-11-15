"""Persona/voice version endpoints: list, snapshot, pin, delete, rollback.

The identity-continuity feature's API half (see ``amadeus/persona_versions``
for the why). Rollback rides the exact same persist path as the settings
API — write the user config YAML, rebuild the runtime config, hot-apply via
``reload_providers`` (which skips the expensive TTS rebuild when the voice
fingerprint didn't actually change).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import Config
from ..logging_utils import get_logger
from ..persona_versions import (
    PersonaVersion,
    PersonaVersionStore,
    describe_diff,
    identity_fingerprint,
    identity_payload,
)
from . import settings_api
from .sessions import SessionStore

LOGGER = get_logger(__name__)


class VersionInfo(BaseModel):
    id: str
    name: str
    created_at: str
    pinned: bool
    fingerprint: str
    is_current: bool
    diff_vs_current: Optional[str] = None


class VersionListResponse(BaseModel):
    versions: List[VersionInfo]
    current_fingerprint: str


class SnapshotRequest(BaseModel):
    name: Optional[str] = Field(default=None, max_length=60)


class PinRequest(BaseModel):
    pinned: bool


def _store_or_503(config: Config) -> PersonaVersionStore:
    path = getattr(config.memory, "persona_versions_path", None)
    if not path:
        raise HTTPException(status_code=503, detail="Persona versioning is disabled")
    try:
        return PersonaVersionStore(Path(path))
    except Exception as exc:  # pragma: no cover - store init failure
        raise HTTPException(
            status_code=503, detail=f"Persona version store unavailable: {exc}"
        ) from exc


def _version_info(version: PersonaVersion, current: Dict[str, Any]) -> VersionInfo:
    current_fp = identity_fingerprint(current)
    is_current = version.fingerprint == current_fp
    return VersionInfo(
        id=version.id,
        name=version.name,
        created_at=version.created_at,
        pinned=version.pinned,
        fingerprint=version.fingerprint,
        is_current=is_current,
        diff_vs_current=(
            None if is_current else (describe_diff(version.payload, current) or None)
        ),
    )


def build_router(store: SessionStore) -> APIRouter:
    router = APIRouter()

    @router.get("/api/persona/versions", response_model=VersionListResponse)
    async def list_versions() -> VersionListResponse:
        vstore = _store_or_503(store.config)
        current = identity_payload(store.config)
        return VersionListResponse(
            versions=[_version_info(v, current) for v in vstore.versions],
            current_fingerprint=identity_fingerprint(current),
        )

    @router.post("/api/persona/versions", response_model=VersionInfo)
    async def snapshot_current(payload: SnapshotRequest) -> VersionInfo:
        vstore = _store_or_503(store.config)
        current = identity_payload(store.config)
        version = vstore.snapshot(current, name=payload.name, pinned=bool(payload.name))
        return _version_info(version, current)

    @router.put("/api/persona/versions/{version_id}/pin", response_model=VersionInfo)
    async def pin_version(version_id: str, payload: PinRequest) -> VersionInfo:
        vstore = _store_or_503(store.config)
        version = vstore.set_pinned(version_id, payload.pinned)
        if version is None:
            raise HTTPException(status_code=404, detail="Unknown version")
        return _version_info(version, identity_payload(store.config))

    @router.delete("/api/persona/versions/{version_id}")
    async def delete_version(version_id: str) -> Dict[str, Any]:
        vstore = _store_or_503(store.config)
        outcome = vstore.delete(version_id)
        if outcome == "missing":
            raise HTTPException(status_code=404, detail="Unknown version")
        if outcome == "pinned":
            raise HTTPException(
                status_code=409, detail="Unpin this version before deleting it"
            )
        return {"deleted": version_id}

    @router.post(
        "/api/persona/versions/{version_id}/rollback",
        response_model=settings_api.SettingsUpdateResponse,
    )
    async def rollback_version(
        version_id: str,
    ) -> settings_api.SettingsUpdateResponse:
        vstore = _store_or_503(store.config)
        version = vstore.get(version_id)
        if version is None:
            raise HTTPException(status_code=404, detail="Unknown version")

        # Keep the pre-rollback identity restorable too — rolling back is
        # itself an identity change the user might want to undo.
        pre_rollback = identity_payload(store.config)
        if identity_fingerprint(pre_rollback) != version.fingerprint:
            vstore.snapshot(pre_rollback)

        settings_path = settings_api.user_config_path()
        raw = {
            **settings_api.effective_raw(store.config),
            **settings_api.load_raw(settings_path),
        }
        # Merge the snapshot at the raw-YAML level: the settings UI can only
        # edit providers.tts itself, but hand-edited voice sub-fields
        # (reference clips, sampling knobs) must restore too.
        raw["persona"] = {
            **raw.get("persona", {}),
            **version.payload.get("persona", {}),
        }
        raw["providers"] = {
            **raw.get("providers", {}),
            **version.payload.get("voice", {}),
        }

        old_tts = store.config.providers.tts
        new_config = settings_api.persist_and_reload(store, raw)
        needs_restart = new_config.providers.tts != old_tts

        return settings_api.SettingsUpdateResponse(
            settings=settings_api.build_settings_response(new_config, llm=store.llm),
            needs_restart=needs_restart,
            restart_reason="Voice provider changed" if needs_restart else None,
        )

    return router
