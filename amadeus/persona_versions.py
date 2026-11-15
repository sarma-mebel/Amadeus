"""Persona/voice identity versioning with rollback.

Why this exists: research on companion apps (the Replika identity-
discontinuity studies) shows users form attachments to a *specific* persona
+ voice, and silent changes to it are experienced as loss — while offering
a rollback to the previous identity measurably softens the reaction. So:
every identity-affecting settings change auto-snapshots the previous state,
changes are announced once in the next greeting, and any stored version can
be restored.

What counts as "identity": the persona dials (PersonaSettings) plus the
voice construction settings (TTS provider + its per-provider fields).
Deliberately excluded: the LLM/model choice (brain, not persona), STT,
embeddings, TTS *cache* knobs, and — critically — API keys/secrets, which
must never be persisted into this store (do not reuse the session layer's
``_tts_fingerprint`` here; it hashes secrets because they affect provider
construction).

Storage mirrors ``memory/profile.py``: one JSON file in the platform data
dir, tolerant load (corrupt/missing → empty store), atomic writes.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config
from .logging_utils import get_logger

LOGGER = get_logger(__name__)

VERSIONS_SCHEMA = 1
MAX_UNPINNED = 20

# Voice fields on ProviderSettings that are NOT part of the voice identity.
_NON_IDENTITY_PROVIDER_FIELDS = {
    "llm",
    "stt",
    "embedding",
    "tts_cache_enabled",
    "tts_cache_dir",
    "tts_cache_max_chars",
}

# Humanized labels for the change announcement.
_FIELD_LABELS = {
    "voice.tts": "voice",
    "persona.tsundere_level": "tsundere level",
    "persona.scientific_formality": "formality",
    "persona.emotional_guardedness": "guardedness",
    "persona.speaking_rate": "speaking rate",
    "persona.pitch_shift": "pitch",
    "persona.safe_mode": "safe mode",
}


def identity_payload(config: Config) -> Dict[str, Any]:
    """The persona+voice snapshot: everything identity-relevant, no secrets."""
    persona = dataclasses.asdict(config.persona)
    voice = {
        key: value
        for key, value in dataclasses.asdict(config.providers).items()
        if key not in _NON_IDENTITY_PROVIDER_FIELDS
    }
    return {"persona": persona, "voice": voice}


def identity_fingerprint(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _flatten(payload: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for section, values in payload.items():
        if isinstance(values, dict):
            for key, value in values.items():
                flat[f"{section}.{key}"] = value
        else:
            flat[section] = values
    return flat


def describe_diff(old: Dict[str, Any], new: Dict[str, Any], limit: int = 3) -> str:
    """Short human phrasing of what changed between two identity payloads."""
    flat_old, flat_new = _flatten(old), _flatten(new)
    changes: List[str] = []
    for key in sorted(set(flat_old) | set(flat_new)):
        before, after = flat_old.get(key), flat_new.get(key)
        if before == after:
            continue
        label = _FIELD_LABELS.get(key, key.split(".", 1)[-1].replace("_", " "))
        if before is None or after is None or key.startswith("voice."):
            # Voice sub-fields (model paths, reference clips, sampling knobs)
            # are noise as raw values; name the dial, not the plumbing.
            changes.append(
                f"{label}: {before} -> {after}"
                if key in _FIELD_LABELS
                else f"{label} changed"
            )
        else:
            changes.append(f"{label}: {before} -> {after}")
    if not changes:
        return ""
    shown = changes[:limit]
    extra = len(changes) - len(shown)
    summary = "; ".join(dict.fromkeys(shown))  # dedupe "x changed" repeats
    if extra > 0:
        summary += f" (+{extra} more)"
    return summary


@dataclass
class PersonaVersion:
    id: str
    name: str
    created_at: str
    pinned: bool
    fingerprint: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> Optional["PersonaVersion"]:
        if not isinstance(data, dict):
            return None
        payload = data.get("payload")
        fingerprint = data.get("fingerprint")
        if not isinstance(payload, dict) or not fingerprint:
            return None
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:8]),
            name=str(data.get("name") or "unnamed"),
            created_at=str(data.get("created_at") or ""),
            pinned=bool(data.get("pinned", False)),
            fingerprint=str(fingerprint),
            payload=payload,
        )


class PersonaVersionStore:
    """JSON-backed version list + the last-announced fingerprint.

    Thread-safety: a single lock guards read-modify-write cycles; writers
    reload from disk first so two processes (web + CLI) sharing the file
    stay coherent enough for this low-write workload.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.versions: List[PersonaVersion] = []
        self.last_announced_fingerprint: Optional[str] = None
        self._load()

    def _load(self) -> None:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        if not isinstance(data, dict):
            LOGGER.warning("Persona version store malformed; starting empty")
            return
        raw_versions = data.get("versions")
        if isinstance(raw_versions, list):
            self.versions = [
                version
                for version in (PersonaVersion.from_dict(v) for v in raw_versions)
                if version is not None
            ]
        announced = data.get("last_announced_fingerprint")
        self.last_announced_fingerprint = str(announced) if announced else None

    def save_atomic(self) -> None:
        payload = {
            "version": VERSIONS_SCHEMA,
            "last_announced_fingerprint": self.last_announced_fingerprint,
            "versions": [v.to_dict() for v in self.versions],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_path, self.path)

    def get(self, version_id: str) -> Optional[PersonaVersion]:
        with self._lock:
            for version in self.versions:
                if version.id == version_id:
                    return version
        return None

    def find_by_fingerprint(self, fingerprint: str) -> Optional[PersonaVersion]:
        with self._lock:
            for version in self.versions:
                if version.fingerprint == fingerprint:
                    return version
        return None

    def snapshot(
        self,
        payload: Dict[str, Any],
        *,
        name: Optional[str] = None,
        pinned: bool = False,
    ) -> PersonaVersion:
        """Store a snapshot; idempotent per fingerprint (auto-snapshot spam
        on repeated saves of the same state collapses to one entry)."""
        fingerprint = identity_fingerprint(payload)
        with self._lock:
            self._load()
            for version in self.versions:
                if version.fingerprint == fingerprint:
                    if name and version.name.startswith("auto "):
                        version.name = name
                    if pinned:
                        version.pinned = True
                    self.save_atomic()
                    return version
            now = dt.datetime.now()
            version = PersonaVersion(
                id=uuid.uuid4().hex[:8],
                name=name or f"auto {now.strftime('%Y-%m-%d %H:%M')}",
                created_at=now.isoformat(timespec="seconds"),
                pinned=pinned,
                fingerprint=fingerprint,
                payload=payload,
            )
            self.versions.append(version)
            self._prune_locked()
            self.save_atomic()
            return version

    def _prune_locked(self) -> None:
        unpinned = [v for v in self.versions if not v.pinned]
        overflow = len(unpinned) - MAX_UNPINNED
        if overflow <= 0:
            return
        drop = {id(v) for v in unpinned[:overflow]}  # oldest first (append order)
        self.versions = [v for v in self.versions if id(v) not in drop]

    def set_pinned(self, version_id: str, pinned: bool) -> Optional[PersonaVersion]:
        with self._lock:
            self._load()
            for version in self.versions:
                if version.id == version_id:
                    version.pinned = pinned
                    self.save_atomic()
                    return version
        return None

    def delete(self, version_id: str) -> str:
        """Returns "deleted", "pinned" (refused), or "missing"."""
        with self._lock:
            self._load()
            for version in self.versions:
                if version.id == version_id:
                    if version.pinned:
                        return "pinned"
                    self.versions.remove(version)
                    self.save_atomic()
                    return "deleted"
        return "missing"

    # -- Change announcement -------------------------------------------------

    def pending_change_note(self, config: Config) -> Optional[str]:
        """Human summary of an unannounced identity change, or None.

        First-ever run (nothing announced yet) deliberately returns None —
        there is no "change from nothing" to announce; ``mark_announced``
        still records the baseline.
        """
        if self.last_announced_fingerprint is None:
            return None
        current = identity_payload(config)
        if identity_fingerprint(current) == self.last_announced_fingerprint:
            return None
        announced = self.find_by_fingerprint(self.last_announced_fingerprint)
        old_payload = announced.payload if announced else {}
        return describe_diff(old_payload, current) or "settings changed"

    def mark_announced(self, config: Config) -> None:
        with self._lock:
            # Reload-before-mutate like every other writer here: saving the
            # in-memory state verbatim would clobber snapshots another
            # process (web + CLI share the file) appended since our load.
            self._load()
            self.last_announced_fingerprint = identity_fingerprint(
                identity_payload(config)
            )
            self.save_atomic()
