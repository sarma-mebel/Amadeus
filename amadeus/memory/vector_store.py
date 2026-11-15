"""Vector store abstractions for long-term memory."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .. import telemetry
from ..logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class VectorRecord:
    """Single vector memory entry."""

    vector: List[float]
    text: str
    metadata: Dict[str, object] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class VectorMatch:
    """Result returned by a similarity search."""

    score: float
    text: str
    metadata: Dict[str, object]
    id: Optional[str] = None
    # Effective recall confidence in [0, 1], attached by recall_detailed and
    # consumed by the render seam (memory/context.py::_recall_line). TRANSIENT
    # by contract: never persisted to metadata — it is derived from `strength`,
    # which _reinforce owns, and storing it would create a second source of
    # truth that drifts on every retrieval. None on any path that did not
    # compute it (raw similarity_search callers, legacy records).
    confidence: Optional[float] = None


class VectorStore:
    """Storage interface for vector-based recall."""

    def add(self, record: VectorRecord) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def count(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def dump(self) -> Iterable[VectorRecord]:  # pragma: no cover - interface
        raise NotImplementedError

    def get(
        self, record_id: str
    ) -> Optional[VectorRecord]:  # pragma: no cover - interface
        """Fetch a single record by id, or None if absent. Backends should
        implement this as a targeted lookup rather than a full dump/scan."""
        raise NotImplementedError

    def similarity_search(
        self, query: Sequence[float], limit: int
    ) -> List[VectorMatch]:  # pragma: no cover - interface
        raise NotImplementedError

    def delete(self, record_id: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def update_metadata(
        self, record_id: str, metadata: Dict[str, object]
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def persist(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def is_empty(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def clear(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    """JSON-backed vector store used by default."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else None
        self._records: List[VectorRecord] = []
        if self.path and self.path.exists():
            self._load()

    def add(self, record: VectorRecord) -> None:
        self._records.append(record)

    def count(self) -> int:
        return len(self._records)

    def dump(self) -> Iterable[VectorRecord]:
        return list(self._records)

    def get(self, record_id: str) -> Optional[VectorRecord]:
        if not record_id:
            return None
        for rec in self._records:
            if rec.id == record_id:
                return rec
        return None

    def similarity_search(
        self, query: Sequence[float], limit: int
    ) -> List[VectorMatch]:
        if not self._records or not query:
            return []
        scores = []
        for item in self._records:
            score = _cosine_similarity(query, item.vector)
            scores.append(
                VectorMatch(
                    score=score, text=item.text, metadata=item.metadata, id=item.id
                )
            )
        scores.sort(key=lambda match: match.score, reverse=True)
        return scores[:limit]

    def delete(self, record_id: str) -> None:
        self._records = [rec for rec in self._records if rec.id != record_id]

    def update_metadata(self, record_id: str, metadata: Dict[str, object]) -> None:
        for rec in self._records:
            if rec.id == record_id:
                rec.metadata = dict(metadata)
                return

    def persist(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "id": rec.id,
                "vector": rec.vector,
                "text": rec.text,
                "metadata": rec.metadata,
            }
            for rec in self._records
        ]
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp_path, self.path)

    def is_empty(self) -> bool:
        return not self._records

    def clear(self) -> None:
        self._records.clear()
        self.persist()

    def _load(self) -> None:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except FileNotFoundError:
            # Normal on a fresh install — not a failure.
            self._records = []
            return
        except json.JSONDecodeError as exc:
            self._quarantine_corrupt(exc)
            return
        try:
            self._records = [
                VectorRecord(
                    vector=[float(x) for x in entry.get("vector", [])],
                    text=str(entry.get("text", "")),
                    metadata=entry.get("metadata", {}) or {},
                    # Legacy rows predate ids; assign fresh ones on load.
                    id=str(entry.get("id") or uuid.uuid4().hex),
                )
                for entry in raw or []
            ]
        except (TypeError, AttributeError, ValueError, KeyError) as exc:
            # Valid JSON, wrong shape ({"a": 1} where a list of record dicts
            # is expected, non-numeric vector entries, ...) is corruption
            # too — same quarantine contract as unparseable JSON, never a
            # crashed construction.
            self._records = []
            self._quarantine_corrupt(exc)

    def _quarantine_corrupt(self, reason: BaseException) -> None:
        # Corruption = silent total loss of long-term memory if we just
        # "start fresh" (and the next persist() would overwrite the
        # evidence). Keep the corrupt file, degrade loudly.
        telemetry.incr("memory.vector_store_load_corrupt")
        backup = self.path.with_name(self.path.name + ".bak-corrupt")
        try:
            os.replace(self.path, backup)
            LOGGER.error(
                "Vector store %s is corrupt (%s); starting fresh. "
                "The unreadable file was kept at %s",
                self.path,
                reason,
                backup,
            )
        except OSError:
            LOGGER.error(
                "Vector store %s is corrupt (%s) and could not be backed "
                "up; starting fresh",
                self.path,
                reason,
            )
        self._records = []


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed persistent vector store."""

    def __init__(
        self, path: Optional[Path] = None, collection: str = "amadeus-memory"
    ) -> None:
        try:
            import chromadb  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "ChromaVectorStore requires the 'chromadb' package. Install with pip install \"amadeus-kurisu[memory]\"."
            ) from exc

        if path:
            self._client = chromadb.PersistentClient(path=str(path))
        else:  # pragma: no cover - in-memory Chroma
            self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, record: VectorRecord) -> None:
        self._collection.add(
            ids=[record.id],
            documents=[record.text],
            embeddings=[record.vector],
            metadatas=[record.metadata],
        )

    def count(self) -> int:
        return self._collection.count()

    def dump(self) -> Iterable[VectorRecord]:
        if self.count() == 0:  # pragma: no cover - empty shortcut
            return []
        result = self._collection.get(include=["documents", "embeddings", "metadatas"])
        records: List[VectorRecord] = []
        for record_id, vector, text, metadata in zip(
            result.get("ids", []) or [],
            result.get("embeddings", []) or [],
            result.get("documents", []) or [],
            result.get("metadatas", []) or [],
        ):
            records.append(
                VectorRecord(
                    vector=[float(x) for x in vector],
                    text=str(text),
                    metadata=metadata or {},
                    id=str(record_id),
                )
            )
        return records

    def get(self, record_id: str) -> Optional[VectorRecord]:
        if not record_id:
            return None
        result = self._collection.get(
            ids=[record_id], include=["documents", "embeddings", "metadatas"]
        )
        ids = result.get("ids") or []
        if not ids:
            return None
        vector = (result.get("embeddings") or [[]])[0]
        text = (result.get("documents") or [""])[0]
        metadata = (result.get("metadatas") or [{}])[0]
        return VectorRecord(
            vector=[float(x) for x in vector],
            text=str(text),
            metadata=metadata or {},
            id=str(ids[0]),
        )

    def similarity_search(
        self, query: Sequence[float], limit: int
    ) -> List[VectorMatch]:
        if not query:
            return []
        res = self._collection.query(
            query_embeddings=[list(query)],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )
        matches: List[VectorMatch] = []
        # Chroma always returns ids for queries; they cannot be requested via
        # the include list.
        ids = (res.get("ids") or [[]])[0]
        documents = (res.get("documents") or [[]])[0]
        metadatas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        for record_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            # Chroma stores cosine distance; convert to similarity in [0,1].
            score = 1.0 - float(distance)
            matches.append(
                VectorMatch(
                    score=score,
                    text=str(text),
                    metadata=metadata or {},
                    id=str(record_id),
                )
            )
        matches.sort(key=lambda match: match.score, reverse=True)
        return matches

    def delete(self, record_id: str) -> None:
        self._collection.delete(ids=[record_id])

    def update_metadata(self, record_id: str, metadata: Dict[str, object]) -> None:
        self._collection.update(ids=[record_id], metadatas=[metadata])

    def persist(self) -> None:
        persist_fn = getattr(self._client, "persist", None)
        if callable(persist_fn):
            persist_fn()

    def is_empty(self) -> bool:
        return self.count() == 0

    def clear(self) -> None:
        if self.is_empty():  # pragma: no cover - fast exit
            return
        # Newer chromadb rejects delete(where={}); delete by explicit ids.
        all_ids = self._collection.get(include=[]).get("ids", [])
        if all_ids:
            self._collection.delete(ids=all_ids)
        self.persist()


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for va, vb in zip(a, b):
        dot += va * vb
        norm_a += va * va
        norm_b += vb * vb
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / ((norm_a**0.5) * (norm_b**0.5))
