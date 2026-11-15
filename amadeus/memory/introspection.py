"""Read-only introspection views over a MemoryManager.

Free functions taking the manager (the sibling-module convention here —
see context.py/backup.py): ``memory_overview`` backs "what do you remember
about me?" and the MemoryPanel, ``memory_graph`` backs the UI "synapse
map". MemoryManager keeps thin delegating methods with the same names.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List

from .profile import tone_trend
from .vector_store import VectorRecord

if TYPE_CHECKING:  # pragma: no cover - import cycle guard (manager imports us)
    from .manager import MemoryManager

# Bounds for the synapse-map graph: caps the O(n^2) similarity pass and
# keeps the payload/canvas readable.
_GRAPH_MAX_RECORDS = 80
_GRAPH_NEIGHBORS = 2
_GRAPH_EDGE_FLOOR = 0.2
_GRAPH_ATTACH_FLOOR = 0.15


def memory_overview(manager: "MemoryManager") -> Dict[str, Any]:
    """Everything Kurisu remembers, for "what do you remember about me?"."""
    overview: Dict[str, Any] = {
        "profile": {},
        "facts": [],
        "episodes": [],
        "fact_records": [],
        "episode_records": [],
        "counts": {"facts": 0, "episodes": 0, "reflections": 0, "moments": 0},
        "relationship": {},
    }
    with manager._state_lock:
        if manager.profile is not None:
            overview["profile"] = {
                "facts": dict(manager.profile.facts),
                "session_count": manager.profile.session_count,
                "first_met": manager.profile.first_met,
                "open_threads": list(manager.profile.open_threads),
                "last_episode": manager.profile.last_episode,
            }
            overview["relationship"] = {
                "trust_level": round(manager.profile.trust_level, 3),
                "arc": {
                    "curiosity_vs_annoyance": round(
                        manager.profile.arc_curiosity_vs_annoyance, 3
                    ),
                    "logic_vs_emotion": round(manager.profile.arc_logic_vs_emotion, 3),
                    "pride_vs_vulnerability": round(
                        manager.profile.arc_pride_vs_vulnerability, 3
                    ),
                },
                "tone_history": list(manager.profile.tone_history[-10:]),
                "tone_trend": tone_trend(manager.profile.tone_history),
                "last_felt": (
                    dict(manager.profile.last_felt)
                    if manager.profile.last_felt
                    else None
                ),
            }
        if manager.vector_store is not None:
            records = list(manager.vector_store.dump())
            for record in records:
                kind = (record.metadata or {}).get("type")
                key = {
                    "fact": "facts",
                    "episode": "episodes",
                    "reflection": "reflections",
                    "moment": "moments",
                }.get(str(kind))
                if key:
                    overview["counts"][key] += 1

            def _record_payload(rec: VectorRecord) -> Dict[str, Any]:
                # timestamp = learn time; valid_at = optional event time
                # (bitemporal split — see fact_archive.store_fact). Extra
                # keys are frontend-safe: MemoryPanel destructures explicitly.
                return {
                    "id": rec.id,
                    "text": rec.text,
                    "pinned": manager._is_protected(rec.metadata),
                    "timestamp": (rec.metadata or {}).get("timestamp"),
                    "valid_at": (rec.metadata or {}).get("valid_at"),
                }

            # Moments ride the fact view rather than getting their own list:
            # they are durable, recallable, editable records in every respect
            # a user cares about, and docs/companion-ethics.md promises a
            # stored moment "shows up in the memory panel". Filtering on
            # type=="fact" alone made that promise false for every moment
            # ever stored.
            fact_records = [
                rec
                for rec in records
                if (rec.metadata or {}).get("type") in ("fact", "moment")
            ]
            fact_records.sort(
                key=lambda rec: str((rec.metadata or {}).get("timestamp") or ""),
                reverse=True,
            )
            fact_records = fact_records[:20]
            overview["fact_records"] = [_record_payload(rec) for rec in fact_records]
            # Legacy plain-text view, derived from the same records so it
            # can never drift from fact_records.
            overview["facts"] = [rec.text for rec in fact_records]

            episode_records = [
                rec for rec in records if (rec.metadata or {}).get("type") == "episode"
            ]
            episode_records.sort(
                key=lambda rec: str((rec.metadata or {}).get("timestamp") or ""),
                reverse=True,
            )
            episode_records = episode_records[:10]
            overview["episode_records"] = [
                _record_payload(rec) for rec in episode_records
            ]
            overview["episodes"] = [rec.text for rec in episode_records]
    return overview


def memory_graph(manager: "MemoryManager") -> Dict[str, Any]:
    """Nodes+edges view of long-term memory for the UI "synapse map".

    Nodes are the vector-store records (facts/episodes/reflections) plus
    profile facts and open threads; edges connect each record to its
    cosine-nearest neighbors, and profile/thread nodes attach to their
    nearest record when an embedder is available. Read-only.
    """
    with manager._state_lock:
        records = (
            list(manager.vector_store.dump())
            if manager.vector_store is not None
            else []
        )
        profile_facts = (
            dict(manager.profile.facts) if manager.profile is not None else {}
        )
        open_threads = (
            list(manager.profile.open_threads) if manager.profile is not None else []
        )

    def record_timestamp(rec: Any) -> str:
        return str((rec.metadata or {}).get("timestamp") or "")

    records = sorted(records, key=record_timestamp, reverse=True)
    records = records[:_GRAPH_MAX_RECORDS]

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    vectored: List[tuple] = []
    for rec in records:
        kind = str((rec.metadata or {}).get("type") or "memory")
        node_id = f"rec:{rec.id}"
        nodes.append(
            {
                "id": node_id,
                "label": rec.text,
                "kind": kind,
                "timestamp": (rec.metadata or {}).get("timestamp"),
                "pinned": bool((rec.metadata or {}).get("pinned")),
            }
        )
        if rec.vector:
            vectored.append((node_id, rec.vector))

    def cosine(a: Any, b: Any) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a <= 0 or norm_b <= 0:
            return 0.0
        return dot / (norm_a * norm_b)

    seen_pairs = set()
    for i, (id_a, vec_a) in enumerate(vectored):
        scored = []
        for j, (id_b, vec_b) in enumerate(vectored):
            if i == j:
                continue
            scored.append((cosine(vec_a, vec_b), id_b))
        scored.sort(key=lambda item: item[0], reverse=True)
        for similarity, id_b in scored[:_GRAPH_NEIGHBORS]:
            if similarity < _GRAPH_EDGE_FLOOR:
                continue
            pair = tuple(sorted((id_a, id_b)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            edges.append(
                {
                    "source": id_a,
                    "target": id_b,
                    "weight": round(float(similarity), 3),
                }
            )

    def attach_nearest(node_id: str, text: str) -> None:
        if manager.embedder is None or not vectored:
            return
        try:
            query = manager.embedder.embed_text(text)
        except Exception:  # pragma: no cover - embedder runtime failure
            return
        if not query:
            return
        best_similarity, best_id = 0.0, None
        for id_b, vec_b in vectored:
            similarity = cosine(query, vec_b)
            if similarity > best_similarity:
                best_similarity, best_id = similarity, id_b
        if best_id is not None and best_similarity >= _GRAPH_ATTACH_FLOOR:
            edges.append(
                {
                    "source": node_id,
                    "target": best_id,
                    "weight": round(float(best_similarity), 3),
                }
            )

    for key, value in list(profile_facts.items())[:20]:
        node_id = f"profile:{key}"
        label = f"{key}: {value}"
        nodes.append(
            {"id": node_id, "label": label, "kind": "profile", "timestamp": None}
        )
        attach_nearest(node_id, label)
    for index, thread in enumerate(open_threads[:10]):
        node_id = f"thread:{index}"
        nodes.append(
            {"id": node_id, "label": thread, "kind": "thread", "timestamp": None}
        )
        attach_nearest(node_id, thread)

    return {"nodes": nodes, "edges": edges}
