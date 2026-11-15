"""Entity-graph memory layer (``memory.graph_enabled``, default off).

Typed entities (person/project/place/date/preference) with relations and
supersede provenance, mined from the SAME combined extraction call
``extraction_jobs.run_extraction_job`` already makes (no extra LLM call —
see the module docstring note below for why that's enough). Free-function-
over-manager, same convention as ``introspection.py``/``context.py``:
``MemoryManager`` keeps thin delegates, this module owns the logic and the
budget.

Storage: one JSON document, either as a row in the SQLite state db
(``sqlite_store.save_graph_doc``/``load_graph_doc``, when
``storage_backend: sqlite``) or a JSON sidecar file next to the message
store (``<persist_path stem>_graph.json``) — same corrupt-file quarantine
convention (``.bak-corrupt`` + a telemetry counter) as the other stores.

Extraction design decision: ``run_extraction_job``'s ``new_facts`` are
already short, single-sentence, English strings such as "User works with
Alice on the CMB segmentation project" or "User's birthday is March 3"
— the LLM has already done the hard part (compressing the conversation
into atomic facts). A second heavy NER call over the same transcript would
mostly re-derive the same nouns at extra cost and latency for a
default-off, best-effort feature. So this module mines entities out of
that already-structured text with a small heuristic parser (capitalized
multi-word spans -> candidate names, keyword/pattern rules -> type,
"replaces" hint -> supersede) rather than an additional model call.
The tradeoff: recall is coarser than a dedicated extraction pass would be
(no real coreference, no relation-type taxonomy beyond a few keyword
buckets) — acceptable for a v1 behind a default-off flag; a follow-up
could promote this to a one-shot structured-output call if the heuristic
proves too lossy in practice.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .. import telemetry
from ..logging_utils import get_logger
from .text_utils import _age_phrase

if TYPE_CHECKING:  # pragma: no cover - import cycle guard (manager imports us)
    from .manager import MemoryManager

LOGGER = get_logger(__name__)

GRAPH_DOC_VERSION = 1

# Small enum on purpose (task says keep it small): the graph exists to give
# recall typed handles, not to be a general knowledge graph.
ENTITY_KINDS = ("person", "project", "place", "date", "preference")

_MAX_ENTITIES = 500  # sanity cap so a runaway extraction can't grow unbounded
_MAX_RELATIONS_PER_ENTITY = 20

# --- heuristic typing --------------------------------------------------

# A "name" candidate: one or more capitalized words, not sentence-initial
# "User"/"The"/etc. Deliberately simple (see module docstring) — this is a
# best-effort mining pass over LLM-produced fact strings, not general NER.
_NAME_RE = re.compile(r"\b([A-Z][a-zA-Z']+(?:\s[A-Z][a-zA-Z']+){0,2})\b")

_NAME_STOPWORDS = frozenset("User I The A An This That These Those Kurisu".split())

_DATE_RE = re.compile(
    r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}"
    r"(?:st|nd|rd|th)?(?:,?\s+\d{4})?|\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)

_PLACE_KEYWORDS = (
    "in ",
    "at ",
    "from ",
    "moved to",
    "lives in",
    "based in",
    "visiting",
)

_PROJECT_KEYWORDS = (
    "project",
    "paper",
    "thesis",
    "app",
    "repo",
    "codebase",
    "startup",
    "manuscript",
)

_PREFERENCE_KEYWORDS = (
    "likes",
    "loves",
    "prefers",
    "enjoys",
    "hates",
    "dislikes",
    "favorite",
    "favourite",
    "is a fan of",
    "into ",
)

_PERSON_KEYWORDS = (
    "friend",
    "colleague",
    "partner",
    "brother",
    "sister",
    "mother",
    "father",
    "mom",
    "dad",
    "wife",
    "husband",
    "boss",
    "coworker",
    "advisor",
    "roommate",
    "with ",
)


@dataclass
class Entity:
    """A typed node in the graph. ``key`` is the lowercased name — identity
    for merge/supersede purposes; ``name`` keeps the original casing for
    display."""

    key: str
    name: str
    kind: str
    first_seen: str
    last_seen: str
    facts: List[str] = field(default_factory=list)
    # The fact text this entity's CURRENT fact (facts[-1]) contradicted and
    # replaced, e.g. "User loves Nolan movies" once facts[-1] becomes "User
    # is bored of Nolan movies now" — kept for "said X, previously Y"
    # provenance in recall (never re-supersedes; only ever overwritten by a
    # later supersede against the entity's new current fact).
    previous_fact: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional["Entity"]:
        try:
            key = str(data["key"])
            name = str(data["name"])
            kind = str(data["kind"])
            first_seen = str(data["first_seen"])
            last_seen = str(data["last_seen"])
        except (KeyError, TypeError, ValueError):
            return None
        if kind not in ENTITY_KINDS:
            return None
        facts = data.get("facts")
        facts = [str(f) for f in facts] if isinstance(facts, list) else []
        previous_fact = data.get("previous_fact")
        return cls(
            key=key,
            name=name,
            kind=kind,
            first_seen=first_seen,
            last_seen=last_seen,
            facts=facts,
            previous_fact=str(previous_fact) if previous_fact else None,
        )


@dataclass
class Relation:
    """A directed edge: ``source`` mentioned alongside ``target`` in one
    fact, e.g. ("Alice", "works_with", "CMB segmentation project")."""

    source: str  # entity key
    target: str  # entity key
    label: str
    fact: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional["Relation"]:
        try:
            return cls(
                source=str(data["source"]),
                target=str(data["target"]),
                label=str(data.get("label") or "related_to"),
                fact=str(data.get("fact") or ""),
                timestamp=str(data["timestamp"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass
class EntityGraph:
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)

    def to_doc(self) -> Dict[str, Any]:
        return {
            "version": GRAPH_DOC_VERSION,
            "entities": [e.to_dict() for e in self.entities.values()],
            "relations": [r.to_dict() for r in self.relations],
        }

    @classmethod
    def from_doc(cls, doc: Optional[Dict[str, Any]]) -> "EntityGraph":
        graph = cls()
        if not isinstance(doc, dict):
            return graph
        for raw in doc.get("entities") or []:
            if not isinstance(raw, dict):
                continue
            entity = Entity.from_dict(raw)
            if entity is not None:
                graph.entities[entity.key] = entity
        for raw in doc.get("relations") or []:
            if not isinstance(raw, dict):
                continue
            relation = Relation.from_dict(raw)
            if relation is not None:
                graph.relations.append(relation)
        return graph


# --- heuristic fact -> entities/relations parse -------------------------


_WINDOW_CHARS = 24  # local context radius around a name candidate


def _classify(name: str, text: str) -> str:
    """Best-effort type for a name candidate given the LOCAL context it
    appears in (not the whole fact — a fact like "User works with Alice on
    the CMB segmentation project" mentions both a person-signal word
    ("with") and a project-signal word ("project"), and only checking
    whole-fact substrings would let whichever keyword list is checked first
    always win, misclassifying every name in a mixed sentence)."""
    if _DATE_RE.search(name):
        return "date"
    idx = text.find(name)
    start = max(0, idx - _WINDOW_CHARS) if idx != -1 else 0
    end = idx + len(name) + _WINDOW_CHARS if idx != -1 else len(text)
    window = text[start:end].lower()
    # Name shape is the strongest signal: a single capitalized word reads as
    # a first name far more often than anything else in companion chat; a
    # multi-word capitalized phrase reads as a title (project/place), UNLESS
    # local context strongly says otherwise (checked below).
    shape_guess = "person" if " " not in name else "project"
    for kw in _PERSON_KEYWORDS:
        if kw in window:
            return "person"
    for kw in _PREFERENCE_KEYWORDS:
        if kw in window:
            return "preference"
    for kw in _PLACE_KEYWORDS:
        if kw in window:
            return "place"
    for kw in _PROJECT_KEYWORDS:
        if kw in window:
            return "project"
    return shape_guess


def parse_fact(fact_text: str, when: str) -> Tuple[List[Entity], List[Relation]]:
    """Mine entities + relations out of one extracted fact string.

    Never raises: malformed/unparseable input yields ``([], [])`` and the
    caller counts it via ``memory.graph_extract_failed``.
    """
    try:
        text = (fact_text or "").strip()
        if not text:
            return [], []
        fact_lower = text.lower()
        names: List[str] = []
        for match in _NAME_RE.finditer(text):
            candidate = match.group(1).strip()
            if candidate in _NAME_STOPWORDS:
                continue
            if candidate not in names:
                names.append(candidate)
        for match in _DATE_RE.finditer(text):
            candidate = match.group(1).strip()
            if candidate not in names:
                names.append(candidate)
        # Preference facts often don't capitalize their object ("User loves
        # hiking") — capture the object of a preference keyword as a
        # lowercase-but-valid preference entity so those aren't lost.
        for kw in _PREFERENCE_KEYWORDS:
            idx = fact_lower.find(kw)
            if idx == -1:
                continue
            tail = text[idx + len(kw) :].strip()
            tail = re.split(r"[.,;]", tail, maxsplit=1)[0].strip()
            tail = " ".join(tail.split()[:4]).strip()
            if tail and tail.lower() not in (n.lower() for n in names):
                names.append(tail)
            break
        entities: List[Entity] = []
        for name in names:
            kind = _classify(name, text)
            entities.append(
                Entity(
                    key=name.lower(),
                    name=name,
                    kind=kind,
                    first_seen=when,
                    last_seen=when,
                    facts=[text],
                )
            )
        relations: List[Relation] = []
        if len(entities) >= 2:
            # Fully connect the entities mentioned in one fact — cheap and,
            # at this scale, more useful than trying to guess directionality
            # from word order in a single heuristic pass.
            for i in range(len(entities)):
                for j in range(len(entities)):
                    if i == j:
                        continue
                    relations.append(
                        Relation(
                            source=entities[i].key,
                            target=entities[j].key,
                            label="mentioned_with",
                            fact=text,
                            timestamp=when,
                        )
                    )
        return entities, relations
    except Exception:  # pragma: no cover - defensive: never crash the job
        return [], []


def extract_into_graph(
    graph: EntityGraph,
    facts: List[Tuple[str, float, Optional[str]]],
    *,
    now: Optional[dt.datetime] = None,
) -> EntityGraph:
    """Merge newly extracted ``(text, salience, replaces)`` facts into
    ``graph`` in place, returns it. Malformed facts are skipped and counted,
    never raise into the caller's job."""
    when = (now or dt.datetime.utcnow()).isoformat()
    # Starred unpack: tolerates both the legacy 3-tuples and the bitemporal
    # 4-tuples (trailing valid_at) — the graph only mines text + replaces.
    for text, _salience, replaces, *_rest in facts:
        entities, relations = parse_fact(text, when)
        if not entities and not relations:
            if (text or "").strip():
                telemetry.incr("memory.graph_extract_failed")
            continue
        # Supersede check runs against each touched entity's PRE-update
        # latest fact — must happen before this fact is appended below, or
        # an entity mentioned in both the new fact and the old one would
        # "supersede" itself with its own just-appended text.
        if replaces:
            _supersede(graph, entities, replaces, text, when)
        for entity in entities:
            if entity.key in graph.entities:
                existing = graph.entities[entity.key]
                existing.last_seen = when
                if text not in existing.facts:
                    existing.facts.append(text)
                    existing.facts = existing.facts[-10:]
            elif len(graph.entities) < _MAX_ENTITIES:
                graph.entities[entity.key] = entity
        for relation in relations:
            graph.relations.append(relation)
    _cap_relations(graph)
    return graph


def _supersede(
    graph: EntityGraph,
    touched_entities: List[Entity],
    replaces_text: str,
    new_text: str,
    when: str,
) -> None:
    """Record supersede provenance on the touched entity whose most recent
    fact matches ``replaces_text`` (loose substring match — the extractor's
    hint is free text, not an id): ``previous_fact`` becomes the OLD latest
    fact, so recall can say "said X on <date>, previously Y" once the new
    fact is appended by the caller. Scoped to entities mentioned in the NEW
    fact (rather than every entity in the graph) so an unrelated entity that
    happens to share wording never gets superseded by mistake."""
    replaces_lower = replaces_text.strip().lower()
    if not replaces_lower:
        return
    for touched in touched_entities:
        existing = graph.entities.get(touched.key)
        if existing is None or not existing.facts:
            continue
        latest = existing.facts[-1]
        if replaces_lower in latest.lower() or latest.lower() in replaces_lower:
            existing.previous_fact = latest
            existing.last_seen = when


def _cap_relations(graph: EntityGraph) -> None:
    """Bound relations-per-entity so a chatty conversation can't grow this
    file without limit; keeps the most recent edges."""
    by_entity: Dict[str, List[Relation]] = {}
    for relation in graph.relations:
        by_entity.setdefault(relation.source, []).append(relation)
    keep_ids = set()
    for source, edges in by_entity.items():
        edges.sort(key=lambda r: r.timestamp)
        for edge in edges[-_MAX_RELATIONS_PER_ENTITY:]:
            keep_ids.add(id(edge))
    graph.relations = [r for r in graph.relations if id(r) in keep_ids]


# --- recall integration ---------------------------------------------------


def matching_entities(graph: EntityGraph, user_text: str) -> List[Entity]:
    """Entities whose name/key is mentioned in ``user_text`` (simple
    substring match on word boundaries — no fuzzy matching, so short/common
    names don't spuriously fire)."""
    if not user_text or not graph.entities:
        return []
    lowered = user_text.lower()
    hits: List[Entity] = []
    for entity in graph.entities.values():
        if len(entity.key) < 3:
            continue
        if re.search(r"\b" + re.escape(entity.key) + r"\b", lowered):
            hits.append(entity)
    return hits


def render_entity_lines(
    graph: EntityGraph,
    entities: List[Entity],
    *,
    now: Optional[dt.datetime] = None,
    max_entities: int = 3,
) -> List[str]:
    """Render matched entities + their one-hop relations as recall-framing
    lines, most-recently-seen first. Supersede provenance is spelled out
    ("said X ..., previously Y") per the task's contradiction handling."""
    now = now or dt.datetime.utcnow()
    ordered = sorted(entities, key=lambda e: e.last_seen, reverse=True)[:max_entities]
    lines: List[str] = []
    seen_keys = {e.key for e in ordered}
    for entity in ordered:
        age = _age_phrase(entity.last_seen, now)
        current = entity.facts[-1] if entity.facts else entity.name
        suffix = f" ({age})" if age else ""
        if entity.previous_fact:
            line = f"- {current}{suffix} (previously: {entity.previous_fact})"
        else:
            line = f"- {current}{suffix}"
        lines.append(line)
        one_hop = [
            r
            for r in graph.relations
            if r.source == entity.key and r.target not in seen_keys
        ]
        for relation in one_hop[:2]:
            target = graph.entities.get(relation.target)
            if target is None:
                continue
            # The relation's fact is often the exact sentence just rendered
            # above (both entities came from the same fact) — skip the
            # no-new-information restatement, only surface a DIFFERENT fact
            # about the related entity.
            if relation.fact == current:
                continue
            lines.append(f"  - related: {relation.fact}")
    return lines


def recall_from_graph(
    manager: "MemoryManager", user_text: str, *, max_entities: int = 3
) -> List[str]:
    """Delegate target for ``MemoryManager.recall_from_graph``: entity-graph
    lines for the current turn, or ``[]`` when the flag is off / nothing
    matches / the graph is empty. Never raises."""
    if not bool(getattr(manager.config.memory, "graph_enabled", False)):
        return []
    try:
        graph = load_graph(manager)
        hits = matching_entities(graph, user_text)
        if not hits:
            return []
        return render_entity_lines(graph, hits, max_entities=max_entities)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Entity-graph recall failed: %s", exc)
        telemetry.incr("memory.graph_extract_failed")
        return []


# --- storage ---------------------------------------------------------------


def _sidecar_path(manager: "MemoryManager") -> Optional[Path]:
    if not manager.persist_path:
        return None
    return manager.persist_path.with_name(manager.persist_path.stem + "_graph.json")


def load_graph(manager: "MemoryManager") -> EntityGraph:
    backend = getattr(manager.config.memory, "storage_backend", "json")
    if backend == "sqlite" and manager.persist_path:
        from . import sqlite_store

        doc = sqlite_store.load_graph_doc(sqlite_store.db_path(manager.persist_path))
        return EntityGraph.from_doc(doc)
    path = _sidecar_path(manager)
    if path is None or not path.exists():
        return EntityGraph()
    try:
        with path.open("r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        telemetry.incr("memory.graph_load_corrupt")
        backup = path.with_name(path.name + ".bak-corrupt")
        try:
            os.replace(path, backup)
            LOGGER.error(
                "Entity graph %s is corrupt (%s); starting fresh. "
                "The unreadable file was kept at %s",
                path,
                exc,
                backup,
            )
        except OSError:
            LOGGER.error(
                "Entity graph %s is corrupt (%s) and could not be backed up; "
                "starting fresh",
                path,
                exc,
            )
        return EntityGraph()
    return EntityGraph.from_doc(doc)


def save_graph(manager: "MemoryManager", graph: EntityGraph) -> None:
    backend = getattr(manager.config.memory, "storage_backend", "json")
    doc = graph.to_doc()
    if backend == "sqlite" and manager.persist_path:
        from . import sqlite_store

        sqlite_store.save_graph_doc(sqlite_store.db_path(manager.persist_path), doc)
        return
    path = _sidecar_path(manager)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(doc, handle, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except OSError as exc:  # pragma: no cover - disk errors
        LOGGER.warning("Failed to save entity graph %s: %s", path, exc)


def update_graph(
    manager: "MemoryManager", facts: List[Tuple[str, float, Optional[str]]]
) -> None:
    """Delegate target for ``MemoryManager._update_graph``: mine ``facts``
    (the same tuples ``_on_summary_ready`` already iterates for
    ``_store_fact``) into the persisted entity graph. No-op, and never
    raises, when the flag is off."""
    if not bool(getattr(manager.config.memory, "graph_enabled", False)):
        return
    if not facts:
        return
    try:
        graph = load_graph(manager)
        extract_into_graph(graph, facts)
        save_graph(manager, graph)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Entity-graph update failed: %s", exc)
        telemetry.incr("memory.graph_extract_failed")
