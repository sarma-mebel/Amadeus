"""Embedding providers for long-term memory recall."""

from __future__ import annotations

import hashlib
import math
import re
from collections import OrderedDict
from typing import List, Sequence


class EmbeddingProvider:
    """Simple embedding provider abstraction."""

    # Minimum cosine similarity below which a recall match is noise for THIS
    # embedder's score distribution. Different embedding families put
    # "completely unrelated text" at very different baselines — lexical
    # hashing near 0, MiniLM ~0.0-0.3, gemini-embedding-2 ~0.45-0.59 — so a
    # single config number can't be right for all of them. The recall path
    # uses max(memory.recall_min_score, relevance_floor); 0.0 here means
    # "the config value alone governs", preserving prior behavior for every
    # embedder that doesn't declare a floor.
    relevance_floor: float = 0.0

    def embed_text(self, text: str) -> List[float]:  # pragma: no cover - interface
        raise NotImplementedError

    def similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        raise NotImplementedError


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Wrap sentence-transformers to produce embeddings."""

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        return list(self.model.encode(text, normalize_embeddings=True))

    def similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        import numpy as np

        va = np.array(a)
        vb = np.array(b)
        return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8))


class HashingEmbedding(EmbeddingProvider):
    """Dependency-free keyword embedder (hashed bag-of-words).

    A no-model fallback for builds that can't ship sentence-transformers/torch
    (notably the packaged desktop sidecar). Each token is hashed into a fixed
    dimension and the resulting vector is L2-normalized, so cosine similarity
    over these vectors measures keyword/term overlap. This is lexical, not
    semantic — "car" and "automobile" won't match — but it makes long-term
    memory (facts, episodes, reflections, user `remember`, recall) actually
    function in the desktop build instead of silently no-opping, at zero bundle
    cost. Deterministic across runs (uses a stable hash, not Python's salted
    ``hash()``), so vectors persisted to disk stay comparable between sessions.
    """

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in re.split(r"[^\w]+", text.lower()) if t]

    def embed_text(self, text: str) -> List[float]:
        import numpy as np

        vec = np.zeros(self.dimension, dtype=np.float32)
        for token in self._tokenize(text):
            # Stable, cross-process hash (Python's built-in hash() is salted
            # per interpreter run, which would make persisted vectors useless).
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        # Plain Python floats (not np.float32) so persisted vectors are
        # JSON-serializable by the InMemory vector store.
        return [float(x) for x in vec]

    def similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        import numpy as np

        va = np.array(a)
        vb = np.array(b)
        return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8))


class GeminiEmbedding(EmbeddingProvider):
    """Semantic embeddings via the Gemini API (gemini-embedding-2).

    ``network = True`` marks per-call latency as a network round-trip, which
    opts recall-query embeds into the recall_embed_timeout_seconds bound
    (local embedders answer in microseconds and never need it).

    Rides the same ``google.generativeai`` SDK + API key as the Gemini chat
    provider — no new dependency or credential. Unlike the hashing embedder
    this is genuinely semantic ("car" matches "automobile") — but that also
    means COMPLETELY unrelated text scores ~0.45-0.59 against a real store
    (measured 2026-07-28: "what's the weather like" hit 0.49, "haha that's
    funny" hit 0.59, while genuinely relevant queries sat 0.62-0.78). The
    config's 0.35 ``recall_min_score`` therefore filters nothing on this
    embedder, which injected top-k memories into every turn regardless of
    relevance ("she keeps routing to memories"). ``relevance_floor`` below
    is the measured noise/signal boundary; the recall path applies
    max(config, floor).

    ``output_dim`` truncates the model's native 3072-d MRL embedding — 768 is
    ample for this store and keeps the persisted JSON a quarter of the size.
    Vectors are L2-normalized defensively (truncated MRL output isn't
    guaranteed normalized across models/API versions).

    Results are LRU-cached: the recall path re-embeds the rolling summary and
    each recalled candidate every turn (see ``recall_summary_dedupe``), and
    those texts repeat across turns — without the cache that would be several
    network calls per turn for identical strings.
    """

    network = True
    # Measured noise ceiling ~0.59 / relevance floor ~0.62 on
    # gemini-embedding-2 @768-d, task_type=semantic_similarity (2026-07-28,
    # real store + live queries — see class docstring). 0.60 splits the gap.
    relevance_floor = 0.60

    def __init__(
        self,
        api_key: str,
        model: str = "models/gemini-embedding-2",
        output_dim: int = 768,
        cache_size: int = 512,
    ) -> None:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._genai = genai
        self.model = model
        self.output_dim = output_dim
        self._cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._cache_size = cache_size

    def embed_text(self, text: str) -> List[float]:
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return list(cached)
        result = self._genai.embed_content(
            model=self.model,
            content=text,
            # Symmetric task type: memory compares query<->record but also
            # record<->record (write dedupe, episode supersede, summary
            # dedupe), so asymmetric retrieval_query/document would skew those.
            task_type="semantic_similarity",
            output_dimensionality=self.output_dim,
        )
        vector = [float(x) for x in result["embedding"]]
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        self._cache[text] = vector
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return list(vector)

    def similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        import numpy as np

        va = np.array(a)
        vb = np.array(b)
        return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8))
