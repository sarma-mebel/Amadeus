"""GeminiEmbedding provider: cache, loader fallback, dimension guard (no live API)."""

import sys
import types
from pathlib import Path

import pytest

from amadeus.config import Config


class _TwoDimEmbedder:
    """Minimal 2-d embedder (mirrors test_memory.VectorEmbedder; tests/ is not
    a package, so it can't be imported across files)."""

    def embed_text(self, text):
        return [1.0, 0.0]

    def similarity(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / ((na * nb) or 1.0)


def _isolated_long_term_config(tmp_path: Path) -> Config:
    config = Config()
    config.memory.persist_path = None
    config.memory.enable_long_term = True
    config.memory.embeddings_path = str(tmp_path / "vectors.json")
    config.memory.dedupe_threshold = 0.0
    config.memory.enable_fact_extraction = False
    config.memory.enable_episodes = False
    config.memory.profile_path = None
    # These tests recall with short synthetic queries; keep the production
    # trivial-query gate out of the way.
    config.memory.recall_min_query_chars = 0
    config.memory.recall_min_score = 0.0
    return config


def _fake_genai(monkeypatch, embed_calls, dimension=4):
    """Install a fake google.generativeai that records embed_content calls."""
    fake = types.ModuleType("google.generativeai")
    fake.configure = lambda **_: None

    def embed_content(model, content, **kwargs):
        embed_calls.append({"model": model, "content": content, **kwargs})
        # Deterministic per-text vector, distinct in DIRECTION (the provider
        # L2-normalizes, so scale-only differences would collapse); honors
        # output_dimensionality like the real API.
        dim = kwargs.get("output_dimensionality", dimension)
        seed = float(len(content) % 7 + 1)
        return {"embedding": [seed, 1.0] + [0.0] * (dim - 2)}

    fake.embed_content = embed_content
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.generativeai", fake)
    return fake


def test_embed_text_calls_api_and_caches(monkeypatch):
    calls = []
    _fake_genai(monkeypatch, calls)
    from amadeus.memory.manager import GeminiEmbedding

    emb = GeminiEmbedding(api_key="test-key")
    v1 = emb.embed_text("hello world")
    v2 = emb.embed_text("hello world")  # cached: no second API call
    v3 = emb.embed_text("different text")

    assert v1 == v2
    assert len(calls) == 2  # one per distinct text
    assert calls[0]["task_type"] == "semantic_similarity"
    assert calls[0]["output_dimensionality"] == 768
    assert len(v1) == 768
    assert v3 != v1
    # Returned vectors are plain floats (JSON-serializable for the store),
    # and L2-normalized.
    assert all(isinstance(x, float) for x in v1)
    assert sum(x * x for x in v1) == pytest.approx(1.0, abs=1e-6)


def test_cache_evicts_oldest(monkeypatch):
    calls = []
    _fake_genai(monkeypatch, calls)
    from amadeus.memory.manager import GeminiEmbedding

    emb = GeminiEmbedding(api_key="test-key", cache_size=2)
    emb.embed_text("a")
    emb.embed_text("b")
    emb.embed_text("c")  # evicts "a"
    emb.embed_text("a")  # re-fetches
    assert len(calls) == 4


def test_similarity_is_cosine(monkeypatch):
    calls = []
    _fake_genai(monkeypatch, calls)
    from amadeus.memory.manager import GeminiEmbedding

    emb = GeminiEmbedding(api_key="test-key")
    assert emb.similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0, abs=1e-6)
    assert emb.similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-6)


def test_load_embedder_gemini(monkeypatch):
    calls = []
    _fake_genai(monkeypatch, calls)
    from amadeus.provider_loader import load_embedder

    config = Config()
    config.providers.embedding = "gemini"
    config.gemini_api_key = "test-key"
    embedder = load_embedder(config)
    assert type(embedder).__name__ == "GeminiEmbedding"


def test_load_embedder_gemini_without_key_falls_back_to_hashing(monkeypatch):
    from amadeus.provider_loader import load_embedder

    config = Config()
    config.providers.embedding = "gemini"
    config.gemini_api_key = None
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    embedder = load_embedder(config)
    assert type(embedder).__name__ == "HashingEmbedding"


def test_dimension_mismatch_disables_recall(tmp_path: Path, caplog):
    """Old-store vectors with a different dimension must not produce garbage
    recall — the guard disables recall and points at the migration script."""
    from amadeus.memory.manager import MemoryManager
    from amadeus.memory.vector_store import VectorRecord

    config = _isolated_long_term_config(tmp_path)
    manager = MemoryManager(config, embedder=_TwoDimEmbedder())
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0, 0.0, 0.0],  # 4-d "old embedder" record
            text="stale-dimension fact",
            metadata={"type": "fact", "strength": 1.0},
        )
    )

    assert manager.recall_detailed("anything", limit=3) == []
    # Once flagged, subsequent recalls stay disabled without re-checking.
    assert manager.recall_detailed("anything else", limit=3) == []
    assert "dimension mismatch" in caplog.text.lower()


def test_dimension_check_resets_on_embedder_swap(tmp_path: Path):
    from amadeus.memory.manager import MemoryManager
    from amadeus.memory.vector_store import VectorRecord

    config = _isolated_long_term_config(tmp_path)
    manager = MemoryManager(config, embedder=_TwoDimEmbedder())
    manager.vector_store.add(
        VectorRecord(
            vector=[1.0, 0.0],
            text="matching-dimension fact",
            metadata={"type": "fact", "strength": 1.0},
        )
    )
    assert manager.recall_detailed("anything", limit=1)  # dims match: works

    class FourDimEmbedder(_TwoDimEmbedder):
        def embed_text(self, text):
            return [1.0, 0.0, 0.0, 0.0]

    manager.set_embedder(FourDimEmbedder())
    assert manager.recall_detailed("anything", limit=1) == []  # re-checked
