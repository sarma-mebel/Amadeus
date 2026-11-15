#!/usr/bin/env python3
"""Benchmark Amadeus memory vector store backends."""

from __future__ import annotations

import argparse
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import List

from amadeus.config import Config
from amadeus.memory.manager import MemoryManager
from amadeus.memory.manager import EmbeddingProvider


class DummyEmbedder(EmbeddingProvider):
    """Lightweight embedder for synthetic benchmarking."""

    def embed_text(self, text: str) -> List[float]:
        return [float(ord(char)) for char in text]

    def similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        min_len = min(len(a), len(b))
        total = sum(x * y for x, y in zip(a[:min_len], b[:min_len]))
        norm_a = sum(x * x for x in a[:min_len]) ** 0.5
        norm_b = sum(y * y for y in b[:min_len]) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return total / (norm_a * norm_b)


def load_test_sentences(path: Path | None) -> List[str]:
    if not path:
        return [
            "Kurisu remembers building the PhoneWave (name subject to change).",
            "El Psy Kongroo. Future Gadget Lab, member 003!",
            "Remember the divergence meter calibration from last night.",
            "The gelatinous bananas experiment is ongoing.",
            "Note the observer effect when dealing with time travel paradoxes.",
            "Track Okabe's vital signs during long gaming sessions.",
            "Tsundere response pack version 1.4 is deployed.",
            "Kurisu prefers Dr. Pepper, not energy drinks.",
            "Maintain backups of all VRM expressions for emergencies.",
            "Review LHC papers before attempting another micro-black hole chat.",
        ]
    if not path.exists():
        raise FileNotFoundError(f"Sentence file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        raise ValueError("Sentence file is empty.")
    return lines


def create_manager(backend: str, base_dir: Path, embedder: EmbeddingProvider) -> MemoryManager:
    config = Config()
    config.memory.enable_long_term = True
    config.memory.persist_path = str(base_dir / "memory.json")
    config.memory.embeddings_path = str(base_dir / "vectors.json")
    config.memory.vector_path = str(base_dir / "vector_store")
    config.memory.vector_backend = backend
    config.memory.dedupe_threshold = 0.0
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "vector_store").mkdir(parents=True, exist_ok=True)
    manager = MemoryManager(config, embedder=embedder)
    manager.clear()  # ensure fresh state
    return manager


def benchmark_backend(name: str, manager: MemoryManager, sentences: List[str], recalls: int) -> dict:
    write_times: List[float] = []
    recall_times: List[float] = []

    for line in sentences:
        start = time.perf_counter()
        manager.add_message("user", line)
        write_times.append((time.perf_counter() - start) * 1000)  # ms

    for _ in range(recalls):
        query = sentences[_ % len(sentences)]
        start = time.perf_counter()
        manager.recall(query)
        recall_times.append((time.perf_counter() - start) * 1000)

    return {
        "backend": name,
        "writes_ms": {
            "avg": statistics.fmean(write_times),
            "p95": statistics.quantiles(write_times, n=20)[-1],
            "max": max(write_times),
        },
        "recall_ms": {
            "avg": statistics.fmean(recall_times),
            "p95": statistics.quantiles(recall_times, n=20)[-1],
            "max": max(recall_times),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Amadeus memory vector stores.")
    parser.add_argument("--recalls", type=int, default=100, help="Number of recall queries to run.")
    parser.add_argument("--sentences", type=Path, help="Path to file with newline-delimited sentences.")
    args = parser.parse_args()

    sentences = load_test_sentences(args.sentences)
    embedder = DummyEmbedder()

    print("=" * 70)
    print("Amadeus Memory Vector Store Benchmark")
    print("=" * 70)
    print(f"Sentences: {len(sentences)} • Recall queries: {args.recalls}")
    print("Backends: JSON (default) vs Chroma (if installed)")
    print()

    tmp_root = Path(tempfile.mkdtemp(prefix="amadeus-memory-bench-"))

    results: List[dict] = []
    json_dir = tmp_root / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    json_manager = create_manager("json", json_dir, embedder)
    results.append(benchmark_backend("json", json_manager, sentences, args.recalls))

    chroma_dir = tmp_root / "chroma"
    try:
        chroma_manager = create_manager("chroma", chroma_dir, embedder)
    except (RuntimeError, Exception) as exc:
        print("⚠️  Skipping Chroma backend:", exc)
    else:
        results.append(benchmark_backend("chroma", chroma_manager, sentences, args.recalls))

    print("=" * 70)
    print("Results (milliseconds)")
    print("=" * 70)
    header = "{:<10} {:>10} {:>10} {:>10}   {:>10} {:>10} {:>10}"
    row = "{:<10} {:>10.2f} {:>10.2f} {:>10.2f}   {:>10.2f} {:>10.2f} {:>10.2f}"
    print(header.format("Backend", "W-avg", "W-p95", "W-max", "R-avg", "R-p95", "R-max"))
    for result in results:
        writes = result["writes_ms"]
        recall = result["recall_ms"]
        print(
            row.format(
                result["backend"],
                writes["avg"],
                writes["p95"],
                writes["max"],
                recall["avg"],
                recall["p95"],
                recall["max"],
            )
        )
    print()
    print("W-* metrics refer to memory writes (adding messages); R-* refer to recall lookups.")
    print("Use --recalls to adjust query volume and --sentences to benchmark real transcripts.")
    os.chdir(tmp_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
