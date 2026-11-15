"""Unit tests for amadeus.web.streaming.iter_in_thread — the sync-generator-
to-async bridge used to drive LLM provider stream() generators.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from amadeus.web.streaming import iter_in_thread


@pytest.mark.asyncio
async def test_yields_all_items_in_order():
    def gen_factory():
        yield "a"
        yield "b"
        yield "c"

    cancel = threading.Event()
    items = [item async for item in iter_in_thread(gen_factory, cancel)]
    assert items == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_propagates_generator_exception():
    def gen_factory():
        yield "a"
        raise RuntimeError("provider exploded")

    cancel = threading.Event()
    collected = []
    with pytest.raises(RuntimeError, match="provider exploded"):
        async for item in iter_in_thread(gen_factory, cancel):
            collected.append(item)
    assert collected == ["a"]


@pytest.mark.asyncio
async def test_cancel_stops_consumption_early():
    produced = {"count": 0}

    def gen_factory():
        for i in range(100):
            produced["count"] = i + 1
            time.sleep(0.01)
            yield str(i)

    cancel = threading.Event()
    collected = []
    async for item in iter_in_thread(gen_factory, cancel):
        collected.append(item)
        if len(collected) == 3:
            cancel.set()
            break

    # The generator worker should stop within a bounded time after cancel;
    # we didn't consume all 100 items.
    assert len(collected) == 3
    assert produced["count"] < 100


@pytest.mark.asyncio
async def test_empty_generator_yields_nothing():
    def gen_factory():
        return
        yield  # pragma: no cover - unreachable, marks this a generator

    cancel = threading.Event()
    items = [item async for item in iter_in_thread(gen_factory, cancel)]
    assert items == []


@pytest.mark.asyncio
async def test_concurrent_streams_do_not_interfere():
    def make_gen(prefix: str):
        def gen_factory():
            for i in range(5):
                yield f"{prefix}{i}"

        return gen_factory

    cancel_a = threading.Event()
    cancel_b = threading.Event()

    async def collect(factory, cancel):
        return [item async for item in iter_in_thread(factory, cancel)]

    results_a, results_b = await asyncio.gather(
        collect(make_gen("a"), cancel_a),
        collect(make_gen("b"), cancel_b),
    )
    assert results_a == ["a0", "a1", "a2", "a3", "a4"]
    assert results_b == ["b0", "b1", "b2", "b3", "b4"]
