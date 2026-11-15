"""Async bridge for synchronous provider generators."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, AsyncIterator, Callable, List

from ...logging_utils import get_logger

LOGGER = get_logger(__name__)


_SENTINEL = object()


async def iter_in_thread(
    sync_gen_factory: Callable[[], Any],
    cancel_event: threading.Event,
) -> AsyncIterator[str]:
    """Run a synchronous generator (an LLM provider's ``stream()``) in a
    worker thread and yield its items on the calling event loop.

    ``asyncio.to_thread`` cannot wrap a generator (it awaits a single
    callable to completion), so the worker posts each item onto an
    ``asyncio.Queue`` via ``call_soon_threadsafe`` and the consumer blocks on
    ``get()`` — no polling, items wake the loop the moment they arrive.
    Cancellation is cooperative: setting ``cancel_event`` causes the worker
    to close the generator and stop after its current item.
    """
    loop = asyncio.get_running_loop()
    items: "asyncio.Queue[Any]" = asyncio.Queue()
    exception_holder: List[BaseException] = []

    def post(item: Any) -> None:
        try:
            loop.call_soon_threadsafe(items.put_nowait, item)
        except RuntimeError:  # pragma: no cover - loop shut down mid-stream
            pass

    def worker() -> None:
        gen = None
        try:
            if cancel_event.is_set():
                # Pre-cancelled turn (e.g. duplex barge-in during STT seeded
                # the event before generation started): don't construct the
                # provider stream at all — a full generation would run only
                # to be discarded. The finally still posts the sentinel, so
                # the async side sees a cleanly-empty stream.
                return
            gen = sync_gen_factory()
            for item in gen:
                if cancel_event.is_set():
                    break
                post(item)
        except BaseException as exc:  # noqa: BLE001 - propagate to async side
            exception_holder.append(exc)
        finally:
            if gen is not None and hasattr(gen, "close"):
                try:
                    gen.close()
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass
            post(_SENTINEL)

    thread = threading.Thread(target=worker, daemon=True, name="amadeus-llm-stream")
    thread.start()

    try:
        while True:
            item = await items.get()
            if item is _SENTINEL:
                break
            yield item
    finally:
        cancel_event.set()
        # Bound the join so a wedged provider can't hang the request forever;
        # the thread is a daemon so the process can still exit cleanly.
        await loop.run_in_executor(None, thread.join, 5.0)

    if exception_holder:
        raise exception_holder[0]
