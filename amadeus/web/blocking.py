"""Shared thread-offload helpers for the web layer.

One executor and one translate cache for every web module: sessions.py and
streaming.py used to carry byte-identical private copies of both, which meant
16 blocking threads for one logical pool and a translation LRU that never
shared a hit between the POST and WS paths.
"""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

_BLOCKING_EXECUTOR = ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="amadeus-web-blocking"
)


async def run_blocking(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    call = functools.partial(func, *args, **kwargs)
    future = _BLOCKING_EXECUTOR.submit(call)
    return await asyncio.wrap_future(future)


@functools.lru_cache(maxsize=256)
def cached_translate(translator: Any, text: str) -> str:
    """Memoized deep-translator hop for the untagged-fallback TTS path only.

    Translation is a deterministic pure function of the input text, so
    caching by (translator instance, text) is safe. Bounded to 256 entries
    so repeated fallback replies (small local models that don't follow the
    tagged grammar) don't re-pay the network round trip every turn.
    """
    return translator.translate(text)
