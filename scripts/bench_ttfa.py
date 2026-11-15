#!/usr/bin/env python3
"""In-process TTFA (time-to-first-audio) benchmark for the streaming chat path.

Drives the exact same code path production traffic uses
(``WebChatSession.handle_stream`` -> ``streaming.run_streaming_turn``) with:

- a mock LLM whose ``stream()`` yields tagged
  ``"[emotion] English || Japanese\\n"`` lines token-by-token with a
  configurable per-token delay (simulating real LLM streaming latency), and
- a mock TTS whose ``synthesize`` sleeps for a configurable duration per
  sentence (simulating ONNX/model synthesis latency).

Runs N turns and reports p50/p95 for ttft_ms / ttfa_ms / total_ms, taken from
the ``timings`` block that ``streaming._wrap_emit_with_timings`` attaches to
the terminal ``done`` frame — the same numbers a real client would see
(server-side emit point, i.e. minus network).

Usage:
    python scripts/bench_ttfa.py [--turns 20] [--token-delay-ms 15] [--synth-delay-ms 300]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amadeus.config import Config
from amadeus.providers.llm.base import LLMProvider
from amadeus.providers.tts.base import TTSProvider
from amadeus.web.app import ChatRequest, create_app

# A handful of scripted replies so turns aren't perfectly identical (mirrors
# real variation in sentence count/length without needing a real LLM).
SCRIPTED_REPLIES = [
    (
        "[happy] Dopamine reinforces reward learning. || ドーパミンは報酬学習を強化する。\n"
        "[relaxed] It also supports memory consolidation, which is a much longer sentence to synthesize. || 記憶の定着も支援する。\n"
        "[surprised] That's the short version of a fairly deep topic. || それが概要です。"
    ),
    (
        "[relaxed] Causality is subtle in closed timelike curves. || 因果律は閉じた時間的曲線では微妙だ。\n"
        "[neutral] The Novikov self-consistency principle is one resolution worth discussing at length here. || ノヴィコフの自己無矛盾原理は一つの解決策だ。"
    ),
    (
        "[happy] Sure, I can explain that. || もちろん、説明できるわ。\n"
        "[relaxed] Neural plasticity underlies most long-term learning effects in the brain. || 神経可塑性が学習効果の基盤よ。\n"
        "[neutral] There is more nuance if you want it. || もっと詳しく知りたいなら教えるわ。\n"
        "[surprised] But that covers the basics. || でもそれが基本ね。"
    ),
]


class TokenDelayLLM(LLMProvider):
    """Streams a scripted reply one whitespace-delimited token at a time,
    sleeping ``token_delay_s`` between tokens to simulate real LLM output
    pacing."""

    name = "bench-token-delay"

    def __init__(self, reply: str, token_delay_s: float) -> None:
        self.reply = reply
        self.token_delay_s = token_delay_s

    def generate(self, messages, **kwargs) -> str:
        return self.reply

    def stream(self, messages, **kwargs):
        text = self.reply
        start = 0
        for idx, char in enumerate(text):
            if char == " ":
                yield text[start : idx + 1]
                start = idx + 1
                if self.token_delay_s:
                    time.sleep(self.token_delay_s)
        if start < len(text):
            yield text[start:]

    def supports_stream(self) -> bool:
        return True


class SynthDelayTTS(TTSProvider):
    """Synchronous TTS mock: sleeps ``synth_delay_s`` per call to simulate
    blocking model inference (e.g. ONNX), then returns dummy bytes."""

    def __init__(self, synth_delay_s: float) -> None:
        self.synth_delay_s = synth_delay_s
        self.calls: List[str] = []

    def synthesize(self, text: str, *, speaking_rate: float = 1.0, pitch_shift: float = 0.0) -> bytes:
        self.calls.append(text)
        if self.synth_delay_s:
            time.sleep(self.synth_delay_s)
        return f"WAV:{text}".encode()


def _build_config(tmp_path: Path) -> Config:
    config = Config()
    config.web.streaming_enabled = True
    config.web.streaming_tagged_output = True
    config.audio.enable_voice = True
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.enable_long_term = False
    return config


async def _run_one_turn(app, llm, message: str) -> Dict[str, Any]:
    store = app.state.session_store
    session_id, session = store.get(None, f"bench-{time.monotonic_ns()}")
    frames: List[Dict[str, Any]] = []
    cancel_event = threading.Event()

    async def emit(frame: Dict[str, Any]) -> None:
        frames.append(frame)

    request = ChatRequest.model_validate({"message": message, "user_id": f"bench-{time.monotonic_ns()}"})
    await session.handle_stream(request, emit, cancel_event)
    done = frames[-1]
    assert done["type"] == "done", f"expected terminal done frame, got {done['type']}"
    return done["timings"]


async def run_bench(turns: int, token_delay_ms: float, synth_delay_ms: float, tmp_path: Path) -> List[Dict[str, Any]]:
    import amadeus.web.app as app_module

    config = _build_config(tmp_path)
    token_delay_s = token_delay_ms / 1000.0
    synth_delay_s = synth_delay_ms / 1000.0

    tts = SynthDelayTTS(synth_delay_s)

    original_load_llm = app_module.load_llm
    original_load_tts = app_module.load_tts
    try:
        app_module.load_llm = lambda cfg: TokenDelayLLM(
            SCRIPTED_REPLIES[0], token_delay_s
        )
        app_module.load_tts = lambda cfg: tts
        app = create_app(config)

        results: List[Dict[str, Any]] = []
        for i in range(turns):
            reply = SCRIPTED_REPLIES[i % len(SCRIPTED_REPLIES)]
            # Swap in a fresh LLM per turn so each session gets its own
            # scripted reply (load_llm is called once per SessionStore, so
            # rebuild the store's llm reference directly instead).
            app.state.session_store.llm = TokenDelayLLM(reply, token_delay_s)
            timings = await _run_one_turn(app, app.state.session_store.llm, f"question {i}")
            results.append(timings)
        return results
    finally:
        app_module.load_llm = original_load_llm
        app_module.load_tts = original_load_tts


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def summarize(results: List[Dict[str, Any]]) -> None:
    for key in ("ttft_ms", "ttfa_ms", "total_ms"):
        values = [r[key] for r in results if r.get(key) is not None]
        if not values:
            print(f"{key}: no data")
            continue
        p50 = _percentile(values, 50)
        p95 = _percentile(values, 95)
        print(f"{key:>10}: p50={p50:7.1f}ms  p95={p95:7.1f}ms  (n={len(values)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--token-delay-ms", type=float, default=15.0)
    parser.add_argument("--synth-delay-ms", type=float, default=300.0)
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default=None,
        help="Directory for scratch memory files (default: a fresh tempdir)",
    )
    args = parser.parse_args()

    import tempfile

    if args.tmp_dir:
        tmp_path = Path(args.tmp_dir)
        tmp_path.mkdir(parents=True, exist_ok=True)
        results = asyncio.run(
            run_bench(args.turns, args.token_delay_ms, args.synth_delay_ms, tmp_path)
        )
    else:
        with tempfile.TemporaryDirectory() as tmp:
            results = asyncio.run(
                run_bench(args.turns, args.token_delay_ms, args.synth_delay_ms, Path(tmp))
            )

    print(
        f"\nbench_ttfa: turns={args.turns} token_delay={args.token_delay_ms}ms "
        f"synth_delay={args.synth_delay_ms}ms\n"
    )
    summarize(results)


if __name__ == "__main__":
    main()
