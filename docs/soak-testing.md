# Long-session soak testing

The unit suite runs single turns against mocked providers, so it can never see
the failure class that only appears after an evening of use: VRAM creeping up
until the TTS engine hits CUDA OOM, server RSS growing turn over turn, latency
drifting as stores bloat, or the memory pipeline slowing down as episodes pile
up. The soak harness exists for exactly that. It drives hundreds of realistic
turns against a live running server and reports degradation over time.

It is not part of `pytest`. It needs a server that is already running.

## Running it

Start the server you want to soak (a real config with the real voice is the
most useful target, since VRAM creep lives there):

```bash
uvicorn amadeus.web.app:app --port 8000
```

Then, in another terminal:

```bash
make soak                                   # 200 turns against 127.0.0.1:8000
make SOAK_URL=http://127.0.0.1:8091 SOAK_TURNS=50 soak
python scripts/soak_test.py --turns 300 --delay 2 --out soak_out/tonight
```

Useful flags:

- `--ws` (default) streams each turn over `/api/chat/stream` and falls back to
  `POST /api/chat` on any WebSocket failure, the same way the frontend does.
  `--post` skips WebSockets entirely; note that server-side timings only ride
  the WS done frame, so POST turns record client wall-clock only.
- `--delay` seconds between turns (default 1). Raise it to pace free-tier
  rate limits when soaking against a real LLM.
- `--prompt-file` points at a file with one prompt per line. The default is a
  built-in rotating list of about 20 casual, technical, and emotional prompts,
  including repeat greetings and long-reply asks, so summarization, recall,
  and the emotion path all get exercised.
- `--status-every` N (default 20) controls how often it snapshots
  `GET /api/status` counters, server RSS, and GPU VRAM.
- `--server-pid` pins RSS tracking to a specific process. Without it the
  harness finds the uvicorn process via psutil, and skips RSS quietly when
  neither is available. VRAM comes from `nvidia-smi` and is skipped silently
  when the binary does not exist.
- `--token` passes the bearer token when `web.api_token` is set.

A failed turn is logged and the run continues. Surviving hundreds of turns is
the point, so a single provider hiccup must not kill the measurement.

## What it measures

Per turn (one JSONL line in `--out/turns.jsonl`): client wall-clock, the
server's own `timings` from the done frame (`ttft_ms`, `first_sentence_ms`,
`ttfa_ms`, `total_ms`), reply length, sentence and audio frame counts,
`tts_failed`, and error details on failure.

Per snapshot (`--out/snapshots.jsonl`): the `/api/status` `counters` block
(including the `latency.*` sum/count pairs, reduced to running averages, and
`tts.qwen3_slow_fallback`), server RSS in MB, and total GPU VRAM used in MB.

## Reading the verdict

The final summary is printed and written to `--out/summary.json`. It computes
p50/p90/p99 latency for the whole run and separately for the first and last
quartile of turns. Comparing those quartiles is the drift detector: a healthy
server produces roughly the same p90 at turn 190 as at turn 10.

Verdict lines are plain sentences, for example:

```
DRIFT: turn total p90 drifted +42% (1200ms -> 1704ms)
GROWTH: GPU VRAM grew +900MB over the run (4100MB -> 5000MB)
ok: server RSS +40MB (610MB -> 650MB)
ok: 1/200 turns failed
```

Thresholds are constants at the top of `scripts/soak_test.py`: 25% p90 drift,
300MB RSS growth, 512MB VRAM growth, and a 10% turn failure rate. The process
exits non-zero when any verdict fires, so the harness can gate a long-running
job.

The behavioral counterpart of this harness is the conversation-quality eval
(docs/conversation-eval.md), which judges what the model says. The soak judges
whether the process stays healthy while saying it.
