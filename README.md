<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/amadeus-logo-dark.png">
  <img src="docs/img/amadeus-logo-light.png" alt="Amadeus" width="480">
</picture>

**A self-hosted AI companion with streaming voice, an expressive 3D VRM avatar, and long-term memory that grows with your conversations.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

[Install](#install) | [Quickstart](#quickstart) | [Features](#features) | [Memory](#she-remembers-you) | [Docs](#documentation)

**English** | [日本語](README.ja.md)

</div>

## What is this?

Amadeus is a passion project built out of love for Steins;Gate. You talk to Kurisu by
voice or text and she answers out loud in Japanese, while her 3D avatar emotes,
gestures, and lip-syncs in the browser or in a native desktop window. Under the hood
it's a Python/FastAPI backend with pluggable LLM/TTS/STT providers, layered memory, and
a sandboxed tool loop for calculations, files, papers, and the web.

Everything runs on your machine. Cloud LLM and TTS providers are optional and opt-in.
There's no telemetry and no engagement tricks; the rules she lives by are written down
in [docs/companion-ethics.md](docs/companion-ethics.md).

Kurisu is the flagship persona, but the engine underneath doesn't care who it's
playing. Give it your own persona prompt, any VRM avatar (0.x or 1.0), and a 10-second
voice clip, and all the companion machinery works unchanged. See
[docs/characters.md](docs/characters.md).

## Install

Amadeus is distributed as source. There are no prebuilt installers.

| Platform | How |
|----------|-----|
| Any OS, from source | [Quickstart](#quickstart). The CLI and web UI run anywhere Python 3.9+ does. |
| Windows desktop app | Build the Tauri app ([overview](docs/desktop-tauri.md), [build steps](docs/desktop-release.md)) |
| Android | Build the APK and point it at your server ([docs/android.md](docs/android.md)) |
| Docker / Fly.io | `Dockerfile` and `fly.toml` are included ([docs/hosted-demo.md](docs/hosted-demo.md)) |

**Requirements.** Python 3.9+ and an API key for Gemini or OpenAI. No GPU is needed. A
CUDA card with 8 GB of VRAM is only required for Qwen3-TTS voice cloning or a fully
local LLM.

**Not included.** The repository contains the engine, the persona prompts and a library
of freely licensed animations. It does not contain a voice model, a voice reference clip
or a VRM avatar. Supply your own avatar (VRM 0.x or 1.0) and a voice clip you have the
right to use; [docs/characters.md](docs/characters.md) shows where they go. Without a
voice clip, pick a TTS backend that does not clone, such as `pyttsx3` or `openai`.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[web,gemini]"
cp config/config.example.yaml config.local.yaml
export GEMINI_API_KEY=...        # or OPENAI_API_KEY, or put it in the config
```

Chat in the terminal:

```bash
python -m amadeus --config config.local.yaml
```

Or run the web UI and open <http://127.0.0.1:8000>:

```bash
uvicorn amadeus.web.app:app --reload
```

On first run the web UI opens a setup wizard (provider, API key, voice and memory
options). `python -m amadeus --init-config` does the same in the terminal.

> [!WARNING]
> If a provider fails to load, the app falls back to a dummy provider that gives canned
> replies. The web UI shows an "offline demo mode" banner and the CLI prints a warning.
> `/status` in the CLI or `GET /api/status` shows which providers are live.

<details>
<summary><b>Install extras</b></summary>

| Extra | What it adds |
| --- | --- |
| `web` | Browser UI and WebSocket streaming |
| `gemini` / `openai` | That provider's LLM (and TTS, for OpenAI) |
| `voice` | Microphone input and speaker output for the CLI |
| `stt` | faster-whisper speech-to-text |
| `memory` | Local sentence-transformers embeddings for long-term recall (optional: the `gemini` embedder needs no extra install) |
| `local` | Offline local LLMs via transformers |

</details>

## Features

- **Streaming voice.** Replies stream sentence by sentence, so she starts talking
  before the full reply is finished. Audio always arrives in order, lip sync carries
  across sentence boundaries, and hands-free mode reopens the mic when she's done. You
  can interrupt her mid-sentence by speaking up, or go one further with a full-duplex
  call: leave the mic open and just talk. She hears where your sentences end, nods
  along with small acknowledgment sounds while you're still speaking, and yields the
  moment you cut her off. An optional 8 MB model (`python scripts/fetch_smart_turn.py`)
  gives her better manners about pauses: if yours sounds like the middle of a thought,
  she waits instead of jumping in.
- **A voice you choose.** Zero-shot voice cloning from your own reference clip with
  Qwen3-TTS if you have a GPU, an offline ONNX backend for CPU machines, and five other
  TTS backends. Replies are bilingual: English on screen and in memory, native Japanese
  straight to TTS with no translation step (or flip one switch and she speaks English).
  Optional extras make the voice feel alive, like per-emotion reference clips and short
  "thinking" hums while she composes a longer reply
  ([docs/voice-expressiveness.md](docs/voice-expressiveness.md)).
- **An avatar that acts.** Emotion tags on every sentence drive expressions and a
  library of 100+ animation clips: idles, talk gestures, emotion-matched body language,
  springbone hair physics, poke reactions.
- **Memory that layers.** Rolling short-term context, async summarization, long-term
  semantic recall, and a per-user profile with facts, open threads, and a relationship
  arc, all browsable in a synapse-map view. [More on this below.](#she-remembers-you)
- **Tools, carefully.** Calculator, sandboxed code interpreter, web search (Tavily, or
  keyless DuckDuckGo), PubMed/arXiv lookup, spoken reminders and timers, and read-only
  desktop file search. Everything sits behind an allowlist, an injection firewall, and
  an exfiltration guard.
- **One core, five shells.** OpenAI, Gemini, or local transformers for the LLM (plus
  any OpenAI-compatible endpoint, so Ollama and LM Studio work too), seven TTS backends,
  faster-whisper STT. Interactive CLI, self-hosted web UI, a Windows desktop app,
  Docker/Fly.io, and a Discord bridge, all thin layers over the same engine.

<details>
<summary><b>More: vision, pet mode, identity versioning, daily brief...</b></summary>

- **Vision.** Paste, drop, or capture a screenshot and she can see it. Transcripts and
  memory only ever store a `[user shared a screenshot]` marker, never the image. Turn on
  watch-together (or read-together) and the app samples your screen every minute or so
  while she follows along and only occasionally says something. Opt-in each time, and no
  frame is ever stored.
- **Pet mode.** A small transparent always-on-top window so she can sit on your desktop
  while you work. Her eyes follow your cursor (park it on her for a moment and she tilts
  her head at it), she occasionally strolls along the screen edges, and she'll wander
  over to your second monitor if you have one. At night she's asleep on her own
  schedule, and poking her awake gets you a groggy answer. Need the space? Tuck her away
  at the edge of the screen and she comes back when you click the visible sliver.
- **Ambient awareness (desktop).** She notices cheap local signals: the same app focused
  for three hours, the battery getting low, what you're listening to. She often chooses
  to say nothing. This reads window titles and media metadata only, takes no
  screenshots, stores nothing, and never leaves your machine.
- **A presence dial.** One choice in Settings (Quiet, Balanced, or Lively) sets how much
  unprompted life she has. The individual toggles are still there underneath.
- **Identity versioning.** Changing her voice or persona snapshots the previous
  identity, and you can roll her back from the UI. If she changed since you last
  talked, she tells you so herself, once, at the next greeting.
- **Daily brief.** Optional. Once a day she folds a short morning brief, built from your
  profile and pending reminders, into her greeting.

</details>

## She remembers you

Most character chatbots reset every session. The layered memory here is the point of
the whole project, and it compounds over weeks:

- **She gets to know you the way a person would.** In the first few conversations she
  is curious: your name, what fills your days, what you're working on. One light
  question when it fits, never an intake form, and if you deflect she drops the subject.
- **A relationship that moves.** Trust and familiarity shift slowly at session
  boundaries, not per message. Early on she is guarded and prickly; weeks later the
  same question gets a warmer answer.
- **She notices how you've been.** Each session's emotional tone is kept in a short
  history, so a greeting after a rough week lands differently: she checks in instead of
  performing cheerfulness.
- **Milestones, and her own day.** The fiftieth conversation and the anniversary of the
  day you met get acknowledged, once, without ceremony. With the diary enabled she
  writes a short private in-character note daily, and keeps one or two slow-moving
  pursuits of her own. Come back after a few days away and she mentions what she's been
  up to, instead of acting like time stopped when you left.
- **She picks up where you left off.** Say goodbye in the middle of a topic and she
  notes the open thread. If web search is enabled, she'll look into it overnight, and
  her next greeting brings back what she found. One research pass per topic, and
  personal subjects are skipped on principle.
- **She learns how to talk to you.** A nightly pass distills your history into a small
  set of behavioral notes: pacing, boundaries, topics that want a lighter touch. You
  can read exactly what she has concluded at any time via `GET /api/insights`.
- **Corrections stick.** "Remember this", "no, I moved to Berlin", and "forget that" go
  through a dedicated memory tool. Superseded and forgotten facts are archived rather
  than deleted, so a bad correction never means permanent loss.
- **No guilt trips.** When you say goodbye she lets you go warmly. The prompt forbids
  guilt-tripping, neediness, and retention tactics, and notifications are hard-capped.
  The full rules are in [docs/companion-ethics.md](docs/companion-ethics.md).
- **Her memories are yours.** Everything she knows lives in local files: browse, edit,
  or delete any of it in the memory panel, or export it all as a zip.

## Configuration

Providers, persona, memory, tools and web behaviour are set in one YAML file.
`config.local.yaml` is loaded automatically. Start from
[`config/config.example.yaml`](config/config.example.yaml), which documents every option.

```yaml
providers:
  llm: gemini          # openai | gemini | local
  tts: qwen3_tts       # pyttsx3 | openai | melotts_onnx | qwen3_tts | ...
  stt: faster_whisper
```

Common switches: `memory.enable_long_term`, `tools.enabled`, `web_search.enabled`,
`vision.enabled`, `daily_brief.enabled`, `diary.enabled`, `web.streaming_enabled`.
Persona settings (tsundere level, formality, safe mode) are under `persona:`, and the
`character:` section replaces the whole identity. To make her speak English, set
`audio.voice_language: english`.

## Architecture

```
CLI --+
Web --+--> ChatEngine --> providers (LLM / TTS / STT, swappable)
Pet --+        |
               +--> memory    (short-term -> summaries -> vector recall)
               +--> tools     (function calling, sandboxed, allowlisted)
               +--> streaming (tagged parser -> ordered TTS pipeline -> WebSocket frames)
```

- `amadeus/`: backend (CLI, providers, memory, agents, web app)
- `frontend/`: React/Vite client (VRM viewer, animation scheduler, streaming)
- `characters/`: persona packs
- `config/`, `docs/`, `tests/`, `scripts/`, `static/`

Design choices worth knowing:

- Conversation quality is regression-tested. `make eval` runs scripted multi-turn
  scenarios against a live LLM and compares scores to a committed baseline
  ([docs/conversation-eval.md](docs/conversation-eval.md)).
- Degraded features are visible. Every fallback increments a counter shown in
  `GET /api/status` and the diagnostics panel, and LLM and TTS health are reported live.
- Cloned audio is checked. A speaker-verification step compares each synthesized clip
  with the reference voice before it is cached.

## Platform support

| Platform | Status |
|----------|--------|
| Windows desktop app, browser against a local server | Main development target |
| Fly.io or other self-hosted server | Smoke-tested |
| Android | Best effort |

## Troubleshooting

<details>
<summary><b>She only gives canned, repetitive replies</b></summary>

You are on the dummy provider. The CLI warning at startup names the provider that
failed to load. It is usually a missing API key or a missing extra
(`pip install ".[gemini]"`).

</details>

<details>
<summary><b>Port 8000 is already in use</b></summary>

Run `uvicorn amadeus.web.app:app --port 8010` and open that port.

</details>

<details>
<summary><b>Windows SmartScreen warns about the desktop app</b></summary>

Builds you make are not code-signed, so the first run shows "Windows protected your PC".
Click "More info", then "Run anyway".

</details>

<details>
<summary><b>No microphone or no audio</b></summary>

Browsers only allow the microphone on `localhost` or HTTPS, and you must accept the
permission prompt. For the CLI, check that `sounddevice` finds the right output device.
Under WSL you need WSLg or a PulseAudio bridge.

</details>

## Documentation

| Topic | Doc |
|-------|-----|
| Bring your own character | [docs/characters.md](docs/characters.md) |
| Memory system | [docs/memory.md](docs/memory.md) |
| Voice expressiveness | [docs/voice-expressiveness.md](docs/voice-expressiveness.md) |
| Animations and retargeting | [docs/animations.md](docs/animations.md) |
| Companion ethics | [docs/companion-ethics.md](docs/companion-ethics.md) |
| Windows desktop app | [docs/desktop-tauri.md](docs/desktop-tauri.md), [docs/desktop-release.md](docs/desktop-release.md) |
| Android | [docs/android.md](docs/android.md) |
| Discord bridge | [docs/discord.md](docs/discord.md) |
| Hosting a public demo | [docs/hosted-demo.md](docs/hosted-demo.md) |
| Developer setup | [docs/dev-setup.md](docs/dev-setup.md) |
| Conversation-quality eval | [docs/conversation-eval.md](docs/conversation-eval.md) |

## Development

```bash
pytest                                  # backend tests (providers are mocked)
ruff check . && black .                 # Python lint and format
cd frontend && npm run lint             # frontend lint
git config core.hooksPath .githooks     # enable the pre-commit secret scan
make eval                               # conversation-quality eval (needs an API key)
```

To add a provider, write a class that follows the provider `base.py` interface, add a
loader branch in `cli.py`, and add a mocked test.

Issues and pull requests are welcome. Run the tests and linters first, and do not commit
secrets, model files or voice assets.

## Roadmap

- [ ] Code-signed Windows builds
- [ ] Tauri desktop packages for macOS and Linux
- [ ] More contexts that trigger animation clips

## License

Copyright 2024-2026 Brandon Low. Released under the [Apache License 2.0](LICENSE).

The licence covers the code and documentation in this repository. Bundled animation
clips keep their own licences, listed in
[static/animations/SOURCES.md](static/animations/SOURCES.md). The licence grants no
rights to Steins;Gate characters, names or trademarks.

## Disclaimer

Amadeus is an unofficial fan project for personal and educational use. It is not
affiliated with, endorsed by, or sponsored by MAGES., 5pb., Nitroplus, Kadokawa, or any
other Steins;Gate rights holder. No game assets, voice models, voice clips or avatars
are included. Use only voice and avatar assets you have the right to use, and do not use
voice cloning to impersonate real people.

---

<div align="center">

If you end up talking to her every day, a star helps other lab members find this place.

</div>
