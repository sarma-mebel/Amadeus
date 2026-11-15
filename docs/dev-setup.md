# Developer setup

This is the bootstrap for a second developer (or a fresh machine) getting
Amadeus running. It assumes WSL2 on Windows, matching how the project has
actually been developed and tested, but the backend steps are otherwise
plain Python and should work on native Linux or macOS too. The frontend
desktop packaging step is Windows-specific by nature (it produces a Windows
installer).

Run `python scripts/doctor.py` at any point to get a PASS/FAIL/WARN report
of where your environment stands against the checks below.

## Prerequisites

- WSL2 with a recent Ubuntu, and conda (miniconda or anaconda) inside it.
  The project has been developed against Python 3.10 in a conda env; a
  plain venv works too since nothing here is conda-specific beyond the
  environment.yml convenience file.
- Node.js. The project doesn't pin a version in package.json, but CI and
  these docs standardize on Node 20 LTS or newer.
- The Rust toolchain, only if you're going to touch the Tauri desktop shell
  (`frontend/src-tauri`). On this kind of WSL2 setup, Rust is usually
  installed on the Windows side, not inside WSL, because the actual desktop
  build runs through Windows PowerShell. That Windows-side cargo is still
  usable from WSL for read-only checks like `cargo check`: it typically
  lives at `/mnt/c/Users/<your-windows-username>/.cargo/bin/cargo.exe`. You
  do not need a WSL-native Rust install unless you specifically want one.

## Backend setup

Create the conda environment and activate it:

```bash
conda env create -f environment.yml
conda activate amadeus
```

Install the package in editable mode plus whatever extras you need. `[web]`
is the one that matters most for development: most of the test suite
imports `fastapi` and skips itself when it's missing, so without `[web]`
you will see a pile of skipped tests, not a clean pass.

```bash
pip install -e ".[web]"
pip install -e ".[voice]" ".[openai]" ".[gemini]" ".[memory]" ".[search]"   # add as needed
```

See `environment.yml` for the scar-tissue pins (torch's CUDA build,
the transformers pin, faster-qwen3-tts) if you're setting up the qwen3
voice path. `requirements-lock.txt` is a working-state snapshot of a real
dev environment, useful for diffing when something behaves differently on
your machine than expected; it is not the primary install path.

## Config bootstrap

Copy the example config and edit it with your own API keys (or leave it as
is to run on dummy providers):

```bash
cp config/config.example.yaml config.local.yaml
```

`Config.load()` looks for `config.local.yaml` first, so this file is your
personal, gitignored config. Without real API keys, providers fall back to
dummies silently; check `GET /api/status` (or the CLI's `/status`) to see
what's actually running.

## Enable the secret-scan hook

The repo ships a pre-commit hook that scans staged files for common secret
patterns (API keys, tokens). It's opt-in per clone:

```bash
git config core.hooksPath .githooks
```

## Running things

CLI chat:

```bash
python -m amadeus --config config.local.yaml
```

Web backend:

```bash
uvicorn amadeus.web.app:app --reload
```

Frontend dev server:

```bash
cd frontend
npm install
npm run dev
```

## Known gotchas

**Run pytest from the repo root, not from frontend/.** `frontend/src-tauri`
can contain a bundled PyInstaller backend build (a full copy of torch and
its own test files) under `src-tauri/target` once you've packaged the
desktop app at least once. If pytest collection starts from inside
`frontend/`, it walks into that tree and dies with collection errors on
files it was never meant to see. Always run `pytest` (or
`python -m pytest`) from the repository root.

**Something often squats port 8000.** On WSL setups it's common for a
Windows-side process to already be listening on 127.0.0.1:8000, which makes
uvicorn fail to bind with an error that looks like a backend problem but
isn't. Check with `scripts/doctor.py` or just try a different port
(`uvicorn amadeus.web.app:app --port 8010`) before assuming the app itself
is broken.

**Plain `npx vitest run` can silently drop test files.** Under a checkout on
`/mnt/c/...` (an NTFS-backed path accessed from WSL), jsdom worker startup
can be slow enough to time out, and vitest has been observed to silently
drop the affected test files while still exiting 0, so a broken run can
look green. Use `npm run test:ci` (a wrapper that guards against this
silent-drop failure mode) instead of a bare `vitest run`. If you need to
run vitest directly, pace it manually: `vitest run --pool=threads
--maxWorkers=1` with generous test/hook timeouts (120 seconds is a safe
starting point).

**Windows packaging goes through PowerShell.** `npm run
package:desktop:windows` shells out to
`scripts/package_desktop_windows.ps1` via `powershell -ExecutionPolicy
Bypass`. That flag is expected, not a workaround for something broken; the
script isn't signed and the default execution policy on most Windows
machines blocks unsigned scripts.

**WSL and Windows can disagree about localhost.** The backend binds IPv4
`127.0.0.1`. Some Windows-side clients (and some tools inside WSL, in the
other direction) resolve `localhost` to the IPv6 loopback `::1` first,
which the backend isn't listening on, and you get a connection refused that
looks like the server isn't running. If `localhost` fails, try
`127.0.0.1` explicitly before troubleshooting further. The frontend's own
dev proxy already talks to `127.0.0.1:8000` for this reason.

**A "working" assistant might be running on dummy providers.** Missing an
API key or an extra doesn't produce a startup error; the loaders in
`cli.py` and `web/app.py` fall back to `DummyProvider` silently. If replies
look canned or generic, check `/status` (CLI) or `GET /api/status` (web)
before assuming there's a bug in prompt or memory handling.

**`black` needs `--fast` in some environments.** Depending on the installed
black version's bundled Python grammar tables, its safety check (which
re-parses the formatted output to confirm it's equivalent) can trip on a
py3.10 target when black's own default target-version detection reaches
for a newer grammar than the interpreter running it supports. If you hit
that, run `black --fast` to skip the re-parse safety check. This project
doesn't pin black as a hard dependency; install it separately
(`pip install black`) if you want it, or use whatever formatter setup you
already have for Python.

## Fast iteration loops (avoid the full packaged build)

The full release build (PyInstaller sidecar + tauri release + NSIS) takes
tens of minutes and is only needed to test *packaging itself*. Pick the
cheapest tier that can catch the bug you're chasing:

**Tier 1 — seconds (most code changes).**
- Backend logic: `uvicorn amadeus.web.app:app --reload` and talk to it from
  a browser (`npm run dev` for the frontend, Vite HMR).
- Shell + UI together: `npm run tauri:dev` — the webview loads the dev
  server with HMR, and Rust changes recompile incrementally in debug mode.
- Frontend tests from WSL: `scripts/vitest-win.sh` (Windows-native node;
  the full suite runs in ~30s — see the project notes for why WSL node is 70x
  slower here).

**Tier 2 — seconds-to-minutes, no build (packaged-only bug classes).**
- Tauri ACL / command wiring: `pytest tests/test_contracts.py` statically
  pins invoke ↔ build.rs ↔ capabilities ↔ generate_handler.
- GPU runtime pack self-sufficiency: `scripts/probe_gpu_pack.py` (the probe
  exe builds once, ~4 min, then gating a changed pack takes seconds).
- Desktop config behavior (write-once, reconcile): `pytest
  tests/test_desktop_runtime.py`.
- Rust compile check without a bundle: `cargo check --features qwen` in
  `frontend/src-tauri` (~2 min warm).

**Tier 3 — the real build, made cheaper.**
- Sidecar-only iteration: `package_backend_windows.ps1 -Iter` reuses the
  venv dependency set (only the amadeus package is reinstalled) and keeps
  PyInstaller's analysis cache (no `--clean`). Never use `-Iter` for a
  release: dependency changes since the last full run won't be picked up.
- Release-mode shell without the installer: `tauri build --no-bundle`.
- Windows Defender: keep the build trees excluded from real-time scanning
  (`frontend/src-tauri/target`, `.venv-package*`, `dist-desktop`) — with no
  exclusions every PyInstaller/cargo/NSIS write is scanned inline, which
  measurably drags exactly these builds.
- After ANY packaged build you intend to hand to someone:
  `python scripts/smoke_desktop.py` (close running Amadeus instances
  first — it refuses to run over them).
