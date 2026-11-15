# Desktop Release (Windows)

The Windows desktop build is a Tauri shell around a PyInstaller-packaged
FastAPI backend (`amadeus-backend.exe`), producing an NSIS installer and a
portable zip. This doc covers the `assets-v1` runtime-asset release, cutting
a desktop release, and building locally.

## Runtime assets (`assets-v1`)

Voice model, VRM avatar, and environment scenes are large binary assets and
are **not committed to git**. They are distributed as a single GitHub
Release asset:

- Repo: `zxKyouma/Amadeus`
- Release tag: `assets-v1`
- Asset file: `amadeus-runtime-assets.zip`

Unzipping `amadeus-runtime-assets.zip` at the repo root must produce:

```
models/kurisu_voice_onnx/model.onnx
models/kurisu_voice_onnx/tokens.txt
models/kurisu_voice_onnx/lexicon.txt
static/models/Kurisu.vrm
static/environment/steinsgate_-_future_gadget_lab.glb
static/environment/steinsgate_time_machine.glb
```

(`assets/kurisu_isms.txt` is already tracked in git and is not part of this
archive.) This list matches `amadeus.desktop.runtime.REQUIRED_ASSET_PATHS` —
that is the single source of truth; if it changes, rebuild the zip to match.

The bundle **also** contains the Qwen3 voice-clone reference clip
(`models/kurisu_voice_qwen/kurisu_ref.wav`,
`amadeus.desktop.runtime.QWEN_REF_ASSET_PATH`). It is deliberately *not* in
`REQUIRED_ASSET_PATHS`: only the GPU build needs it, so its absence must not
fail the default CPU packaging. See "GPU variant" below.

### Creating/updating the `assets-v1` release

From the repo root, with the required files already in place on disk:

```bash
cd /path/to/Amadeus
zip -r amadeus-runtime-assets.zip \
    models/kurisu_voice_onnx/model.onnx \
    models/kurisu_voice_onnx/tokens.txt \
    models/kurisu_voice_onnx/lexicon.txt \
    static/models/Kurisu.vrm \
    static/environment/steinsgate_-_future_gadget_lab.glb \
    static/environment/steinsgate_time_machine.glb

# First time:
gh release create assets-v1 amadeus-runtime-assets.zip \
    --repo zxKyouma/Amadeus \
    --title "Runtime assets v1" \
    --notes "Voice model, VRM avatar, and environment assets for the desktop/web build."

# Updating an existing assets-v1 release:
gh release upload assets-v1 amadeus-runtime-assets.zip --repo zxKyouma/Amadeus --clobber
```

## Cutting a desktop release

The `release.yml` workflow builds the Windows installer + portable zip and
publishes them to a GitHub Release.

1. Bump versions as needed (`frontend/src-tauri/tauri.conf.json`,
   `frontend/package.json`, `pyproject.toml`).
2. Tag and push. (`vX.Y.Z` here and in every command below is a
   placeholder: substitute the version being cut, e.g. `v0.16.0`.)

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

   Pushing a `v*` tag triggers the workflow on the **self-hosted Windows
   runner** (see docs/self-hosted-runner.md — GitHub's hosted Windows
   runners are unavailable on this plan, and the self-hosted box builds
   faster anyway). Three Windows jobs run, serially on the single runner:

   - `build-windows` (~25 min): downloads `assets-v1`, builds the CPU
     sidecar (`package_backend_windows.ps1 -SkipAssetFetch`), produces the
     **signed** NSIS installer via the empty-password node wrapper,
     assembles the portable zip, verifies the zip's file manifest (exe +
     `frontend/dist` + `static` + `assets` present), smoke-tests the built
     sidecar (`/healthz` within 120s on a throwaway config — the
     dummy-provider fallback means no API keys needed), generates
     `latest.json` with `make_update_manifest.py`, and uploads the
     installer + `.sig` + manifest + portable zip to the tag's release.
   - `build-windows-gpu-slim` (~20 min): builds the **signed slim GPU
     installer** (`package_desktop_windows.ps1 -GpuSlim`), generates
     `latest-gpu.json` (`make_update_manifest.py --variant gpu`), uploads
     both. The auto-updatable GPU channel is fully CI-built.
   - `build-windows-gpu` (~30 min): builds the full CUDA sidecar and the
     no-bundle GPU shell, assembles the offline portable folder, splits it
     into <2GiB 7z volumes, verifies contents, boot-smokes the GPU
     sidecar, uploads the `.7z.0*` parts. See "GPU variant" below for what
     this does and doesn't prove.

   All three run under PowerShell 5.1 with the ExecutionPolicy-bypass shell
   template (workflow comments explain; keep `run:` blocks ASCII-only — a
   literal em dash in a step script breaks 5.1's parser because the runner
   writes scripts UTF-8-without-BOM and 5.1 reads them as ANSI). The
   updater signing key resolves from the `TAURI_SIGNING_PRIVATE_KEY`
   repository secret when set, else falls back to the runner machine's
   `local-assets/updater-keys/amadeus_updater.key`. Runner machine deps
   beyond the toolchains: `gh` (authenticated), 7-Zip, and SoX for the
   full-GPU job (resolved via `SOX_DIR` → choco's lib dir → `PATH`).

   Note: pushing the tag does **not** create the GitHub Release itself —
   create it first (e.g. `gh release create vX.Y.Z --generate-notes`) so
   there is a release for the workflow to upload assets to.

   Once the installer is built and signed, run
   `docs/pet-release-checklist.md` on the packaged Windows app before
   calling the release done, that surface only fails on a real Windows
   machine, never in CI.

3. To test the build without publishing, run the workflow manually via
   **Actions → Release Desktop (Windows) → Run workflow**
   (`workflow_dispatch`). In that mode the installer and zip are attached as
   a downloadable build artifact instead of being uploaded to a release. The
   GPU jobs are skipped on dispatch unless the `include_gpu` input is
   checked (they add ~50 min of runner time).

This workflow is independent of `ci.yml` (ubuntu-only backend tests/lint)
and `fly-deploy.yml` (Fly.io web deploy on push to `main`) — neither of
those is touched by desktop releases.

## Auto-updater

Installed (NSIS) copies of the app update themselves. On launch the app asks
`https://github.com/zxKyouma/Amadeus/releases/latest/download/latest.json`
for a newer signed build. When one exists, the settings panel shows an
"Update & restart" button that downloads the installer with a progress bar,
stops the backend sidecar, runs the installer passively, and relaunches.
Portable and GPU zips never auto-update: they contain a `portable.marker`
file next to the exe, and the app shows the plain releases-page nudge
instead. Auto-installing would put a fresh copy in Program Files and leave
the portable folder stale, and for the GPU build it would silently swap the
cloned voice for the ONNX one.

> **Why the FULL GPU build can never be the auto-update artifact** (investigated
> 2026-07-31, so nobody re-litigates it): two independent hard walls apply to
> the *full* GPU build (CUDA torch bundled directly). First, makensis cannot
> pack the CUDA payload — `torch_cuda.dll` alone is ~900MB and the build dies
> with "error mmapping datablock" (the reason the GPU variant went
> portable-only in the first place; compression settings don't rescue a
> 32-bit address-space limit). Second, even if it packed, the compressed GPU
> payload measures ~2.3GB (v0.12.0 shipped it as split 7z parts of 1900MB +
> 422MB) — over GitHub's 2GB per-asset cap — and tauri-plugin-updater
> consumes exactly one download URL, so split archives can't ride
> `latest.json`. The full-GPU portable zip is therefore structurally
> never auto-updatable.
>
> The workaround, shipped in this pass: a **slim GPU installer** that
> excludes the CUDA venv entirely and fetches it as a separate "GPU runtime
> pack" release on first launch (the same download-on-first-run pattern the
> ~8.5GB Qwen3 HF weights already use). Because the installer payload is then
> CPU-sized, both walls disappear — makensis packs it fine and it's well
> under the 2GB asset cap — so the slim installer rides `latest-gpu.json` and
> auto-updates like the CPU build. See "GPU variant" below for the full
> shape.

What this means for cutting a release:

1. Every release must include three updater pieces: the signed
   `*-setup.exe`, its `*.sig`, and a `latest.json` manifest. The CI workflow
   does all of it. For a local release, build with the signing key in the
   environment, then run the manifest script and upload:

   ```powershell
   # Tauri v2 reads TAURI_SIGNING_PRIVATE_KEY (a path or the key content).
   # There is no _PATH variant: with only _PATH set the build fails at the
   # very end with "A public key has been found, but no private key".
   $env:TAURI_SIGNING_PRIVATE_KEY = "C:\path\to\amadeus_updater.key"
   # The password var must ALSO be present (the key has none, so present but
   # EMPTY) or the CLI prompts interactively after makensis and hangs any
   # scripted build at "Decrypting updater signing key". Windows shells
   # CANNOT set an empty env var - $env:VAR = "", cmd's `set VAR=`, and
   # SetEnvironmentVariable all DELETE it instead (this hung the v0.15.0
   # build for an hour). Wrap the build in the node helper, which builds the
   # child environment block itself and preserves the empty entry:
   node scripts\win_empty_signing_password.cjs npm run tauri:build
   ```

   If a build did run without the password var (installer exists, no `.sig`,
   build hung at the prompt), sign the existing installer manually instead of
   rebuilding — note the '""' quoting, which is how an empty argument
   survives PowerShell:

   ```powershell
   npx tauri signer sign --private-key-path "C:\path\to\amadeus_updater.key" `
       --password '""' "src-tauri\target\release\bundle\nsis\<name>-setup.exe"
   ```

   ```bash
   python scripts/make_update_manifest.py --tag vX.Y.Z
   # Upload the STAGED copies the script printed (dist-desktop/release-assets/),
   # not the originals: GitHub renames assets on upload (spaces become dots),
   # which would 404 the manifest URL.
   gh release upload vX.Y.Z latest.json \
       "dist-desktop/release-assets/<name>-setup.exe" \
       "dist-desktop/release-assets/<name>-setup.exe.sig" --clobber
   ```

   The script refuses to write a manifest when the `.sig` is missing or the
   tag does not match the version in `tauri.conf.json`, and it stages
   rename-proof copies of the installer + sig for upload.

2. The signing keypair lives OUTSIDE git, in
   `local-assets/updater-keys/amadeus_updater.key` (private) and `.key.pub`
   (public, also baked into `tauri.conf.json`). Back the private key up
   somewhere safe. If it is ever lost, installed apps will reject all future
   updates and every user has to reinstall by hand once. The key currently
   has no password. For CI, store the private key contents in the
   `TAURI_SIGNING_PRIVATE_KEY` repository secret (and the password in
   `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` if one is ever set).

3. A release that ships no `latest.json` is harmless: installed apps simply
   see no update and the old releases-page nudge still works. A release with
   a BAD latest.json (wrong signature, wrong URL) makes updates fail
   quietly, so prefer letting the script generate it.

The Rust side lives in `check_app_update` and `install_app_update` in
`frontend/src-tauri/src/main.rs` (the updater plugin behind app commands,
same pattern as autostart). The UI lives in
`frontend/src/components/UpdateNudge.tsx` with `frontend/src/appUpdater.ts`.

## GPU variant (Qwen3 cloned voice)

The default desktop build is **CPU-only and ships the ONNX voice** — the
sidecar deliberately excludes `torch`/`transformers` to stay small. Three GPU
artifacts exist on top of that, each with a different tradeoff:

| Artifact | Payload | Auto-updates? | Built by |
|---|---|---|---|
| **Slim GPU installer** (`Amadeus Kurisu (GPU)` NSIS, `-GpuSlim`) | CPU-sized (no CUDA venv); fetches the GPU runtime pack on first launch | Yes, via `latest-gpu.json` | `package_desktop_windows.ps1 -GpuSlim` / `build-windows-gpu-slim` CI job |
| **Full GPU portable** (`Amadeus-Kurisu-GPU-portable-windows`, `-Gpu`) | ~2.3GB compressed, CUDA torch bundled directly | No — fully offline, `portable.marker` disables the updater | `package_desktop_windows.ps1 -Gpu` / `build-windows-gpu` CI job |
| **GPU runtime pack** (`gpu-runtime-v2` release) | The CUDA torch + qwen-tts package set + `sox`, split-zipped | N/A — it's the thing being fetched, not an app | `package_gpu_runtime.ps1`, local only, republished only when the torch/qwen dependency set changes (NOT per app release) |

The slim installer is the one to hand most users going forward — it behaves
like the CPU installer (small download, auto-updates) and self-upgrades to
the cloned voice in the background after install. The full GPU portable zip
remains for fully-offline use (air-gapped machines, or anyone who wants the
voice available immediately with no first-run download). Both still fall
back to the ONNX voice on a non-NVIDIA machine or if CUDA setup fails — see
"Runtime selection with fallback" below.

Both GPU app artifacts are CI-built on every `v*` tag: the slim installer
by `build-windows-gpu-slim` (signed, with its `latest-gpu.json` manifest)
and the full portable by `build-windows-gpu` (split 7z volumes). Only the
**GPU runtime pack** stays a local build — it needs the full GPU packaging
venv plus the frozen-import probe, and it republishes on its own
`gpu-runtime-v2` release only when the torch/qwen dependency set changes,
not per app release. The checklist below remains the recipe for that.

How it works end to end:

- **Runtime selection with fallback.** The GPU installer's Tauri shell is
  built with the `qwen` Cargo feature, so it passes `--prefer-qwen` to the
  sidecar. `build_desktop_config(prefer_qwen=True)` selects `qwen3_tts` **only
  when both** a CUDA GPU is present (`runtime.cuda_available()`) **and** the
  bundled `kurisu_ref.wav` exists; otherwise it falls back to the ONNX voice.
  On top of that, if the Qwen3 provider fails to *load* at runtime (weights not
  downloaded yet, CUDA OOM, missing `sox`), `WebChatSession` retries with the
  ONNX voice (`amadeus/web/app.py`) rather than going silent. So a GPU
  installer launched on a machine without an NVIDIA GPU still speaks — just in
  the ONNX voice.
- **Model weights are NOT bundled.** The ~8.5GB `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
  weights are downloaded from Hugging Face on first run (the provider's
  existing `snapshot_download` behavior). Only the ~483KB reference clip ships
  in the installer, via the `assets-v1` bundle. First voice reply therefore
  waits on that one-time download plus CUDA-graph warmup.
- **`sox` is bundled.** `qwen-tts` shells out to the native `sox` binary (not a
  pip package). `package_backend_windows.ps1 -Gpu` bundles `sox.exe` + its DLLs
  into the sidecar via PyInstaller `--add-binary`. The CI job resolves SoX
  from the runner machine — `$env:SOX_DIR` if set, else choco's
  `sox.portable` lib dir, else `sox.exe` on `PATH` (with its DLLs beside
  it); for local builds set `SOX_DIR` to a folder containing `sox.exe` (or
  put it on `PATH`).
  The **slim** installer's sidecar has none of this bundled — `sox` and the
  CUDA/qwen packages both arrive in the GPU runtime pack instead (see below).

Build the full GPU portable locally (on a Windows machine with an NVIDIA GPU
to validate):

```powershell
# Full GPU desktop build
powershell -ExecutionPolicy Bypass -File scripts/package_desktop_windows.ps1 -Gpu

# Or just the GPU sidecar
powershell -ExecutionPolicy Bypass -File scripts/package_backend_windows.ps1 -Gpu
# then, in frontend/:  npm run tauri:build:gpu
```

The `-Gpu` switch uses a separate packaging venv (`.venv-package-gpu`) so it
never clobbers the CPU build's dependencies.

> **CI cannot validate the voice.** `build-windows-gpu` proves the artifact
> *builds* and its sidecar *boots* (`/healthz` answers on a throwaway
> config), but a green job ≠ a working voice. The self-hosted runner does
> have an NVIDIA GPU, so the smoke exercises more than the old hosted
> runners could — but nothing in CI listens to the output, and the
> speaker-drift/quality class of regression only shows up by ear. Validate
> the packaged GPU build on a real machine (per
> docs/pet-release-checklist.md) before trusting a release.

### Slim GPU installer + GPU runtime pack

The slim installer ships the same CPU-sized sidecar as the default build
(built WITHOUT `-Gpu`, so no CUDA venv), but its Tauri shell is still built
with the `qwen` Cargo feature and `tauri.gpuslim.conf.json` (product name,
bundle identifier, and its own updater endpoint `latest-gpu.json`, so it
never gets offered the CPU build as an "update"). It must NOT reuse
`tauri.gpu.conf.json`: that config serves the no-bundle full-GPU path and
bundles the staged onedir CUDA backend as a resource (with `externalBin`
cleared), which fed makensis a ~1.96GB payload and reproduced the exact
mmap death the slim build exists to dodge (found live cutting v0.15.0).
`amadeus/gpu_runtime.py` (owned by the backend, not this doc's scripts)
downloads the CUDA torch + qwen-tts package set from the `gpu-runtime-v2`
release on first launch, in the background, while the app speaks in the
ONNX voice; once the download completes the app needs one restart to
activate the pack (`ready_restart` state) and switch to the cloned voice.

`tauri.gpuslim.conf.json` only sets `plugins.updater.endpoints` — it does
not repeat `pubkey` or `windows.installMode`. That's safe because Tauri v2's
`--config` override deep-merges each top-level key onto the base
`tauri.conf.json` (a JSON merge-patch, not a wholesale replace), so a
partial `plugins.updater` object in the override file adds/replaces just
`endpoints` while `pubkey` and `windows.installMode` fall through from the
base config unchanged. (JSON has no comment syntax, which is why this note
lives here instead of inline in the config file.)

Backend env overrides for testing this locally (see `amadeus/gpu_runtime.py`
for the authoritative behavior):

- `AMADEUS_GPU_RUNTIME_AUTOFETCH=0` — disable the automatic first-launch
  download (useful for testing the ONNX-fallback path without waiting on a
  multi-gigabyte fetch).
- `AMADEUS_GPU_RUNTIME_DIR` — override where the pack is downloaded/extracted
  (default: under the app data dir).
- `AMADEUS_GPU_RUNTIME_URL_BASE` — override the manifest/parts URL base
  (default: `https://github.com/zxKyouma/Amadeus/releases/download/gpu-runtime-v2`);
  point this at a local `file://` or HTTP server to test the fetch/extract
  path without publishing a real release.

The ~2.3GB first-run download must be disclosed up front — put it in the
release notes and in the installer/app description (something like "the GPU
voice pack downloads automatically on first launch, ~2.3GB over your
connection"), since it's easy to mistake for a hang if the user doesn't
expect it.

#### First slim GPU release checklist

Steps 5–7 (slim installer build, `latest-gpu.json`, upload) are what the
`build-windows-gpu-slim` CI job now does on every tag — this checklist is
the manual recipe for the **runtime pack** (steps 1–5, which stay local)
and for cutting a release without CI.

Order matters: the runtime pack needs BOTH the CPU and GPU packaging venvs to
compute its diff, and it should exist on the release before anyone installs
the slim build that will try to fetch it.

1. **Build the CPU backend venv** (if not already built):
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/package_backend_windows.ps1
   ```
2. **Build the GPU backend venv** (this does NOT produce the
   artifact you ship, just populates `.venv-package-gpu`):
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/package_backend_windows.ps1 -Gpu
   ```
3. **Build the runtime pack** (full GPU site-packages + stdlib overlay —
   NEVER a CPU-venv diff; v1 diffed and `import torch` died inside the
   frozen sidecar on modules the diff assumed the sidecar had):
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/package_gpu_runtime.ps1
   ```
4. **Run the frozen-import probe against the staged pack** — this is the
   gate that would have caught the v1 silent-ONNX-fallback bug. A plain
   venv python proves nothing (it has the full stdlib the frozen sidecar
   prunes). Build + usage instructions live in `scripts/probe_gpu_pack.py`;
   publish only on `RESULT: ALL-OK`.
5. **Publish `gpu-runtime-v2`** using the `gh release create`/`upload` command
   the script prints (it does not run `gh` itself). Do this BEFORE
   distributing the slim installer — a slim install's first launch will
   fail its download otherwise (it retries on next launch, but don't ship
   that experience on purpose).
5. **Build the slim installer** (signed, same env vars as the CPU NSIS build
   — see "Auto-updater" above):
   ```powershell
   $env:TAURI_SIGNING_PRIVATE_KEY = "C:\path\to\amadeus_updater.key"
   # node wrapper = the only way to hand the build an EMPTY password var on
   # Windows (see "Auto-updater" above; $env: assignment deletes it).
   node scripts\win_empty_signing_password.cjs powershell -ExecutionPolicy Bypass -File scripts\package_desktop_windows.ps1 -GpuSlim
   ```
6. **Generate the manifest**:
   ```bash
   python scripts/make_update_manifest.py --tag vX.Y.Z --variant gpu
   ```
7. **Upload** the setup exe + its `.sig` + `latest-gpu.json` to the app
   release (the same `vX.Y.Z` release the CPU installer goes to — the GPU
   runtime pack is the only piece that lives on its own `gpu-runtime-v2`
   release):
   ```bash
   # Staged rename-proof copies, written by make_update_manifest.py — never
   # upload the original "(GPU)" filenames (GitHub rewrites them and the
   # manifest URL 404s).
   gh release upload vX.Y.Z latest-gpu.json \
       "dist-desktop/release-assets/Amadeus.Kurisu.GPU_X.Y.Z_x64-setup.exe" \
       "dist-desktop/release-assets/Amadeus.Kurisu.GPU_X.Y.Z_x64-setup.exe.sig" \
       --repo zxKyouma/Amadeus --clobber
   ```

**Validation** (all Windows-only; first exercised live cutting v0.15.0, and
the qwen voice path was validated end-to-end on 2026-08-01 — re-run this
list whenever the pack or the slim packaging path changes):

- [ ] `makensis` succeeded (the whole point of the slim build) — no "error
      mmapping datablock".
- [ ] The produced `*-setup.exe` is under 1.9GB (`package_desktop_windows.ps1
      -GpuSlim` checks this and fails loudly if not).
- [ ] A fresh install on a clean machine launches and speaks in the ONNX
      voice immediately (no waiting on the GPU pack for first sound).
- [ ] `/api/status`'s `gpu_runtime` block shows `downloading` shortly after
      launch, then `ready_restart` once the pack finishes.
- [ ] After restarting the app, the Qwen3 cloned voice is actually in use —
      cross-check with `nvidia-smi` that the backend process shows up with
      GPU memory allocated (confirms `torch.cuda.is_available()` is True and
      the pack activated, not a silent ONNX fallback wearing the GPU shell).
- [ ] Updater round-trip: install an OLDER slim build, publish a newer
      `vX.Y.Z` with a fresh `latest-gpu.json`, confirm the in-app "Update &
      restart" flow finds and installs it (and does NOT offer the CPU
      installer or vice versa — the two channels must stay isolated).

### Downloading and reassembling the full GPU portable release (for users)

The GPU release is published as **split 7-Zip volumes** instead of one file,
because GitHub release assets are capped at 2 GiB and the GPU payload
(bundled CUDA torch) is around 3.2 GB. You'll see multiple files attached to
the release named like:

```
Amadeus-Kurisu-GPU-portable-windows.7z.001
Amadeus-Kurisu-GPU-portable-windows.7z.002
Amadeus-Kurisu-GPU-portable-windows.7z.003
...
```

To reassemble and extract:

1. Download **all** of the `.7z.001`, `.7z.002`, ... parts into the same
   folder (missing a part makes extraction fail partway through).
2. Install [7-Zip](https://www.7-zip.org/) if you don't already have it.
3. Select all of the `.7z.*` parts together (e.g. in File Explorer, click
   `.7z.001`, shift-click the last part to select the whole run).
4. Right-click the selection → **7-Zip → Extract Here** (or **Extract to
   "Amadeus-Kurisu-GPU-portable-windows\"**).
   7-Zip automatically joins the volumes starting from `.001` — you do not
   need to run a separate "combine" step, and you do not need to extract
   from any part other than `.001`.
5. Run `Amadeus Kurisu (GPU).exe` from the extracted folder.

If extraction fails or looks truncated, re-download the parts — a partial or
corrupted download of any single volume will break the whole archive.

## Building locally

Requirements: Windows, Python 3.11 (or `py -3.11` via the Python launcher),
Node 20, Rust (`rustup`), and the GitHub CLI (`gh`, authenticated) if you
want automatic asset fetching.

```powershell
# Full desktop build: frontend, sidecar, Tauri installer, portable zip
powershell -ExecutionPolicy Bypass -File scripts/package_desktop_windows.ps1

# Just the backend sidecar (amadeus-backend.exe)
powershell -ExecutionPolicy Bypass -File scripts/package_backend_windows.ps1
```

Useful switches on `package_backend_windows.ps1` (both scripts accept
`-IncludeSTT` / `-SkipAssetFetch`, forwarded by `package_desktop_windows.ps1`):

- `-VenvPath <dir>` — packaging venv location, default `.venv-package`.
  The script always installs into a **clean, isolated venv** (created via
  `py -3.11 -m venv`, falling back to `python -m venv`), never the global
  Python — this is what keeps the PyInstaller output free of unrelated
  packages (previously a global TensorFlow install bloated the installer to
  368MB and forced defensive `--exclude-module` flags). Delete the venv
  directory to force a fully clean rebuild.
- `-IncludeSTT` — also installs `.[stt]` (faster-whisper) and bundles it
  into the sidecar (`--hidden-import faster_whisper --collect-all
  ctranslate2`). Off by default; only needed if the desktop build should
  ship local speech-to-text.
- `-SkipAssetFetch` — skip the runtime-asset presence check/download
  entirely. Use this when assets are already unpacked at the repo root (as
  CI does, since it fetches them in a separate step) or when you're
  intentionally building without them for a quick smoke test.
- `-Python <name>` / `-TargetTriple <triple>` — as before, controls which
  Python bootstraps the venv and the Tauri sidecar filename suffix.

Without `-SkipAssetFetch`, the script checks for the required assets
(`amadeus.desktop.runtime.REQUIRED_ASSET_PATHS`) and, if any are missing,
downloads and unpacks `assets-v1` automatically via `gh release download`
(requires `gh` to be installed and authenticated). If assets are still
missing afterward, the script fails with a clear list of what's absent.

### Known fix baked into the script

The PyInstaller build now passes `--collect-data pykakasi` — without it the
packaged app fails to find `pykakasi/data/kanwadict4.db` at runtime
(`No such file or directory: ..._MEIxxxx\pykakasi\data\kanwadict4.db`),
which silently disables voice playback. `unidic_lite` data collection was
already covered via `--collect-all unidic_lite`.
