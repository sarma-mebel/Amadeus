param(
    [string]$Python = "python",
    [string]$TargetTriple = "x86_64-pc-windows-msvc",
    [string]$VenvPath = ".venv-package",
    [switch]$IncludeSTT,
    [switch]$SkipAssetFetch,
    # GPU variant: bundle the Qwen3 cloned voice (torch CUDA + qwen-tts +
    # faster-qwen3-tts + soxr). Produces a much larger sidecar that only runs
    # the cloned voice when a CUDA GPU is present at runtime, falling back to
    # the ONNX voice otherwise. Off by default (the shipped desktop build is
    # CPU/ONNX-only). Uses a separate venv so the CPU and GPU builds don't
    # clobber each other's dependencies.
    [switch]$Gpu,
    # Fast iteration rebuild: reuse the venv's dependency set as-is (only the
    # amadeus package itself is reinstalled, --no-deps, so code changes land)
    # and keep PyInstaller's analysis cache (no --clean). Cuts the
    # change-code -> new-sidecar loop to roughly PyInstaller pack time.
    # NEVER use for a release build: a dependency added or updated since the
    # venv was built silently won't be in the artifact.
    [switch]$Iter
)

if ($Gpu -and $VenvPath -eq ".venv-package") {
    $VenvPath = ".venv-package-gpu"
}

$ErrorActionPreference = "Stop"
$env:Path = "$env:USERPROFILE\.cargo\bin;C:\Program Files\nodejs;$env:Path"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$TauriDir = Join-Path $RepoRoot "frontend\src-tauri"
$BinaryDir = Join-Path $TauriDir "binaries"
$SidecarName = "amadeus-backend"
$SidecarExe = "$SidecarName.exe"
$TauriSidecarExe = "$SidecarName-$TargetTriple.exe"

Set-Location $RepoRoot

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [ScriptBlock]$Command,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

# ---------------------------------------------------------------------------
# Runtime asset fetch (models/kurisu_voice_onnx, static/models, static/environment)
# ---------------------------------------------------------------------------
function Test-RequiredAssets {
    & $VenvPython -c "
import sys
from pathlib import Path
from amadeus.desktop.runtime import validate_required_assets
result = validate_required_assets(Path('.'))
if not result.ok:
    print('MISSING:' + ','.join(result.missing))
    sys.exit(1)
sys.exit(0)
"
    return $LASTEXITCODE -eq 0
}

function Invoke-AssetFetch {
    Write-Host "Fetching runtime assets from GitHub Release assets-v1 ..."
    $gh = Get-Command gh -ErrorAction SilentlyContinue
    if (-not $gh) {
        throw "GitHub CLI ('gh') not found. Install it or run with -SkipAssetFetch after placing assets manually (see docs/desktop-release.md)."
    }
    $zipPath = Join-Path $RepoRoot "amadeus-runtime-assets.zip"
    Invoke-Checked { gh release download assets-v1 --pattern "amadeus-runtime-assets.zip" --clobber --dir $RepoRoot } "Runtime asset download (gh release download assets-v1)"
    if (-not (Test-Path -LiteralPath $zipPath)) {
        throw "Runtime asset '$zipPath' was not downloaded from GitHub Release 'assets-v1'. Confirm the release contains an asset named 'amadeus-runtime-assets.zip'."
    }
    Expand-Archive -Path $zipPath -DestinationPath $RepoRoot -Force
    Remove-Item -Force $zipPath -ErrorAction SilentlyContinue
}

# ---------------------------------------------------------------------------
# Clean venv setup
# ---------------------------------------------------------------------------
$VenvDir = Join-Path $RepoRoot $VenvPath
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating clean packaging venv at $VenvPath ..."
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        Invoke-Checked { py -3.11 -m venv $VenvDir } "Create venv (py -3.11)"
    } else {
        Invoke-Checked { & $Python -m venv $VenvDir } "Create venv (python -m venv)"
    }
} else {
    Write-Host "Reusing existing packaging venv at $VenvPath"
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    # Non-Windows python -m venv layout fallback (e.g. local dev on WSL/mac for smoke tests)
    $VenvPython = Join-Path $VenvDir "bin/python"
}

# ---------------------------------------------------------------------------
# Dependency install
# ---------------------------------------------------------------------------
if ($Iter) {
    if (-not (Test-Path $VenvPython)) {
        throw "-Iter needs an existing packaging venv at $VenvPath - run once without -Iter first."
    }
    # Iteration path: dependencies are trusted as-is; only the app package is
    # refreshed (the venv install is non-editable, so without this the
    # sidecar would freeze STALE amadeus code from the last full run).
    Write-Host "-Iter: refreshing only the amadeus package (deps reused as-is)."
    Invoke-Checked { & $VenvPython -m pip install --no-deps --force-reinstall --quiet . } "amadeus package refresh (--no-deps)"
} else {
    Invoke-Checked { & $VenvPython -m pip install --upgrade pip } "pip upgrade"
    Invoke-Checked { & $VenvPython -m pip install ".[desktop-sidecar,web]" } "Python dependency install"
    Invoke-Checked { & $VenvPython -m pip install --no-deps "https://github.com/myshell-ai/MeloTTS/archive/refs/heads/main.zip" } "MeloTTS source install"
}

if ($IncludeSTT -and -not $Iter) {
    Invoke-Checked { & $VenvPython -m pip install ".[stt]" } "STT dependency install"
}

if ($Gpu -and -not $Iter) {
    # IMPORTANT: the DEFAULT PyPI torch wheel on Windows is CPU-ONLY. Installing
    # ".[qwen-tts]" alone gives a torch that reports torch.cuda.is_available()
    # == False even on an NVIDIA machine, so the app silently uses the ONNX
    # voice. Install the CUDA torch wheel EXPLICITLY and FIRST, so the extra's
    # "torch>=2.1" is already satisfied by the CUDA build and pip won't swap it
    # for the CPU wheel. Override the CUDA channel with $env:TORCH_CUDA_INDEX.
    #
    # Default is cu128: it is the first CUDA build with Blackwell / sm_120
    # kernels (RTX 50-series, e.g. RTX 5060). Older channels like cu124 install
    # but fail at model load with "CUDA error: no kernel image is available for
    # execution on the device" on those GPUs. cu128 wheels also run older
    # architectures, so it is the safe default.
    $torchIndex = $env:TORCH_CUDA_INDEX
    if (-not $torchIndex) { $torchIndex = "https://download.pytorch.org/whl/cu128" }
    # --force-reinstall so a stale torch already in the venv (e.g. a cu124 build
    # from a previous run) is actually replaced by the requested CUDA channel.
    Invoke-Checked {
        & $VenvPython -m pip install torch torchaudio --index-url $torchIndex --force-reinstall
    } "CUDA torch install ($torchIndex)"

    Invoke-Checked { & $VenvPython -m pip install ".[qwen-tts]" } "Qwen3 GPU dependency install"

    # Fail loudly if we somehow still ended up with a CPU-only torch — a green
    # build that silently falls back to ONNX is exactly the bug this guards.
    # (CUDA hardware is absent on CI, so we check the build's CUDA support flag,
    # not is_available(), which needs a GPU.)
    Invoke-Checked {
        & $VenvPython -c "import torch, sys; v = torch.version.cuda; print('torch', torch.__version__, 'cuda', v); sys.exit(0 if v else 1)"
    } "Verify torch is a CUDA build (torch.version.cuda must be set)"
}

# Asset fetch/validation runs once deps (and therefore a working `amadeus`
# import) are available, using the venv's python so the import path matches
# what PyInstaller will bundle.
if (-not $SkipAssetFetch) {
    $env:PYTHONPATH = "$RepoRoot;$env:PYTHONPATH"
    $assetsOk = Test-RequiredAssets
    if (-not $assetsOk) {
        Invoke-AssetFetch
        $assetsOk = Test-RequiredAssets
        if (-not $assetsOk) {
            throw "Runtime assets still missing after fetch. Check GitHub Release 'assets-v1' on the repo and see docs/desktop-release.md."
        }
    } else {
        Write-Host "Required runtime assets already present; skipping fetch."
    }

    # The Qwen3 reference clip ships in the same assets-v1 bundle but is not in
    # REQUIRED_ASSET_PATHS (it only matters to the GPU build), so ensure it is
    # present separately when building the GPU variant. Invoke-AssetFetch
    # unpacks the whole bundle, which includes the clip.
    if ($Gpu) {
        $refClip = Join-Path $RepoRoot "models\kurisu_voice_qwen\kurisu_ref.wav"
        if (-not (Test-Path -LiteralPath $refClip)) {
            Invoke-AssetFetch
        }
        if (-not (Test-Path -LiteralPath $refClip)) {
            throw "Qwen3 reference clip (models\kurisu_voice_qwen\kurisu_ref.wav) missing after asset fetch. Confirm assets-v1 contains it (see docs/desktop-release.md)."
        }
    }
}

# ---------------------------------------------------------------------------
# PyInstaller build
# ---------------------------------------------------------------------------
# CPU build ships a single --onefile exe (packed by NSIS via externalBin).
# The GPU sidecar with CUDA torch is ~3GB, which NSIS cannot pack as one file
# (makensis: "File: failed creating mmap"), so it is built --onedir (a folder
# of normal-sized files) and shipped as a Tauri resource instead.
$OnefileMode = if ($Gpu) { "--onedir" } else { "--onefile" }
# --clean wipes PyInstaller's analysis cache. Correct for releases (no stale
# analysis can leak into the artifact), wasteful for iteration - -Iter keeps
# the cache and rebuilds only what changed. NOTE the [string[]] type and the
# += shape: a single-element array coming out of an `if` EXPRESSION unrolls
# to a bare string, and string + array is string CONCATENATION - that exact
# trap turned this whole argument list into one giant string on the first
# v0.15.1 build (PyInstaller: "Script file '-' does not exist").
[string[]]$pyinstallerArgs = @()
if (-not $Iter) { $pyinstallerArgs += "--clean" }
$pyinstallerArgs += @(
    "--noconfirm",
    "--name", $SidecarName,
    $OnefileMode,
    "--collect-all", "onnxruntime",
    "--collect-all", "deep_translator",
    "--collect-data", "melo",
    "--collect-all", "unidic_lite",
    "--collect-data", "pykakasi",
    "--hidden-import", "amadeus.provider_loader",
    "--hidden-import", "amadeus.desktop.server",
    "--hidden-import", "amadeus.web.app",
    "--hidden-import", "amadeus.web.streaming",
    "--hidden-import", "amadeus.providers.llm.gemini_provider",
    "--hidden-import", "amadeus.providers.llm.fallback",
    "--hidden-import", "amadeus.agents.tool_loop",
    "--hidden-import", "amadeus.tools.literature",
    "--hidden-import", "amadeus.tools.desktop_files",
    "--hidden-import", "amadeus.reminders",
    "--hidden-import", "amadeus.providers.tts.melotts_onnx_provider",
    "--hidden-import", "google.generativeai",
    "--hidden-import", "onnxruntime",
    "--hidden-import", "melo.text",
    "--hidden-import", "melo.text.japanese",
    "--hidden-import", "melo.text.symbols",
    "--hidden-import", "unidic_lite",
    "--exclude-module", "tensorflow",
    "--exclude-module", "pandas",
    "--exclude-module", "matplotlib",
    "--exclude-module", "sklearn"
)

# The CPU build excludes torch/transformers/scipy to keep the sidecar small.
# The GPU build needs all three (torch for CUDA, transformers for the Qwen3
# model, scipy pulled in by the audio stack), so those excludes are dropped
# only for the CPU build.
if (-not $Gpu) {
    $pyinstallerArgs += @(
        "--exclude-module", "torch",
        "--exclude-module", "transformers",
        "--exclude-module", "scipy",
        "--exclude-module", "torchvision"
    )
} else {
    $pyinstallerArgs += @(
        "--hidden-import", "amadeus.providers.tts.qwen3_tts_provider",
        "--hidden-import", "qwen_tts",
        "--hidden-import", "soxr",
        "--collect-all", "qwen_tts",
        "--collect-all", "faster_qwen3_tts",
        "--collect-all", "transformers",
        "--exclude-module", "torchvision"
    )

    # qwen-tts shells out to the native `sox` binary (see the project notes and the
    # Qwen3TTSProvider error message); it is NOT a pip package, so PyInstaller
    # cannot collect it automatically. Bundle sox.exe (+ its DLLs) into the
    # onefile payload so it unpacks next to the sidecar. Point $env:SOX_DIR at
    # a folder containing sox.exe; otherwise we look for sox on PATH.
    $soxDir = $env:SOX_DIR
    if (-not $soxDir) {
        $soxCmd = Get-Command sox.exe -ErrorAction SilentlyContinue
        if ($soxCmd) { $soxDir = Split-Path $soxCmd.Source -Parent }
    }
    if ($soxDir -and (Test-Path (Join-Path $soxDir "sox.exe"))) {
        # Refuse chocolatey shims: a shim sox.exe resolves via PATH but only
        # redirects into the choco lib dir, so it exits -1 on user machines
        # and the app silently falls back to the ONNX voice (shipped that way
        # in v0.8.3). Every real SoX for Windows layout keeps at least one
        # DLL (zlib1.dll etc.) next to sox.exe; a chocolatey shim dir has
        # none. NOTE: this file is parsed by Windows PowerShell 5.1, which
        # reads BOM-less UTF-8 as ANSI - keep strings ASCII-only.
        if (-not (Get-ChildItem -Path $soxDir -Filter *.dll -File)) {
            throw "No DLLs found next to sox.exe in $soxDir - this looks like a chocolatey shim dir, not a real SoX install. Set `$env:SOX_DIR to the actual SoX folder (containing sox.exe and its DLLs)."
        }
        Get-ChildItem -Path $soxDir -File | ForEach-Object {
            $pyinstallerArgs += @("--add-binary", "$($_.FullName);.")
        }
        Write-Host "Bundling sox from $soxDir"
    } else {
        throw "GPU build requires the native 'sox' binary but none was found. Install SoX for Windows and set `$env:SOX_DIR to its folder (containing sox.exe), or add sox.exe to PATH. See docs/desktop-release.md."
    }
}

if ($IncludeSTT) {
    $pyinstallerArgs += @(
        "--hidden-import", "faster_whisper",
        "--collect-all", "ctranslate2"
    )
}

$pyinstallerArgs += "amadeus\desktop\server.py"

Invoke-Checked { & $VenvPython -m PyInstaller @pyinstallerArgs } "PyInstaller backend build"

if ($Gpu) {
    # --onedir output: dist\amadeus-backend\ (exe + all DLLs). Tauri's GPU
    # config (tauri.gpu.conf.json) bundles this folder as a resource, and
    # main.rs (feature "qwen") launches dist\amadeus-backend\amadeus-backend.exe
    # by absolute path. Stage it at frontend\src-tauri\amadeus-backend so the
    # resource mapping "amadeus-backend" -> "amadeus-backend" resolves.
    $OneDirSrc = Join-Path $RepoRoot "dist\$SidecarName"
    if (-not (Test-Path (Join-Path $OneDirSrc $SidecarExe))) {
        throw "Expected --onedir output '$OneDirSrc\$SidecarExe' not found."
    }
    $OneDirDest = Join-Path $TauriDir $SidecarName
    Remove-Item -Recurse -Force $OneDirDest -ErrorAction SilentlyContinue
    Copy-Item -Recurse -Force $OneDirSrc $OneDirDest
    Write-Host "Backend sidecar (onedir) ready: frontend\src-tauri\$SidecarName\$SidecarExe"
} else {
    New-Item -ItemType Directory -Force -Path $BinaryDir | Out-Null
    Copy-Item -Force (Join-Path $RepoRoot "dist\$SidecarExe") (Join-Path $BinaryDir $TauriSidecarExe)
    Write-Host "Backend sidecar ready: frontend\src-tauri\binaries\$TauriSidecarExe"
}
