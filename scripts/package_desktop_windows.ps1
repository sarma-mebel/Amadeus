param(
    [string]$Python = "python",
    [string]$TargetTriple = "x86_64-pc-windows-msvc",
    [switch]$IncludeSTT,
    [switch]$SkipAssetFetch,
    # Build the GPU variant (Qwen3 cloned voice). Forwarded to
    # package_backend_windows.ps1; see that script and docs/desktop-release.md.
    [switch]$Gpu,
    # Build the SLIM GPU installer: a CPU-sized NSIS installer (built from the
    # CPU backend sidecar, no CUDA venv) whose Tauri shell targets the GPU
    # updater manifest and downloads the GPU runtime pack (gpu-runtime-v2) on
    # first launch. This is the auto-updatable GPU artifact; -Gpu (full
    # portable, CUDA bundled) remains the fully-offline option and is
    # unaffected by this switch. See docs/desktop-release.md "GPU variant".
    [switch]$GpuSlim
)

if ($Gpu -and $GpuSlim) {
    throw "-Gpu and -GpuSlim are mutually exclusive build paths (full portable vs slim installer). Pick one."
}

$ErrorActionPreference = "Stop"
$env:Path = "$env:USERPROFILE\.cargo\bin;C:\Program Files\nodejs;$env:Path"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$FrontendDir = Join-Path $RepoRoot "frontend"
$TauriDir = Join-Path $FrontendDir "src-tauri"
$PortableDir = Join-Path $RepoRoot "dist-desktop\portable"
$PortableZip = Join-Path $RepoRoot "dist-desktop\Amadeus-Kurisu-portable-windows.zip"

Set-Location $FrontendDir
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

if (-not (Test-Path "node_modules")) {
    Invoke-Checked { npm install } "npm install"
}
Invoke-Checked { npm run build:desktop } "frontend build (desktop mode, relative URLs)"

Set-Location $RepoRoot
$backendArgs = @{
    Python       = $Python
    TargetTriple = $TargetTriple
}
# GpuSlim ships the CPU sidecar (no CUDA venv, no -Gpu) - GPU-ness arrives at
# runtime via the downloaded GPU runtime pack, not via PyInstaller bundling.
# It always needs -IncludeSTT regardless of the caller's own -IncludeSTT
# switch, same rationale as the full GPU build below.
if ($IncludeSTT -or $GpuSlim) { $backendArgs["IncludeSTT"] = $true }
if ($SkipAssetFetch) { $backendArgs["SkipAssetFetch"] = $true }
if ($Gpu) { $backendArgs["Gpu"] = $true }
Invoke-Checked { & (Join-Path $PSScriptRoot "package_backend_windows.ps1") @backendArgs } "backend sidecar packaging"

Set-Location $FrontendDir
if ($Gpu) {
    # GPU build ships as a PORTABLE ZIP ONLY, never an NSIS installer: the CUDA
    # torch DLLs are too large for NSIS to pack (torch_cuda.dll alone is ~900MB;
    # makensis dies "error mmapping datablock"). Build with --no-bundle to
    # compile the Tauri shell without attempting installer packaging, then
    # assemble a portable tree by hand below.
    Invoke-Checked { npm run tauri:build:gpu:nobundle } "Tauri desktop build (GPU, no-bundle)"
} elseif ($GpuSlim) {
    # Stage character packs for bundling (tauri.gpuslim.conf.json maps
    # dist-desktop\packs-staged -> resources\characters\packs). Selective:
    # yaml + prompt + voice_ref only, never candidates/ audition clips.
    # Voice refs are private-use only, which is why this staging is
    # GpuSlim-exclusive (a personal-channel artifact) and the public CPU
    # installer deliberately ships no packs. Without this, an installed
    # app's packs came only from in-app zip installs - which predated the
    # voice blocks, so switching characters changed the outfit but never
    # the voice (found live, 2026-08-03 shakedown). Dir is created even
    # when empty so the tauri resource map never dangles.
    $PacksStage = Join-Path $RepoRoot "dist-desktop\packs-staged"
    Remove-Item -Recurse -Force $PacksStage -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $PacksStage | Out-Null
    $PacksSrc = Join-Path $RepoRoot "characters\packs"
    if (Test-Path $PacksSrc) {
        Get-ChildItem $PacksSrc -Directory | ForEach-Object {
            $packDest = Join-Path $PacksStage $_.Name
            New-Item -ItemType Directory -Force -Path $packDest | Out-Null
            foreach ($f in @("character.yaml", "prompt.txt", "voice_ref.wav")) {
                $srcFile = Join-Path $_.FullName $f
                if (Test-Path $srcFile) { Copy-Item -Force $srcFile $packDest }
            }
        }
    }
    # GpuSlim uses its OWN bundled config (tauri.gpuslim.conf.json), NOT
    # tauri.gpu.conf.json: the gpu config serves the no-bundle full-GPU path
    # and bundles the staged onedir CUDA backend as a resource with
    # externalBin cleared - reusing it here packed the ~5GB onedir tree and
    # makensis died at ~1.96GB with the same mmap error the slim build exists
    # to dodge (found live cutting v0.15.0). The slim config keeps the CPU
    # onefile sidecar via the base externalBin, adds the qwen voice refs, and
    # points the updater endpoint at latest-gpu.json so this installer
    # auto-updates on its own channel. Signing env passthrough is identical
    # to the CPU NSIS path below (see docs/desktop-release.md).
    Invoke-Checked { npm run tauri:build:gpuslim } "Tauri desktop build (GPU slim, bundled)"
} else {
    Invoke-Checked { npm run tauri:build } "Tauri desktop build"
}

Set-Location $RepoRoot
Remove-Item -Recurse -Force $PortableDir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $PortableDir | Out-Null

$ReleaseRoot = Join-Path $TauriDir "target\release"

if ($Gpu) {
    # GPU portable layout: Tauri shell exe + the onedir backend (staged at
    # $TauriDir\amadeus-backend by package_backend_windows.ps1 -Gpu) + the
    # runtime resources the shell resolves relative to its resource dir. The
    # shell (feature "qwen") launches amadeus-backend\amadeus-backend.exe.
    $ReleaseExe = Join-Path $ReleaseRoot "amadeus-desktop.exe"
    if (-not (Test-Path $ReleaseExe)) {
        throw "GPU shell exe not found at $ReleaseExe (did tauri build --no-bundle succeed?)."
    }
    Copy-Item -Force $ReleaseExe (Join-Path $PortableDir "Amadeus Kurisu (GPU).exe")

    $backendOneDir = Join-Path $TauriDir "amadeus-backend"
    if (-not (Test-Path (Join-Path $backendOneDir "amadeus-backend.exe"))) {
        throw "GPU onedir backend not found at $backendOneDir (run package_backend_windows.ps1 -Gpu first)."
    }
    Copy-Item -Recurse -Force $backendOneDir (Join-Path $PortableDir "amadeus-backend")

    # Runtime resources, from the repo (same paths tauri.gpu.conf.json bundles).
    Copy-Item -Recurse -Force (Join-Path $FrontendDir "dist") (Join-Path $PortableDir "frontend\dist")
    foreach ($res in @("static", "assets")) {
        Copy-Item -Recurse -Force (Join-Path $RepoRoot $res) (Join-Path $PortableDir $res)
    }
    New-Item -ItemType Directory -Force -Path (Join-Path $PortableDir "models") | Out-Null
    foreach ($m in @("kurisu_voice_onnx", "kurisu_voice_qwen")) {
        $src = Join-Path $RepoRoot "models\$m"
        if (Test-Path $src) { Copy-Item -Recurse -Force $src (Join-Path $PortableDir "models\$m") }
    }

    # Character packs for in-app switching (/api/characters resolves them
    # next to the static dir). Selective copy: candidates/ audition clips
    # never ship, and any voice_ref.wav is private-use only - this portable
    # build is personal, do not distribute it with packs inside.
    $PacksSrc = Join-Path $RepoRoot "characters\packs"
    if (Test-Path $PacksSrc) {
        Get-ChildItem $PacksSrc -Directory | ForEach-Object {
            $packDest = Join-Path $PortableDir "characters\packs\$($_.Name)"
            New-Item -ItemType Directory -Force -Path $packDest | Out-Null
            foreach ($f in @("character.yaml", "prompt.txt", "voice_ref.wav")) {
                $srcFile = Join-Path $_.FullName $f
                if (Test-Path $srcFile) { Copy-Item -Force $srcFile $packDest }
            }
        }
    }

    $PortableZip = Join-Path $RepoRoot "dist-desktop\Amadeus-Kurisu-GPU-portable-windows.zip"
    Remove-Item -Force $PortableZip -ErrorAction SilentlyContinue
    # portable.marker keeps the in-app NSIS auto-updater off for this copy
    # (it would install into Program Files and leave this dir stale); the
    # app falls back to the releases-page nudge when the marker is present.
    Set-Content -Path (Join-Path $PortableDir "portable.marker") -Value "portable build - in-app auto-update disabled"
    Compress-Archive -Path (Join-Path $PortableDir "*") -DestinationPath $PortableZip
    Write-Host "GPU portable ZIP (no NSIS installer): $PortableZip"
} elseif ($GpuSlim) {
    # No portable assembly here - GpuSlim's whole point is the NSIS installer
    # produced by `npm run tauri:build:gpuslim` above (tauri.gpuslim.conf.json's
    # nsis section — NOT tauri.gpu.conf.json, which serves the no-bundle
    # full-GPU path and would re-trigger the makensis mmap death). Verify it
    # landed and is under GitHub's 2GB asset cap (1.9GB, matching the margin
    # used elsewhere in this repo for release assets) - a slim installer
    # ballooning past that would defeat the whole point of this build and
    # silently break the next `gh release upload`.
    #
    # Newest-first: tauri:build:gpu and tauri:build:gpuslim both emit
    # "*GPU*-setup.exe" into this SAME directory, so a stale full-GPU
    # artifact matches the glob. Shipping that one means an installer whose
    # shell expects an onedir tree GpuSlim doesn't bundle (backend never
    # spawns). Picking the newest write is the guard.
    $NsisDir = Join-Path $TauriDir "target\release\bundle\nsis"
    $installer = Get-ChildItem -Path $NsisDir -Filter "*-setup.exe" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "*GPU*" } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $installer) {
        throw "GPU slim installer not found under $NsisDir (expected a '*GPU*-setup.exe'). Did 'npm run tauri:build:gpuslim' actually produce an NSIS bundle? Check bundle.active and bundle.targets in tauri.gpuslim.conf.json / tauri.conf.json."
    }
    Write-Host "GpuSlim installer candidate: $($installer.Name) (LastWriteTime $($installer.LastWriteTime))"
    $MaxInstallerBytes = 1900MB
    if ($installer.Length -gt $MaxInstallerBytes) {
        throw "GPU slim installer '$($installer.FullName)' is $($installer.Length) bytes, over the $MaxInstallerBytes byte (1.9GB) GitHub release asset budget. It should be roughly CPU-sized now that CUDA torch moved to the GPU runtime pack - investigate what got bundled (e.g. STT/ctranslate2 bloat, a stray torch install in node_modules-adjacent paths, or an accidental -Gpu backend build feeding this step)."
    }
    Write-Host ""
    Write-Host "GPU slim installer: $($installer.FullName) ($($installer.Length) bytes)"
    Write-Host "Next: python scripts/make_update_manifest.py --tag <vX.Y.Z> --variant gpu"
} else {
    $ReleaseExe = Join-Path $ReleaseRoot "amadeus-desktop.exe"
    if (Test-Path $ReleaseExe) {
        Copy-Item -Force $ReleaseExe (Join-Path $PortableDir "Amadeus Kurisu.exe")
    }
    $portableEntries = @("amadeus-backend.exe", "frontend", "static", "assets", "models")
    foreach ($entry in $portableEntries) {
        $source = Join-Path $ReleaseRoot $entry
        if (Test-Path $source) {
            Copy-Item -Recurse -Force $source (Join-Path $PortableDir $entry)
        }
    }
    Remove-Item -Force $PortableZip -ErrorAction SilentlyContinue
    # portable.marker keeps the in-app NSIS auto-updater off for this copy
    # (it would install into Program Files and leave this dir stale); the
    # app falls back to the releases-page nudge when the marker is present.
    Set-Content -Path (Join-Path $PortableDir "portable.marker") -Value "portable build - in-app auto-update disabled"
    Compress-Archive -Path (Join-Path $PortableDir "*") -DestinationPath $PortableZip
    Write-Host "Installer output: frontend\src-tauri\target\release\bundle\nsis"
    Write-Host "Portable ZIP: dist-desktop\Amadeus-Kurisu-portable-windows.zip"
}
