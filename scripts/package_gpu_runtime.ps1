param(
    [string]$GpuVenvPath = ".venv-package-gpu",
    [string]$OutputDir = "dist-desktop\gpu-runtime",
    # Max size per zip part, in bytes. GitHub release assets cap at 2 GiB;
    # 1.9GB matches the cross-agent contract (amadeus/gpu_runtime.py expects
    # parts "<=1.9GB").
    [long]$MaxPartBytes = 1900MB
)

# ---------------------------------------------------------------------------
# Builds the GPU runtime pack (gpu-runtime-v2): the CUDA Python packages and
# sox binary that the slim GPU installer fetches on first launch instead of
# bundling. Cross-agent contract lives in amadeus/gpu_runtime.py's module
# docstring - this script must produce exactly what that module consumes:
#
#   Release tag:  gpu-runtime-v2
#   Assets:       gpu-runtime-manifest.json
#                 gpu-runtime.zip.part01, .part02, ... (each <=1.9GB,
#                 concatenated in order = one zip)
#   Zip layout:   site-packages/<FULL GPU venv site-packages + stdlib overlay>
#                 sox/<sox.exe + DLLs>
#
# v1 HISTORY - why the pack is no longer a CPU-venv diff: v1 shipped only
# the top-level entries present in the GPU venv but not the CPU venv, on the
# assumption that the frozen sidecar provides everything the CPU venv has.
# That assumption is FALSE twice over (found live, first real slim install):
#   1. PyInstaller freezes only what its import analysis reaches - numpy,
#      tokenizers, filelock etc. were in the CPU venv, hence dropped from the
#      pack, but transformers/torch need them at runtime and the frozen
#      sidecar never bundled some of them (or bundled the module without its
#      *.dist-info, which transformers' version check also requires).
#   2. Same for the stdlib: the frozen sidecar prunes stdlib modules nothing
#      in the CPU graph imports (pickletools, filecmp, cProfile - all needed
#      by torch/transformers imports).
# Result: `import torch` died inside the installed app and the voice silently
# fell back to ONNX. The pack now ships the FULL GPU site-packages plus a
# stdlib overlay (top-level Lib/*.py + DLLs/*.pyd from the venv's base
# python) merged into site-packages/. Frozen modules always win - activation
# APPENDS to sys.path - so overlap with the sidecar is harmless by design.
#
# This script does NOT publish anything - it stages dist-desktop/gpu-runtime/
# and prints the exact `gh release` commands to run by hand.
# ---------------------------------------------------------------------------

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
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
# Preconditions
# ---------------------------------------------------------------------------
$GpuVenvDir = Join-Path $RepoRoot $GpuVenvPath
$GpuSitePackages = Join-Path $GpuVenvDir "Lib\site-packages"
if (-not (Test-Path $GpuSitePackages)) {
    throw "GPU packaging venv not found at $GpuVenvDir (expected $GpuSitePackages). Run 'powershell -ExecutionPolicy Bypass -File scripts/package_backend_windows.ps1 -Gpu' first - it creates and populates this venv with the CUDA torch + qwen-tts dependency set."
}

# The stdlib overlay comes from the GPU venv's base interpreter (pyvenv.cfg
# "home"), which is by construction the same Python version the sidecar was
# frozen with (both venvs share one base install).
$PyVenvCfg = Join-Path $GpuVenvDir "pyvenv.cfg"
$BasePythonDir = $null
if (Test-Path $PyVenvCfg) {
    $homeLine = (Get-Content $PyVenvCfg | Where-Object { $_ -match '^\s*home\s*=' } | Select-Object -First 1)
    if ($homeLine -match '=\s*(.+)$') { $BasePythonDir = $Matches[1].Trim() }
}
if (-not ($BasePythonDir -and (Test-Path (Join-Path $BasePythonDir "Lib")))) {
    throw "Could not resolve the GPU venv's base python (pyvenv.cfg home = '$BasePythonDir'). The stdlib overlay needs it - see the v1 history note at the top of this script."
}

$soxDir = $env:SOX_DIR
if (-not $soxDir) {
    $soxCmd = Get-Command sox.exe -ErrorAction SilentlyContinue
    if ($soxCmd) { $soxDir = Split-Path $soxCmd.Source -Parent }
}
if (-not ($soxDir -and (Test-Path (Join-Path $soxDir "sox.exe")))) {
    throw "sox.exe not found. Set `$env:SOX_DIR to a folder containing sox.exe (and its DLLs), or put sox.exe on PATH. See docs/desktop-release.md."
}
if (-not (Get-ChildItem -Path $soxDir -Filter *.dll -File)) {
    throw "No DLLs found next to sox.exe in $soxDir - this looks like a chocolatey shim dir, not a real SoX install. Set `$env:SOX_DIR to the actual SoX folder (containing sox.exe and its DLLs)."
}

Write-Host "GPU venv site-packages: $GpuSitePackages"
Write-Host "Stdlib overlay source: $BasePythonDir"
Write-Host "sox source: $soxDir"

# ---------------------------------------------------------------------------
# Selection: EVERY top-level site-packages entry except installer machinery.
# "Top-level entry" = a package dir, a *.dist-info dir, or a standalone
# *.pth / *.py file directly under site-packages - the same granularity pip
# installs/uninstalls at. No CPU-venv diff (see v1 history at the top).
# ---------------------------------------------------------------------------
$AlwaysExclude = @("pip", "pip-*", "setuptools", "setuptools-*", "wheel", "wheel-*", "__pycache__")

$packEntries = @()
foreach ($entry in (Get-ChildItem -Path $GpuSitePackages -Force | ForEach-Object { $_.Name })) {
    $excluded = $false
    foreach ($pattern in $AlwaysExclude) {
        if ($entry -like $pattern) { $excluded = $true; break }
    }
    if ($excluded) { continue }
    $packEntries += $entry
}

if (-not ($packEntries -contains "torch")) {
    throw "GPU venv has no 'torch' - did package_backend_windows.ps1 -Gpu actually install the CUDA dependency set into $GpuVenvPath?"
}

Write-Host ""
Write-Host "Pack set: $($packEntries.Count) top-level site-packages entries (full GPU venv)."
Write-Host ""

# ---------------------------------------------------------------------------
# Stage: <tmp>/site-packages/<entries + stdlib overlay> + <tmp>/sox/
# ---------------------------------------------------------------------------
$StageDir = Join-Path $RepoRoot "dist-desktop\gpu-runtime-stage"
Remove-Item -Recurse -Force $StageDir -ErrorAction SilentlyContinue
$StageSitePackages = Join-Path $StageDir "site-packages"
$StageSox = Join-Path $StageDir "sox"
New-Item -ItemType Directory -Force -Path $StageSitePackages | Out-Null
New-Item -ItemType Directory -Force -Path $StageSox | Out-Null

Write-Host "Staging GPU venv site-packages into $StageSitePackages ..."
foreach ($entry in $packEntries) {
    $src = Join-Path $GpuSitePackages $entry
    $dest = Join-Path $StageSitePackages $entry
    if (Test-Path $src -PathType Container) {
        Copy-Item -Recurse -Force $src $dest
    } else {
        Copy-Item -Force $src $dest
    }
}

# Stdlib overlay: top-level Lib/*.py and DLLs/*.pyd from the base install,
# merged directly into site-packages (the layout validated live - activation
# appends one dir and frozen modules always win, so these only ever FILL
# GAPS the sidecar's analysis pruned: pickletools, filecmp, cProfile, ...).
# Entries already staged (a venv package shadowing a stdlib name) are kept.
Write-Host "Staging stdlib overlay from $BasePythonDir ..."
$overlayCount = 0
foreach ($file in (Get-ChildItem -Path (Join-Path $BasePythonDir "Lib") -Filter *.py -File)) {
    $dest = Join-Path $StageSitePackages $file.Name
    if (-not (Test-Path $dest)) { Copy-Item -Force $file.FullName $dest; $overlayCount++ }
}
$BaseDlls = Join-Path $BasePythonDir "DLLs"
if (Test-Path $BaseDlls) {
    foreach ($file in (Get-ChildItem -Path $BaseDlls -Filter *.pyd -File)) {
        $dest = Join-Path $StageSitePackages $file.Name
        if (-not (Test-Path $dest)) { Copy-Item -Force $file.FullName $dest; $overlayCount++ }
    }
}
Write-Host "Stdlib overlay: $overlayCount files."

Write-Host "Staging sox from $soxDir into $StageSox ..."
Get-ChildItem -Path $soxDir -File | ForEach-Object {
    Copy-Item -Force $_.FullName (Join-Path $StageSox $_.Name)
}

# Informational torch version for the manifest (best-effort; the GPU venv's
# python reports it directly rather than parsing dist-info by hand).
$GpuVenvPython = Join-Path $GpuVenvDir "Scripts\python.exe"
if (-not (Test-Path $GpuVenvPython)) {
    $GpuVenvPython = Join-Path $GpuVenvDir "bin/python"
}
$torchVersion = ""
try {
    $torchVersion = (& $GpuVenvPython -c "import torch; print(torch.__version__)").Trim()
} catch {
    Write-Host "WARNING: could not determine torch version from $GpuVenvPython (non-fatal, manifest 'torch' field left blank)."
}

# ---------------------------------------------------------------------------
# Zip (System.IO.Compression - no external 7z dependency)
# ---------------------------------------------------------------------------
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$OutDir = Join-Path $RepoRoot $OutputDir
Remove-Item -Recurse -Force $OutDir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$ZipPath = Join-Path $OutDir "gpu-runtime.zip"
Write-Host "Compressing $StageDir -> $ZipPath ..."
# Manual per-entry zip instead of CreateFromDirectory: under Windows
# PowerShell 5.1 (.NET Framework) CreateFromDirectory writes BACKSLASH
# entry names, violating the zip spec (the backend loader now tolerates
# both styles, but the artifact should be spec-clean). Entry names are
# forced to forward slashes here.
$archive = [System.IO.Compression.ZipFile]::Open(
    $ZipPath, [System.IO.Compression.ZipArchiveMode]::Create
)
try {
    $stagePrefix = (Get-Item $StageDir).FullName.TrimEnd('\') + '\'
    Get-ChildItem -Recurse -File -Path $StageDir | ForEach-Object {
        $entryName = $_.FullName.Substring($stagePrefix.Length).Replace('\', '/')
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $archive, $_.FullName, $entryName,
            [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    }
} finally {
    $archive.Dispose()
}

function Get-Sha256Hex {
    param([string]$Path)
    (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$zipInfo = Get-Item $ZipPath
$zipSha256 = Get-Sha256Hex -Path $ZipPath
Write-Host "Assembled zip: $($zipInfo.Length) bytes, sha256 $zipSha256"

# ---------------------------------------------------------------------------
# Split into <=$MaxPartBytes parts: gpu-runtime.zip.part01, .part02, ...
# ---------------------------------------------------------------------------
Write-Host "Splitting into <= $MaxPartBytes byte parts ..."
$parts = @()
$buffer = New-Object byte[] (16MB)
$reader = [System.IO.File]::OpenRead($ZipPath)
try {
    $partIndex = 0
    $bytesRemainingInPart = 0
    $writer = $null
    try {
        while ($true) {
            if ($bytesRemainingInPart -le 0) {
                if ($writer) { $writer.Dispose() }
                $partIndex++
                $partName = "gpu-runtime.zip.part{0:D2}" -f $partIndex
                $partPath = Join-Path $OutDir $partName
                $writer = [System.IO.File]::Create($partPath)
                $parts += [PSCustomObject]@{ Name = $partName; Path = $partPath }
                $bytesRemainingInPart = $MaxPartBytes
            }
            $toRead = [Math]::Min($buffer.Length, $bytesRemainingInPart)
            $read = $reader.Read($buffer, 0, [int]$toRead)
            if ($read -le 0) { break }
            $writer.Write($buffer, 0, $read)
            $bytesRemainingInPart -= $read
        }
    } finally {
        if ($writer) { $writer.Dispose() }
    }
} finally {
    $reader.Dispose()
}

if ($parts.Count -eq 0) {
    throw "Zip split produced no parts - is $ZipPath empty?"
}
if ($parts.Count -gt 99) {
    throw "Zip split produced $($parts.Count) parts (>99) - the partNN naming scheme only supports two digits. Something is unexpectedly huge; investigate before publishing."
}

Remove-Item -Force $ZipPath

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
$manifestParts = @()
foreach ($p in $parts) {
    $size = (Get-Item $p.Path).Length
    $sha = Get-Sha256Hex -Path $p.Path
    $manifestParts += [PSCustomObject]@{
        name   = $p.Name
        size   = $size
        sha256 = $sha
    }
    Write-Host "  $($p.Name): $size bytes, sha256 $sha"
}

$manifest = [ordered]@{
    version    = "gpu-runtime-v2"
    created    = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    zip_sha256 = $zipSha256
    total_size = $zipInfo.Length
    parts      = $manifestParts
    layout     = [ordered]@{
        site_packages = "site-packages"
        sox           = "sox"
    }
    torch      = $torchVersion
}

$ManifestPath = Join-Path $OutDir "gpu-runtime-manifest.json"
# WriteAllText with a BOM-less UTF8Encoding: PowerShell 5.1's -Encoding utf8
# writes a UTF-8 BOM, which crashed the backend's plain-utf8 json.loads on
# the first real pack build (the loader is now BOM-tolerant too, but the
# artifact should be clean regardless).
[System.IO.File]::WriteAllText(
    $ManifestPath,
    ($manifest | ConvertTo-Json -Depth 6),
    (New-Object System.Text.UTF8Encoding($false))
)
Write-Host ""
Write-Host "Wrote manifest: $ManifestPath"

# The stage dir is deliberately KEPT: the frozen-import probe gate (see the
# instructions printed below) runs against it before publishing. Delete it
# by hand once the gate passes and the release is up.

# ---------------------------------------------------------------------------
# Publish instructions (not run automatically)
# ---------------------------------------------------------------------------
$assetList = @($ManifestPath)
$assetList += ($parts | ForEach-Object { $_.Path })
$quotedAssets = ($assetList | ForEach-Object { "`"$_`"" }) -join " "

$releaseNotes = "CUDA torch + qwen-tts runtime pack fetched by the slim GPU desktop build on first launch."

Write-Host ""
Write-Host "Output dir: $OutDir"
Write-Host ""
Write-Host "This script does NOT publish. Review the package list above, then run:"
Write-Host ""
Write-Host "  gh release view gpu-runtime-v2 --repo zxKyouma/Amadeus"
Write-Host "  # First time (release doesn't exist yet):"
# --latest=false: a freshly published side-release steals the "Latest" flag
# from the vX.Y.Z app release, which breaks the updater's
# /releases/latest/download/latest.json URL. Bitten twice (v1 and v2 packs).
Write-Host "  gh release create gpu-runtime-v2 $quotedAssets --repo zxKyouma/Amadeus --latest=false --title `"GPU runtime pack v2`" --notes `"$releaseNotes`""
Write-Host ""
Write-Host "  # Updating an existing gpu-runtime-v2 release:"
Write-Host "  gh release upload gpu-runtime-v2 $quotedAssets --repo zxKyouma/Amadeus --clobber"
Write-Host ""
Write-Host "Re-publish gpu-runtime-v2 only when the torch/qwen dependency set changes - NOT per app release."
Write-Host ""
Write-Host "GATE (do not skip): before publishing, run the frozen-import probe against the"
Write-Host "staged pack - a plain venv python proves NOTHING about the frozen sidecar."
Write-Host "Build + usage instructions in scripts/probe_gpu_pack.py. Publish only on ALL-OK."
