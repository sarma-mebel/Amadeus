# Build the self-contained Amadeus desktop app (Windows).
# Output: dist\AmadeusKurisu\AmadeusKurisu.exe
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "==> Building frontend (desktop mode: relative URLs)"
Push-Location frontend
if (-not (Test-Path node_modules)) {
    npm ci --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { throw "npm ci failed" }
}
npm run build:desktop
if ($LASTEXITCODE -ne 0) { throw "frontend build failed" }
Pop-Location

Write-Host "==> Installing Python build deps"
python -m pip install --quiet ".[web,desktop]" pyinstaller
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

Write-Host "==> Packaging with PyInstaller"
pyinstaller --noconfirm packaging/amadeus-desktop.spec
if ($LASTEXITCODE -ne 0) { throw "pyinstaller failed" }

Write-Host "==> Done: dist\AmadeusKurisu\AmadeusKurisu.exe"

# Optional: build the installer when Inno Setup is on PATH.
$iscc = Get-Command iscc -ErrorAction SilentlyContinue
if ($iscc) {
    Write-Host "==> Building installer (Inno Setup)"
    iscc packaging\installer.iss
    if ($LASTEXITCODE -ne 0) { throw "iscc failed" }
    Write-Host "==> Installer: dist\AmadeusKurisu-Setup.exe"
} else {
    Write-Host "==> Skipping installer (Inno Setup's iscc not on PATH)"
}
