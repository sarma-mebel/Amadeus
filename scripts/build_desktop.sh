#!/usr/bin/env bash
# Build the self-contained Amadeus desktop app (Linux/macOS).
# Output: dist/AmadeusKurisu/AmadeusKurisu
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Building frontend (desktop mode: relative URLs)"
pushd frontend >/dev/null
if [ ! -d node_modules ]; then
  npm ci --no-audit --no-fund
fi
npm run build:desktop
popd >/dev/null

echo "==> Installing Python build deps"
python -m pip install --quiet ".[web,desktop]" pyinstaller

echo "==> Packaging with PyInstaller"
pyinstaller --noconfirm packaging/amadeus-desktop.spec

echo "==> Done: dist/AmadeusKurisu/AmadeusKurisu"
