# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Amadeus desktop app.

Build (after `npm run build:desktop` in frontend/):

    pyinstaller --noconfirm packaging/amadeus-desktop.spec

Produces dist/AmadeusKurisu/ — a self-contained folder app that needs no
Python, Node, or hosting. Bundled data mirrors the repo layout (config/,
static/, frontend/dist) because that is where Config.load and
web.app.create_app already resolve paths relative to the package.
"""

import pkgutil
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

repo = Path(SPECPATH).resolve().parent  # noqa: F821 — SPECPATH is injected by PyInstaller
sys.path.insert(0, str(repo))

frontend_dist = repo / "frontend" / "dist"
if not frontend_dist.exists():
    raise SystemExit(
        "frontend/dist not found — run `npm run build:desktop` in frontend/ first."
    )

datas = [
    (str(repo / "config" / "config.example.yaml"), "config"),
    (str(frontend_dist), "frontend/dist"),
]

# Large VRM/animation assets are optional downloads; bundle them when present.
static_dir = repo / "static"
if static_dir.exists():
    datas.append((str(static_dir), "static"))

# Character packs, so in-app switching (/api/characters) has something to
# switch to. Selective on purpose: candidates/ holds the voice-ref audition
# clips (dozens of wavs) that must never ship, and any bundled voice_ref.wav
# is private-use only — fine in a personal build, not for distribution.
packs_dir = repo / "characters" / "packs"
if packs_dir.is_dir():
    for pack in sorted(packs_dir.iterdir()):
        if not pack.is_dir():
            continue
        for name in ("character.yaml", "prompt.txt", "voice_ref.wav"):
            file = pack / name
            if file.is_file():
                datas.append((str(file), f"characters/packs/{pack.name}"))

# The web app is referenced by import string ("amadeus.web.app:app"), which
# static analysis can't follow — bundle every amadeus submodule explicitly.
amadeus_modules = ["amadeus"] + [
    name
    for _, name, _ in pkgutil.walk_packages([str(repo / "amadeus")], prefix="amadeus.")
]

hiddenimports = amadeus_modules + [
    # uvicorn resolves these lazily by string; PyInstaller can't see them.
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
]

# Optional runtime deps with bundled data files (e.g. the VADER lexicon).
for package in ("vaderSentiment", "deep_translator"):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    except Exception:
        continue
    datas += pkg_datas
    hiddenimports += pkg_hidden

a = Analysis(
    [str(repo / "packaging" / "desktop_entry.py")],
    pathex=[str(repo)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "transformers", "chromadb"],  # opt-in extras; keep the bundle lean
    noarchive=False,
)

pyz = PYZ(a.pure)

# PyInstaller only honours icons on Windows/macOS builds.
icon_path = repo / "packaging" / "assets" / "amadeus.ico"
icon = str(icon_path) if sys.platform in ("win32", "darwin") and icon_path.exists() else None

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="AmadeusKurisu",
    debug=False,
    strip=False,
    upx=False,
    # No console window; runtime logs go to ~/.amadeus/desktop.log instead
    # (see amadeus.desktop_webview._configure_logging).
    console=False,
    icon=icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AmadeusKurisu",
)
