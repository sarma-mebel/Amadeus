#!/usr/bin/env python3
"""Generate the Tauri updater manifest (latest.json) for a desktop release.

The auto-updater in the desktop app polls
  https://github.com/zxKyouma/Amadeus/releases/latest/download/latest.json
so every release that ships the NSIS installer must also upload a latest.json
describing it. This script finds the signed installer + its .sig (produced by
`npm run tauri:build` when TAURI_SIGNING_PRIVATE_KEY is set and
bundle.createUpdaterArtifacts is on) and writes the manifest.

Two variants ship on separate updater channels (one manifest per variant,
because a Tauri updater manifest carries exactly one URL per platform):
  cpu   the default desktop build (latest.json, tauri.conf.json)
  gpu   the slim GPU installer (latest-gpu.json, tauri.gpu.conf.json) that
        fetches the GPU runtime pack on first launch — see
        docs/desktop-release.md "GPU variant"

Usage (from the repo root, after a build):

    python scripts/make_update_manifest.py --tag v0.14.0
    gh release upload v0.14.0 latest.json \
        "frontend/src-tauri/target/release/bundle/nsis/<setup>.exe" --clobber

    python scripts/make_update_manifest.py --tag v0.15.0 --variant gpu
    gh release upload v0.15.0 latest-gpu.json \
        "frontend/src-tauri/target/release/bundle/nsis/<name (GPU)-setup>.exe" --clobber

Options:
    --tag vX.Y.Z     release tag the download URL points at (required)
    --variant        cpu (default) or gpu — selects which installer glob to
                     match and the default output filename
    --bundle-dir     where to look for the installer (default:
                     frontend/src-tauri/target/release/bundle/nsis)
    --notes          release notes line embedded in the manifest (optional)
    --output         manifest path (default: latest.json for cpu,
                     latest-gpu.json for gpu)

The script fails loudly when the .sig is missing — an unsigned installer
would make every installed app reject the update, which looks exactly like
"the updater silently does nothing".
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path

REPO = "zxKyouma/Amadeus"
DEFAULT_BUNDLE_DIR = Path("frontend/src-tauri/target/release/bundle/nsis")
RELEASE_ASSETS_DIR = Path("dist-desktop/release-assets")

# Tauri's NSIS bundle names installers "${productName}_${version}_${arch}-setup.exe"
# verbatim (spaces/parens included) — confirmed against real build output:
#   "Amadeus Kurisu_0.14.0_x64-setup.exe"          (cpu, productName "Amadeus Kurisu")
#   "Amadeus Kurisu (GPU)_0.15.0_x64-setup.exe"    (gpu, productName "Amadeus Kurisu (GPU)")
# "GPU" only ever appears in the GPU productName, so a substring match on it
# unambiguously partitions the two variants without collision in either
# direction: the gpu glob picks only installers containing "GPU", and the cpu
# glob explicitly excludes anything containing "GPU" so a bundle dir holding
# both variants (e.g. a shared CI artifact dir) still resolves the cpu
# manifest to the one cpu installer instead of failing on "multiple
# installers". Each variant's OWN match still fails loudly if it is itself
# ambiguous (e.g. two stale cpu builds) — cleaning the bundle dir before
# building remains the fix for that case.
GPU_MARKER = "GPU"


def github_asset_name(installer_name: str) -> str:
    """GitHub-safe asset name for the manifest URL and the uploaded file.

    GitHub renames release assets on upload: characters outside
    [0-9A-Za-z_.+-@] become dots (observed live — v0.12.0's
    "Amadeus Kurisu_0.12.0_x64-setup.exe" is served as
    "Amadeus.Kurisu_0.12.0_x64-setup.exe"). A manifest URL built from the
    LOCAL filename therefore 404s and the updater silently does nothing.
    Instead of guessing GitHub's exact rewrite for every character, produce a
    name GitHub keeps verbatim and upload a copy under that name.
    """
    name = installer_name.replace(" (GPU)", ".GPU").replace(" ", ".")
    leftover = re.findall(r"[^0-9A-Za-z_.+-]", name)
    if leftover:
        sys.exit(
            f"Installer name {installer_name!r} still has characters GitHub "
            f"would rewrite after sanitizing: {sorted(set(leftover))}. Extend "
            "github_asset_name() for the new productName."
        )
    return name


def find_installer(bundle_dir: Path, variant: str) -> Path:
    all_candidates = sorted(bundle_dir.glob("*-setup.exe"))
    if variant == "gpu":
        candidates = [c for c in all_candidates if GPU_MARKER in c.name]
        variant_desc = f"GPU (*{GPU_MARKER}*-setup.exe)"
    else:
        candidates = [c for c in all_candidates if GPU_MARKER not in c.name]
        variant_desc = "CPU (*-setup.exe, excluding GPU-named installers)"

    if not candidates:
        sys.exit(
            f"No {variant_desc} installer found in {bundle_dir} — run the desktop "
            f"build first (npm run tauri:build{'::gpu' if variant == 'gpu' else ''} "
            "with TAURI_SIGNING_PRIVATE_KEY set)."
        )
    if len(candidates) > 1:
        sys.exit(
            f"Multiple {variant_desc} installers in {bundle_dir}: "
            f"{[c.name for c in candidates]} — clean the bundle dir so the "
            "manifest can't point at a stale build."
        )
    return candidates[0]


def read_version(repo_root: Path, variant: str) -> str:
    conf_name = (
        "frontend/src-tauri/tauri.gpu.conf.json"
        if variant == "gpu"
        else "frontend/src-tauri/tauri.conf.json"
    )
    conf = json.loads((repo_root / conf_name).read_text(encoding="utf-8"))
    version = conf.get("version")
    if version is None and variant == "gpu":
        # tauri.gpu.conf.json is a partial overlay merged onto tauri.conf.json
        # at build time (no "version" key of its own) — the base config is
        # the source of truth for both variants.
        conf = json.loads(
            (repo_root / "frontend/src-tauri/tauri.conf.json").read_text(
                encoding="utf-8"
            )
        )
        version = conf.get("version")
    if not version:
        sys.exit(f"{conf_name} (or the base tauri.conf.json) has no version field")
    return str(version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="release tag, e.g. v0.14.0")
    parser.add_argument(
        "--variant",
        choices=("cpu", "gpu"),
        default="cpu",
        help="cpu (default, latest.json) or gpu (slim installer, latest-gpu.json)",
    )
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or Path(
        "latest-gpu.json" if args.variant == "gpu" else "latest.json"
    )

    repo_root = Path(__file__).resolve().parent.parent
    bundle_dir = (
        args.bundle_dir
        if args.bundle_dir.is_absolute()
        else repo_root / args.bundle_dir
    )

    installer = find_installer(bundle_dir, args.variant)
    sig_path = installer.with_name(installer.name + ".sig")
    if not sig_path.exists():
        sys.exit(
            f"{sig_path.name} not found next to the installer. The build was not "
            "signed — set TAURI_SIGNING_PRIVATE_KEY (see docs/desktop-release.md, "
            "'Auto-updater') and rebuild. Shipping an unsigned installer would "
            "make every installed app silently reject the update."
        )

    version = read_version(repo_root, args.variant)
    tag_version = args.tag.lstrip("vV")
    if tag_version != version:
        conf_desc = (
            "tauri.gpu.conf.json (or base tauri.conf.json)"
            if args.variant == "gpu"
            else "tauri.conf.json"
        )
        sys.exit(
            f"--tag {args.tag} does not match {conf_desc} version {version} — "
            "bump the version (tauri.conf.json / package.json / pyproject.toml) "
            "or fix the tag."
        )

    # Stage upload copies under the GitHub-safe name so the uploaded asset,
    # and thus the manifest URL, is exactly what we wrote here.
    asset_name = github_asset_name(installer.name)
    assets_dir = repo_root / RELEASE_ASSETS_DIR
    assets_dir.mkdir(parents=True, exist_ok=True)
    staged_installer = assets_dir / asset_name
    staged_sig = assets_dir / (asset_name + ".sig")
    shutil.copyfile(installer, staged_installer)
    shutil.copyfile(sig_path, staged_sig)

    url = f"https://github.com/{REPO}/releases/download/{args.tag}/{asset_name}"
    manifest = {
        "version": version,
        "notes": args.notes or f"Amadeus {args.tag}",
        "pub_date": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "platforms": {
            "windows-x86_64": {
                "signature": sig_path.read_text(encoding="utf-8").strip(),
                "url": url,
            }
        },
    }
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output} for {installer.name}")
    print(f"Staged upload copies: {staged_installer} (+ .sig)")
    print("Upload the manifest AND the staged copies (NOT the original —")
    print("GitHub would rename it and break the manifest URL), e.g.:")
    print(
        f'  gh release upload {args.tag} {output} "{staged_installer}" '
        f'"{staged_sig}" --repo {REPO} --clobber'
    )


if __name__ == "__main__":
    main()
