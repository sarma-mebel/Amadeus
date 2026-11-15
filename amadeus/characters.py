"""Character pack installer: the shareable format behind BYO-character.

A pack is a folder (or zip) that fully describes a character:

    character.yaml    required: name, full_name; optional metadata below
    prompt.txt        the persona system prompt (name set by prompt_file)
    avatar.vrm        optional VRM 1.0 avatar (name set by avatar)
    voice_ref.wav     optional zero-shot voice reference (voice.ref_audio)

character.yaml:

    name: Rin                  # short in-prompt name
    full_name: Rin Asakura     # used by greetings/farewell/brief/diary
    description: one line for humans
    author: ...
    license: ...
    birthday: "03-14"          # optional; greeting milestone on that day
    voice_style: "..."         # optional; 1-3 sentence register spec for
                               # presence/greeting/brief/diary prompts
    prompt_file: prompt.txt
    avatar: avatar.vrm
    spoken_language: english   # audio.voice_language (japanese|english)
    voice:
      ref_audio: voice_ref.wav
      ref_text: "exact transcript of the reference clip"
      language: English        # qwen3_tts_language

Installing copies the assets where the app expects them (VRM into
static/models/ plus a models.json registry entry; voice ref into
models/characters/<id>/; prompt into characters/installed/<id>/) and
produces the config patch that activates the character. Run as:

    python -m amadeus.characters path/to/pack [--apply] [--config FILE]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - bundled fallback parser
    from .utils import simple_yaml as yaml

from .logging_utils import get_logger

LOGGER = get_logger(__name__)

PACK_MANIFEST = "character.yaml"
MAX_PACK_UNCOMPRESSED_BYTES = 512 * 1024 * 1024


def _repo_root() -> Path:
    """The tree that holds characters/, static/, models/.

    In a checkout that is the repo root. In the packaged desktop app the
    amadeus package lives inside the PyInstaller payload, but the shell
    passes the real asset tree via --static-dir (exported as
    AMADEUS_STATIC_DIR) — its parent is where characters/packs and the
    install destinations actually live."""
    static_override = os.environ.get("AMADEUS_STATIC_DIR")
    if static_override:
        return Path(static_override).resolve().parent
    return Path(__file__).resolve().parent.parent


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    if not slug:
        raise ValueError("Character name produces an empty id.")
    return slug


@dataclasses.dataclass
class InstallResult:
    """What an install did, plus the config patch that activates it."""

    character_id: str
    name: str
    full_name: str
    prompt_path: Path
    avatar_path: Optional[Path]
    voice_ref_path: Optional[Path]
    config_patch: Dict[str, Any]
    installed_files: List[Path]

    def config_yaml_snippet(self) -> str:
        return yaml.safe_dump(self.config_patch, sort_keys=False).rstrip()


def _load_manifest(pack_dir: Path) -> Dict[str, Any]:
    manifest_path = pack_dir / PACK_MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"Not a character pack: {PACK_MANIFEST} is missing.")
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{PACK_MANIFEST} must be a YAML mapping.")
    for required in ("name", "full_name"):
        if not str(data.get(required) or "").strip():
            raise ValueError(f"{PACK_MANIFEST} is missing required field: {required}.")
    return data


def _pack_file(pack_dir: Path, name: str, *, kind: str) -> Path:
    """A file referenced by the manifest, confined to the pack directory."""

    candidate = (pack_dir / name).resolve()
    if not str(candidate).startswith(str(pack_dir.resolve())):
        raise ValueError(f"Pack {kind} path escapes the pack: {name!r}.")
    if not candidate.is_file():
        raise ValueError(f"Pack {kind} file not found: {name!r}.")
    return candidate


def _update_models_registry(
    models_dir: Path, character_id: str, label: str, vrm_name: str
) -> Path:
    """Merge/append the avatar into static/models/models.json."""

    registry_path = models_dir / "models.json"
    registry: Dict[str, Any] = {"models": []}
    if registry_path.is_file():
        try:
            loaded = json.loads(registry_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("models"), list):
                registry = loaded
        except ValueError:
            LOGGER.warning("models.json is unreadable; rewriting it.")
    entry = {
        "id": character_id,
        "label": label,
        "file": f"/static/models/{vrm_name}",
    }
    models = [m for m in registry["models"] if m.get("id") != character_id]
    models.append(entry)
    registry["models"] = models
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return registry_path


def install_character_pack(
    source: Path,
    *,
    static_models_dir: Optional[Path] = None,
    voice_assets_dir: Optional[Path] = None,
    prompts_dir: Optional[Path] = None,
) -> InstallResult:
    """Install a pack from a folder or zip. Returns what was installed.

    The three destination roots default to the repo layout
    (static/models, models/characters, characters/installed) and exist as
    parameters so tests and packaged builds can redirect them.
    """

    root = _repo_root()
    static_models_dir = static_models_dir or (root / "static" / "models")
    voice_assets_dir = voice_assets_dir or (root / "models" / "characters")
    prompts_dir = prompts_dir or (root / "characters" / "installed")

    source = Path(source)
    if source.is_file() and source.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory(prefix="amadeus-pack-") as tmp:
            _extract_pack_zip(source, Path(tmp))
            return install_character_pack(
                _pack_root_in(Path(tmp)),
                static_models_dir=static_models_dir,
                voice_assets_dir=voice_assets_dir,
                prompts_dir=prompts_dir,
            )
    if not source.is_dir():
        raise ValueError(f"Character pack not found: {source}")

    manifest = _load_manifest(source)
    name = str(manifest["name"]).strip()
    full_name = str(manifest["full_name"]).strip()
    character_id = slugify(name)
    installed: List[Path] = []

    # Prompt (required).
    prompt_src = _pack_file(
        source, str(manifest.get("prompt_file") or "prompt.txt"), kind="prompt"
    )
    prompt_text = prompt_src.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError("Pack prompt file is empty.")
    prompt_dest = prompts_dir / character_id / "prompt.txt"
    prompt_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(prompt_src, prompt_dest)
    installed.append(prompt_dest)

    patch: Dict[str, Any] = {
        "character": {
            "name": name,
            "full_name": full_name,
            "system_prompt_path": str(prompt_dest),
        }
    }
    # The avatar to auto-select in the UI: an explicit manifest model_id
    # (avatar installed separately into models.json), or this pack's own
    # bundled avatar, which registers under character_id below.
    manifest_model_id = str(manifest.get("model_id") or "").strip()
    if manifest_model_id:
        patch["character"]["model_id"] = manifest_model_id

    # Birthday (optional, "MM-DD"/"July 25"/ISO): greeting milestone on the
    # matching day. Only ever set from the manifest — a pack without one
    # must NOT inherit built-in Kurisu's July 25 (see CharacterSettings).
    manifest_birthday = str(manifest.get("birthday") or "").strip()
    if manifest_birthday:
        patch["character"]["birthday"] = manifest_birthday

    # Voice style (optional): 1–3 sentence register spec for the satellite
    # prompts (persona.persona_capsule). Distinct from the ``voice:`` block,
    # which is TTS reference audio.
    manifest_voice_style = str(manifest.get("voice_style") or "").strip()
    if manifest_voice_style:
        patch["character"]["voice_style"] = manifest_voice_style

    # Avatar (optional).
    avatar_dest: Optional[Path] = None
    avatar_name = manifest.get("avatar") or (
        "avatar.vrm" if (source / "avatar.vrm").is_file() else None
    )
    if avatar_name:
        avatar_src = _pack_file(source, str(avatar_name), kind="avatar")
        vrm_name = f"{character_id}.vrm"
        static_models_dir.mkdir(parents=True, exist_ok=True)
        avatar_dest = static_models_dir / vrm_name
        shutil.copy2(avatar_src, avatar_dest)
        installed.append(avatar_dest)
        installed.append(
            _update_models_registry(static_models_dir, character_id, name, vrm_name)
        )
        patch["character"].setdefault("model_id", character_id)

    # Voice (optional).
    voice_dest: Optional[Path] = None
    voice = manifest.get("voice") or {}
    if isinstance(voice, dict) and voice.get("ref_audio"):
        voice_src = _pack_file(source, str(voice["ref_audio"]), kind="voice reference")
        ref_text = str(voice.get("ref_text") or "").strip()
        if not ref_text:
            raise ValueError(
                "voice.ref_text (the clip transcript) is required with voice.ref_audio."
            )
        voice_dir = voice_assets_dir / character_id
        voice_dir.mkdir(parents=True, exist_ok=True)
        voice_dest = voice_dir / f"ref{voice_src.suffix.lower() or '.wav'}"
        shutil.copy2(voice_src, voice_dest)
        installed.append(voice_dest)
        providers_patch: Dict[str, Any] = {
            "qwen3_tts_ref_audio": str(voice_dest),
            "qwen3_tts_ref_text": ref_text,
        }
        if voice.get("language"):
            providers_patch["qwen3_tts_language"] = str(voice["language"])
        patch["providers"] = providers_patch

    spoken = str(manifest.get("spoken_language") or "").strip().lower()
    if spoken in ("japanese", "english"):
        patch["audio"] = {"voice_language": spoken}

    return InstallResult(
        character_id=character_id,
        name=name,
        full_name=full_name,
        prompt_path=prompt_dest,
        avatar_path=avatar_dest,
        voice_ref_path=voice_dest,
        config_patch=patch,
        installed_files=installed,
    )


def _extract_pack_zip(source: Path, dest: Path) -> None:
    with zipfile.ZipFile(source) as archive:
        total = sum(info.file_size for info in archive.infolist())
        if total > MAX_PACK_UNCOMPRESSED_BYTES:
            raise ValueError("Character pack zip is too large.")
        for info in archive.infolist():
            target = (dest / info.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise ValueError(
                    f"Pack zip entry escapes the archive: {info.filename!r}."
                )
        archive.extractall(dest)


def _pack_root_in(extracted: Path) -> Path:
    """The directory holding character.yaml (allows a single top-level folder)."""

    if (extracted / PACK_MANIFEST).is_file():
        return extracted
    subdirs = [p for p in extracted.iterdir() if p.is_dir()]
    for sub in subdirs:
        if (sub / PACK_MANIFEST).is_file():
            return sub
    raise ValueError(f"Not a character pack: {PACK_MANIFEST} is missing.")


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def apply_patch_to_config(config_path: Path, patch: Dict[str, Any]) -> Path:
    """Merge the pack's patch into a config YAML file.

    Comments in the file are lost (plain YAML round-trip), so the previous
    file is kept as ``<name>.bak-character`` next to it.
    """

    existing: Dict[str, Any] = {}
    if config_path.is_file():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            existing = loaded
        backup = config_path.with_name(config_path.name + ".bak-character")
        shutil.copy2(config_path, backup)
    merged = _deep_merge(existing, patch)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    return config_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m amadeus.characters",
        description="Install a character pack (folder or zip).",
    )
    parser.add_argument("pack", help="Path to the pack folder or .zip")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Merge the config patch into the config file (a .bak-character copy is kept; comments are lost)",
    )
    parser.add_argument(
        "--config",
        default="config.local.yaml",
        help="Config file for --apply (default: config.local.yaml)",
    )
    args = parser.parse_args(argv)

    try:
        result = install_character_pack(Path(args.pack))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Installed character pack: {result.name} ({result.character_id})")
    for path in result.installed_files:
        print(f"  + {path}")
    if args.apply:
        target = apply_patch_to_config(Path(args.config), result.config_patch)
        print(f"Config updated: {target} (previous file kept as *.bak-character)")
        print("Restart the app to meet them.")
    else:
        print("\nAdd this to your config (or re-run with --apply):\n")
        print(result.config_yaml_snippet())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
