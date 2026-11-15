"""Character pack installer (amadeus/characters.py)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import yaml

from amadeus.characters import (
    apply_patch_to_config,
    install_character_pack,
    main,
    slugify,
)

PROMPT = "You are Rin Asakura, a retired pro gamer. Deadpan, competitive."


def _make_pack(tmp_path, *, avatar=False, voice=False, extra_yaml="") -> Path:
    pack = tmp_path / "pack"
    pack.mkdir()
    manifest = "name: Rin\nfull_name: Rin Asakura\n"
    if avatar:
        (pack / "avatar.vrm").write_bytes(b"fake-vrm-bytes")
        manifest += "avatar: avatar.vrm\n"
    if voice:
        (pack / "voice_ref.wav").write_bytes(b"fake-wav-bytes")
        manifest += 'voice:\n  ref_audio: voice_ref.wav\n  ref_text: "Hello there."\n  language: English\n'
    manifest += extra_yaml
    (pack / "character.yaml").write_text(manifest, encoding="utf-8")
    (pack / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    return pack


def _dests(tmp_path):
    return {
        "static_models_dir": tmp_path / "static-models",
        "voice_assets_dir": tmp_path / "voices",
        "prompts_dir": tmp_path / "prompts",
    }


def test_minimal_pack_installs_prompt_and_patch(tmp_path):
    pack = _make_pack(tmp_path)
    result = install_character_pack(pack, **_dests(tmp_path))

    assert result.character_id == "rin"
    assert result.prompt_path.read_text(encoding="utf-8") == PROMPT
    assert result.config_patch["character"]["name"] == "Rin"
    assert result.config_patch["character"]["full_name"] == "Rin Asakura"
    assert result.config_patch["character"]["system_prompt_path"] == str(
        result.prompt_path
    )
    assert result.avatar_path is None and result.voice_ref_path is None


def test_avatar_installs_and_registers_in_models_json(tmp_path):
    pack = _make_pack(tmp_path, avatar=True)
    dests = _dests(tmp_path)
    result = install_character_pack(pack, **dests)

    assert result.avatar_path is not None and result.avatar_path.is_file()
    registry = json.loads((dests["static_models_dir"] / "models.json").read_text())
    entry = next(m for m in registry["models"] if m["id"] == "rin")
    assert entry["file"] == "/static/models/rin.vrm"

    # Reinstall replaces, not duplicates.
    install_character_pack(pack, **dests)
    registry = json.loads((dests["static_models_dir"] / "models.json").read_text())
    assert sum(1 for m in registry["models"] if m["id"] == "rin") == 1


def test_voice_installs_ref_and_patches_providers(tmp_path):
    pack = _make_pack(tmp_path, voice=True, extra_yaml="spoken_language: english\n")
    result = install_character_pack(pack, **_dests(tmp_path))

    assert result.voice_ref_path is not None and result.voice_ref_path.is_file()
    providers = result.config_patch["providers"]
    assert providers["qwen3_tts_ref_audio"] == str(result.voice_ref_path)
    assert providers["qwen3_tts_ref_text"] == "Hello there."
    assert providers["qwen3_tts_language"] == "English"
    assert result.config_patch["audio"] == {"voice_language": "english"}


def test_voice_without_transcript_is_rejected(tmp_path):
    pack = _make_pack(tmp_path)
    (pack / "voice_ref.wav").write_bytes(b"x")
    (pack / "character.yaml").write_text(
        "name: Rin\nfull_name: Rin Asakura\nvoice:\n  ref_audio: voice_ref.wav\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ref_text"):
        install_character_pack(pack, **_dests(tmp_path))


def test_zip_pack_installs(tmp_path):
    pack = _make_pack(tmp_path)
    zip_path = tmp_path / "rin.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in pack.iterdir():
            archive.write(path, f"rin/{path.name}")  # single top-level folder

    result = install_character_pack(zip_path, **_dests(tmp_path))
    assert result.character_id == "rin"
    assert result.prompt_path.is_file()


def test_missing_manifest_is_rejected(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="character.yaml"):
        install_character_pack(empty, **_dests(tmp_path))


def test_manifest_path_escape_is_rejected(tmp_path):
    pack = _make_pack(tmp_path, extra_yaml="prompt_file: ../outside.txt\n")
    (tmp_path / "outside.txt").write_text("evil", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        install_character_pack(pack, **_dests(tmp_path))


def test_apply_patch_merges_and_backs_up(tmp_path):
    config_path = tmp_path / "config.local.yaml"
    config_path.write_text(
        yaml.safe_dump({"providers": {"llm": "gemini"}, "llm_model": "x"}),
        encoding="utf-8",
    )
    patch = {"character": {"name": "Rin"}, "providers": {"qwen3_tts_ref_audio": "y"}}

    apply_patch_to_config(config_path, patch)

    merged = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert merged["character"]["name"] == "Rin"
    assert merged["providers"]["llm"] == "gemini"  # deep merge kept siblings
    assert merged["providers"]["qwen3_tts_ref_audio"] == "y"
    assert merged["llm_model"] == "x"
    assert config_path.with_name("config.local.yaml.bak-character").is_file()


def test_bundled_example_pack_installs(tmp_path):
    example = Path(__file__).resolve().parent.parent / "characters" / "examples" / "rin"
    result = install_character_pack(example, **_dests(tmp_path))
    assert result.character_id == "rin"
    assert "Rin Asakura" in result.prompt_path.read_text(encoding="utf-8")
    assert result.config_patch["audio"] == {"voice_language": "english"}


def test_cli_main_prints_snippet(tmp_path, capsys, monkeypatch):
    pack = _make_pack(tmp_path)
    # Redirect the default destinations away from the repo.
    import amadeus.characters as characters_module

    monkeypatch.setattr(characters_module, "_repo_root", lambda: tmp_path / "root")
    code = main([str(pack)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Installed character pack: Rin" in out
    assert "system_prompt_path" in out


def test_slugify():
    assert slugify("Rin Asakura!") == "rin-asakura"
    with pytest.raises(ValueError):
        slugify("!!!")
