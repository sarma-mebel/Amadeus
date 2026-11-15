"""Tests for platform data-dir resolution and legacy CWD migration."""

from pathlib import Path

from amadeus.config import Config
from amadeus.paths import (
    default_data_dir,
    migrate_legacy_cwd_files,
    resolve_user_data_path,
)


def test_data_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("AMADEUS_DATA_DIR", str(tmp_path / "custom"))
    assert default_data_dir() == tmp_path / "custom"


def test_resolve_relative_anchors_to_data_dir(tmp_path):
    resolved = resolve_user_data_path("memory_store.json", data_dir=tmp_path)
    assert resolved == str(tmp_path / "memory_store.json")


def test_resolve_absolute_and_empty_pass_through(tmp_path):
    absolute = str(tmp_path / "explicit.json")
    assert resolve_user_data_path(absolute, data_dir=tmp_path) == absolute
    assert resolve_user_data_path(None) is None


def _write_relative_paths_yaml(path: Path) -> Path:
    # The test conftest patches MemorySettings' defaults to absolute tmp
    # paths, so exercise resolution with explicit relative values (the
    # shipped defaults) instead of relying on the dataclass defaults.
    path.write_text(
        "memory:\n"
        "  persist_path: memory_store.json\n"
        "reminders:\n"
        "  store_path: reminders.json\n",
        encoding="utf-8",
    )
    return path


def test_config_load_resolves_defaults_to_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AMADEUS_DATA_DIR", str(tmp_path))
    config = Config.load(_write_relative_paths_yaml(tmp_path / "config.yaml"))
    assert config.memory.persist_path == str(tmp_path / "memory_store.json")
    assert config.reminders.store_path == str(tmp_path / "reminders.json")


def test_config_load_respects_absolute_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("AMADEUS_DATA_DIR", str(tmp_path / "data"))
    explicit = tmp_path / "elsewhere" / "store.json"
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(f"memory:\n  persist_path: {explicit}\n", encoding="utf-8")
    config = Config.load(yaml_path)
    assert config.memory.persist_path == str(explicit)


def test_migration_moves_legacy_files_and_users_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "memory_store.json").write_text("{}", encoding="utf-8")
    users_dir = cwd / "memory_store_users"
    users_dir.mkdir()
    (users_dir / "u1.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("AMADEUS_DATA_DIR", str(data_dir))
    monkeypatch.delenv("AMADEUS_NO_LEGACY_MIGRATION", raising=False)
    monkeypatch.chdir(cwd)

    config = Config.load(_write_relative_paths_yaml(tmp_path / "config.yaml"))
    migrate_legacy_cwd_files(config)

    assert not (cwd / "memory_store.json").exists()
    assert (data_dir / "memory_store.json").read_text(encoding="utf-8") == "{}"
    assert (data_dir / "memory_store_users" / "u1.json").exists()


def test_migration_skips_when_target_exists(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "memory_store.json").write_text('{"new": true}', encoding="utf-8")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "memory_store.json").write_text('{"old": true}', encoding="utf-8")

    monkeypatch.setenv("AMADEUS_DATA_DIR", str(data_dir))
    monkeypatch.delenv("AMADEUS_NO_LEGACY_MIGRATION", raising=False)
    monkeypatch.chdir(cwd)

    config = Config.load(tmp_path / "missing.yaml")
    migrate_legacy_cwd_files(config)

    assert (cwd / "memory_store.json").exists()
    assert '"new"' in (data_dir / "memory_store.json").read_text(encoding="utf-8")


def test_migration_ignores_custom_configured_paths(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "memory_store.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("AMADEUS_DATA_DIR", str(data_dir))
    monkeypatch.delenv("AMADEUS_NO_LEGACY_MIGRATION", raising=False)
    monkeypatch.chdir(cwd)

    config = Config()
    config.memory.persist_path = str(tmp_path / "elsewhere" / "store.json")
    config.memory.embeddings_path = None
    config.memory.profile_path = None
    config.memory.transcripts_path = None
    config.memory.vector_path = None
    config.reminders.store_path = str(tmp_path / "elsewhere" / "reminders.json")
    migrate_legacy_cwd_files(config)

    assert (cwd / "memory_store.json").exists()
    assert not (data_dir / "memory_store.json").exists()


def test_migration_disabled_by_env(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "memory_store.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("AMADEUS_DATA_DIR", str(data_dir))
    monkeypatch.setenv("AMADEUS_NO_LEGACY_MIGRATION", "1")
    monkeypatch.chdir(cwd)

    config = Config.load(tmp_path / "missing.yaml")
    migrate_legacy_cwd_files(config)

    assert (cwd / "memory_store.json").exists()
