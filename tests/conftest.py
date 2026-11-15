import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Memory tiers default ON now, and their default paths are cwd-relative — a
# test that builds MemoryManager(Config()) without isolating paths would
# silently create/mutate these files in the repo root (including a user's
# real profile). Guard: fail the offending test and remove what it wrote.
_GUARDED_MEMORY_FILES = (
    "memory_profile.json",
    "memory_store.json",
    "memory_embeddings.json",
    "memory_transcripts.json",
    "persona_versions.json",
    "daily_brief.json",
    "kurisu_diary.json",
    "insights.json",
)


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path_factory, monkeypatch):
    """Point the platform data dir at a temp dir and disable legacy migration.

    Config.load() resolves relative user-data paths against
    amadeus.paths.default_data_dir(); without this, tests would read/write
    the developer's real data dir — and the entrypoints' one-time CWD
    migration could move real repo-root files into a test directory.
    """

    monkeypatch.setenv("AMADEUS_DATA_DIR", str(tmp_path_factory.mktemp("data-dir")))
    monkeypatch.setenv("AMADEUS_NO_LEGACY_MIGRATION", "1")
    yield


@pytest.fixture(autouse=True)
def _redirect_default_memory_paths(tmp_path_factory, monkeypatch):
    """Point the cwd-relative memory-path DEFAULTS at a temp dir per test.

    MemorySettings' defaults (memory_store.json / memory_embeddings.json /
    memory_profile.json) are cwd-relative, and with the memory tiers on by
    default any test that builds Config() without overriding them would write
    into the repo root. Dataclass defaults live in __init__.__defaults__ (all
    MemorySettings fields have defaults, so positions line up with fields()).
    Tests that set explicit paths are unaffected.
    """
    import dataclasses

    from amadeus.config import MemorySettings

    base = tmp_path_factory.mktemp("default-memory")
    field_names = [f.name for f in dataclasses.fields(MemorySettings)]
    new_defaults = list(MemorySettings.__init__.__defaults__)
    for name, filename in (
        ("persist_path", "memory_store.json"),
        ("embeddings_path", "memory_embeddings.json"),
        ("profile_path", "memory_profile.json"),
        ("transcripts_path", "memory_transcripts.json"),
        ("persona_versions_path", "persona_versions.json"),
    ):
        new_defaults[field_names.index(name)] = str(base / filename)
    monkeypatch.setattr(MemorySettings.__init__, "__defaults__", tuple(new_defaults))
    yield


@pytest.fixture(autouse=True)
def _no_repo_root_memory_writes():
    existed = {name: (REPO_ROOT / name).exists() for name in _GUARDED_MEMORY_FILES}
    yield
    leaked = []
    for name, was_there in existed.items():
        path = REPO_ROOT / name
        if not was_there and path.exists():
            path.unlink()
            leaked.append(name)
    if leaked:
        pytest.fail(
            f"test wrote {leaked} into the repo root — isolate memory paths "
            "(memory.persist_path/embeddings_path/profile_path) to tmp_path"
        )
