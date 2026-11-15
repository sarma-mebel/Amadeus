"""play_animation tool: registration, allowlist, and turn plumbing."""

from amadeus.agents.factory import create_tool_registry
from amadeus.agents.tools import AnimationTool
from amadeus.config import Config


def test_registered_by_default_with_curated_clips():
    registry = create_tool_registry(Config())
    tool = registry.tools.get("play_animation")
    assert tool is not None
    assert "Spin" in tool.clips
    # The definition advertises the allowlist so the model can pick.
    assert "Spin" in tool.definition.description
    assert tool.definition.parameters[0].enum == tool.clips


def test_description_restricts_to_explicit_requests():
    """Gemini called play_animation on 3 of 4 casual turns ('Greeting' for
    good-morning, 'Clapping' for good news) with the old 'physical flourish'
    wording — she clapped after every reply on the desktop app and burned tool
    iterations. The description must demand an explicit user request and say
    automatic body language already exists."""
    tool = AnimationTool(["Spin"])
    desc = tool.definition.description
    assert "ONLY when the user explicitly asks" in desc
    assert "automatically" in desc
    assert "physical flourish" not in desc


def test_default_allowlist_clips_exist_in_manifest():
    """Every default animation_clips name must resolve in the committed
    manifest with its file present — a rename would otherwise make the tool
    silently no-op in every build (including the packaged desktop app, which
    ships static/ wholesale)."""
    import json
    from pathlib import Path

    manifest_path = Path("static/animations/manifest.json")
    if not manifest_path.exists():  # minimal checkouts without assets
        import pytest

        pytest.skip("animation manifest not present")
    manifest = json.loads(manifest_path.read_text())
    entries = {clip["name"]: clip for section in manifest.values() for clip in section}
    for name in Config().tools.animation_clips:
        assert name in entries, f"allowlisted clip {name!r} missing from manifest"
        entry = entries[name]
        assert not entry.get(
            "disabled"
        ), f"allowlisted clip {name!r} is disabled in the manifest"
        clip_file = (
            Path("static/animations") / entry.get("directory", "") / entry["file"]
        )
        assert (
            clip_file.exists()
        ), f"{name!r} manifest entry points at missing {clip_file}"


def test_empty_allowlist_disables_the_tool():
    config = Config()
    config.tools.animation_clips = []
    registry = create_tool_registry(config)
    assert "play_animation" not in registry.tools


def test_execute_records_request_and_pop_drains():
    tool = AnimationTool(["Spin", "Jump"])
    result = tool.execute(name="Spin")
    assert result.success
    assert tool.pop_requested() == ["Spin"]
    assert tool.pop_requested() == []


def test_execute_rejects_unknown_clip():
    tool = AnimationTool(["Spin"])
    result = tool.execute(name="Backflip")
    assert not result.success
    assert "Backflip" in result.error
    assert tool.pop_requested() == []


def test_engine_drains_requests_for_the_response(tmp_path):
    from amadeus.engine import ChatEngine
    from amadeus.providers.llm.base import DummyProvider

    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    engine = ChatEngine(config, DummyProvider())

    engine.tool_registry.tools["play_animation"].execute(name="Jump")
    engine._drain_animation_requests()
    assert engine.pop_pending_animations() == ["Jump"]
    # Drained exactly once.
    assert engine.pop_pending_animations() == []


def test_keyword_gate_catches_dance_requests(tmp_path):
    from amadeus.engine import ChatEngine
    from amadeus.providers.llm.base import DummyProvider

    config = Config()
    config.tools.enabled = True
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    engine = ChatEngine(config, DummyProvider())
    # Dummy provider has no native function calling -> keyword gate path.
    assert engine._should_use_tools("dance for me!") is True
    assert engine._should_use_tools("hi") is False


def test_chat_response_carries_pending_animations(tmp_path):
    """The web turn drains engine._pending_animations into the response."""
    import pytest

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from amadeus.web.app import create_app

    config = Config()
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    app = create_app(config)
    client = TestClient(app)
    store = app.state.session_store
    _, session = store.get(None, "user-1")
    session._pending_animations = ["Dance 1"]

    response = client.post("/api/chat", json={"message": "hello", "user_id": "user-1"})
    assert response.status_code == 200
    assert response.json()["animations"] == ["Dance 1"]
    # Drained: the next turn has none.
    second = client.post("/api/chat", json={"message": "hi", "user_id": "user-1"})
    assert second.json()["animations"] == []
