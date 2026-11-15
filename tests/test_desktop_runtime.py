from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from starlette.routing import Mount

from amadeus.config import Config
from amadeus.desktop.runtime import (
    REQUIRED_ASSET_PATHS,
    build_desktop_config,
    reconcile_desktop_config,
    validate_required_assets,
    write_desktop_config,
)
from amadeus.web.app import create_app


def _make_required_assets(root: Path) -> None:
    for relative in REQUIRED_ASSET_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("asset", encoding="utf-8")


def test_validate_required_assets_reports_missing_and_present(tmp_path):
    (tmp_path / "static" / "models").mkdir(parents=True)
    (tmp_path / "static" / "models" / "Kurisu.vrm").write_text("vrm", encoding="utf-8")

    result = validate_required_assets(tmp_path)

    assert not result.ok
    assert "static/models/Kurisu.vrm" in result.present
    assert "models/kurisu_voice_onnx/model.onnx" in result.missing


def test_build_desktop_config_uses_app_data_and_bundled_resources(tmp_path):
    resources = tmp_path / "resources"
    app_data = tmp_path / "app-data"
    _make_required_assets(resources)

    config = build_desktop_config(app_data_dir=app_data, resources_dir=resources)

    assert isinstance(config, Config)
    assert config.gemini_api_key is None
    assert config.providers.llm == "gemini"
    assert config.providers.tts == "melotts_onnx"
    assert config.audio.enable_voice is True
    assert config.web.streaming_enabled is True
    assert config.providers.melotts_onnx_model_path == str(
        resources / "models" / "kurisu_voice_onnx" / "model.onnx"
    )
    assert config.kurisu_isms_path == str(resources / "assets" / "kurisu_isms.txt")
    assert config.memory.persist_path == str(app_data / "memory" / "memory_store.json")
    assert config.memory.embeddings_path == str(
        app_data / "memory" / "memory_embeddings.json"
    )
    assert config.memory.profile_path == str(
        app_data / "memory" / "memory_profile.json"
    )
    assert config.transcript_path == str(app_data / "transcripts")
    assert config.logging.logfile == str(app_data / "logs" / "amadeus.log")

    # Memory-as-personality: facts, episodes, greeting, reflection all on.
    assert config.memory.enable_fact_extraction is True
    assert config.memory.enable_episodes is True
    assert config.memory.enable_greeting is True
    assert config.memory.enable_reflection is True
    # Long-term recall is now ON in the desktop build via the dependency-free
    # HashingEmbedding + JSON vector store (no torch/sentence-transformers).
    assert config.memory.enable_long_term is True
    assert config.providers.embedding == "hashing"
    assert config.memory.vector_backend == "json"
    # Summarizer must not use the gpt-3.5-turbo default — it 404s on Gemini.
    assert config.memory.summary_model == config.llm_model
    # Presence features on: the frontend is fully wired for poke/idle.
    assert config.memory.enable_poke_reactions is True
    assert config.memory.enable_idle_chatter is True
    # CPU/ONNX build keeps the single synthesis worker default.
    assert config.web.tts_stream_workers == 1


def _make_qwen_ref(root: Path) -> Path:
    clip = root / "models" / "kurisu_voice_qwen" / "kurisu_ref.wav"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"RIFF")
    return clip


def test_prefer_qwen_selects_qwen_when_gpu_and_clip_present(tmp_path, monkeypatch):
    import amadeus.desktop.runtime as runtime

    monkeypatch.setattr(runtime, "cuda_available", lambda: True)
    resources = tmp_path / "resources"
    _make_required_assets(resources)
    clip = _make_qwen_ref(resources)

    config = build_desktop_config(
        app_data_dir=tmp_path / "app-data",
        resources_dir=resources,
        prefer_qwen=True,
    )

    assert config.providers.tts == "qwen3_tts"
    assert config.providers.qwen3_tts_ref_audio == str(clip)
    # ONNX fallback paths must still be populated for the runtime fallback.
    assert config.providers.melotts_onnx_model_path.endswith("model.onnx")
    # GPU build overlaps sentence synthesis with two workers.
    assert config.web.tts_stream_workers == 2


def test_prefer_qwen_falls_back_to_onnx_without_gpu(tmp_path, monkeypatch):
    import amadeus.desktop.runtime as runtime

    # Both probes must say "no GPU" for the ONNX fallback: cuda_available()
    # (torch-based, always False in a slim build) OR'd with
    # gpu_runtime.gpu_likely_present() (torch-free nvidia-smi probe) is the
    # actual bake decision since the slim GPU installer landed.
    monkeypatch.setattr(runtime, "cuda_available", lambda: False)
    monkeypatch.setattr(runtime.gpu_runtime, "gpu_likely_present", lambda: False)
    resources = tmp_path / "resources"
    _make_required_assets(resources)
    _make_qwen_ref(resources)

    config = build_desktop_config(
        app_data_dir=tmp_path / "app-data",
        resources_dir=resources,
        prefer_qwen=True,
    )

    assert config.providers.tts == "melotts_onnx"


def test_prefer_qwen_selects_qwen_via_driver_probe_without_cuda_torch(
    tmp_path, monkeypatch
):
    """Slim GPU installer: no CUDA torch bundled (cuda_available() is always
    False), but nvidia-smi is visible — bakes qwen3_tts optimistically so the
    runtime-pack download (amadeus/gpu_runtime.py) has something to activate
    into once it lands; the ONNX fallback covers every launch until then."""
    import amadeus.desktop.runtime as runtime

    monkeypatch.setattr(runtime, "cuda_available", lambda: False)
    monkeypatch.setattr(runtime.gpu_runtime, "gpu_likely_present", lambda: True)
    resources = tmp_path / "resources"
    _make_required_assets(resources)
    _make_qwen_ref(resources)

    config = build_desktop_config(
        app_data_dir=tmp_path / "app-data",
        resources_dir=resources,
        prefer_qwen=True,
    )

    assert config.providers.tts == "qwen3_tts"


def test_prefer_qwen_falls_back_to_onnx_without_clip(tmp_path, monkeypatch):
    import amadeus.desktop.runtime as runtime

    monkeypatch.setattr(runtime, "cuda_available", lambda: True)
    resources = tmp_path / "resources"
    _make_required_assets(resources)  # no qwen ref clip

    config = build_desktop_config(
        app_data_dir=tmp_path / "app-data",
        resources_dir=resources,
        prefer_qwen=True,
    )

    assert config.providers.tts == "melotts_onnx"


def test_default_build_ignores_qwen_even_with_gpu_and_clip(tmp_path, monkeypatch):
    import amadeus.desktop.runtime as runtime

    monkeypatch.setattr(runtime, "cuda_available", lambda: True)
    resources = tmp_path / "resources"
    _make_required_assets(resources)
    _make_qwen_ref(resources)

    # prefer_qwen defaults to False (the CPU build): always ONNX.
    config = build_desktop_config(
        app_data_dir=tmp_path / "app-data",
        resources_dir=resources,
    )

    assert config.providers.tts == "melotts_onnx"


def test_write_desktop_config_never_persists_secret(tmp_path):
    resources = tmp_path / "resources"
    app_data = tmp_path / "app-data"
    _make_required_assets(resources)
    config = build_desktop_config(app_data_dir=app_data, resources_dir=resources)
    config.gemini_api_key = "must-not-be-written"

    config_path = tmp_path / "config.yaml"
    write_desktop_config(config_path, config)

    text = config_path.read_text(encoding="utf-8")
    assert "must-not-be-written" not in text
    assert "gemini_api_key" not in text
    loaded = Config.load(config_path)
    assert loaded.providers.tts == "melotts_onnx"


def _write_config_from_old_install(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A config whose resource paths point at a now-deleted previous install."""

    old_resources = tmp_path / "old-install"
    resources = tmp_path / "resources"
    app_data = tmp_path / "app-data"
    _make_required_assets(old_resources)
    _make_required_assets(resources)
    for root in (old_resources, resources):
        clip = root / "models" / "kurisu_voice_qwen" / "kurisu_ref.wav"
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_text("clip", encoding="utf-8")

    config = build_desktop_config(app_data_dir=app_data, resources_dir=old_resources)
    config.providers.tts = "qwen3_tts"
    config.providers.qwen3_tts_ref_audio = str(
        old_resources / "models" / "kurisu_voice_qwen" / "kurisu_ref.wav"
    )
    config_path = tmp_path / "config.yaml"
    write_desktop_config(config_path, config)
    return config_path, resources, app_data


def test_reconcile_rewrites_dead_resource_paths(tmp_path):
    config_path, resources, app_data = _write_config_from_old_install(tmp_path)
    import shutil

    shutil.rmtree(tmp_path / "old-install")  # previous install is gone

    changed = reconcile_desktop_config(
        config_path, app_data_dir=app_data, resources_dir=resources
    )

    assert changed
    loaded = Config.load(config_path)
    assert loaded.providers.melotts_onnx_model_path == str(
        resources / "models" / "kurisu_voice_onnx" / "model.onnx"
    )
    assert loaded.providers.qwen3_tts_ref_audio == str(
        resources / "models" / "kurisu_voice_qwen" / "kurisu_ref.wav"
    )
    assert loaded.kurisu_isms_path == str(resources / "assets" / "kurisu_isms.txt")
    # Untouched settings survive the rewrite.
    assert loaded.providers.tts == "qwen3_tts"


def test_reconcile_keeps_existing_custom_paths(tmp_path):
    config_path, resources, app_data = _write_config_from_old_install(tmp_path)
    # The "old install" still exists: the paths are alive, maybe deliberate.

    changed = reconcile_desktop_config(
        config_path, app_data_dir=app_data, resources_dir=resources
    )

    assert not changed
    loaded = Config.load(config_path)
    assert str(tmp_path / "old-install") in loaded.providers.melotts_onnx_model_path


def test_reconcile_anchors_relative_tts_cache_dir(tmp_path):
    config_path, resources, app_data = _write_config_from_old_install(tmp_path)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace(
        "tts_cache_dir: " + str(app_data / "cache" / "tts"),
        "tts_cache_dir: cache/tts",
    )
    config_path.write_text(text, encoding="utf-8")

    changed = reconcile_desktop_config(
        config_path, app_data_dir=app_data, resources_dir=resources
    )

    assert changed
    loaded = Config.load(config_path)
    assert loaded.providers.tts_cache_dir == str(app_data / "cache" / "tts")


def test_server_main_reconciles_config_and_disables_access_log(tmp_path, monkeypatch):
    """The sidecar boot must heal dead inherited paths and never enable
    uvicorn's per-request access log (console writes are pipe pressure in the
    packaged app; a full pipe blocks the event loop — seen live v0.15.0)."""

    import shutil

    from amadeus.desktop import server

    config_path, resources, app_data = _write_config_from_old_install(tmp_path)
    shutil.rmtree(tmp_path / "old-install")

    for key in (
        "AMADEUS_CONFIG",
        "AMADEUS_USER_CONFIG",
        "KURISU_FRONTEND_DIR",
        "AMADEUS_STATIC_DIR",
        "AMADEUS_NO_MODULE_APP",
    ):
        monkeypatch.setenv(key, "sentinel")  # registers teardown restore

    captured: dict = {}

    def fake_run(app, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(server.uvicorn, "run", fake_run)
    monkeypatch.setattr("amadeus.web.app.create_app", lambda: object())

    rc = server.main(
        [
            "--port",
            "8123",
            "--config",
            str(config_path),
            "--frontend-dir",
            str(tmp_path / "fe"),
            "--static-dir",
            str(tmp_path / "st"),
            "--resources-dir",
            str(resources),
            "--app-data-dir",
            str(app_data),
            "--write-default-config",
        ]
    )

    assert rc == 0
    assert captured.get("access_log") is False
    loaded = Config.load(config_path)
    assert loaded.providers.melotts_onnx_model_path == str(
        resources / "models" / "kurisu_voice_onnx" / "model.onnx"
    )


def test_build_desktop_config_anchors_tts_cache_to_app_data(tmp_path):
    resources = tmp_path / "resources"
    app_data = tmp_path / "app-data"
    _make_required_assets(resources)

    config = build_desktop_config(app_data_dir=app_data, resources_dir=resources)

    cache_dir = Path(config.providers.tts_cache_dir)
    assert cache_dir.is_absolute()
    assert str(app_data.resolve()) in str(cache_dir)


def test_web_app_mounts_static_dir_override(monkeypatch, tmp_path):
    import amadeus.web.app as app_module

    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "probe.txt").write_text("desktop-static", encoding="utf-8")
    monkeypatch.setenv("AMADEUS_STATIC_DIR", str(static_dir))
    monkeypatch.setenv("KURISU_FRONTEND_DIR", str(tmp_path / "no-frontend"))
    monkeypatch.setattr(app_module, "load_llm", lambda cfg: None)
    monkeypatch.setattr(app_module, "load_embedder", lambda cfg: None)
    monkeypatch.setattr(app_module, "load_tts", lambda cfg: None)

    config = Config()
    config.audio.enable_voice = False
    config.memory.persist_path = str(tmp_path / "memory.json")
    config.memory.embeddings_path = str(tmp_path / "embeddings.json")
    config.memory.profile_path = str(tmp_path / "profile.json")
    app = create_app(config)

    static_mount = next(
        route
        for route in app.routes
        if isinstance(route, Mount) and route.path == "/static"
    )

    assert Path(static_mount.app.directory) == static_dir
