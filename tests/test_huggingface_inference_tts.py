import base64
import sys
import types

import pytest


def _install_dummy_gradio(monkeypatch, calls, result=b"audio-bytes"):
    class DummyClient:
        def __init__(self, model, hf_token=None):
            calls["init"] = {"model": model, "hf_token": hf_token}
            calls["init_count"] = calls.get("init_count", 0) + 1

        def predict(self, *args, api_name=None, **kwargs):
            calls["call"] = {"args": args, "api_name": api_name, "kwargs": kwargs}
            return result

    dummy_module = types.SimpleNamespace(Client=DummyClient)
    monkeypatch.setitem(sys.modules, "gradio_client", dummy_module)


def _make_provider(monkeypatch, calls):
    _install_dummy_gradio(monkeypatch, calls)
    from amadeus.providers.tts.huggingface_inference import (
        HuggingFaceInferenceTTSProvider,
    )

    return HuggingFaceInferenceTTSProvider(
        model="zxKyouma/Amadeus",
        api_token="secret",
        api_name="/predict",
    )


def test_huggingface_inference_provider(monkeypatch):
    calls: dict[str, dict] = {}
    provider = _make_provider(monkeypatch, calls)

    audio = provider.synthesize("Hello, world!", speaking_rate=1.3, pitch_shift=-0.2)

    assert audio == b"audio-bytes"
    assert calls["init"] == {"model": "zxKyouma/Amadeus", "hf_token": "secret"}
    assert calls["call"]["args"] == ("Hello, world!", 1.3)
    assert calls["call"]["api_name"] == "/predict"


def test_huggingface_inference_defers_connection_until_first_synthesize(monkeypatch):
    """Constructing the provider must not touch the network — the Gradio
    handshake would otherwise block every app boot (see SessionStore, which
    builds providers eagerly), even offline or before a user sends anything."""
    calls: dict[str, dict] = {}
    _install_dummy_gradio(monkeypatch, calls)
    from amadeus.providers.tts.huggingface_inference import (
        HuggingFaceInferenceTTSProvider,
    )

    provider = HuggingFaceInferenceTTSProvider(
        model="zxKyouma/Amadeus", api_token="secret", api_name="/predict"
    )
    assert "init" not in calls, "Client() must not connect until synthesize() is called"

    provider.synthesize("Hello", speaking_rate=1.0, pitch_shift=0.0)
    assert calls["init"] == {"model": "zxKyouma/Amadeus", "hf_token": "secret"}

    # A second call reuses the same client rather than reconnecting.
    provider.synthesize("Again", speaking_rate=1.0, pitch_shift=0.0)
    assert calls["init_count"] == 1


def test_huggingface_inference_requires_model(monkeypatch):
    _install_dummy_gradio(monkeypatch, {})
    from amadeus.providers.tts.huggingface_inference import (
        HuggingFaceInferenceTTSProvider,
    )

    with pytest.raises(ValueError):
        HuggingFaceInferenceTTSProvider()


@pytest.mark.parametrize(
    "response,expected",
    [
        (b"bytes", b"bytes"),
        (bytearray(b"chunk"), b"chunk"),
        # Gradio returns a tuple/list of outputs; the first item is the audio widget.
        ([b"a", b"b"], b"a"),
        (
            "data:audio/wav;base64," + base64.b64encode(b"wav-data").decode("ascii"),
            b"wav-data",
        ),
    ],
)
def test_huggingface_inference_coerce(response, expected, monkeypatch):
    calls: dict[str, dict] = {}
    provider = _make_provider(monkeypatch, calls)
    assert provider._coerce_audio_bytes(response) == expected


def test_huggingface_inference_coerce_file_path(monkeypatch, tmp_path):
    calls: dict[str, dict] = {}
    provider = _make_provider(monkeypatch, calls)

    audio_file = tmp_path / "reply.wav"
    audio_file.write_bytes(b"wav-from-file")

    assert provider._coerce_audio_bytes(str(audio_file)) == b"wav-from-file"


def test_huggingface_inference_coerce_rejects_bogus_string(monkeypatch, tmp_path):
    calls: dict[str, dict] = {}
    provider = _make_provider(monkeypatch, calls)

    # A string that is neither a data URI nor an existing file (e.g. an error
    # message or a stale temp path) must raise instead of being returned as audio.
    with pytest.raises(TypeError):
        provider._coerce_audio_bytes(str(tmp_path / "gone.wav"))
    with pytest.raises(TypeError):
        provider._coerce_audio_bytes("Internal Server Error")
