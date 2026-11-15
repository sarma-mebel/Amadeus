import importlib.util
from importlib.machinery import ModuleSpec
import sys
import types
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "export_kurisu_to_onnx.py"
)
SPEC = importlib.util.spec_from_file_location("export_kurisu_to_onnx", SCRIPT_PATH)
export = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(export)


def fake_clean_text(text: str, language: str):
    assert language == "JP"
    normalized = {
        "牧": "牧",
        "牧瀬": "牧瀬",
        "ま": "マ",
        "マ": "マ",
        "。": "。",
        "、": "、",
    }.get(text, text)
    phones_by_text = {
        "牧": ["m", "a"],
        "牧瀬": ["m", "a", "s", "e"],
        "ま": ["m", "a"],
        "マ": ["m", "a"],
    }
    phones = phones_by_text.get(text)
    if phones is None:
        phones = ["a"] if text and text[0] in "あア" else ["m", "a"]
    tones = [0] * len(phones)
    word2ph = [len(phones)]
    return normalized, phones, tones, word2ph


def test_write_tokens_file(tmp_path):
    output = tmp_path / "tokens.txt"

    export.write_tokens_file(["_", "m", "a"], output)

    assert output.read_text(encoding="utf-8").splitlines() == ["_ 0", "m 1", "a 2"]


def test_build_japanese_lexicon_entries_adds_tone_start_and_punctuation():
    entries = export.build_japanese_lexicon_entries(
        texts=["牧瀬"],
        symbols=["_", ",", ".", "!", "?", "…", "m", "a", "s", "e"],
        clean_text=fake_clean_text,
        tone_start=6,
    )

    assert entries["牧"] == (["m", "a"], [6, 6])
    assert entries["牧瀬"] == (["m", "a", "s", "e"], [6, 6, 6, 6])
    assert entries["。"] == (["."], [0])
    assert entries["、"] == ([","], [0])


def test_write_lexicon_file_rejects_empty_output(tmp_path):
    with pytest.raises(RuntimeError, match="empty lexicon"):
        export.write_lexicon_file({}, tmp_path / "lexicon.txt")


def test_write_lexicon_file_uses_sherpa_row_format(tmp_path):
    output = tmp_path / "lexicon.txt"

    export.write_lexicon_file({"牧": (["m", "a"], [6, 6])}, output)

    assert output.read_text(encoding="utf-8") == "牧 m a 6 6\n"


def test_load_lexicon_seed_texts_prefers_last_pipe_field(tmp_path):
    seed_file = tmp_path / "train.list"
    seed_file.write_text(
        "wav/path.wav|Kurisu|こんにちは\n# ignored\nplain text\n",
        encoding="utf-8",
    )

    assert export.load_lexicon_seed_texts([seed_file]) == ["こんにちは", "plain text"]


def install_fake_direct_runtime(monkeypatch):
    captured = {}

    class Meta:
        def __init__(self, key, value):
            self.key = key
            self.value = value

    class Model:
        metadata_props = [
            Meta("sample_rate", "44100"),
            Meta("add_blank", "1"),
            Meta("lang_id", "1"),
            Meta("tone_start", "6"),
        ]

    fake_onnx = types.ModuleType("onnx")
    fake_onnx.__spec__ = ModuleSpec("onnx", loader=None)
    fake_onnx.load = lambda path: Model()

    class SessionOptions:
        def __init__(self):
            self.intra_op_num_threads = None

    class InferenceSession:
        def __init__(self, model_path, sess_options, providers):
            captured["model_path"] = model_path
            captured["threads"] = sess_options.intra_op_num_threads
            captured["providers"] = providers

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.__spec__ = ModuleSpec("onnxruntime", loader=None)
    fake_ort.SessionOptions = SessionOptions
    fake_ort.InferenceSession = InferenceSession

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__spec__ = ModuleSpec("transformers", loader=None)
    fake_transformers.AutoTokenizer = object

    fake_melo = types.ModuleType("melo")
    fake_melo_text = types.ModuleType("melo.text")
    fake_melo_text.language_id_map = {"JP": 1}
    fake_melo_text.language_tone_start_map = {"JP": 6}
    fake_cleaner = types.ModuleType("melo.text.cleaner")
    fake_cleaner.clean_text = lambda text, language: (text, ["m", "a"], [0, 0], [2])

    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "melo", fake_melo)
    monkeypatch.setitem(sys.modules, "melo.text", fake_melo_text)
    monkeypatch.setitem(sys.modules, "melo.text.cleaner", fake_cleaner)
    return captured


def test_provider_loads_direct_onnx_runtime(monkeypatch, tmp_path):
    captured = install_fake_direct_runtime(monkeypatch)
    model = tmp_path / "model.onnx"
    tokens = tmp_path / "tokens.txt"
    model.write_bytes(b"onnx")
    tokens.write_text("_ 0\nm 1\na 2\n", encoding="utf-8")

    from amadeus.providers.tts.melotts_onnx_provider import MeloTTSONNXProvider

    provider = MeloTTSONNXProvider(str(model), str(tokens))

    assert provider.sample_rate == 44100
    phones, tones = provider._encode("牧")
    assert phones.tolist() == [[0, 1, 0, 2, 0]]
    assert tones.tolist() == [[0, 6, 0, 6, 0]]
    assert captured["model_path"] == str(model)
    assert captured["providers"] == ["CPUExecutionProvider"]


def test_provider_requires_tokens_file(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx")

    from amadeus.providers.tts.melotts_onnx_provider import MeloTTSONNXProvider

    with pytest.raises(FileNotFoundError, match="Tokens file"):
        MeloTTSONNXProvider(str(model), str(tmp_path / "missing-tokens.txt"))
