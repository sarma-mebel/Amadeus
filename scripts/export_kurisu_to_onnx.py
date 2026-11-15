#!/usr/bin/env python3
"""Export custom Kurisu voice model to sherpa-onnx compatible artifacts."""

from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple
import argparse
import os
import sys
import types
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DEFAULT_LEXICON_TEXTS = (
    "こんにちは、私は牧瀬紅莉栖です。",
    "助手じゃないと言っているでしょう。",
    "エル・プサイ・コングルゥ。",
    "な、何を言っているのよ。",
    "それは科学的に興味深いわね。",
    "実験を始めましょう。",
    "未来ガジェット研究所へようこそ。",
    "ちゃんと説明しなさい。",
)

KANA_LEXICON_SEED = (
    "あいうえおかきくけこさしすせそたちつてと"
    "なにぬねのはひふへほまみむめもやゆよ"
    "らりるれろわをんがぎぐげござじずぜぞ"
    "だぢづでどばびぶべぼぱぴぷぺぽ"
    "ぁぃぅぇぉゃゅょっー"
    "アイウエオカキクケコサシスセソタチツテト"
    "ナニヌネノハヒフヘホマミムメモヤユヨ"
    "ラリルレロワヲンガギグゲゴザジズゼゾ"
    "ダヂヅデドバビブベボパピプペポ"
    "ァィゥェォャュョッ"
)

JAPANESE_PUNCTUATION_ALIASES = {
    "、": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "・": ",",
    "…": "…",
}

CleanTextFn = Callable[[str, str], Tuple[str, List[str], List[int], List[int]]]


def add_meta_data(filename: str, meta_data: Dict[str, Any]) -> None:
    """Add metadata to an ONNX model in-place."""

    import onnx

    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


def write_tokens_file(symbols: Sequence[str], output_path: Path) -> None:
    """Write sherpa-onnx token ids from the Melo symbol table."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, symbol in enumerate(symbols):
            handle.write(f"{symbol} {idx}\n")


def _dedupe_texts(texts: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for text in texts:
        stripped = text.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        result.append(stripped)
    return result


def load_lexicon_seed_texts(paths: Sequence[Path]) -> List[str]:
    """Load optional Japanese seed text used to build phrase/kanji lexicon rows."""

    texts: List[str] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Training lists are often pipe-delimited. Prefer the last
                # field because it is commonly the transcript.
                parts = [part.strip() for part in line.split("|") if part.strip()]
                texts.append(parts[-1] if parts else line)
    return _dedupe_texts(texts)


def _entry_for_text(
    surface: str,
    *,
    clean_text: CleanTextFn,
    language: str,
    symbols: set[str],
    tone_start: int,
) -> Tuple[str, List[str], List[int]] | None:
    """Convert one surface form into a sherpa lexicon row."""

    if not surface.strip():
        return None

    if surface in JAPANESE_PUNCTUATION_ALIASES:
        phone = JAPANESE_PUNCTUATION_ALIASES[surface]
        if phone in symbols:
            return surface, [phone], [0]
        return None

    try:
        _norm_text, phones, tones, _word2ph = clean_text(surface, language)
    except Exception as exc:
        print(f"Skipping lexicon entry for {surface!r}: {exc}")
        return None

    if not phones or len(phones) != len(tones):
        return None
    missing = [phone for phone in phones if phone not in symbols]
    if missing:
        print(f"Skipping lexicon entry for {surface!r}; unknown phones: {sorted(set(missing))}")
        return None
    return surface, phones, [int(tone) + tone_start for tone in tones]


def build_japanese_lexicon_entries(
    *,
    texts: Iterable[str],
    symbols: Sequence[str],
    clean_text: CleanTextFn,
    tone_start: int,
    language: str = "JP",
) -> Dict[str, Tuple[List[str], List[int]]]:
    """Build phrase and character entries for sherpa's lexicon-based frontend."""

    symbol_set = set(symbols)
    entries: Dict[str, Tuple[List[str], List[int]]] = {}
    seed_texts = _dedupe_texts([*DEFAULT_LEXICON_TEXTS, KANA_LEXICON_SEED, *texts])

    def add(surface: str) -> None:
        entry = _entry_for_text(
            surface,
            clean_text=clean_text,
            language=language,
            symbols=symbol_set,
            tone_start=tone_start,
        )
        if entry is None:
            return
        key, phones, tones = entry
        entries.setdefault(key, (phones, tones))

    for text in seed_texts:
        add(text)
        try:
            norm_text, _phones, _tones, _word2ph = clean_text(text, language)
        except Exception:
            norm_text = ""
        if norm_text and norm_text != text:
            add(norm_text)
        for char in text:
            add(char)
        for char in norm_text:
            add(char)

    for surface in JAPANESE_PUNCTUATION_ALIASES:
        add(surface)

    return entries


def write_lexicon_file(
    entries: Dict[str, Tuple[List[str], List[int]]],
    output_path: Path,
) -> None:
    """Write sherpa-onnx Melo lexicon rows: surface phones... tones...."""

    if not entries:
        raise RuntimeError("Refusing to write an empty lexicon; check Japanese tokenizer setup.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for surface in sorted(entries):
            phones, tones = entries[surface]
            handle.write(
                f"{surface} {' '.join(phones)} {' '.join(str(tone) for tone in tones)}\n"
            )


class _OfflineTokenizer:
    """Small tokenizer stub for Melo's import-time Japanese BERT tokenizer."""

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def tokenize(self, text: str) -> List[str]:
        return list(text)


def _prepare_melo_import_environment() -> None:
    """Avoid export-time network/dictionary failures that are irrelevant here.

    The ONNX wrapper feeds zero BERT tensors, so Melo's Japanese BERT tokenizer
    is not needed for export. Melo also prefers the full ``unidic`` package
    when installed; if its dictionary was not downloaded, point MeCab at the
    already bundled ``unidic_lite`` dictionary instead.
    """

    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

    try:
        import unidic
        import unidic_lite

        if not Path(getattr(unidic, "DICDIR", "")).exists():
            unidic.DICDIR = unidic_lite.DICDIR
            sys.modules["unidic"] = unidic
    except ImportError:
        try:
            import unidic_lite

            stub = types.ModuleType("unidic")
            stub.DICDIR = unidic_lite.DICDIR
            sys.modules["unidic"] = stub
        except ImportError:
            pass

    try:
        import transformers

        transformers.AutoTokenizer = _OfflineTokenizer
    except ImportError:
        pass

    if "g2p_en" not in sys.modules:
        g2p_stub = types.ModuleType("g2p_en")

        class _OfflineG2p:
            def __call__(self, text: str) -> List[str]:
                return list(text)

        g2p_stub.G2p = _OfflineG2p
        sys.modules["g2p_en"] = g2p_stub


def _import_export_dependencies():
    try:
        _prepare_melo_import_environment()
        import torch
        from melo.api import TTS
        from melo.text import language_id_map, language_tone_start_map
        from melo.text.cleaner import clean_text
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export requires the MeloTTS export toolchain: torch, onnx, "
            "MeloTTS/melo, and Japanese tokenizer dependencies such as "
            "mecab-python3, unidic-lite, pykakasi, and transformers. Use a "
            "Python 3.10/3.11 environment; the current runtime is only suitable "
            "for lightweight tests."
        ) from exc

    return torch, TTS, language_id_map, language_tone_start_map, clean_text


def _build_model_wrapper(torch_module, language_id_map):
    class ModelWrapper(torch_module.nn.Module):
        def __init__(self, model: "SynthesizerTrn"):
            super().__init__()
            self.model = model
            self.lang_id = language_id_map[model.language]

        def forward(
            self,
            x,
            x_lengths,
            tones,
            sid,
            noise_scale,
            length_scale,
            noise_scale_w,
            max_len=None,
        ):
            bert = torch_module.zeros(x.shape[0], 1024, x.shape[1], dtype=torch_module.float32)
            ja_bert = torch_module.zeros(x.shape[0], 768, x.shape[1], dtype=torch_module.float32)
            lang_id = torch_module.zeros_like(x)
            lang_id[:, 1::2] = self.lang_id
            return self.model.model.infer(
                x=x,
                x_lengths=x_lengths,
                sid=sid,
                tone=tones,
                language=lang_id,
                bert=bert,
                ja_bert=ja_bert,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0]

    return ModelWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "models" / "kurisu_voice" / "config.json",
        help="MeloTTS config.json path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root / "models" / "kurisu_voice" / "G_55000.pth",
        help="MeloTTS generator checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "models" / "kurisu_voice_onnx",
        help="Directory for model.onnx, tokens.txt, and lexicon.txt.",
    )
    parser.add_argument(
        "--lexicon-text",
        action="append",
        default=[],
        help="Extra Japanese text to seed the lexicon. Can be passed multiple times.",
    )
    parser.add_argument(
        "--lexicon-text-file",
        action="append",
        type=Path,
        default=[],
        help="UTF-8 text/training-list file with Japanese lines to seed the lexicon.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch, TTS, language_id_map, language_tone_start_map, clean_text = _import_export_dependencies()
    ModelWrapper = _build_model_wrapper(torch, language_id_map)

    config_path = args.config
    checkpoint_path = args.checkpoint
    output_dir = args.output_dir

    print(f"Loading custom Kurisu model from {checkpoint_path}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    language = "JP"
    model = TTS(
        language=language,
        device="cpu",
        use_hf=False,
        config_path=str(config_path),
        ckpt_path=str(checkpoint_path),
    )

    print(f"Model loaded. Language: {language}")
    print(f"Sample rate: {model.hps.data.sampling_rate}")
    print(f"Speakers: {model.hps.data.spk2id}")

    tokens_file = output_dir / "tokens.txt"
    write_tokens_file(model.hps["symbols"], tokens_file)
    print(f"Generated tokens file: {tokens_file}")

    seed_texts = [
        *load_lexicon_seed_texts(args.lexicon_text_file),
        *args.lexicon_text,
    ]
    lexicon_file = output_dir / "lexicon.txt"
    lexicon_entries = build_japanese_lexicon_entries(
        texts=seed_texts,
        symbols=model.hps["symbols"],
        clean_text=clean_text,
        tone_start=language_tone_start_map[language],
        language=language,
    )
    write_lexicon_file(lexicon_entries, lexicon_file)
    print(f"Generated lexicon file: {lexicon_file} ({len(lexicon_entries)} entries)")

    torch_model = ModelWrapper(model)

    opset_version = 18
    x = torch.randint(low=0, high=10, size=(60,), dtype=torch.int64)
    x_lengths = torch.tensor([x.size(0)], dtype=torch.int64)
    sid = torch.tensor([list(model.hps.data.spk2id.values())[0]], dtype=torch.int64)
    tones = torch.zeros_like(x)

    noise_scale = torch.tensor([1.0], dtype=torch.float32)
    length_scale = torch.tensor([1.0], dtype=torch.float32)
    noise_scale_w = torch.tensor([1.0], dtype=torch.float32)

    x = x.unsqueeze(0)
    tones = tones.unsqueeze(0)

    onnx_file = output_dir / "model.onnx"
    print(f"Exporting to ONNX: {onnx_file}...")

    torch.onnx.export(
        torch_model,
        (
            x,
            x_lengths,
            tones,
            sid,
            noise_scale,
            length_scale,
            noise_scale_w,
        ),
        str(onnx_file),
        opset_version=opset_version,
        input_names=[
            "x",
            "x_lengths",
            "tones",
            "sid",
            "noise_scale",
            "length_scale",
            "noise_scale_w",
        ],
        output_names=["y"],
        dynamic_axes={
            "x": {0: "N", 1: "L"},
            "x_lengths": {0: "N"},
            "tones": {0: "N", 1: "L"},
            "y": {0: "N", 1: "S", 2: "T"},
        },
        dynamo=False,
    )

    meta_data = {
        "model_type": "melo-vits",
        "comment": f"Custom Kurisu voice ({checkpoint_path.name})",
        "version": 2,
        "language": "Japanese",
        "add_blank": int(model.hps.data.add_blank),
        "n_speakers": len(model.hps.data.spk2id),
        "jieba": 1,
        "sample_rate": model.hps.data.sampling_rate,
        "bert_dim": 1024,
        "ja_bert_dim": 768,
        "speaker_id": list(model.hps.data.spk2id.values())[0],
        "lang_id": language_id_map[language],
        "tone_start": language_tone_start_map[language],
        "url": "https://github.com/myshell-ai/MeloTTS",
        "license": "MIT license",
        "description": "Custom trained Kurisu voice from Steins;Gate",
    }
    add_meta_data(str(onnx_file), meta_data)

    print("\nONNX export complete!")
    print(f"  Model: {onnx_file}")
    print(f"  Tokens: {tokens_file}")
    print(f"  Lexicon: {lexicon_file}")
    print(f"  Sample rate: {model.hps.data.sampling_rate} Hz")
    print(f"  Language: {language}")


if __name__ == "__main__":
    main()
