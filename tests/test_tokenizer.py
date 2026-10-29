"""Tests for the WordPiece tokenizer: special IDs, round trip, first-subword alignment."""

import json

import pytest

from src.config import deep_merge, load_config
from src.data.dialogsum import Dialogue
from src.data.io import write_jsonl, write_lines
from src.tokenizer.corpus import iter_corpus
from src.tokenizer.tokenizer import (
    CLS_ID,
    MASK_ID,
    PAD_ID,
    SEP_ID,
    SPECIAL_TOKENS,
    UNK_ID,
    check_special_tokens,
    encode_words,
    load_tokenizer,
)
from src.tokenizer.train import main, tokenizer_report, train_tokenizer

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello, Mr. Smith. I'm Doctor Hawkins. Why are you here today?",
    "I don't know; maybe we should go to the café tomorrow!",
    "Unbelievable results were reported by the researchers in 2011.",
] * 50


@pytest.fixture(scope="module")
def tokenizer(request):
    from pathlib import Path

    cfg = load_config(Path(request.config.rootpath) / "configs" / "tokenizer.yaml")
    cfg = deep_merge(cfg, {"vocab_size": 300, "min_frequency": 1})
    return train_tokenizer(lambda: CORPUS, cfg)


def test_special_token_ids_fixed(tokenizer):
    assert [tokenizer.token_to_id(t) for t in SPECIAL_TOKENS] == list(range(13))
    assert (PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID) == (0, 1, 2, 3, 4)
    assert [tokenizer.token_to_id(f"[S{s}]") for s in (1, 8)] == [5, 12]


def test_config_special_tokens_match_code(configs_dir):
    assert load_config(configs_dir / "tokenizer.yaml")["special_tokens"] == SPECIAL_TOKENS


def test_speaker_tokens_survive_lowercasing(tokenizer):
    enc = tokenizer.encode("[S1] hello [S8] there", add_special_tokens=False)
    assert enc.ids[0] == tokenizer.token_to_id("[S1]")
    assert tokenizer.token_to_id("[S8]") in enc.ids


def test_round_trip_lowercase_text(tokenizer):
    text = "the quick brown fox jumps over the lazy dog unbelievable results"
    assert tokenizer.decode(tokenizer.encode(text).ids) == text


def test_normalisation_lowercases_and_strips_accents(tokenizer):
    assert tokenizer.encode("CAFÉ").tokens == tokenizer.encode("cafe").tokens


def test_punctuation_is_its_own_token(tokenizer):
    tokens = tokenizer.encode("hello, smith. why?").tokens
    assert [t for t in tokens if t in {",", ".", "?"}] == [",", ".", "?"]


def test_first_subword_alignment(tokenizer):
    words = ["unbelievable", "results", "don't", "zzzzqqq", "fox"]
    enc = encode_words(tokenizer, words)
    assert len(enc.ids) > len(words)  # some words split into several pieces
    for w, pos in enumerate(enc.word_to_token):
        assert pos is not None
        assert enc.word_ids[pos] == w and enc.first_subword[pos]
        assert pos == 0 or enc.word_ids[pos - 1] != w
    assert sum(enc.first_subword) == len(words)
    pieces = [
        t
        for t, w in zip(
            tokenizer.encode(words, is_pretokenized=True).tokens, enc.word_ids, strict=True
        )
        if w == 2
    ]
    assert pieces == ["don", "'", "t"]


def test_encode_words_with_cls_sep(tokenizer):
    enc = encode_words(tokenizer, ["hello", "fox"], add_cls_sep=True)
    assert enc.ids[0] == CLS_ID and enc.ids[-1] == SEP_ID
    assert enc.word_ids[0] is None and enc.word_ids[-1] is None
    assert not enc.first_subword[0] and not enc.first_subword[-1]
    assert enc.word_to_token[0] == 1


def test_encode_words_empty_word(tokenizer):
    enc = encode_words(tokenizer, ["hello", "\x00", "fox"])
    assert enc.word_to_token[1] is None
    assert enc.word_to_token[0] is not None and enc.word_to_token[2] is not None


def test_report_counts(tokenizer):
    rep = tokenizer_report(tokenizer, ["hello , fox", "zzzzqqq"], batch_size=1)
    assert rep["texts"] == 2 and rep["words"] == 4
    assert rep["subwords_per_word"] >= 1.0
    assert 0.0 <= rep["unk_rate"] <= 1.0


def test_train_rejects_wrong_special_tokens(configs_dir):
    cfg = load_config(configs_dir / "tokenizer.yaml")
    cfg["special_tokens"] = ["[UNK]", "[PAD]"]
    with pytest.raises(ValueError):
        train_tokenizer(lambda: CORPUS, cfg)


def test_check_special_tokens_detects_mismatch(tokenizer):
    from tokenizers import Tokenizer

    other = Tokenizer.from_str(tokenizer.to_str())
    other.add_special_tokens(["[EXTRA]"])
    check_special_tokens(other)  # appending keeps existing IDs
    vocab_only = Tokenizer.from_str(tokenizer.to_str().replace('"[PAD]"', '"[PADX]"'))
    with pytest.raises(ValueError):
        check_special_tokens(vocab_only)


def _write_raw(raw_dir):
    write_lines(raw_dir / "wikitext" / "train.txt", ["", " = T = ", *[f" {s} " for s in CORPUS]])
    write_lines(raw_dir / "wikitext" / "test.txt", [" = U = ", " the lazy fox , again . "])
    d = Dialogue("train_0", "train", [1, 2], ["Hi, how are you?", "Fine, thanks."], ["s"], ["t"])
    write_jsonl(raw_dir / "dialogsum" / "train.jsonl", [d.to_dict()] * 20)
    t = Dialogue("test_0", "test", [1, 2], ["Hello there.", "Why?"], ["s"] * 3, ["t"] * 3)
    write_jsonl(raw_dir / "dialogsum" / "test.jsonl", [t.to_dict()])


def test_corpus_sources(tmp_path):
    _write_raw(tmp_path)
    texts = list(iter_corpus(tmp_path, ["wikitext_test", "dialogsum_test"]))
    assert texts == ["the lazy fox, again.", "Hello there.", "Why?"]
    assert len(list(iter_corpus(tmp_path, ["wikitext_train"], 3))) == 3
    with pytest.raises(ValueError):
        list(iter_corpus(tmp_path, ["other_train"]))


def test_main_end_to_end(tmp_path, configs_dir):
    _write_raw(tmp_path / "raw")
    cfg_path = tmp_path / "tok.yaml"
    cfg_path.write_text(
        f"base: {configs_dir / 'tokenizer.yaml'}\nname: t\nvocab_size: 300\nmin_frequency: 1\n"
        f"paths:\n  raw_dir: {tmp_path / 'raw'}\n  output_dir: {tmp_path / 'out'}\n"
        f"  report_path: {tmp_path / 'out' / 'report.json'}\n"
    )
    main(["--config", str(cfg_path)])
    tok = load_tokenizer(tmp_path / "out")
    vocab = (tmp_path / "out" / "vocab.txt").read_text().splitlines()
    assert vocab[:13] == SPECIAL_TOKENS and len(vocab) == tok.get_vocab_size()
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert set(report) == {"vocab_size", "wikitext_test", "dialogsum_test"}
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["stage"] == "tokenizer" and len(manifest["vocab_sha256"]) == 64


def test_canonical_ids_do_not_depend_on_trainer_order(tokenizer):
    """Same token set with shuffled IDs (what the trainer produces) -> identical files."""
    import random

    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece

    from src.tokenizer.train import canonicalize_ids

    tokens = [t for t in tokenizer.get_vocab() if t not in SPECIAL_TOKENS]
    random.Random(0).shuffle(tokens)
    shuffled = Tokenizer.from_str(tokenizer.to_str())
    shuffled.model = WordPiece(
        {t: i for i, t in enumerate([*SPECIAL_TOKENS, *tokens])}, unk_token="[UNK]"
    )
    canonicalize_ids(shuffled, "##")
    assert shuffled.to_str() == tokenizer.to_str()


def test_canonical_ids_keep_encoding(tokenizer):
    vocab = tokenizer.get_vocab()
    rest = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])][len(SPECIAL_TOKENS) :]
    assert rest == sorted(rest)
    assert tokenizer.encode("unbelievable [S2] fox?").tokens[1] != "[UNK]"


def test_compute_alphabet_breaks_ties_by_character():
    from tokenizers.normalizers import BertNormalizer

    from src.tokenizer.train import compute_alphabet

    norm = BertNormalizer(lowercase=True, strip_accents=True)
    texts = ["aaa bb c d", "B É e"]  # counts: a3 b3 c1 d1 e2 (É -> e)
    assert compute_alphabet(texts, norm, 3) == ["a", "b", "e"]
    assert compute_alphabet(texts, norm, 4) == ["a", "b", "c", "e"]  # c beats d on the tie
