"""Tests for Model 2 example encoding (first-subword labels, truncation, padding)."""

import numpy as np

from src.layers.losses import IGNORE
from src.model2_cleanup.examples import encode_windows, label_counts, unit_windows
from src.model2_cleanup.labels import label_text
from src.tokenizer.tokenizer import CLS_ID, PAD_ID, SEP_ID, load_tokenizer


def test_labels_sit_on_first_subwords(tiny_assets):
    tok = load_tokenizer(tiny_assets["tokenizer"])
    words = label_text("Unbelievably, NASA launched it. Why?")
    enc = encode_windows(tok, [words], max_len=32)
    ids, casing, punct = enc.token_ids[0], enc.casing[0], enc.punct[0]
    pieces = tok.encode([w.form for w in words], is_pretokenized=True, add_special_tokens=False)
    assert ids[0] == CLS_ID and ids[len(pieces.ids) + 1] == SEP_ID
    assert (ids[len(pieces.ids) + 2 :] == PAD_ID).all()
    labelled = np.flatnonzero(punct != IGNORE)
    assert len(labelled) == len(words)
    for pos, word in zip(labelled, words, strict=True):
        assert pieces.word_ids[pos - 1] == words.index(word)
        assert (casing[pos], punct[pos]) == (word.casing, word.punct)
    others = np.setdiff1d(np.arange(32), labelled)
    assert (punct[others] == IGNORE).all() and (casing[others] == IGNORE).all()
    assert enc.truncated_words == 0 and enc.total_words == len(words)


def test_truncation_drops_and_counts_words(tiny_assets):
    tok = load_tokenizer(tiny_assets["tokenizer"])
    words = label_text(" ".join(["fox"] * 20))
    enc = encode_windows(tok, [words], max_len=8)
    assert enc.token_ids.shape == (1, 8)
    assert enc.token_ids[0, -1] == SEP_ID
    assert (enc.punct[0] != IGNORE).sum() == 6
    assert enc.truncated_words == 14


def test_unit_windows_and_counts():
    windows = unit_windows("One two. Three four five. Six.", window_words=4)
    assert [len(w) for w in windows] == [2, 4]
    assert label_counts(np.array([[IGNORE, 0, 2], [2, IGNORE, 3]]), 4) == [1, 0, 2, 1]
