"""Tests for Model 2 noise: labels must stay aligned after each operation."""

import numpy as np
import pytest

from src.model2_cleanup.labels import CAPITALISED, COMMA, LOWER, NONE, PERIOD, label_text
from src.model2_cleanup.noise import NoiseConfig, apply_noise

VOCAB = ["zzz"]


class FixedRng:
    """Stand-in generator returning preset draws (column 0: sub/del, column 1: insertion)."""

    def __init__(self, draws):
        self.draws = np.asarray(draws, dtype=float)

    def random(self, shape):
        return self.draws

    def integers(self, low, high, size):
        return np.zeros(size, dtype=int)


def _noise(words, draws, sub=0.0, dele=0.0, ins=0.0):
    return apply_noise(words, NoiseConfig(sub, dele, ins), VOCAB, FixedRng(draws))


def test_zero_noise_is_identity():
    words = label_text("Hello, world. How are you?")
    rng = np.random.default_rng(0)
    assert apply_noise(words, NoiseConfig(0, 0, 0), VOCAB, rng) == words


def test_substitution_keeps_labels():
    words = label_text("Hello, world.")
    out = _noise(words, [[0.0, 1.0], [1.0, 1.0]], sub=0.5)
    assert [w.form for w in out] == ["zzz", "world"]
    assert [(w.casing, w.punct) for w in out] == [(w.casing, w.punct) for w in words]


def test_deletion_moves_punctuation_to_previous_word():
    words = label_text("Hello there, friend. Bye")
    out = _noise(words, [[1, 1], [0.0, 1], [1, 1], [1, 1]], dele=0.5)  # delete "there,"
    assert [(w.form, w.punct) for w in out] == [("hello", COMMA), ("friend", PERIOD), ("bye", NONE)]


def test_deleting_sentence_start_capitalises_next_word():
    words = label_text("Done. The cat sat.")
    out = _noise(words, [[1, 1], [0.0, 1], [1, 1], [1, 1]], dele=0.5)  # delete "The"
    assert [(w.form, w.casing) for w in out] == [
        ("done", CAPITALISED),
        ("cat", CAPITALISED),
        ("sat", LOWER),
    ]


def test_deleting_mid_sentence_capital_does_not_propagate():
    words = label_text("I met Paris friends.")
    out = _noise(words, [[1, 1], [1, 1], [0.0, 1], [1, 1]], dele=0.5)  # delete "Paris"
    assert [(w.form, w.casing) for w in out] == [("i", CAPITALISED), ("met", LOWER),
                                                 ("friends", LOWER)]  # fmt: skip


def test_deleting_first_word_drops_its_punctuation():
    out = _noise(label_text("Yes, sure."), [[0.0, 1], [1, 1]], dele=0.5)
    assert [(w.form, w.casing, w.punct) for w in out] == [("sure", CAPITALISED, PERIOD)]


def test_insertion_adds_lower_none_word_before():
    words = label_text("Hello world.")
    out = _noise(words, [[1, 1], [1, 0.0]], ins=0.5)
    assert [(w.form, w.casing, w.punct) for w in out] == [
        ("hello", CAPITALISED, NONE),
        ("zzz", LOWER, NONE),
        ("world", LOWER, PERIOD),
    ]


def test_random_noise_rates_and_label_conservation():
    words = label_text(" ".join(["Alpha beta, gamma delta."] * 2000))
    noise = NoiseConfig(0.05, 0.03, 0.02)
    out = apply_noise(words, noise, ["zzz"], np.random.default_rng(0))
    n = len(words)
    inserted = sum(w.form == "zzz" and w.punct == NONE and w.casing == LOWER for w in out)
    assert len(out) == pytest.approx(n * (1 - 0.03 + 0.02), rel=0.05)
    assert inserted > 0
    # Sentence ends are only lost in rare cases (a whole sentence deleted, or the first word).
    kept = sum(w.punct == PERIOD for w in out)
    assert 0.995 * 2000 <= kept <= 2000


def test_scaled_keeps_ratios():
    scaled = NoiseConfig(0.05, 0.03, 0.02).scaled(0.2)
    assert (scaled.substitution, scaled.deletion, scaled.insertion) == pytest.approx(
        (0.1, 0.06, 0.04)
    )
    assert scaled.total == pytest.approx(0.2)
