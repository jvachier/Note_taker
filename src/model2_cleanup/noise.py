"""Synthetic word errors for Model 2 inputs (PLAN.md section 4, step 4).

Per word, independently:
    substitution (p_sub): the input form is replaced by a random word from the frequency list;
                          labels stay (they describe the position, not the word),
    deletion (p_del):     the word is removed; a non-``none`` punctuation label moves to the
                          previous word. If the deleted word started a sentence, the next word
                          takes over its ``capitalised`` label (only if it was ``lower``),
    insertion (p_ins):    a random word is inserted before the word, labelled lower / none.
Substitution and deletion are exclusive; insertion is drawn separately.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from src.model2_cleanup.labels import CAPITALISED, LOWER, NONE, SENTENCE_END, Word


@dataclass(frozen=True)
class NoiseConfig:
    """Per-word probabilities; their sum is the "total noise" level."""

    substitution: float
    deletion: float
    insertion: float

    @property
    def total(self) -> float:
        return self.substitution + self.deletion + self.insertion

    def scaled(self, total: float) -> NoiseConfig:
        """Same ratios, rescaled to a new total (for the noise ablation)."""
        if self.total == 0:
            raise ValueError("Cannot rescale a zero noise config.")
        f = total / self.total
        return NoiseConfig(self.substitution * f, self.deletion * f, self.insertion * f)


def apply_noise(
    words: Sequence[Word], noise: NoiseConfig, vocab: Sequence[str], rng: np.random.Generator
) -> list[Word]:
    """Return a noisy copy of ``words`` with labels kept aligned (see module docstring)."""
    if noise.total == 0:
        return list(words)
    out: list[Word] = []
    pending_capital = False  # a deleted sentence-initial word hands its capital to the next
    sentence_start = True
    n = len(words)
    draws = rng.random((n, 2))
    picks = rng.integers(0, len(vocab), size=(n, 2))
    for i, word in enumerate(words):
        starts_sentence = sentence_start
        sentence_start = word.punct in SENTENCE_END
        if draws[i, 1] < noise.insertion:
            out.append(Word(vocab[picks[i, 1]], LOWER, NONE))
        u = draws[i, 0]
        if u < noise.substitution:
            word = replace(word, form=vocab[picks[i, 0]])
        elif u < noise.substitution + noise.deletion:
            if word.punct != NONE and out:
                out[-1] = replace(out[-1], punct=word.punct)
            if starts_sentence and word.casing == CAPITALISED:
                pending_capital = True
            if word.punct in SENTENCE_END:
                pending_capital = False
            continue
        if pending_capital:
            if word.casing == LOWER:
                word = replace(word, casing=CAPITALISED)
            pending_capital = False
        out.append(word)
    return out
