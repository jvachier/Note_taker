"""Model 2 example builder: text units -> labelled windows -> noisy token arrays.

A *unit* is a WikiText article or a DialogSum dialogue. Its words are labelled
(:mod:`labels`), split into sentences and packed into windows of whole sentences. Noise is
applied per window, then windows are tokenised as ``[CLS] subwords [SEP]`` and padded or
truncated to ``max_len``. Labels sit on the first subword of each word; every other position
holds ``IGNORE``. Words whose first subword falls beyond the truncation point are lost and
counted.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from src.data.io import read_jsonl, read_lines
from src.data.wikitext import iter_articles
from src.layers.losses import IGNORE
from src.model2_cleanup.labels import Word, label_text, pack_windows, split_sentences
from src.model2_cleanup.noise import NoiseConfig, apply_noise
from src.tokenizer.tokenizer import CLS_ID, PAD_ID, SEP_ID


def iter_units(raw_dir: str | Path, source: str, max_units: int | None = None) -> Iterator[str]:
    """Yield the text of each unit of ``source`` (``wikitext_<split>`` / ``dialogsum_<split>``).

    A WikiText unit is an article (paragraphs joined by spaces); a DialogSum unit is a
    dialogue (utterances joined by spaces, speaker tags already removed).
    """
    dataset, _, split = source.partition("_")
    raw_dir = Path(raw_dir)
    if dataset == "wikitext":
        units: Iterable[str] = (
            " ".join(p) for p in iter_articles(read_lines(raw_dir / "wikitext" / f"{split}.txt"))
        )
    elif dataset == "dialogsum":
        units = (
            " ".join(r["utterances"]) for r in read_jsonl(raw_dir / "dialogsum" / f"{split}.jsonl")
        )
    else:
        raise ValueError(f"Unknown source: {source!r}")
    for i, unit in enumerate(units):
        if max_units is not None and i >= max_units:
            return
        yield unit


def unit_windows(text: str, window_words: int) -> list[list[Word]]:
    """Label a unit and pack it into windows of whole sentences."""
    return pack_windows(split_sentences(label_text(text)), window_words)


def word_frequencies(texts: Iterable[str]) -> Counter[str]:
    """Counts of word input forms (used for the substitution/insertion vocabulary)."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(w.form for w in label_text(text))
    return counts


@dataclass
class EncodedWindows:
    """Arrays for a set of windows: ``token_ids`` int32, ``casing`` / ``punct`` int8."""

    token_ids: np.ndarray
    casing: np.ndarray
    punct: np.ndarray
    truncated_words: int = 0
    total_words: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def encode_windows(
    tokenizer: Tokenizer, windows: Sequence[Sequence[Word]], max_len: int
) -> EncodedWindows:
    """Tokenise windows into fixed-length arrays with first-subword labels."""
    n = len(windows)
    token_ids = np.full((n, max_len), PAD_ID, dtype=np.int32)
    casing = np.full((n, max_len), IGNORE, dtype=np.int8)
    punct = np.full((n, max_len), IGNORE, dtype=np.int8)
    if n == 0:
        return EncodedWindows(token_ids, casing, punct)
    encodings = tokenizer.encode_batch(
        [[w.form for w in window] for window in windows],
        is_pretokenized=True,
        add_special_tokens=False,
    )
    budget = max_len - 2
    truncated = total = 0
    for row, (window, enc) in enumerate(zip(windows, encodings, strict=True)):
        ids = enc.ids[:budget]
        token_ids[row, 0] = CLS_ID
        token_ids[row, 1 : 1 + len(ids)] = ids
        token_ids[row, 1 + len(ids)] = SEP_ID
        seen = set()
        for pos, w in enumerate(enc.word_ids[:budget]):
            if w is not None and w not in seen:
                seen.add(w)
                casing[row, pos + 1] = window[w].casing
                punct[row, pos + 1] = window[w].punct
        total += len(window)
        truncated += len(window) - len(seen)
    return EncodedWindows(token_ids, casing, punct, truncated, total)


def build_split(
    tokenizer: Tokenizer,
    units: Iterable[tuple[str, int]],
    *,
    window_words: int,
    max_len: int,
    noise: NoiseConfig,
    vocab: Sequence[str],
    rng: np.random.Generator,
    chunk_windows: int = 20000,
) -> EncodedWindows:
    """Build arrays from ``(unit_text, copies)`` pairs; each copy gets its own noise draw."""
    chunks: list[EncodedWindows] = []
    pending: list[list[Word]] = []

    def flush() -> None:
        chunks.append(encode_windows(tokenizer, pending, max_len))
        pending.clear()

    for text, copies in units:
        windows = unit_windows(text, window_words)
        for _ in range(copies):
            pending.extend(apply_noise(w, noise, vocab, rng) for w in windows)
        if len(pending) >= chunk_windows:
            flush()
    flush()
    return EncodedWindows(
        token_ids=np.concatenate([c.token_ids for c in chunks]),
        casing=np.concatenate([c.casing for c in chunks]),
        punct=np.concatenate([c.punct for c in chunks]),
        truncated_words=sum(c.truncated_words for c in chunks),
        total_words=sum(c.total_words for c in chunks),
    )


def label_counts(labels: np.ndarray, num_classes: int) -> list[int]:
    """Number of labelled positions per class (``IGNORE`` excluded)."""
    valid = labels[labels != IGNORE].astype(np.int64)
    return np.bincount(valid, minlength=num_classes).tolist()
