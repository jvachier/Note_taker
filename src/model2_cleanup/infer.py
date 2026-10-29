"""Model 2 inference on raw text: noisy lowercase text in, punctuated and cased text out.

The input is split on whitespace and every token is reduced to its model input form (as in
training, :func:`~src.model2_cleanup.labels.label_word`): lowercase, punctuation removed.
Words are cut into windows of ``window_words`` and tokenised like the training windows
(``[CLS] subwords [SEP]``). The prediction at each word's first subword gives its casing and
trailing punctuation. Words cut off by ``max_len`` keep their input form.

Training windows always end at a sentence end, so the model tends to put a period on the last
word of a window. Longer texts therefore use overlapping windows (stride ``window_words // 2``)
and every word takes its prediction from the window where it is farthest from both edges.
"""

from __future__ import annotations

from collections.abc import Sequence

import keras
import numpy as np
from tokenizers import Tokenizer

import src.model2_cleanup.model  # noqa: F401  (registers the custom layers for loading)
from src.model2_cleanup.labels import CAPITALISED, COMMA, PERIOD, QUESTION, UPPER, label_word
from src.tokenizer.tokenizer import CLS_ID, PAD_ID, SEP_ID

_PUNCT_TEXT = {COMMA: ",", PERIOD: ".", QUESTION: "?"}


def input_words(text: str) -> list[str]:
    """Model input forms of the words of ``text`` (tokens without letters or digits dropped)."""
    return [w.form for w in (label_word(t) for t in text.split()) if w is not None]


def encode_words(
    tokenizer: Tokenizer, windows: Sequence[Sequence[str]], max_len: int
) -> tuple[np.ndarray, list[list[int | None]]]:
    """Token IDs ``(n, max_len)`` and, per window, the first-subword position of each word."""
    token_ids = np.full((len(windows), max_len), PAD_ID, dtype=np.int32)
    positions: list[list[int | None]] = []
    encodings = tokenizer.encode_batch(
        [list(w) for w in windows], is_pretokenized=True, add_special_tokens=False
    )
    budget = max_len - 2
    for row, (window, enc) in enumerate(zip(windows, encodings, strict=True)):
        ids = enc.ids[:budget]
        token_ids[row, 0] = CLS_ID
        token_ids[row, 1 : 1 + len(ids)] = ids
        token_ids[row, 1 + len(ids)] = SEP_ID
        first: list[int | None] = [None] * len(window)
        for pos, w in enumerate(enc.word_ids[:budget]):
            if w is not None and first[w] is None:
                first[w] = pos + 1
        positions.append(first)
    return token_ids, positions


def render_word(form: str, casing: int, punct: int) -> str:
    """``form`` with its predicted casing and trailing punctuation applied."""
    if casing == UPPER:
        form = form.upper()
    elif casing == CAPITALISED:
        form = form[:1].upper() + form[1:]
    return form + _PUNCT_TEXT.get(punct, "")


def clean_text(
    model: keras.Model,
    tokenizer: Tokenizer,
    text: str,
    window_words: int,
    max_len: int,
) -> str:
    """Restore punctuation and casing in ``text`` (see module docstring)."""
    words = input_words(text)
    if not words:
        return ""
    starts = window_starts(len(words), window_words)
    windows = [words[s : s + window_words] for s in starts]
    token_ids, positions = encode_words(tokenizer, windows, max_len)
    out = model.predict_on_batch(token_ids)
    casing = np.argmax(np.asarray(out["casing"]), axis=-1)
    punct = np.argmax(np.asarray(out["punct"]), axis=-1)
    result = []
    for j, form in enumerate(words):
        row = max(
            (r for r, s in enumerate(starts) if s <= j < s + len(windows[r])),
            key=lambda r: min(j - starts[r], starts[r] + len(windows[r]) - 1 - j),
        )
        pos = positions[row][j - starts[row]]
        if pos is None:
            result.append(form)
        else:
            result.append(render_word(form, int(casing[row, pos]), int(punct[row, pos])))
    return " ".join(result)


def window_starts(n_words: int, window_words: int) -> list[int]:
    """Start of each window: one window if it fits, else stride ``window_words // 2``."""
    if n_words <= window_words:
        return [0]
    stride = max(1, window_words // 2)
    starts = list(range(0, n_words - window_words, stride))
    return starts + [n_words - window_words]
