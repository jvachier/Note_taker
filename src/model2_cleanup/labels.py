"""Per-word casing and trailing-punctuation labels from clean text (PLAN.md section 4).

A clean text is split on whitespace. Each whitespace token gives one :class:`Word` with
    ``form``    the model input: lowercase, punctuation removed (apostrophes / hyphens between
                letters or digits and ``.``/``,`` between digits are kept: "don't", "1,000"),
    ``casing``  ``lower`` / ``capitalised`` / ``upper`` (``IGNORE`` for mixed case such as
                "iPhone", and for words without cased letters such as numbers),
    ``punct``   ``none`` / ``comma`` / ``period`` / ``question`` from the trailing punctuation
                (``!`` ``;`` -> period, ``:`` -> comma, everything else dropped).
Tokens whose form is empty (a lone "-" or "&") are dropped. The final period of an
abbreviation ("Mr.", "U.S.", "J.") is part of the word, not a label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.layers.losses import IGNORE

CASING_CLASSES = ("lower", "capitalised", "upper")
PUNCT_CLASSES = ("none", "comma", "period", "question")
LOWER, CAPITALISED, UPPER = 0, 1, 2
NONE, COMMA, PERIOD, QUESTION = 0, 1, 2, 3
SENTENCE_END = {PERIOD, QUESTION}

_PUNCT_MAP = {",": COMMA, ":": COMMA, ".": PERIOD, "!": PERIOD, ";": PERIOD, "?": QUESTION}
_PRIORITY = (QUESTION, PERIOD, COMMA)  # when a word ends with several marks: 'end?",'
ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "inc", "ltd", "co",
        "corp", "no", "gen", "col", "lt", "sgt", "capt", "rev", "jan", "feb", "mar", "apr",
        "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "approx", "dept", "est",
        "fig", "vol",
    }
)  # fmt: skip
_EDGE_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)
_INNER_RE = re.compile(r"(?<=[^\W_])['’\-](?=[^\W_])|(?<=\d)[.,](?=\d)")


@dataclass(frozen=True)
class Word:
    """One labelled word: model input ``form`` plus ``casing`` and ``punct`` labels."""

    form: str
    casing: int
    punct: int


def casing_label(core: str) -> int:
    """Casing class of a word from its cased letters (``IGNORE`` if mixed or caseless)."""
    letters = [c for c in core if c.lower() != c.upper()]
    if not letters:
        return IGNORE
    if all(c.islower() for c in letters):
        return LOWER
    if len(letters) == 1 or letters[0].isupper() and all(c.islower() for c in letters[1:]):
        return CAPITALISED
    if all(c.isupper() for c in letters):
        return UPPER
    return IGNORE


def punct_label(trailing: str) -> int:
    """Punctuation class of a word's trailing characters."""
    marks = {_PUNCT_MAP[c] for c in trailing if c in _PUNCT_MAP}
    return next((p for p in _PRIORITY if p in marks), NONE)


def input_form(core: str) -> str:
    """Lowercase ``core`` and drop punctuation, keeping inner apostrophes/hyphens/digit marks."""
    kept = {m.start() for m in _INNER_RE.finditer(core)}
    return "".join(c for i, c in enumerate(core) if c.isalnum() or i in kept).lower()


def _is_abbreviation(core: str) -> bool:
    bare = core.lower()
    has_letters = any(c.isalpha() for c in core)
    return (
        bare in ABBREVIATIONS
        or (has_letters and "." in core)
        or (len(core) == 1 and core.isupper())
    )


def label_word(token: str) -> Word | None:
    """Label one whitespace token; ``None`` if it has no letters or digits."""
    _, core, trailing = _EDGE_RE.match(token).groups()
    if trailing.startswith(".") and _is_abbreviation(core):
        core, trailing = core + ".", trailing[1:]
    form = input_form(core)
    if not form:
        return None
    return Word(form=form, casing=casing_label(core), punct=punct_label(trailing))


def label_text(text: str) -> list[Word]:
    """Label every whitespace token of ``text``, dropping tokens with an empty form."""
    return [w for w in (label_word(t) for t in text.split()) if w is not None]


def split_sentences(words: list[Word]) -> list[list[Word]]:
    """Split labelled words into sentences ending at a period/question label."""
    sentences: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        current.append(word)
        if word.punct in SENTENCE_END:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def pack_windows(sentences: list[list[Word]], window_words: int) -> list[list[Word]]:
    """Pack consecutive whole sentences into windows of at most ``window_words`` words.

    A sentence longer than ``window_words`` becomes its own window (it is truncated later at
    the token level). No sentence is split across windows.
    """
    windows: list[list[Word]] = []
    current: list[Word] = []
    for sentence in sentences:
        if current and len(current) + len(sentence) > window_words:
            windows.append(current)
            current = []
        current = current + sentence
    if current:
        windows.append(current)
    return windows
