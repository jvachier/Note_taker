"""Pure text helpers of the app: dialogue parsing and the cleanup diff (no model, no Dash)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.model2_cleanup.infer import input_words

MAX_SPEAKERS = 8  # the speaker embedding of Model 1 (more speakers share the last ID in v3)
# Longest input the demo accepts (about 3,000 words, a one-hour meeting is ~9,000): keeps one
# request to a few seconds of CPU so it cannot block the single server worker.
MAX_CHARS = 20_000
_SPEAKER = re.compile(r"^\s*([^:]{1,40}?)\s*:\s*(.+)$")


def check_length(text: str) -> None:
    """Raise ``ValueError`` if ``text`` is longer than :data:`MAX_CHARS`."""
    if len(text) > MAX_CHARS:
        raise ValueError(
            f"The text is {len(text):,} characters long; the demo accepts up to {MAX_CHARS:,}."
        )


@dataclass
class Dialogue:
    """A parsed dialogue: speaker IDs (1..8), their names and the utterances."""

    speakers: list[int]
    names: list[str]
    utterances: list[str]

    def name(self, i: int) -> str:
        """Name of the speaker of utterance ``i``."""
        return self.names[self.speakers[i] - 1]


def parse_dialogue(text: str) -> Dialogue:
    """One utterance per line, ``Speaker: text``.

    Speakers get IDs 1, 2, ... in order of first appearance. A line without a ``name:`` prefix
    continues the previous speaker's turn as a new utterance (or speaker 1 at the start).

    Raises:
        ValueError: no utterance, more than :data:`MAX_SPEAKERS` speakers or more than
            :data:`MAX_CHARS` characters.
    """
    check_length(text)
    names: list[str] = []
    speakers: list[int] = []
    utterances: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = _SPEAKER.match(line)
        if match:
            name, utterance = match.group(1).strip("# "), match.group(2).strip()
            if name not in names:
                names.append(name)
            speaker = names.index(name) + 1
        else:
            if not names:
                names.append("Speaker 1")
            speaker, utterance = speakers[-1] if speakers else 1, line.strip()
        speakers.append(speaker)
        utterances.append(utterance)
    if not utterances:
        raise ValueError("Write at least one line, for example `Anna: Can we meet on Monday?`")
    if len(names) > MAX_SPEAKERS:
        raise ValueError(f"At most {MAX_SPEAKERS} speakers are supported, found {len(names)}.")
    return Dialogue(speakers, names, utterances)


# A new sentence starts with a capital, digit or quote: "the rent. increase, too." (a period
# Model 2 put inside a phrase) stays one sentence.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str) -> list[str]:
    """Split a turn into sentences after ``.``, ``!`` or ``?`` (Model 1 v3 scores sentences)."""
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def meeting_sentences(dialogue: Dialogue) -> tuple[list[str], list[str]]:
    """Speaker name and text of every sentence of ``dialogue``, in order."""
    speakers, sentences = [], []
    for i, utterance in enumerate(dialogue.utterances):
        for sentence in split_sentences(utterance) or [utterance]:
            speakers.append(dialogue.name(i))
            sentences.append(sentence)
    return speakers, sentences


def changed_words(text: str, cleaned: str) -> list[tuple[str, bool]]:
    """Words of ``cleaned`` with ``True`` where the model changed casing or punctuation.

    ``cleaned`` has one word per model input word (see :func:`input_words`), in order.
    """
    forms = input_words(text)
    words = cleaned.split()
    if len(words) != len(forms):  # never expected; fall back to no highlighting
        return [(w, False) for w in words]
    return [(w, w != f) for w, f in zip(words, forms, strict=True)]
