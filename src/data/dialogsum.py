"""DialogSum parsing: speaker turns, and merging the 3 test references into one dialogue.

Raw dialogues look like ``"#Person1#: Hi.\\n#Person2#: Hello."``. Each line starting with a
``#PersonN#:`` tag is one utterance. The tag is removed from the text and kept as the integer
speaker ID ``N`` (1-based, matching the ``[S1]``..``[S8]`` tokens).

The official test split stores each dialogue three times (``test_<n>_1`` .. ``test_<n>_3``),
once per reference summary; :func:`group_test_rows` merges them into one record ``test_<n>``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

# Case-insensitive: the train split contains one "#PErson2#:".
_SPEAKER_RE = re.compile(r"^\s*#person(\d+)#\s*:\s*", re.IGNORECASE)
_TEST_ID_RE = re.compile(r"^(test_\d+)_(\d+)$")


@dataclass
class Dialogue:
    """One DialogSum dialogue with parallel ``speakers`` / ``utterances`` lists."""

    id: str
    split: str
    speakers: list[int]
    utterances: list[str]
    summaries: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dialogue:
        """Inverse of :meth:`to_dict`."""
        return cls(**data)


def count_empty_turns(text: str) -> int:
    """Number of speaker tags followed by no text (dropped by :func:`parse_dialogue`)."""
    return sum(
        1
        for line in text.replace("\r\n", "\n").split("\n")
        if (m := _SPEAKER_RE.match(line)) and not line[m.end() :].strip()
    )


def parse_dialogue(text: str, max_speakers: int = 8) -> tuple[list[int], list[str]]:
    """Split a raw dialogue into speaker IDs and utterance texts.

    Lines without a speaker tag are appended to the previous utterance. Blank lines and
    empty turns (a tag with no text; 6 occurrences in DialogSum train) are skipped.

    Raises:
        ValueError: if text appears before the first speaker tag, no utterance is left,
            or a speaker ID is outside ``1..max_speakers``.
    """
    turns: list[tuple[int, list[str]]] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if not line.strip():
            continue
        match = _SPEAKER_RE.match(line)
        if match is None:
            if not turns:
                raise ValueError(f"Text before the first speaker tag: {line[:60]!r}")
            turns[-1][1].append(line.strip())
            continue
        speaker = int(match.group(1))
        if not 1 <= speaker <= max_speakers:
            raise ValueError(f"Speaker {speaker} outside 1..{max_speakers}")
        first = line[match.end() :].strip()
        turns.append((speaker, [first] if first else []))
    kept = [(speaker, " ".join(parts)) for speaker, parts in turns if parts]
    if not kept:
        raise ValueError("Dialogue has no utterances.")
    return [s for s, _ in kept], [u for _, u in kept]


def build_dialogue(row: dict[str, str], split: str, max_speakers: int = 8) -> Dialogue:
    """Build a :class:`Dialogue` from one raw train/validation row."""
    speakers, utterances = parse_dialogue(row["dialogue"], max_speakers)
    return Dialogue(
        id=row["id"],
        split=split,
        speakers=speakers,
        utterances=utterances,
        summaries=[row["summary"].strip()],
        topics=[row["topic"].strip()],
    )


def group_test_rows(rows: Iterable[dict[str, str]], max_speakers: int = 8) -> list[Dialogue]:
    """Merge test rows ``test_<n>_<k>`` into one dialogue ``test_<n>`` with all references.

    References and topics are ordered by ``k``. Output is ordered by ``n``.

    Raises:
        ValueError: on an unexpected ID, or if rows of one group have different dialogues.
    """
    groups: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for row in rows:
        match = _TEST_ID_RE.match(row["id"])
        if match is None:
            raise ValueError(f"Unexpected test ID: {row['id']!r}")
        groups[match.group(1)].append((int(match.group(2)), row))

    dialogues = []
    for dialogue_id in sorted(groups, key=lambda s: int(s.split("_")[1])):
        members = [row for _, row in sorted(groups[dialogue_id], key=lambda kr: kr[0])]
        texts = {row["dialogue"].strip() for row in members}
        if len(texts) != 1:
            raise ValueError(f"{dialogue_id}: rows have different dialogue text.")
        speakers, utterances = parse_dialogue(members[0]["dialogue"], max_speakers)
        dialogues.append(
            Dialogue(
                id=dialogue_id,
                split="test",
                speakers=speakers,
                utterances=utterances,
                summaries=[row["summary"].strip() for row in members],
                topics=[row["topic"].strip() for row in members],
            )
        )
    return dialogues
