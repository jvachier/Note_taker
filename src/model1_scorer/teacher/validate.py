"""Validation of teacher answers (PLAN.md section 5.3).

An answer is valid when it contains a JSON array with exactly one object per utterance id
``1..n``, each with an integer-valued ``score`` inside ``score_range`` and a ``tag`` from the
tag set. Surrounding text (e.g. the ``Summary:`` line of the prompt without a reference) or a
Markdown code fence is tolerated; anything else is rejected with a short reason (fed back to
the teacher in the retry prompt).

Self-consistency (:func:`combine_samples`): several sampled answers for one dialogue are
validated separately and the valid ones are merged: mean score per utterance, majority tag.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_ARRAY_RE = re.compile(r"\[\s*[{\]]")  # an array of objects (or empty), not "[" in prose
# Majority-vote ties go to the rarer, more specific tag.
TAG_PRIORITY = ("decision", "action_item", "key_point", "none")


class InvalidAnswer(ValueError):
    """The teacher answer cannot be used; the message says why."""


@dataclass(frozen=True)
class TeacherLabels:
    """Validated labels for one dialogue, in utterance order."""

    raw_scores: list[float]  # teacher scale (int for one answer, mean over samples otherwise)
    scores: list[float]  # normalised to [0, 1]
    tags: list[str]
    samples: int = 1  # valid answers merged into these labels


def extract_json_array(text: str) -> list:
    """The first JSON array in ``text`` (inside a code fence if there is one)."""
    fenced = _FENCE_RE.search(text)
    body = fenced.group(1) if fenced else text
    match = _ARRAY_RE.search(body)
    if match is None:
        raise InvalidAnswer("no JSON array found")
    start = match.start()
    try:
        value, _ = json.JSONDecoder().raw_decode(body[start:])
    except json.JSONDecodeError as err:
        raise InvalidAnswer(f"invalid JSON ({err.msg})") from None
    if not isinstance(value, list):
        raise InvalidAnswer("the answer is not a JSON array")
    return value


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def validate_answer(
    text: str, n: int, score_range: Sequence[int], tags: Sequence[str]
) -> TeacherLabels:
    """Parse and check a teacher answer for a dialogue of ``n`` utterances.

    Raises:
        InvalidAnswer: with the first problem found.
    """
    items = extract_json_array(text)
    lo, hi = score_range
    by_id: dict[int, tuple[int, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise InvalidAnswer("every array element must be an object")
        uid, score, tag = _as_int(item.get("id")), _as_int(item.get("score")), item.get("tag")
        if uid is None or not 1 <= uid <= n:
            raise InvalidAnswer(f"id {item.get('id')!r} is not an utterance number 1..{n}")
        if uid in by_id:
            raise InvalidAnswer(f"id {uid} appears more than once")
        if score is None or not lo <= score <= hi:
            raise InvalidAnswer(
                f"id {uid}: score {item.get('score')!r} is not an integer {lo}-{hi}"
            )
        if tag not in tags:
            raise InvalidAnswer(f"id {uid}: tag {tag!r} is not one of {list(tags)}")
        by_id[uid] = (score, tag)
    missing = [i for i in range(1, n + 1) if i not in by_id]
    if missing:
        raise InvalidAnswer(f"missing ids {missing[:10]}")
    raw = [by_id[i][0] for i in range(1, n + 1)]
    return TeacherLabels(
        raw_scores=raw,
        scores=[(s - lo) / (hi - lo) for s in raw],
        tags=[by_id[i][1] for i in range(1, n + 1)],
    )


def combine_samples(samples: Sequence[TeacherLabels], score_range: Sequence[int]) -> TeacherLabels:
    """Merge valid answers for one dialogue: mean raw score, majority tag (ties by priority).

    Raises:
        ValueError: if ``samples`` is empty or the samples disagree on the utterance count.
    """
    if not samples:
        raise ValueError("no valid samples to combine")
    n = len(samples[0].raw_scores)
    if any(len(s.raw_scores) != n for s in samples):
        raise ValueError("samples have different utterance counts")
    lo, hi = score_range
    raw = [round(sum(s.raw_scores[i] for s in samples) / len(samples), 4) for i in range(n)]
    rank = {t: k for k, t in enumerate(TAG_PRIORITY)}
    tags = []
    for i in range(n):
        votes = Counter(s.tags[i] for s in samples)
        best = max(votes.values())
        tied = [t for t in votes if votes[t] == best]
        tags.append(min(tied, key=lambda t: rank.get(t, len(rank))))
    return TeacherLabels(
        raw_scores=raw,
        scores=[round((r - lo) / (hi - lo), 4) for r in raw],
        tags=tags,
        samples=len(samples),
    )
