"""Validate a Qwen answer and turn its token log-probabilities into soft scores (PLAN.md 10.3).

An answer is a JSON array of ``[id, score, "reason", depends_on]`` entries. :func:`parse_answer`
is the strict reading (every id 1..n exactly once, ``score`` an integer 0-3, ``reason`` one of
:data:`~src.model1_notes.data.REASONS`, ``depends_on`` -1 or an earlier id).

**Lenient reading** (:func:`parse_entries`, used for labelling). Qwen3-0.6B mostly answers with
objects ``{"id": .., "score": .., "reason": .., "depends_on": ..}``, often wrapped in an extra
array, ends with ``<|im_end|>`` and is sometimes cut off by the token budget. Every well-formed
entry is kept, in either format; an unknown reason becomes ``None`` and an invalid
``depends_on`` becomes -1 (only that field is lost); entries with an invalid id or score are
dropped. A window counts when at least ``min_coverage`` of its sentences are labelled.

**Soft score.** The answer is generated with the top-k log-probabilities of every token. For
each entry, the token that emitted its score digit is found by character offset; the
probabilities of ``0``..``3`` among its top-k alternatives are renormalised and the expected
score / 3 is the soft score. If no digit is among the alternatives (or no log-probabilities
are given) the emitted score / 3 is used.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

SCORE_RANGE = (0, 3)
#: ``[ <id> , <score>`` at the start of an entry: group 2 is the score digit.
ENTRY_RE = re.compile(r"\[\s*(\d+)\s*,\s*(-?\d+)")
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
#: Chat special tokens such as ``<|im_end|>`` left in the decoded text.
SPECIAL_RE = re.compile(r"<\|[^|>]*\|>")
#: One flat JSON object (an entry in object form).
OBJECT_RE = re.compile(r"\{[^{}]*\}")
#: ``[id, score, "reason", depends_on]`` (an entry in list form).
LIST_RE = re.compile(r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*"([^"]*)"\s*,\s*(-?\d+)\s*\]')
#: ``"id": <id>, "score": <score>`` in an object entry: group 2 is the score digit.
OBJECT_SCORE_RE = re.compile(r'"id"\s*:\s*(\d+)\s*,\s*"score"\s*:\s*(-?\d+)')
TOO_LONG = "prompt too long for the model context"


class InvalidAnswer(ValueError):
    """The teacher's answer does not follow the format."""


@dataclass
class Label:
    """One sentence's teacher label (``depends_on`` is a 1-based id, -1 = none)."""

    score: int
    reason: str | None  # None: the teacher gave a reason outside the list
    depends_on: int
    soft: float


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def answer_body(text: str | None) -> tuple[str, int]:
    """The JSON part of an answer and its character offset in ``text``."""
    if text is None:
        raise InvalidAnswer(TOO_LONG)
    fenced = FENCE_RE.search(text)
    if fenced:
        return fenced.group(1), fenced.start(1)
    start = text.find("[")
    if start < 0:
        raise InvalidAnswer("no JSON array found")
    return text[start:], start


def parse_answer(text: str | None, n: int, reasons: Sequence[str]) -> list[tuple[int, str, int]]:
    """``(score, reason, depends_on)`` for sentence ids 1..n, or :class:`InvalidAnswer`."""
    body, _ = answer_body(text)
    try:
        items, _ = json.JSONDecoder().raw_decode(body)
    except json.JSONDecodeError as err:
        raise InvalidAnswer(f"invalid JSON ({err.msg})") from None
    if not isinstance(items, list):
        raise InvalidAnswer("the answer is not a JSON array")
    lo, hi = SCORE_RANGE
    by_id: dict[int, tuple[int, str, int]] = {}
    for item in items:
        if not isinstance(item, list) or len(item) != 4:
            raise InvalidAnswer('every entry must be [id, score, "reason", depends_on]')
        uid, score, reason, dep = _as_int(item[0]), _as_int(item[1]), item[2], _as_int(item[3])
        if uid is None or not 1 <= uid <= n:
            raise InvalidAnswer(f"id {item[0]!r} is not a sentence number 1..{n}")
        if uid in by_id:
            raise InvalidAnswer(f"id {uid} appears more than once")
        if score is None or not lo <= score <= hi:
            raise InvalidAnswer(f"id {uid}: score {item[1]!r} is not an integer {lo}-{hi}")
        if reason not in reasons:
            raise InvalidAnswer(f"id {uid}: reason {reason!r} is not one of the reasons")
        if dep is None or not (dep == -1 or 1 <= dep < uid):
            raise InvalidAnswer(f"id {uid}: depends_on {item[3]!r} is not -1 or an earlier id")
        by_id[uid] = (score, reason, dep)
    missing = [i for i in range(1, n + 1) if i not in by_id]
    if missing:
        raise InvalidAnswer(f"missing ids {missing[:10]}")
    return [by_id[i] for i in range(1, n + 1)]


def score_positions(text: str) -> dict[int, int]:
    """``id -> character offset`` of each entry's score digit in ``text`` (both formats)."""
    positions = {int(m.group(1)): m.start(2) for m in OBJECT_SCORE_RE.finditer(text)}
    for m in LIST_RE.finditer(text):
        positions.setdefault(int(m.group(1)), m.start(2))
    return positions


def parse_entries(
    text: str | None, n: int, reasons: Sequence[str]
) -> dict[int, tuple[int, str | None, int]]:
    """Lenient reading (see module docstring): ``id -> (score, reason, depends_on)``."""
    if not text:
        return {}
    text = SPECIAL_RE.sub("", text)
    lo, hi = SCORE_RANGE
    raw: dict[int, tuple[object, object, object]] = {}
    for m in OBJECT_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        uid = _as_int(obj.get("id")) if isinstance(obj, dict) else None
        if uid is not None and uid not in raw:
            raw[uid] = (obj.get("score"), obj.get("reason"), obj.get("depends_on", -1))
    for m in LIST_RE.finditer(text):
        uid = int(m.group(1))
        if uid not in raw:
            raw[uid] = (int(m.group(2)), m.group(3), int(m.group(4)))
    entries: dict[int, tuple[int, str | None, int]] = {}
    for uid, (score, reason, dep) in raw.items():
        score, dep = _as_int(score), _as_int(dep)
        if not 1 <= uid <= n or score is None or not lo <= score <= hi:
            continue
        reason = reason if reason in reasons else None
        dep = dep if dep is not None and 1 <= dep < uid else -1
        entries[uid] = (score, reason, dep)
    return entries


def digit_distribution(top: dict[str, float]) -> dict[int, float] | None:
    """Renormalised probabilities of the scores among a token's top log-probabilities."""
    lo, hi = SCORE_RANGE
    probs: dict[int, float] = {}
    for token, logprob in top.items():
        token = token.strip()
        if token.isdigit() and lo <= int(token) <= hi:
            probs[int(token)] = probs.get(int(token), 0.0) + math.exp(logprob)
    total = sum(probs.values())
    return None if total <= 0 else {k: v / total for k, v in probs.items()}


def soft_scores(
    text: str,
    scores: Sequence[int],
    tokens: Sequence[tuple[str, dict[str, float]]] | None,
) -> list[float]:
    """Soft score in [0, 1] per sentence (see module docstring).

    ``tokens`` are the generated tokens as ``(text, {alternative: logprob})``; their texts
    concatenated must equal ``text``.
    """
    hi = SCORE_RANGE[1]
    soft = [s / hi for s in scores]
    if not tokens:
        return soft
    ends, pos = [], 0
    for token_text, _ in tokens:
        pos += len(token_text)
        ends.append(pos)
    for uid, char in score_positions(text).items():
        if not 1 <= uid <= len(scores):
            continue
        k = next((i for i, end in enumerate(ends) if end > char), None)
        dist = None if k is None else digit_distribution(tokens[k][1])
        if dist:
            soft[uid - 1] = sum(score * p for score, p in dist.items()) / hi
    return [round(s, 6) for s in soft]


def labels_from_answer(
    text: str | None,
    n: int,
    reasons: Sequence[str],
    tokens: Sequence[tuple[str, dict[str, float]]] | None = None,
    min_coverage: float = 0.0,
) -> dict[int, Label]:
    """Labels with soft scores for the sentences of one window that the answer covers
    (``id -> Label``); raises :class:`InvalidAnswer` below ``min_coverage``."""
    if text is None:
        raise InvalidAnswer(TOO_LONG)
    entries = parse_entries(text, n, reasons)
    if not entries or len(entries) < min_coverage * n:
        raise InvalidAnswer(f"only {len(entries)} of {n} sentences have a valid entry")
    scores = [entries[i][0] if i in entries else 0 for i in range(1, n + 1)]
    soft = soft_scores(text, scores, tokens)
    return {i: Label(s, r, d, soft[i - 1]) for i, (s, r, d) in sorted(entries.items())}
