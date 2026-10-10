"""Inference rule shared by every system (PLAN.md section 5.5).

Select the top-k utterances by score (ties -> earlier utterance), output them in dialogue
order, grouped by predicted tag. :func:`notes_text` is the plain text that ROUGE / BERTScore
score (no headings, so systems with and without tags are compared on the same words);
:func:`render_notes` is the human-readable version with headings.
"""

from __future__ import annotations

from collections.abc import Sequence

TAG_ORDER = ("decision", "action_item", "key_point", "none")
HEADINGS = {
    "decision": "Decisions",
    "action_item": "Action items",
    "key_point": "Key points",
    "none": "Other",
}


def select_top_k(scores: Sequence[float], k: int) -> list[int]:
    """Indices of the ``k`` highest scores, returned in dialogue order."""
    ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return sorted(ranked[:k])


def grouped(selected: Sequence[int], tags: Sequence[str] | None) -> list[tuple[str, list[int]]]:
    """``[(tag, indices in dialogue order), ...]`` in :data:`TAG_ORDER`, empty groups dropped."""
    if tags is None:
        return [("none", sorted(selected))]
    return [
        (tag, [i for i in sorted(selected) if tags[i] == tag])
        for tag in TAG_ORDER
        if any(tags[i] == tag for i in selected)
    ]


def notes_text(
    utterances: Sequence[str], selected: Sequence[int], tags: Sequence[str] | None = None
) -> str:
    """Selected utterances, grouped by tag, joined with spaces (what the metrics score)."""
    return " ".join(utterances[i] for _, group in grouped(selected, tags) for i in group)


def render_notes(
    utterances: Sequence[str],
    speakers: Sequence[int],
    selected: Sequence[int],
    tags: Sequence[str] | None = None,
) -> str:
    """Markdown notes: one heading per tag group, one bullet per utterance with its speaker."""
    lines = []
    for tag, group in grouped(selected, tags):
        if tags is not None:
            lines.append(f"**{HEADINGS[tag]}**")
        lines += [f"- #Person{speakers[i]}#: {utterances[i]}" for i in group]
    return "\n".join(lines)
