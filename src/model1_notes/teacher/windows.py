"""Meeting windows and the constants shared by the teacher and the student (no ML imports).

The student (``src.model1_notes.data``) and the Qwen teacher cut long meetings the same way, so
this module has no dependency beyond the standard library and can run in the teacher's vLLM
environment.
"""

from __future__ import annotations

#: Qwen teacher reason classes (PLAN.md 10.3); index = the student's ``reason`` head output.
REASONS = (
    "agreed_outcome",
    "commitment_with_owner",
    "risk_or_blocker",
    "key_fact",
    "status_update",
    "rejected_proposal",
    "tentative",
    "question",
    "filler",
)


def window_starts(length: int, size: int, overlap: int) -> list[int]:
    """Start offsets of windows of ``size`` sentences overlapping by ``overlap``.

    Windows advance by ``size - overlap``; the last one is aligned to the end of the meeting so
    every window except a short meeting's single window is full.
    """
    if not 0 <= overlap < size:
        raise ValueError(f"window_overlap must be in [0, {size}); got {overlap}")
    if length <= size:
        return [0]
    starts = list(range(0, length - size, size - overlap))
    return [*starts, length - size]


def best_window(length: int, size: int, overlap: int) -> list[int]:
    """For each sentence, the window (index into :func:`window_starts`) where it is farthest
    from both window edges (ties go to the earlier window)."""
    starts = window_starts(length, size, overlap)
    best = []
    for i in range(length):
        margins = [min(i - s, s + size - 1 - i) if s <= i < s + size else -1 for s in starts]
        best.append(max(range(len(starts)), key=lambda w: (margins[w], -w)))
    return best
