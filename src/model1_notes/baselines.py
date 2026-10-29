"""Untrained baselines for Model 1 v3 (PLAN.md 10.6): Lead-k and a keyword rule.

No ML imports, so the teacher quality gate (vLLM environment) can use them too.

- **Lead-k** keeps the first ``fraction`` of a meeting's sentences (meetings vary from 20 to
  1,500 sentences, so k is a fraction, tuned on validation).
- **Keyword rule** keeps a sentence when it matches a decision or action trigger (regexes from
  the config) and predicts its zone from the trigger kind; everything else is discussion.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

ZONE_DISCUSSION, ZONE_DECISION, ZONE_ACTION = "discussion", "decision", "action_item"


def lead_fraction(n: int, fraction: float) -> list[int]:
    """1 for the first ``ceil(fraction * n)`` sentences (at least one), else 0."""
    k = max(1, math.ceil(fraction * n - 1e-9))
    return [1 if i < k else 0 for i in range(n)]


class KeywordRule:
    """Decision / action trigger regexes (case-insensitive) from ``cfg``."""

    def __init__(self, cfg: dict[str, Sequence[str]]) -> None:
        self.decision = [re.compile(p, re.IGNORECASE) for p in cfg["decision"]]
        self.action = [re.compile(p, re.IGNORECASE) for p in cfg["action"]]

    def zone(self, text: str) -> str:
        """Predicted zone: action triggers win over decision triggers."""
        if any(p.search(text) for p in self.action):
            return ZONE_ACTION
        if any(p.search(text) for p in self.decision):
            return ZONE_DECISION
        return ZONE_DISCUSSION

    def importance(self, text: str) -> int:
        """1 if the sentence matches any trigger."""
        return int(self.zone(text) != ZONE_DISCUSSION)


def binary_f1(truth: Sequence[int], pred: Sequence[int]) -> float:
    """F1 of the positive class (0 when nothing is predicted or labelled positive)."""
    tp = sum(1 for t, p in zip(truth, pred, strict=True) if t and p)
    denom = sum(truth) + sum(pred)
    return 0.0 if denom == 0 else 2 * tp / denom


def tune_lead(meetings: Sequence[Sequence[int]], grid: Sequence[float]) -> tuple[float, float]:
    """Fraction in ``grid`` with the best pooled importance F1 over ``meetings`` (gold 0/1)."""
    best = (grid[0], -1.0)
    for fraction in grid:
        truth = [t for m in meetings for t in m]
        pred = [p for m in meetings for p in lead_fraction(len(m), fraction)]
        f1 = binary_f1(truth, pred)
        if f1 > best[1]:
            best = (fraction, f1)
    return best


def baseline_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """The ``baselines`` section of a v3 config."""
    return cfg["baselines"]
