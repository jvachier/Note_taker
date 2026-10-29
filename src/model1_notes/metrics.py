"""Sentence-level metrics for Model 1 v3 (PLAN.md 10.7), on plain NumPy arrays.

Shared by training (validation / test metrics), the evaluation tables and the notebooks' copy.
No ML imports.

- **Importance**: F1 at a threshold, average precision (PR-AUC) of the probabilities.
- **Zones**: macro F1 over the zones present in the gold labels.
- **Decisions / action items**: a sentence counts as a predicted decision when it is kept
  *and* its predicted zone is ``decision`` (gold: ``importance = 1`` and zone ``decision``);
  same for action items.
- **Hard negatives**: share of gold-0 hard-negative sentences that are kept (lower is better).
- **Spans**: exact-match F1 of owner / deadline entities decoded from BIO token tags.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

ZONES = ("context", "discussion", "decision", "action_item", "closing")
SPAN_TAGS = ("O", "B-owner", "I-owner", "B-deadline", "I-deadline")
DECISION, ACTION = ZONES.index("decision"), ZONES.index("action_item")


def f1_score(truth: np.ndarray, pred: np.ndarray) -> float:
    """Binary F1 (0 when there is no positive prediction or label)."""
    truth, pred = np.asarray(truth, dtype=bool), np.asarray(pred, dtype=bool)
    tp = float(np.sum(truth & pred))
    denom = float(np.sum(truth) + np.sum(pred))
    return 0.0 if denom == 0 else 2 * tp / denom


def tune_threshold(truth: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    """Threshold in 0.05..0.95 that maximises importance F1, and that F1."""
    truth = np.asarray(truth).astype(bool)
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    scores = [f1_score(truth, np.asarray(prob) >= t) for t in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


def average_precision(truth: np.ndarray, prob: np.ndarray) -> float:
    """Area under the precision-recall curve (step-wise, as in scikit-learn)."""
    truth = np.asarray(truth, dtype=bool)
    if not truth.any():
        return 0.0
    order = np.argsort(-np.asarray(prob, dtype=np.float64), kind="stable")
    hits = truth[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(np.sum(precision[hits]) / truth.sum())


def zone_macro_f1(truth: np.ndarray, pred: np.ndarray) -> float:
    """Macro F1 over the zones present in ``truth``."""
    truth, pred = np.asarray(truth), np.asarray(pred)
    present = np.unique(truth)
    if not len(present):
        return 0.0
    return float(np.mean([f1_score(truth == z, pred == z) for z in present]))


def span_entities(tags: Sequence[int]) -> set[tuple[int, int, str]]:
    """``(start, end, role)`` entities of one BIO sequence (an ``I`` without ``B`` starts one)."""
    entities, start, role = set(), None, None
    for i, t in enumerate([*tags, 0]):
        name = SPAN_TAGS[t] if 0 <= t < len(SPAN_TAGS) else "O"
        inside = name.startswith("I-") and role == name[2:]
        if start is not None and not inside:
            entities.add((start, i, role))
            start, role = None, None
        if name != "O" and not inside:
            start, role = i, name[2:]
    return entities


def span_f1(gold: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Exact-match entity F1 per role and overall; ``gold`` / ``pred`` are (N, T) BIO tags
    (``-1`` positions are ignored, i.e. treated as ``O``)."""
    counts = {"owner": [0, 0, 0], "deadline": [0, 0, 0]}  # tp, n_gold, n_pred
    for g_row, p_row in zip(np.asarray(gold), np.asarray(pred), strict=True):
        valid = g_row >= 0
        g = span_entities(np.where(valid, g_row, 0).tolist())
        p = span_entities(np.where(valid, p_row, 0).tolist())
        for role, c in counts.items():
            gr, pr = {e for e in g if e[2] == role}, {e for e in p if e[2] == role}
            c[0] += len(gr & pr)
            c[1] += len(gr)
            c[2] += len(pr)

    def f1(tp: int, n_gold: int, n_pred: int) -> float:
        return 0.0 if n_gold + n_pred == 0 else 2 * tp / (n_gold + n_pred)

    total = [sum(c[k] for c in counts.values()) for k in range(3)]
    return {
        "owner_f1": round(f1(*counts["owner"]), 4),
        "deadline_f1": round(f1(*counts["deadline"]), 4),
        "span_f1": round(f1(*total), 4),
        "gold_spans": total[1],
    }


def selection_metrics(
    truth: np.ndarray,
    zone_true: np.ndarray,
    picked: np.ndarray,
    prob: np.ndarray | None = None,
    zone_pred: np.ndarray | None = None,
    hard_negative: np.ndarray | None = None,
) -> dict[str, Any]:
    """Metrics of one system on one set of sentences (see module docstring); ``None`` for
    what the system does not predict."""
    truth, picked = np.asarray(truth, dtype=bool), np.asarray(picked, dtype=bool)
    zone_true = np.asarray(zone_true)
    out: dict[str, Any] = {
        "sentences": int(len(truth)),
        "importance_f1": round(f1_score(truth, picked), 4),
        "pr_auc": None if prob is None else round(average_precision(truth, prob), 4),
        "zone_macro_f1": None,
        "decision_f1": None,
        "action_f1": None,
        "hard_negative_fp": None,
    }
    if zone_pred is not None:
        zone_pred = np.asarray(zone_pred)
        out["zone_macro_f1"] = round(zone_macro_f1(zone_true, zone_pred), 4)
        for key, z in (("decision_f1", DECISION), ("action_f1", ACTION)):
            out[key] = round(f1_score(truth & (zone_true == z), picked & (zone_pred == z)), 4)
    if hard_negative is not None:
        hn = np.asarray(hard_negative, dtype=bool) & ~truth
        out["hard_negative_fp"] = round(float(picked[hn].mean()), 4) if hn.any() else None
    return out
