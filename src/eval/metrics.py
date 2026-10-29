"""Classification metrics shared by both models (NumPy only)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """``num_classes`` x ``num_classes`` counts; rows = true class, columns = predicted."""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    flat = y_true * num_classes + y_pred
    return np.bincount(flat, minlength=num_classes**2).reshape(num_classes, num_classes)


def per_class_prf(confusion: np.ndarray, names: Sequence[str]) -> dict[str, dict[str, float]]:
    """Precision, recall, F1 and support per class (0 when undefined)."""
    tp = np.diag(confusion).astype(np.float64)
    predicted = confusion.sum(axis=0)
    support = confusion.sum(axis=1)
    out = {}
    for i, name in enumerate(names):
        p = tp[i] / predicted[i] if predicted[i] else 0.0
        r = tp[i] / support[i] if support[i] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out[name] = {"precision": p, "recall": r, "f1": f1, "support": int(support[i])}
    return out


def classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, names: Sequence[str], macro_over: Sequence[str]
) -> dict[str, Any]:
    """Per-class P/R/F1, accuracy, and macro-F1 over the classes in ``macro_over``.

    A class that is neither present nor predicted is left out of the macro average (its F1
    is undefined); a class that is present or predicted counts, with F1 = 0 if missed.
    """
    conf = confusion_matrix(y_true, y_pred, len(names))
    per_class = per_class_prf(conf, names)
    total = conf.sum()
    index = {n: i for i, n in enumerate(names)}
    seen = [n for n in macro_over if conf[index[n]].sum() or conf[:, index[n]].sum()]
    return {
        "per_class": per_class,
        "accuracy": float(np.trace(conf) / total) if total else 0.0,
        "macro_f1": float(np.mean([per_class[n]["f1"] for n in seen])) if seen else 0.0,
        "n": int(total),
    }
