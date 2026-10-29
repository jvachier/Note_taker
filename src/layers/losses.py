"""Masked token-level loss utilities.

Label arrays use ``IGNORE = -1`` for positions without a label (padding, non-first subwords,
words excluded from a task). :func:`label_weights` turns them into per-position
``sample_weight`` (0 at ignored positions, optionally scaled by a per-class weight), and the
loss averages only over weighted positions (``reduction="mean_with_sample_weight"``).
"""

from __future__ import annotations

from collections.abc import Sequence

import keras
import numpy as np

IGNORE = -1


def masked_sparse_ce() -> keras.losses.Loss:
    """Sparse categorical cross-entropy on logits, averaged over the sample weights."""
    return keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="mean_with_sample_weight"
    )


def label_weights(labels: np.ndarray, class_weights: Sequence[float] | None = None) -> np.ndarray:
    """Per-position float32 weights: 0 where ``labels == IGNORE``, else the class weight (or 1)."""
    valid = labels != IGNORE
    if class_weights is None:
        return valid.astype(np.float32)
    table = np.asarray(class_weights, dtype=np.float32)
    return np.where(valid, table[np.where(valid, labels, 0)], 0.0).astype(np.float32)


def safe_labels(labels: np.ndarray) -> np.ndarray:
    """Replace ``IGNORE`` by 0 so the loss can index; pair with :func:`label_weights`."""
    return np.where(labels == IGNORE, 0, labels).astype(np.int32)


def inverse_sqrt_frequency_weights(counts: Sequence[int]) -> list[float]:
    """Class weights proportional to ``1 / sqrt(frequency)``, scaled so the average weight over
    labelled positions is 1 (``sum_c f_c * w_c = 1``). Classes with zero count get weight 0.
    """
    c = np.asarray(counts, dtype=np.float64)
    freq = c / c.sum()
    w = np.zeros_like(freq)
    nz = freq > 0
    w[nz] = 1.0 / np.sqrt(freq[nz])
    w /= (freq * w).sum()
    return [round(float(x), 6) for x in w]
