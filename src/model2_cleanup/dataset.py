"""Memory-mapped Model 2 arrays and a shuffling ``keras.utils.PyDataset`` over them."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import keras
import numpy as np

from src.layers.losses import label_weights, safe_labels


@dataclass
class ArraySet:
    """One array set on disk (see :mod:`src.model2_cleanup.build_data`)."""

    token_ids: np.ndarray
    casing: np.ndarray
    punct: np.ndarray
    meta: dict[str, Any]

    @classmethod
    def load(cls, directory: str | Path, mmap: bool = True) -> ArraySet:
        d = Path(directory)
        mode = "r" if mmap else None
        return cls(
            token_ids=np.load(d / "token_ids.npy", mmap_mode=mode),
            casing=np.load(d / "casing.npy", mmap_mode=mode),
            punct=np.load(d / "punct.npy", mmap_mode=mode),
            meta=json.loads((d / "meta.json").read_text()),
        )

    @classmethod
    def concat(cls, sets: Sequence[ArraySet]) -> ArraySet:
        """Concatenate several sets in memory (used for the combined validation set)."""
        return cls(
            token_ids=np.concatenate([s.token_ids for s in sets]),
            casing=np.concatenate([s.casing for s in sets]),
            punct=np.concatenate([s.punct for s in sets]),
            meta={"parts": [s.meta for s in sets]},
        )

    def __len__(self) -> int:
        return int(self.token_ids.shape[0])


def make_targets(
    casing: np.ndarray, punct: np.ndarray, punct_class_weights: Sequence[float] | None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Labels and per-position sample weights for the two heads."""
    y = {"casing": safe_labels(casing), "punct": safe_labels(punct)}
    sw = {"casing": label_weights(casing), "punct": label_weights(punct, punct_class_weights)}
    return y, sw


class CleanupDataset(keras.utils.PyDataset):
    """Batches ``(token_ids, labels, sample_weights)``, reshuffled every epoch.

    Args:
        arrays: the array set.
        batch_size: windows per batch.
        punct_class_weights: per-class weights for the punctuation loss (``None`` = uniform).
        shuffle: shuffle windows each epoch with ``seed`` (epoch-dependent).
        max_batches: cap on batches per epoch (smoke runs).
        drop_last: drop the last incomplete batch (keeps shapes static for JAX).
    """

    def __init__(
        self,
        arrays: ArraySet,
        batch_size: int,
        punct_class_weights: Sequence[float] | None = None,
        shuffle: bool = True,
        seed: int = 0,
        max_batches: int | None = None,
        drop_last: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.arrays = arrays
        self.batch_size = batch_size
        self.punct_class_weights = punct_class_weights
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        n = len(arrays)
        n_batches = n // batch_size if drop_last else math.ceil(n / batch_size)
        if n_batches == 0:
            raise ValueError(f"{n} windows is fewer than one batch of {batch_size}")
        self.n_batches = n_batches if max_batches is None else min(n_batches, max_batches)
        self._order = self._make_order()

    def _make_order(self) -> np.ndarray:
        n = len(self.arrays)
        if not self.shuffle:
            return np.arange(n)
        return np.random.default_rng([self.seed, self.epoch]).permutation(n)

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, index: int):
        idx = np.sort(self._order[index * self.batch_size : (index + 1) * self.batch_size])
        ids = np.asarray(self.arrays.token_ids[idx], dtype=np.int32)
        y, sw = make_targets(
            np.asarray(self.arrays.casing[idx]),
            np.asarray(self.arrays.punct[idx]),
            self.punct_class_weights,
        )
        return ids, y, sw

    def on_epoch_end(self) -> None:
        self.epoch += 1
        self._order = self._make_order()
