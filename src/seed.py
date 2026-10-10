"""Global seeding for reproducible runs."""

from __future__ import annotations

import random

import keras
import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python's ``random``, NumPy and Keras (which also seeds the backend)."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  (global state is what we seed)
    keras.utils.set_random_seed(seed)
