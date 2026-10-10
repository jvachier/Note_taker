"""Tests for global seeding."""

import random

import keras
import numpy as np

from src.seed import set_seed


def _draw() -> tuple[float, float, np.ndarray]:
    return (
        random.random(),
        float(np.random.rand()),  # noqa: NPY002
        keras.ops.convert_to_numpy(keras.random.uniform((3,))),
    )


def test_set_seed_is_reproducible():
    set_seed(123)
    first = _draw()
    set_seed(123)
    second = _draw()
    assert first[:2] == second[:2]
    np.testing.assert_array_equal(first[2], second[2])


def test_different_seeds_differ():
    set_seed(1)
    first = _draw()
    set_seed(2)
    assert first[1] != _draw()[1]
