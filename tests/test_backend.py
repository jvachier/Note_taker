"""Keras must run on the JAX backend."""

import keras

import src  # noqa: F401


def test_backend_is_jax():
    assert keras.backend.backend() == "jax"
