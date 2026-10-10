"""Dialogue models package.

Importing anything under ``src`` pins Keras to the JAX backend before Keras is loaded.
"""

import os

_backend = os.environ.setdefault("KERAS_BACKEND", "jax")
if _backend != "jax":
    raise RuntimeError(f"KERAS_BACKEND must be 'jax', got {_backend!r}.")
