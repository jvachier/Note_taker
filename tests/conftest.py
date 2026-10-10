"""Shared pytest fixtures. Pins the Keras backend before any test imports keras."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def configs_dir() -> Path:
    """Path to the repo's ``configs/`` directory."""
    return REPO_ROOT / "configs"
