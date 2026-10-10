"""Training utilities shared by both models: optimiser, time guard, devices, manifests."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import keras

log = logging.getLogger(__name__)

# Parameters matching these name fragments are excluded from weight decay.
NO_DECAY = ["bias", "gamma", "beta", "position_embedding"]


def build_optimizer(opt_cfg: dict[str, Any], total_steps: int) -> keras.optimizers.Optimizer:
    """AdamW with linear warmup to ``peak_lr`` then cosine decay to 0 at ``total_steps``.

    ``warmup_steps`` (absolute) or ``warmup_fraction`` (of ``total_steps``) sets the warmup.
    """
    if opt_cfg["name"] != "adamw" or opt_cfg["schedule"] != "cosine":
        raise ValueError("Only adamw with a cosine schedule is implemented.")
    if "warmup_steps" in opt_cfg:
        warmup = int(opt_cfg["warmup_steps"])
    else:
        warmup = int(opt_cfg["warmup_fraction"] * total_steps)
    warmup = min(warmup, max(total_steps - 1, 0))
    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=max(total_steps - warmup, 1),
        warmup_target=opt_cfg["peak_lr"],
        warmup_steps=warmup,
    )
    optimizer = keras.optimizers.AdamW(learning_rate=schedule, weight_decay=opt_cfg["weight_decay"])
    optimizer.exclude_from_weight_decay(var_names=NO_DECAY)
    return optimizer


class TimeGuard(keras.callbacks.Callback):
    """Stop training cleanly before ``max_hours`` after ``session_start`` (default: train start).

    Pass the same ``session_start`` to every run of a session (several seeds in one Kaggle
    session share one budget). At each epoch end, stops if another epoch (estimated as 1.2 x
    the longest so far) would overrun. Also stops mid-epoch if the hard limit is reached.
    ``stopped`` tells the caller whether training was cut short (so it can skip the final
    evaluation and resume later).
    """

    def __init__(
        self, max_hours: float, session_start: float | None = None, check_every: int = 50
    ) -> None:
        super().__init__()
        self.limit = max_hours * 3600.0
        self.session_start = session_start
        self.check_every = check_every
        self.stopped = False
        self._start = 0.0
        self._epoch_start = 0.0
        self._longest_epoch = 0.0

    def on_train_begin(self, logs=None) -> None:
        self._start = self.session_start if self.session_start is not None else time.time()

    def on_epoch_begin(self, epoch, logs=None) -> None:
        self._epoch_start = time.time()

    def on_train_batch_end(self, batch, logs=None) -> None:
        if batch % self.check_every == 0 and time.time() - self._start > self.limit:
            log.warning("TimeGuard: hard limit reached mid-epoch, stopping.")
            self.stopped = self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None) -> None:
        now = time.time()
        self._longest_epoch = max(self._longest_epoch, now - self._epoch_start)
        if now - self._start + 1.2 * self._longest_epoch > self.limit:
            log.warning("TimeGuard: next epoch would exceed %.2f h, stopping.", self.limit / 3600)
            self.stopped = self.model.stop_training = True


def configure_devices(mixed_precision: str) -> dict[str, Any]:
    """Set the dtype policy and, with several GPUs, data parallelism. Returns device info."""
    keras.mixed_precision.set_dtype_policy(mixed_precision)
    devices = jax.devices()
    if len(devices) > 1:
        keras.distribution.set_distribution(keras.distribution.DataParallel())
    return {
        "devices": [str(d) for d in devices],
        "platform": devices[0].platform,
        "mixed_precision": mixed_precision,
        "keras": keras.__version__,
        "jax": jax.__version__,
    }


def best_logged_value(csv_path: Path, key: str, mode: str) -> float | None:
    """Best value of ``key`` in a CSVLogger file (for resuming checkpoint thresholds)."""
    if not csv_path.exists():
        return None
    with csv_path.open() as f:
        values = [float(row[key]) for row in csv.DictReader(f) if row.get(key)]
    if not values:
        return None
    return max(values) if mode == "max" else min(values)


def commit_sha() -> str | None:
    """``COMMIT_SHA`` env var, else ``git rev-parse HEAD``, else ``None``."""
    if sha := os.environ.get("COMMIT_SHA"):
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` as indented JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def utc_now() -> str:
    """Current UTC time, ISO format, seconds precision."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def steps_for(
    n_examples: int,
    batch_size: int,
    epochs: int,
    max_steps: int | None = None,
    steps_per_epoch: int | None = None,
) -> tuple[int, int, int]:
    """``(steps_per_epoch, epochs, total_steps)``.

    An epoch is one pass over the data unless ``steps_per_epoch`` caps it; ``max_steps`` caps
    the total (reducing the number of epochs).
    """
    full = n_examples // batch_size
    steps_per_epoch = full if steps_per_epoch is None else min(full, steps_per_epoch)
    if steps_per_epoch == 0:
        raise ValueError(f"{n_examples} examples is fewer than one batch of {batch_size}")
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
        epochs = min(epochs, math.ceil(max_steps / steps_per_epoch))
    return steps_per_epoch, epochs, steps_per_epoch * epochs
