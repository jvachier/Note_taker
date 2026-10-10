"""Training utilities shared by both models: optimiser, time guard, devices, manifests."""

from __future__ import annotations

import csv
import json
import logging
import math
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import keras
import numpy as np

log = logging.getLogger(__name__)

# Parameters matching these name fragments are excluded from weight decay.
NO_DECAY = ["bias", "gamma", "beta", "position_embedding"]


@keras.saving.register_keras_serializable(package="dialogue_models")
class LayerwiseAdamW(keras.optimizers.AdamW):
    """AdamW whose learning rate (and decoupled weight decay) is scaled per layer.

    ``lr_multipliers`` maps a layer name to a factor: a variable whose path contains that
    name as one segment (``.../encoder/...``) uses ``learning_rate * factor``. Other variables
    use the base learning rate.
    """

    def __init__(self, lr_multipliers: dict[str, float] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.lr_multipliers = dict(lr_multipliers or {})

    def _multiplier(self, variable) -> float:
        segments = variable.path.split("/")
        for name, factor in self.lr_multipliers.items():
            if name in segments:
                return factor
        return 1.0

    def update_step(self, gradient, variable, learning_rate):
        return super().update_step(gradient, variable, learning_rate * self._multiplier(variable))

    def _apply_weight_decay(self, variables) -> None:
        if self.weight_decay is None:
            return
        for variable in variables:
            if self._use_weight_decay(variable):
                lr = keras.ops.cast(self.learning_rate, variable.dtype) * self._multiplier(variable)
                wd = keras.ops.cast(self.weight_decay, variable.dtype)
                variable.assign(variable - variable * wd * lr)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "lr_multipliers": self.lr_multipliers}


def build_optimizer(opt_cfg: dict[str, Any], total_steps: int) -> keras.optimizers.Optimizer:
    """AdamW with linear warmup to the peak learning rate, then cosine decay to 0.

    The peak is ``peak_lr`` for every weight, or ``lr_heads`` with ``lr_encoder`` for the
    weights of the layer named ``encoder`` (students: smaller steps for pretrained weights).
    ``warmup_steps`` (absolute) or ``warmup_fraction`` (of ``total_steps``) sets the warmup.
    """
    if opt_cfg["name"] != "adamw" or opt_cfg["schedule"] != "cosine":
        raise ValueError("Only adamw with a cosine schedule is implemented.")
    if "warmup_steps" in opt_cfg:
        warmup = int(opt_cfg["warmup_steps"])
    else:
        warmup = int(opt_cfg["warmup_fraction"] * total_steps)
    warmup = min(warmup, max(total_steps - 1, 0))
    if "lr_heads" in opt_cfg:
        peak = opt_cfg["lr_heads"]
        multipliers = {"encoder": opt_cfg["lr_encoder"] / peak}
    else:
        peak, multipliers = opt_cfg["peak_lr"], {}
    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=max(total_steps - warmup, 1),
        warmup_target=peak,
        warmup_steps=warmup,
    )
    optimizer = LayerwiseAdamW(
        lr_multipliers=multipliers, learning_rate=schedule, weight_decay=opt_cfg["weight_decay"]
    )
    optimizer.exclude_from_weight_decay(var_names=NO_DECAY)
    return optimizer


class TimeGuard(keras.callbacks.Callback):
    """Stop training cleanly before ``max_hours`` after ``session_start`` (default: train start).

    Pass the same ``session_start`` to every run of a session (several seeds in one Kaggle
    session share one budget). At each epoch end, stops if another epoch (estimated as 1.2 x
    the longest so far) would overrun. Also stops mid-epoch if the hard limit is reached
    (checked on every batch-end call: with ``steps_per_execution`` > 1 Keras calls it once per
    execution, with ``batch`` = the last step of that execution).
    ``stopped`` tells the caller whether training was cut short (so it can skip the final
    evaluation and resume later).
    """

    def __init__(self, max_hours: float, session_start: float | None = None) -> None:
        super().__init__()
        self.limit = max_hours * 3600.0
        self.session_start = session_start
        self.stopped = False
        self._start = 0.0
        self._epoch_start = 0.0
        self._longest_epoch = 0.0

    def on_train_begin(self, logs=None) -> None:
        self._start = self.session_start if self.session_start is not None else time.time()

    def on_epoch_begin(self, epoch, logs=None) -> None:
        self._epoch_start = time.time()

    def on_train_batch_end(self, batch, logs=None) -> None:
        if time.time() - self._start > self.limit:
            log.warning("TimeGuard: hard limit reached mid-epoch, stopping.")
            self.stopped = self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None) -> None:
        now = time.time()
        self._longest_epoch = max(self._longest_epoch, now - self._epoch_start)
        if now - self._start + 1.2 * self._longest_epoch > self.limit:
            log.warning("TimeGuard: next epoch would exceed %.2f h, stopping.", self.limit / 3600)
            self.stopped = self.model.stop_training = True


class SyncEpoch(keras.callbacks.Callback):
    """Call ``dataset.set_epoch(epoch)`` when each epoch starts.

    A dataset that counted epochs itself would restart at 0 after ``BackupAndRestore``
    resumes a run (and replay the first epoch's shuffle); Keras passes the true epoch here.
    """

    def __init__(self, dataset: Any) -> None:
        super().__init__()
        self.dataset = dataset

    def on_epoch_begin(self, epoch, logs=None) -> None:
        self.dataset.set_epoch(epoch)


def pad_rows(rows: np.ndarray, batch_size: int) -> tuple[np.ndarray, int]:
    """Pad a partial batch of row indices to ``batch_size`` by repeating the last row.

    Returns ``(rows, n_real)``; give rows ``n_real:`` zero weight (or drop their outputs).
    Data parallelism splits every batch evenly across devices, so batches are always full.
    """
    rows = np.asarray(rows)
    n_real = len(rows)
    if 0 < n_real < batch_size:
        rows = np.concatenate([rows, np.repeat(rows[-1:], batch_size - n_real)])
    return rows, n_real


def configure_devices(mixed_precision: str, *batch_sizes: int) -> dict[str, Any]:
    """Set the dtype policy and, with several GPUs, data parallelism. Returns device info.

    Every ``batch_sizes`` value must divide evenly across the devices.
    """
    keras.mixed_precision.set_dtype_policy(mixed_precision)
    devices = jax.devices()
    for bs in batch_sizes:
        if bs % len(devices):
            raise ValueError(f"batch size {bs} is not divisible by the {len(devices)} devices")
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
    """``git rev-parse HEAD``, or ``None`` outside a git checkout."""
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
