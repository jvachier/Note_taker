"""Stage 5b: MLM pretraining of the Model 1 encoder (PLAN.md section 5.1).

Usage::

    uv run python -m src.model1_scorer.mlm_train --config configs/mlm.yaml

Masking (dynamic, per batch): ``mask_prob`` of the non-special tokens of each row are chosen
(exact count, at most ``n_pred = ceil(mask_prob * seq_len)``); of those, ``mask_token_frac``
become ``[MASK]``, ``random_token_frac`` a random non-special token, the rest stay unchanged.
Held-out sets use fixed masks (same every evaluation).

Step budget: with ``training.max_steps: null`` a short throughput probe sets
``max_steps = target_hours * steps/s``; the choice is stored in ``<output_dir>/plan.json`` and
reused when a cut-off run resumes. An "epoch" is ``steps_per_epoch`` steps (the checkpoint and
evaluation interval).

Outputs:
    ``<output_dir>/encoder.keras``     encoder only, updated every epoch (what students load)
    ``<output_dir>/mlm.keras``         encoder + MLM head, at the end
    ``<output_dir>/backup/``           BackupAndRestore state, removed when complete
    ``<results_dir>/<name>/``          ``history.csv``, ``metrics.json`` (held-out MLM loss and
                                       accuracy per set), ``manifest.json``
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any

import keras
import numpy as np

from src.config import load_config
from src.layers.losses import masked_sparse_ce
from src.model1_scorer.mlm_model import build_mlm_model, encoder_model
from src.seed import set_seed
from src.tokenizer.tokenizer import MASK_ID, SPECIAL_TOKENS
from src.training import (
    SyncEpoch,
    TimeGuard,
    build_optimizer,
    commit_sha,
    configure_devices,
    pad_rows,
    utc_now,
    write_json,
)

log = logging.getLogger(__name__)
FIRST_REGULAR_ID = len(SPECIAL_TOKENS)  # IDs below this are special tokens, never masked


def num_predictions(seq_len: int, mask_prob: float) -> int:
    """Fixed number of prediction slots per row."""
    return math.ceil(mask_prob * seq_len)


def mask_batch(
    token_ids: np.ndarray, data_cfg: dict[str, Any], vocab_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mask a batch of rows.

    Returns ``(inputs, positions, labels, weights)``: masked token IDs (batch, length), and per
    prediction slot (batch, n_pred) the position, the original token and a 0/1 weight (0 for
    unused slots).
    """
    batch, length = token_ids.shape
    n_pred = num_predictions(length, data_cfg["mask_prob"])
    inputs = token_ids.copy()
    positions = np.zeros((batch, n_pred), dtype=np.int32)
    labels = np.zeros((batch, n_pred), dtype=np.int32)
    weights = np.zeros((batch, n_pred), dtype=np.float32)
    p_mask = data_cfg["mask_token_frac"]
    p_random = data_cfg["random_token_frac"]
    for row in range(batch):
        eligible = np.flatnonzero(token_ids[row] >= FIRST_REGULAR_ID)
        k = min(n_pred, round(data_cfg["mask_prob"] * len(eligible)))
        if len(eligible) and k == 0:
            k = 1
        chosen = np.sort(rng.choice(eligible, size=k, replace=False)) if k else eligible[:0]
        positions[row, :k] = chosen
        labels[row, :k] = token_ids[row, chosen]
        weights[row, :k] = 1.0
        u = rng.random(k)
        inputs[row, chosen[u < p_mask]] = MASK_ID
        random_slots = chosen[(u >= p_mask) & (u < p_mask + p_random)]
        inputs[row, random_slots] = rng.integers(FIRST_REGULAR_ID, vocab_size, len(random_slots))
    return inputs, positions, labels, weights


class MLMDataset(keras.utils.PyDataset):
    """Masked batches from memory-mapped rows.

    Training (``fixed=False``): global batch ``g = epoch * steps_per_epoch + index`` walks a
    fresh permutation of the rows on every pass, and masks are drawn from ``(seed, g)``; the
    epoch is set by :class:`~src.training.SyncEpoch`, so a resumed run continues the same
    sequence.
    Evaluation (``fixed=True``): rows in order, masks from ``(seed, index)``, identical every
    time.
    """

    def __init__(
        self,
        token_ids: np.ndarray,
        speaker_ids: np.ndarray,
        batch_size: int,
        data_cfg: dict[str, Any],
        vocab_size: int,
        seed: int = 0,
        fixed: bool = False,
        steps_per_epoch: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.token_ids, self.speaker_ids = token_ids, speaker_ids
        self.batch_size, self.data_cfg, self.vocab_size = batch_size, data_cfg, vocab_size
        self.seed, self.fixed, self.epoch = seed, fixed, 0
        n = len(token_ids)
        self.batches_per_pass = math.ceil(n / batch_size) if fixed else n // batch_size
        if self.batches_per_pass == 0:
            raise ValueError(f"{n} rows is fewer than one batch of {batch_size}")
        self.steps_per_epoch = steps_per_epoch or self.batches_per_pass
        self._orders: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.batches_per_pass if self.fixed else self.steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Training position (called by :class:`~src.training.SyncEpoch`)."""
        self.epoch = epoch

    def _rows(self, g: int) -> np.ndarray:
        if self.fixed:
            end = min((g + 1) * self.batch_size, len(self.token_ids))
            return np.arange(g * self.batch_size, end)
        n_pass, i = divmod(g, self.batches_per_pass)
        if n_pass not in self._orders:
            self._orders = {
                n_pass: np.random.default_rng([self.seed, n_pass]).permutation(len(self.token_ids))
            }
        return np.sort(self._orders[n_pass][i * self.batch_size : (i + 1) * self.batch_size])

    def __getitem__(self, index: int):
        g = index if self.fixed else self.epoch * self.steps_per_epoch + index
        rows, n_real = pad_rows(self._rows(g), self.batch_size)
        ids = np.asarray(self.token_ids[rows], dtype=np.int32)
        rng = np.random.default_rng([self.seed, g])
        inputs, positions, labels, weights = mask_batch(ids, self.data_cfg, self.vocab_size, rng)
        weights[n_real:] = 0.0  # padding rows of a partial last batch do not count
        x = {
            "token_ids": inputs,
            "speaker_ids": np.asarray(self.speaker_ids[rows], dtype=np.int32),
            "masked_positions": positions,
        }
        return x, labels, weights


class SaveEncoder(keras.callbacks.Callback):
    """Save the encoder alone as ``encoder.keras`` at the end of every epoch."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def on_epoch_end(self, epoch, logs=None) -> None:
        encoder_model(self.model).save(self.path)


def load_set(processed_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    d = processed_dir / name
    return (
        np.load(d / "token_ids.npy", mmap_mode="r"),
        np.load(d / "speaker_ids.npy", mmap_mode="r"),
    )


def compile_model(
    model: keras.Model, optimizer: keras.optimizers.Optimizer, steps_per_execution: int = 1
) -> None:
    model.compile(
        optimizer=optimizer,
        loss=masked_sparse_ce(),
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
        steps_per_execution=steps_per_execution,
    )


class _EpochTimer(keras.callbacks.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.durations: list[float] = []

    def on_epoch_begin(self, epoch, logs=None) -> None:
        self._t = time.time()

    def on_epoch_end(self, epoch, logs=None) -> None:
        self.durations.append(time.time() - self._t)


def measure_throughput(cfg: dict[str, Any], train: MLMDataset) -> float:
    """Training steps per second, timed on a second epoch of ``probe_steps`` (after compile)."""
    probe = cfg["training"]["probe_steps"]
    model = build_mlm_model(cfg["model"])
    tr = cfg["training"]
    compile_model(model, keras.optimizers.AdamW(learning_rate=1e-5), tr["steps_per_execution"])
    ds = MLMDataset(
        train.token_ids,
        train.speaker_ids,
        train.batch_size,
        train.data_cfg,
        train.vocab_size,
        seed=train.seed,
        steps_per_epoch=probe,
        workers=tr["workers"],
    )
    timer = _EpochTimer()
    model.fit(ds, epochs=2, callbacks=[timer], verbose=0)
    return probe / timer.durations[-1]


def plan_steps(cfg: dict[str, Any], train: MLMDataset, out_dir: Path) -> dict[str, Any]:
    """Total steps for the run: config value, earlier plan (resume), or throughput probe."""
    plan_path = out_dir / "plan.json"
    if plan_path.exists():
        return json.loads(plan_path.read_text())
    tr = cfg["training"]
    if tr["max_steps"] is not None:
        plan = {"max_steps": int(tr["max_steps"]), "source": "config"}
    else:
        rate = measure_throughput(cfg, train)
        steps = int(tr["target_hours"] * 3600 * rate)
        plan = {
            "max_steps": steps,
            "source": "throughput",
            "steps_per_second": round(rate, 3),
            "target_hours": tr["target_hours"],
        }
        log.info("throughput %.2f steps/s -> max_steps %d", rate, steps)
    write_json(plan_path, plan)
    return plan


def evaluate_held_out(model: keras.Model, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """MLM loss and accuracy on every held-out set (fixed masks)."""
    processed = Path(cfg["paths"]["processed_dir"])
    results = {}
    for name in cfg["eval"]["held_out"]:
        ids, speakers = load_set(processed, name)
        ds = MLMDataset(
            ids,
            speakers,
            cfg["training"]["batch_size"],
            cfg["data"],
            cfg["model"]["vocab_size"],
            seed=seed,
            fixed=True,
        )
        scores = model.evaluate(ds, return_dict=True, verbose=0)
        results[name] = {"loss": scores["loss"], "accuracy": scores["acc"], "rows": len(ids)}
        log.info("%s: MLM loss %.4f, accuracy %.4f", name, scores["loss"], scores["acc"])
    return results


def train(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Pretrain (or resume) and evaluate. Returns the manifest."""
    paths, tr = cfg["paths"], cfg["training"]
    out = Path(paths["output_dir"])
    results = Path(paths["results_dir"]) / cfg["name"]
    if (results / "metrics.json").exists():
        raise FileExistsError(f"{results} is complete; use a new experiment name.")
    start, started = time.time(), utc_now()
    set_seed(seed)
    device_info = configure_devices(tr["mixed_precision"], tr["batch_size"])
    processed = Path(paths["processed_dir"])
    vocab = cfg["model"]["vocab_size"]

    train_ids, train_speakers = load_set(processed, "train")
    train_ds = MLMDataset(
        train_ids,
        train_speakers,
        tr["batch_size"],
        cfg["data"],
        vocab,
        seed=seed,
        steps_per_epoch=tr["steps_per_epoch"],
        workers=tr["workers"],
    )
    out.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    plan = plan_steps(cfg, train_ds, out)
    steps_per_epoch = min(tr["steps_per_epoch"], plan["max_steps"])
    train_ds.steps_per_epoch = steps_per_epoch
    epochs = math.ceil(plan["max_steps"] / steps_per_epoch)
    total_steps = steps_per_epoch * epochs

    val_parts = [load_set(processed, name) for name in cfg["eval"]["held_out"]]
    val_ds = MLMDataset(
        np.concatenate([p[0] for p in val_parts]),
        np.concatenate([p[1] for p in val_parts]),
        tr["batch_size"],
        cfg["data"],
        vocab,
        seed=seed,
        fixed=True,
    )

    set_seed(seed)  # the probe consumed random state
    model = build_mlm_model(cfg["model"])
    compile_model(model, build_optimizer(cfg["optimizer"], total_steps), tr["steps_per_execution"])
    log.info("%d params, %d steps/epoch x %d epochs", model.count_params(), steps_per_epoch, epochs)
    guard = TimeGuard(cfg["max_hours"])
    callbacks = [
        SyncEpoch(train_ds),
        guard,
        SaveEncoder(out / "encoder.keras"),
        keras.callbacks.BackupAndRestore(str(out / "backup"), delete_checkpoint=False),
        keras.callbacks.CSVLogger(str(results / "history.csv"), append=True),
    ]
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2
    )
    manifest = {
        "stage": "mlm_pretrain",
        "name": cfg["name"],
        "seed": seed,
        "commit": commit_sha(),
        "config": cfg,
        "plan": plan,
        "device": device_info,
        "params": model.count_params(),
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "epochs_this_session": len(history.history.get("loss", [])),
        "started": started,
        "finished": utc_now(),
        "seconds": round(time.time() - start, 1),
    }
    if guard.stopped:
        manifest["status"] = "interrupted"
        write_json(results / "manifest.json", manifest)
        log.warning("Interrupted by the time guard; rerun to resume.")
        return manifest
    model.save(out / "mlm.keras")
    encoder_model(model).save(out / "encoder.keras")
    shutil.rmtree(out / "backup", ignore_errors=True)
    held_out = evaluate_held_out(model, cfg, seed)
    metrics = {"seed": seed, "params": model.count_params(), "held_out": held_out}
    write_json(results / "metrics.json", metrics)
    manifest.update(status="complete", seconds=round(time.time() - start, 1))
    write_json(results / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    """Entry point: pretrain from a config file (first seed unless ``--seed``)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/mlm.yaml"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    train(cfg, cfg["seeds"][0] if args.seed is None else args.seed)


if __name__ == "__main__":
    main()
