"""Stage 7b: train a Model 1 student (A/B/C/D) for one or all seeds.

Usage::

    uv run python -m src.model1_scorer.train --config configs/model1_student_A.yaml --all-seeds

Students differ only by their config: targets (7B teacher, 1.5B teacher, oracle), whether the
tag head is trained, and the initialisation (MLM ``encoder.keras`` or random). Only dialogues
with labels are used. Early stopping and checkpoint selection use ``val_spearman``: per-dialogue
Spearman between predicted importance and the student's own validation targets.

Per seed:
    ``<paths.output_dir>/seed_<n>/``          ``best.keras``, ``backup/`` (while running)
    ``<paths.results_dir>/<name>/seed_<n>/``  ``history.csv``, ``metrics.json`` (validation),
                                              ``test_predictions.jsonl`` (scores and tags per
                                              test utterance, for :mod:`evaluate`),
                                              ``manifest.json``
This module never imports ROUGE or BERTScore: summary metrics are computed by :mod:`evaluate`.
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any

import keras
import numpy as np

from src.config import load_config
from src.data.io import write_jsonl
from src.eval.metrics import mean_spearman
from src.layers.losses import label_weights, masked_sparse_ce, safe_labels
from src.model1_scorer.data import TAGS, StudentSet, Targets, load_targets
from src.model1_scorer.losses import ImportanceLoss
from src.model1_scorer.model import build_student, load_encoder_weights
from src.seed import set_seed
from src.training import (
    SyncEpoch,
    TimeGuard,
    best_logged_value,
    build_optimizer,
    commit_sha,
    configure_devices,
    pad_rows,
    steps_for,
    utc_now,
    write_json,
)

log = logging.getLogger(__name__)


def run_paths(cfg: dict[str, Any], seed: int) -> dict[str, Path]:
    """Checkpoint and results locations for one seed."""
    out = Path(cfg["paths"]["output_dir"]) / f"seed_{seed}"
    results = Path(cfg["paths"]["results_dir"]) / cfg["name"] / f"seed_{seed}"
    return {
        "checkpoints": out,
        "backup": out / "backup",
        "best_model": out / "best.keras",
        "results": results,
        "history": results / "history.csv",
    }


def model_inputs(data: StudentSet, rows: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "token_ids": np.asarray(data.token_ids[rows], dtype=np.int32),
        "speaker_ids": np.asarray(data.speaker_ids[rows], dtype=np.int32),
        "cls_positions": np.asarray(data.cls_positions[rows], dtype=np.int32),
    }


class StudentDataset(keras.utils.PyDataset):
    """Batches of labelled dialogues ``(inputs, targets, sample_weights)``.

    ``rows`` are the dialogues to use; with ``shuffle`` they are reshuffled per epoch from
    ``(seed, epoch)`` (epoch set by :class:`~src.training.SyncEpoch`).
    """

    def __init__(
        self,
        data: StudentSet,
        targets: Targets,
        rows: np.ndarray,
        batch_size: int,
        use_tags: bool,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        max_batches: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.data, self.targets, self.rows = data, targets, np.asarray(rows)
        self.batch_size, self.use_tags, self.shuffle, self.seed = (
            batch_size,
            use_tags,
            shuffle,
            seed,
        )
        n = len(self.rows)
        n_batches = n // batch_size if drop_last else math.ceil(n / batch_size)
        if n_batches == 0:
            raise ValueError(f"{n} dialogues is fewer than one batch of {batch_size}")
        self.n_batches = n_batches if max_batches is None else min(n_batches, max_batches)
        self.epoch = 0
        self._order = self._make_order()

    def _make_order(self) -> np.ndarray:
        if not self.shuffle:
            return self.rows
        return self.rows[np.random.default_rng([self.seed, self.epoch]).permutation(len(self.rows))]

    def set_epoch(self, epoch: int) -> None:
        if epoch != self.epoch:
            self.epoch = epoch
            self._order = self._make_order()

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, index: int):
        rows = np.sort(self._order[index * self.batch_size : (index + 1) * self.batch_size])
        rows, n_real = pad_rows(rows, self.batch_size)
        y = {"importance": self.targets.scores[rows]}
        sw = {"importance": np.ones(len(rows), dtype=np.float32)}
        if self.use_tags:
            tags = self.targets.tags[rows]
            y["tags"], sw["tags"] = safe_labels(tags), label_weights(tags)
        for weights in sw.values():
            weights[n_real:] = 0.0  # padding rows of a partial last batch do not count
        return model_inputs(self.data, rows), y, sw


def predict_utterances(
    model: keras.Model, data: StudentSet, rows: np.ndarray, batch_size: int
) -> list[dict[str, list]]:
    """Per dialogue: importance scores (sigmoid) and, if the model has them, tag names."""
    out: list[dict[str, list]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        padded, _ = pad_rows(batch, batch_size)
        pred = model.predict(model_inputs(data, padded), batch_size=batch_size, verbose=0)
        scores = 1.0 / (1.0 + np.exp(-np.asarray(pred["importance"], dtype=np.float64)))
        tags = np.argmax(pred["tags"], axis=-1) if "tags" in pred else None
        for j, i in enumerate(batch):
            k = int(data.n_utterances[i])
            entry = {"scores": [round(float(s), 6) for s in scores[j, :k]]}
            if tags is not None:
                entry["tags"] = [TAGS[t] for t in tags[j, :k]]
            out.append(entry)
    return out


def validation_spearman(model, data: StudentSet, targets: Targets, rows, batch_size) -> float:
    """Mean per-dialogue Spearman between predicted importance and the targets."""
    preds = predict_utterances(model, data, rows, batch_size)
    truth = [targets.scores[i, : int(data.n_utterances[i])] for i in rows]
    return mean_spearman([p["scores"] for p in preds], truth)


class ValidationSpearman(keras.callbacks.Callback):
    """Adds ``val_spearman`` to the epoch logs (before checkpoint / early stopping read it)."""

    def __init__(self, data: StudentSet, targets: Targets, rows: np.ndarray, batch_size: int):
        super().__init__()
        self.args = (data, targets, rows, batch_size)

    def on_epoch_end(self, epoch, logs=None) -> None:
        data, targets, rows, batch_size = self.args
        value = validation_spearman(self.model, data, targets, rows, batch_size)
        if logs is not None:
            logs["val_spearman"] = value


def build_compiled_student(cfg: dict[str, Any], total_steps: int) -> keras.Model:
    """Student with MLM or random init, compiled with the importance (+ tag) losses."""
    use_tags = cfg["targets"]["use_tags"]
    model = build_student(cfg["model"], use_tags)
    if cfg["init"] == "mlm":
        load_encoder_weights(model, cfg["paths"]["init_checkpoint"])
    elif cfg["init"] != "random":
        raise ValueError(f"Unknown init: {cfg['init']!r}")
    loss_cfg = cfg["loss"]
    losses = {
        "importance": ImportanceLoss(
            loss_cfg["listwise_lambda"], loss_cfg["temperature"], loss_cfg["score_scale"]
        )
    }
    weights = {"importance": 1.0}
    if use_tags:
        losses["tags"], weights["tags"] = masked_sparse_ce(), loss_cfg["tag_weight"]
    model.compile(
        optimizer=build_optimizer(cfg["optimizer"], total_steps), loss=losses, loss_weights=weights
    )
    return model


def train_seed(cfg: dict[str, Any], seed: int, session_start: float | None = None) -> dict:
    """Train (or resume) one seed, then write validation metrics and test predictions."""
    paths = run_paths(cfg, seed)
    if (paths["results"] / "metrics.json").exists():
        raise FileExistsError(f"{paths['results']} is complete; use a new experiment name.")
    tr, es = cfg["training"], cfg["training"]["early_stopping"]
    start, started = time.time(), utc_now()
    set_seed(seed)
    device_info = configure_devices(tr["mixed_precision"], tr["batch_size"])

    processed, labels_dir = Path(cfg["paths"]["processed_dir"]), cfg["paths"]["labels_dir"]
    source = cfg["targets"]["source"]
    train_data, val_data = (
        StudentSet.load(processed / "train"),
        StudentSet.load(processed / "validation"),
    )
    train_t = load_targets(labels_dir, "train", train_data, source)
    val_t = load_targets(labels_dir, "validation", val_data, source)
    train_rows, val_rows = np.flatnonzero(train_t.labelled), np.flatnonzero(val_t.labelled)
    spe, epochs, total = steps_for(len(train_rows), tr["batch_size"], tr["epochs"], tr["max_steps"])
    use_tags = cfg["targets"]["use_tags"]
    train_ds = StudentDataset(
        train_data, train_t, train_rows, tr["batch_size"], use_tags, seed=seed, max_batches=spe
    )
    val_ds = StudentDataset(
        val_data, val_t, val_rows, tr["batch_size"], use_tags, shuffle=False, drop_last=False
    )

    model = build_compiled_student(cfg, total)
    log.info(
        "%s seed %d: %d params, %d labelled train dialogues, %d steps/epoch x %d epochs",
        cfg["name"],
        seed,
        model.count_params(),
        len(train_rows),
        spe,
        epochs,
    )
    paths["checkpoints"].mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)
    guard = TimeGuard(cfg["max_hours"], session_start)
    eval_bs = tr["batch_size"]
    callbacks = [
        SyncEpoch(train_ds),
        ValidationSpearman(val_data, val_t, val_rows, eval_bs),
        guard,
        keras.callbacks.ModelCheckpoint(
            str(paths["best_model"]),
            monitor=es["monitor"],
            mode=es["mode"],
            save_best_only=True,
            initial_value_threshold=best_logged_value(paths["history"], es["monitor"], es["mode"]),
        ),
        keras.callbacks.EarlyStopping(
            monitor=es["monitor"], mode=es["mode"], patience=es["patience"]
        ),
        keras.callbacks.BackupAndRestore(str(paths["backup"]), delete_checkpoint=False),
        keras.callbacks.CSVLogger(str(paths["history"]), append=True),
    ]
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2
    )
    manifest = {
        "stage": "model1_student",
        "name": cfg["name"],
        "seed": seed,
        "commit": commit_sha(),
        "config": cfg,
        "device": device_info,
        "params": model.count_params(),
        "train_dialogues": len(train_rows),
        "validation_dialogues": len(val_rows),
        "steps_per_epoch": spe,
        "epochs_this_session": len(history.history.get("loss", [])),
        "started": started,
        "finished": utc_now(),
        "seconds": round(time.time() - start, 1),
    }
    if guard.stopped:
        manifest["status"] = "interrupted"
        write_json(paths["results"] / "manifest.json", manifest)
        log.warning("seed %d interrupted by the time guard; rerun to resume.", seed)
        return manifest

    shutil.rmtree(paths["backup"], ignore_errors=True)
    best = keras.saving.load_model(paths["best_model"])
    test_data = StudentSet.load(processed / "test")
    preds = predict_utterances(best, test_data, np.arange(len(test_data)), eval_bs)
    write_jsonl(
        paths["results"] / "test_predictions.jsonl",
        ({"id": i, **p} for i, p in zip(test_data.ids, preds, strict=True)),
    )
    val_spearman = validation_spearman(best, val_data, val_t, val_rows, eval_bs)
    write_json(
        paths["results"] / "metrics.json",
        {"seed": seed, "params": best.count_params(), "validation": {"spearman": val_spearman}},
    )
    manifest.update(status="complete", seconds=round(time.time() - start, 1))
    write_json(paths["results"] / "manifest.json", manifest)
    log.info("%s seed %d done: val Spearman %.4f", cfg["name"], seed, val_spearman)
    return manifest


def main(argv: list[str] | None = None) -> None:
    """Train one seed (``--seed``) or every config seed not yet complete (``--all-seeds``)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int)
    group.add_argument("--all-seeds", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    session_start = time.time()
    if not args.all_seeds:
        train_seed(cfg, args.seed, session_start)
        return
    for seed in cfg["seeds"]:
        if (run_paths(cfg, seed)["results"] / "metrics.json").exists():
            log.info("seed %d already complete, skipping", seed)
            continue
        if train_seed(cfg, seed, session_start)["status"] == "interrupted":
            break


if __name__ == "__main__":
    main()
