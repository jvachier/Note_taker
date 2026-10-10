"""Stage 4b: train Model 2 for one or all seeds, then evaluate on the test sets.

Usage::

    uv run python -m src.model2_cleanup.train --config configs/model2.yaml --seed 0
    uv run python -m src.model2_cleanup.train --config configs/model2.yaml --all-seeds

Per seed:
    ``<paths.output_dir>/seed_<n>/``           ``best.keras`` (best validation punct macro-F1),
                                               ``backup/`` (BackupAndRestore, for resuming)
    ``<paths.results_dir>/<name>/seed_<n>/``   ``history.csv``, ``metrics.json``, ``manifest.json``
A seed whose ``metrics.json`` exists is complete and never overwritten. A run stopped by the
time guard writes ``manifest.json`` with ``status: interrupted``; rerunning resumes it.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import keras

from src.config import load_config
from src.layers.losses import masked_sparse_ce
from src.model2_cleanup.build_data import validation_set_name
from src.model2_cleanup.dataset import ArraySet, CleanupDataset
from src.model2_cleanup.evaluate import evaluate_all, predict, run_paths, score
from src.model2_cleanup.model import build_cleanup_model
from src.seed import set_seed
from src.tokenizer.tokenizer import load_tokenizer
from src.training import (
    SyncEpoch,
    TimeGuard,
    best_logged_value,
    build_optimizer,
    commit_sha,
    configure_devices,
    steps_for,
    utc_now,
    write_json,
)

log = logging.getLogger(__name__)


class ValidationMetrics(keras.callbacks.Callback):
    """Adds word-level ``val_punct_macro_f1`` and casing scores to the epoch logs.

    Must come before the callbacks that read these keys (checkpoint, early stopping, CSV).
    """

    def __init__(self, arrays: ArraySet, batch_size: int) -> None:
        super().__init__()
        self.arrays = arrays
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None) -> None:
        report = score(self.arrays, predict(self.model, self.arrays.token_ids, self.batch_size))
        logs = logs if logs is not None else {}
        logs["val_punct_macro_f1"] = report["punct"]["macro_f1"]
        logs["val_casing_accuracy"] = report["casing"]["accuracy"]
        logs["val_casing_macro_f1"] = report["casing"]["macro_f1"]


def load_validation(cfg: dict[str, Any]) -> ArraySet:
    """All validation sets concatenated (early stopping uses the combined score)."""
    processed = Path(cfg["paths"]["processed_dir"])
    sources = cfg["data"]["validation_sources"]
    return ArraySet.concat(
        [ArraySet.load(processed / validation_set_name(s), mmap=False) for s in sources]
    )


def punct_class_weights(cfg: dict[str, Any]) -> list[float] | None:
    """Class weights from the build step's meta.json, or ``None`` if disabled."""
    if cfg["loss"]["punct_class_weighting"] == "none":
        return None
    if cfg["loss"]["punct_class_weighting"] != "inverse_sqrt_freq":
        raise ValueError(f"Unknown punct_class_weighting: {cfg['loss']['punct_class_weighting']}")
    meta = json.loads((Path(cfg["paths"]["processed_dir"]) / "meta.json").read_text())
    return meta["punct_class_weights"]


def train_seed(
    cfg: dict[str, Any], seed: int, session_start: float | None = None
) -> dict[str, Any]:
    """Train (or resume) one seed and evaluate it. Returns the manifest.

    ``session_start`` (a ``time.time()`` value) makes ``max_hours`` a budget for the whole
    session rather than for this seed.
    """
    paths = run_paths(cfg, seed)
    if (paths["results"] / "metrics.json").exists():
        raise FileExistsError(f"{paths['results']} is complete; use a new experiment name.")
    tr, es_cfg = cfg["training"], cfg["training"]["early_stopping"]
    start, started = time.time(), utc_now()
    set_seed(seed)
    device_info = configure_devices(
        tr["mixed_precision"], tr["batch_size"], cfg["eval"]["batch_size"]
    )

    weights = punct_class_weights(cfg)
    train_arrays = ArraySet.load(Path(cfg["paths"]["processed_dir"]) / "train")
    val_arrays = load_validation(cfg)
    spe, epochs, total_steps = steps_for(
        len(train_arrays), tr["batch_size"], tr["epochs"], tr["max_steps"], tr["steps_per_epoch"]
    )
    train_ds = CleanupDataset(train_arrays, tr["batch_size"], weights, seed=seed, max_batches=spe)
    val_ds = CleanupDataset(val_arrays, tr["batch_size"], weights, shuffle=False, drop_last=False)

    model = build_cleanup_model(cfg["model"])
    model.compile(
        optimizer=build_optimizer(cfg["optimizer"], total_steps),
        loss={"casing": masked_sparse_ce(), "punct": masked_sparse_ce()},
        loss_weights={"casing": cfg["loss"]["casing_weight"], "punct": cfg["loss"]["punct_weight"]},
    )
    log.info(
        "seed %d: %d params, %d steps/epoch x %d epochs", seed, model.count_params(), spe, epochs
    )

    paths["checkpoints"].mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)
    guard = TimeGuard(cfg["max_hours"], session_start)
    callbacks = [
        SyncEpoch(train_ds),
        ValidationMetrics(val_arrays, cfg["eval"]["batch_size"]),
        guard,
        keras.callbacks.ModelCheckpoint(
            str(paths["best_model"]),
            monitor=es_cfg["monitor"],
            mode=es_cfg["mode"],
            save_best_only=True,
            initial_value_threshold=best_logged_value(
                paths["history"], es_cfg["monitor"], es_cfg["mode"]
            ),
        ),
        keras.callbacks.EarlyStopping(
            monitor=es_cfg["monitor"], mode=es_cfg["mode"], patience=es_cfg["patience"]
        ),
        # Keras deletes the backup on *any* train end (including a time-guard stop), so the
        # backup is kept here and removed below only once the seed is complete.
        keras.callbacks.BackupAndRestore(str(paths["backup"]), delete_checkpoint=False),
        keras.callbacks.CSVLogger(str(paths["history"]), append=True),
    ]
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2
    )

    manifest = {
        "stage": "model2_train",
        "name": cfg["name"],
        "seed": seed,
        "commit": commit_sha(),
        "config": cfg,
        "data_meta": {
            "train": train_arrays.meta,
            "validation": val_arrays.meta,
        },
        "device": device_info,
        "params": model.count_params(),
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
    tokenizer = load_tokenizer(cfg["paths"]["tokenizer_dir"])
    val_report = score(val_arrays, predict(best, val_arrays.token_ids, cfg["eval"]["batch_size"]))
    metrics = {
        "seed": seed,
        "params": model.count_params(),
        "validation": val_report,
        "test": evaluate_all(best, cfg, tokenizer),
    }
    write_json(paths["results"] / "metrics.json", metrics)
    manifest["status"] = "complete"
    manifest["seconds"] = round(time.time() - start, 1)
    write_json(paths["results"] / "manifest.json", manifest)
    log.info("seed %d done: val punct macro-F1 %.4f", seed, val_report["punct"]["macro_f1"])
    return manifest


def main(argv: list[str] | None = None) -> None:
    """Train one seed (``--seed``) or every config seed not yet complete (``--all-seeds``)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/model2.yaml"))
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
            break  # out of time for this session


if __name__ == "__main__":
    main()
