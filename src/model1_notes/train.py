"""Model 1 v3: train the meeting-notes student (E4 / E1 / tiny) for one or all seeds.

Usage::

    uv run python -m src.model1_notes.train --config configs/model1_v3_E4.yaml --all-seeds

Batches are windows of meetings (``src.model1_notes.data``), grouped by sentence-count bucket so
XLA compiles one shape per bucket. Training windows of hard-negative meetings are repeated
``data.hard_negative_oversample`` times per epoch.

Teacher targets (E4) are read from the Notebook 1 export (PLAN.md 10.3), one Parquet table per
teacher and split, keyed by ``(note_id, sentence_id)``:
    ``<paths.teacher_dir>/qwen/<split>.parquet``     ``score`` (soft, [0, 1]), ``reason``
                                                     (:data:`~src.model1_notes.data.REASONS`),
                                                     ``depends_on`` (a ``sentence_id``, -1 = none)
    ``<paths.teacher_dir>/roberta/<split>.parquet``  ``logit``
and aligned to the processed sentence rows (:func:`align_teacher`). Sentences a teacher did not
label, or a missing table, mean "no target". Teacher losses with weight 0 are not compiled.

Per seed:
    ``<paths.output_dir>/seed_<n>/``          ``best.keras``, ``backup/`` (while running)
    ``<paths.results_dir>/<name>/seed_<n>/``  ``history.csv``, ``metrics.json`` (validation,
                                              threshold tuned for importance F1),
                                              ``test_predictions.jsonl``, ``manifest.json``

This module never imports ROUGE or BERTScore.
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
import pyarrow.parquet as pq

from src.config import load_config
from src.data.io import write_jsonl
from src.layers.losses import (
    IGNORE,
    inverse_sqrt_frequency_weights,
    label_weights,
    masked_sparse_ce,
    safe_labels,
)
from src.model1_notes.calibrate import assign_zones, tune_zone_thresholds
from src.model1_notes.data import (
    REASONS,
    SOURCES,
    ZONES,
    NotesSet,
    bucket_for,
    gather_windows,
    sentence_rows,
)
from src.model1_notes.losses import NotesImportanceLoss, ZoneLoss
from src.model1_notes.metrics import f1_score, selection_metrics, span_f1, tune_threshold  # noqa: F401
from src.model1_notes.model import build_notes_model, load_sentence_encoder
from src.seed import set_seed
from src.training import (
    SyncEpoch,
    TimeGuard,
    best_logged_value,
    build_optimizer,
    commit_sha,
    configure_devices,
    pad_rows,
    utc_now,
    write_json,
)

log = logging.getLogger(__name__)
AMI = SOURCES.index("ami")


def empty_teacher(n_rows: int) -> dict[str, np.ndarray]:
    """Per-row teacher arrays with no labels."""
    return {
        "qwen_score": np.full(n_rows, -1.0, dtype=np.float32),
        "qwen_reason": np.full(n_rows, IGNORE, dtype=np.int8),
        "qwen_depends": np.full(n_rows, -2, dtype=np.int64),  # row; -1 = none, -2 = unlabelled
        "roberta_logit": np.full(n_rows, np.nan, dtype=np.float32),
    }


def align_teacher(
    dataset: NotesSet, teacher_dir: str | Path | None, split: str, teachers: list[str]
) -> dict[str, np.ndarray]:
    """Teacher targets for every sentence row of ``dataset`` (see module docstring)."""
    arrays = empty_teacher(len(dataset["importance"]))
    if teacher_dir is None:
        return arrays
    rows = sentence_rows(dataset)
    path = Path(teacher_dir) / "qwen" / f"{split}.parquet"
    if "qwen" in teachers and path.exists():
        for r in pq.read_table(path).to_pylist():
            row = rows.get((r["note_id"], int(r["sentence_id"])))
            if row is None:
                continue
            arrays["qwen_score"][row] = r["score"]
            reason = r["reason"]
            arrays["qwen_reason"][row] = REASONS.index(reason) if reason in REASONS else IGNORE
            dep = int(r["depends_on"])
            arrays["qwen_depends"][row] = -1 if dep < 0 else rows.get((r["note_id"], dep), -1)
    path = Path(teacher_dir) / "roberta" / f"{split}.parquet"
    if "roberta" in teachers and path.exists():
        for r in pq.read_table(path).to_pylist():
            row = rows.get((r["note_id"], int(r["sentence_id"])))
            if row is not None:
                arrays["roberta_logit"][row] = r["logit"]
    return arrays


def bucket_batch_sizes(buckets: list[int], sentences_per_batch: int) -> dict[int, int]:
    """Windows per batch for each bucket: a fixed sentence budget (``sentences_per_batch``)
    keeps the memory of a step about the same whatever the window length."""
    sizes = {b: max(1, sentences_per_batch // b) for b in buckets}
    if sentences_per_batch % max(buckets):
        raise ValueError(f"sentences_per_batch must be a multiple of {max(buckets)}")
    return sizes


def window_batches(
    dataset: NotesSet,
    windows: np.ndarray,
    buckets: list[int],
    sentences_per_batch: int,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, int]]:
    """``(window indices, bucket)`` batches; windows of a batch share the smallest bucket and a
    batch holds ``sentences_per_batch // bucket`` windows.

    With ``rng`` (training) windows are shuffled within each bucket and the batch order is
    shuffled; without, the order is deterministic. The last batch of a bucket may be partial.
    """
    lengths = np.asarray(dataset["window_length"])
    sizes = bucket_batch_sizes(buckets, sentences_per_batch)
    by_bucket: dict[int, list[int]] = {}
    for w in windows:
        by_bucket.setdefault(bucket_for(int(lengths[w]), buckets), []).append(int(w))
    batches = []
    for bucket in sorted(by_bucket):
        ws, bs = np.asarray(by_bucket[bucket]), sizes[bucket]
        if rng is not None:
            ws = ws[rng.permutation(len(ws))]
        batches += [(ws[i : i + bs], bucket) for i in range(0, len(ws), bs)]
    if rng is not None:
        batches = [batches[i] for i in rng.permutation(len(batches))]
    return batches


def depends_targets(rows: np.ndarray, depends_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pointer labels for a batch: position of the dependency in the window, or ``S`` = none.

    ``rows`` (B, S) are the sentence rows of each slot (``-1`` = padding). A dependency outside
    the window (or not earlier) becomes "none"; unlabelled sentences get weight 0.
    """
    b, s = rows.shape
    labels = np.full((b, s), s, dtype=np.int32)
    weights = np.zeros((b, s), dtype=np.float32)
    for i in range(b):
        position = {int(r): j for j, r in enumerate(rows[i]) if r >= 0}
        for j, r in enumerate(rows[i]):
            if r < 0 or depends_row[r] == -2:
                continue
            target = position.get(int(depends_row[r]), s)
            labels[i, j] = target if target < j else s
            weights[i, j] = 1.0
    return labels, weights


class NotesDataset(keras.utils.PyDataset):
    """Batches ``(inputs, targets, sample_weights)`` of meeting windows (see module docstring)."""

    def __init__(
        self,
        dataset: NotesSet,
        teacher: dict[str, np.ndarray],
        windows: np.ndarray,
        cfg: dict[str, Any],
        zone_weights: list[float] | None,
        shuffle: bool,
        seed: int = 0,
        max_batches: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset, self.teacher, self.windows = dataset, teacher, np.asarray(windows)
        self.buckets = cfg["data"]["sentence_buckets"]
        self.sentences_per_batch = cfg["training"]["sentences_per_batch"]
        self.zone_weights, self.shuffle, self.seed = zone_weights, shuffle, seed
        self.epoch = 0
        self.max_batches = max_batches
        self._batches = self._make_batches()

    def _make_batches(self) -> list[tuple[np.ndarray, int]]:
        rng = np.random.default_rng([self.seed, self.epoch]) if self.shuffle else None
        batches = window_batches(
            self.dataset, self.windows, self.buckets, self.sentences_per_batch, rng
        )
        return batches if self.max_batches is None else batches[: self.max_batches]

    def set_epoch(self, epoch: int) -> None:
        if epoch != self.epoch:
            self.epoch = epoch
            self._batches = self._make_batches()

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, index: int):
        windows, bucket = self._batches[index]
        windows, n_real = pad_rows(windows, max(1, self.sentences_per_batch // bucket))
        return batch_arrays(self.dataset, self.teacher, windows, bucket, self.zone_weights, n_real)


def batch_arrays(
    dataset: NotesSet,
    teacher: dict[str, np.ndarray],
    windows: np.ndarray,
    slots: int,
    zone_weights: list[float] | None = None,
    n_real: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Model inputs, targets and sample weights for one batch of windows."""
    g = gather_windows(dataset, windows, slots)
    mask, rows = g["sentence_mask"], g["rows"]
    safe = np.where(mask, rows, 0)
    is_ami = np.broadcast_to((g["source"] == AMI)[:, None], mask.shape)

    def teacher_values(name: str, fill: float) -> np.ndarray:
        return np.where(mask, teacher[name][safe], fill)

    qwen_p = teacher_values("qwen_score", -1.0)
    rob = teacher_values("roberta_logit", np.nan)
    importance = np.stack(
        [
            np.where(mask, g["importance"], 0),
            np.where(mask, g["confidence"], 0.0),
            np.clip(qwen_p, 0.0, 1.0),
            (qwen_p >= 0).astype(np.float32),
            np.nan_to_num(rob),
            (~np.isnan(rob)).astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)
    reason = teacher_values("qwen_reason", IGNORE).astype(np.int32)
    depends, depends_w = depends_targets(rows, teacher["qwen_depends"])
    x = {"token_ids": g["token_ids"], "speaker_ids": g["speaker"].astype(np.int32)}
    y = {
        "importance": importance,
        "zone": np.stack([safe_labels(g["zone"]), is_ami.astype(np.int32)], axis=-1),
        "reason": safe_labels(reason),
        "depends_on": depends,
        "spans": safe_labels(g["span_tags"]),
    }
    sw = {
        "importance": mask.astype(np.float32),
        "zone": label_weights(g["zone"], zone_weights),
        "reason": label_weights(reason),
        "depends_on": depends_w,
        "spans": label_weights(g["span_tags"]),
    }
    if n_real is not None:
        for weights in sw.values():
            weights[n_real:] = 0.0  # repeated windows of a partial batch do not count
    return x, y, sw


def train_windows(dataset: NotesSet, oversample: int) -> np.ndarray:
    """All windows, with windows of hard-negative meetings repeated ``oversample`` times."""
    windows = np.arange(dataset.num_windows)
    hn = np.asarray(dataset["source"])[np.asarray(dataset["window_note"])] == 1
    return np.concatenate([windows, *[windows[hn]] * (oversample - 1)])


def zone_class_weights(cfg: dict[str, Any], dataset: NotesSet) -> list[float] | None:
    """Inverse-sqrt-frequency zone weights from the train labels, or ``None``."""
    mode = cfg["loss"].get("zone_class_weights")
    if mode is None:
        return None
    if mode != "inverse_sqrt":
        raise ValueError(f"Unknown zone_class_weights: {mode!r}")
    counts = np.bincount(np.asarray(dataset["zone"]), minlength=len(ZONES))
    return inverse_sqrt_frequency_weights(counts.tolist())


def compile_model(cfg: dict[str, Any], total_steps: int) -> keras.Model:
    """Student (MLM or random sentence encoder) compiled with the configured losses."""
    data, loss_cfg = cfg["data"], cfg["loss"]
    model = build_notes_model(cfg["model"], data["max_sentences"], data["max_sentence_tokens"])
    checkpoint = cfg["paths"].get("init_checkpoint")
    if checkpoint:
        load_sentence_encoder(model, checkpoint)
    losses = {
        "importance": NotesImportanceLoss(
            loss_cfg["importance"],
            loss_cfg["qwen_kl"],
            loss_cfg["roberta_kl"],
            loss_cfg["temperature"],
        ),
        "zone": ZoneLoss(),
        "spans": masked_sparse_ce(),
        "reason": masked_sparse_ce(),
        "depends_on": masked_sparse_ce(),
    }
    weights = {
        "importance": 1.0,
        "zone": loss_cfg["zone"],
        "spans": loss_cfg["spans"],
        "reason": loss_cfg["reason"],
        "depends_on": loss_cfg["depends_on"],
    }
    losses = {k: v for k, v in losses.items() if weights[k]}
    opt = cfg["optimizer"]
    optimizer = build_optimizer(
        {
            "name": opt["name"],
            "schedule": opt["schedule"],
            "lr_heads": opt["lr_meeting"],
            "lr_encoder": opt["lr_sentence_encoder"],
            "warmup_fraction": opt["warmup_fraction"],
            "weight_decay": opt["weight_decay"],
        },
        total_steps,
    )
    model.compile(optimizer=optimizer, loss=losses, loss_weights={k: weights[k] for k in losses})
    return model


def predict_sentences(
    model: keras.Model, dataset: NotesSet, cfg: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Per sentence row: importance probability, zone probabilities and span tags per token.

    Sentences covered by several windows get the mean of their importance / zone predictions
    (span tags: the last window's).
    """
    n = len(dataset["importance"])
    prob, zones, seen = np.zeros(n), np.zeros((n, len(ZONES))), np.zeros(n)
    spans = np.zeros(np.asarray(dataset["span_tags"]).shape, dtype=np.int8)
    per_batch = cfg["training"]["sentences_per_batch"]
    teacher = empty_teacher(n)
    for windows, bucket in window_batches(
        dataset, np.arange(dataset.num_windows), cfg["data"]["sentence_buckets"], per_batch
    ):
        padded, n_real = pad_rows(windows, max(1, per_batch // bucket))
        x, _, _ = batch_arrays(dataset, teacher, padded, bucket)
        rows = gather_windows(dataset, padded[:n_real], bucket)["rows"]
        out = model.predict_on_batch(x)
        p = 1.0 / (1.0 + np.exp(-np.asarray(out["importance"], dtype=np.float64)))
        z = np.asarray(out["zone"], dtype=np.float64)
        z = np.exp(z - z.max(-1, keepdims=True))
        z /= z.sum(-1, keepdims=True)
        valid = rows >= 0
        np.add.at(prob, rows[valid], p[:n_real][valid])
        np.add.at(zones, rows[valid], z[:n_real][valid])
        np.add.at(seen, rows[valid], 1.0)
        spans[rows[valid]] = np.argmax(np.asarray(out["spans"]), axis=-1)[:n_real][valid]
    seen = np.maximum(seen, 1.0)
    return {"importance": prob / seen, "zone": zones / seen[:, None], "spans": spans}


def sentence_metrics(
    dataset: NotesSet,
    preds: dict[str, np.ndarray],
    threshold: float,
    zone_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """:func:`~src.model1_notes.metrics.selection_metrics` overall and per source, plus span
    F1 on the sentences with gold spans. Zones use ``zone_thresholds`` when given
    (:mod:`~src.model1_notes.calibrate`), else the argmax."""
    note_of_row = np.repeat(np.arange(len(dataset.note_ids)), np.asarray(dataset["note_length"]))
    source = np.asarray(dataset["source"])[note_of_row]
    truth = np.asarray(dataset["importance"]).astype(bool)
    zone_true = np.asarray(dataset["zone"])
    zone_pred = assign_zones(preds["zone"], zone_thresholds)
    picked = preds["importance"] >= threshold
    hard = np.asarray(dataset["hard_negative"])

    def block(rows: np.ndarray) -> dict[str, Any]:
        return selection_metrics(
            truth[rows],
            zone_true[rows],
            picked[rows],
            preds["importance"][rows],
            zone_pred[rows],
            hard[rows],
        )

    result = {"all": block(np.ones_like(truth))}
    for k, name in enumerate(SOURCES):
        if (source == k).any():
            result[name] = block(source == k)
    if "spans" in preds:
        result["spans"] = span_f1(np.asarray(dataset["span_tags"]), preds["spans"])
    return result


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


def train_seed(cfg: dict[str, Any], seed: int, session_start: float | None = None) -> dict:
    """Train (or resume) one seed, then write validation metrics and test predictions."""
    paths = run_paths(cfg, seed)
    if (paths["results"] / "metrics.json").exists():
        raise FileExistsError(f"{paths['results']} is complete; use a new experiment name.")
    tr, es = cfg["training"], cfg["training"]["early_stopping"]
    start, started = time.time(), utc_now()
    set_seed(seed)
    sizes = bucket_batch_sizes(cfg["data"]["sentence_buckets"], tr["sentences_per_batch"])
    device_info = configure_devices(
        tr["mixed_precision"], *sizes.values(), data_parallel=tr["data_parallel"]
    )
    processed, teachers = Path(cfg["paths"]["processed_dir"]), cfg.get("teachers", [])
    train_data = NotesSet.load(processed / "train")
    val_data = NotesSet.load(processed / "validation")

    def teacher_for(split: str, data: NotesSet) -> dict[str, np.ndarray]:
        return align_teacher(data, cfg["paths"].get("teacher_dir"), split, teachers)

    zone_w = zone_class_weights(cfg, train_data)
    windows = train_windows(train_data, cfg["data"]["hard_negative_oversample"])
    probe = NotesDataset(train_data, {}, windows, cfg, zone_w, shuffle=False)
    spe, epochs = len(probe), tr["epochs"]
    if tr["max_steps"] is not None:
        spe = min(spe, tr["max_steps"])
        epochs = min(epochs, math.ceil(tr["max_steps"] / spe))
    train_ds = NotesDataset(
        train_data,
        teacher_for("train", train_data),
        windows,
        cfg,
        zone_w,
        shuffle=True,
        seed=seed,
        max_batches=spe,
    )
    val_ds = NotesDataset(
        val_data,
        teacher_for("validation", val_data),
        np.arange(val_data.num_windows),
        cfg,
        zone_w,
        shuffle=False,
    )
    model = compile_model(cfg, spe * epochs)
    log.info(
        "%s seed %d: %d params, %d train windows, %d steps/epoch x %d epochs",
        cfg["name"],
        seed,
        model.count_params(),
        len(windows),
        spe,
        epochs,
    )
    paths["checkpoints"].mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)
    guard = TimeGuard(cfg["max_hours"], session_start)
    callbacks = [
        SyncEpoch(train_ds),
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
        "stage": "model1_v3",
        "name": cfg["name"],
        "seed": seed,
        "commit": commit_sha(),
        "config": cfg,
        "device": device_info,
        "params": model.count_params(),
        "train_windows": len(windows),
        "zone_class_weights": zone_w,
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
    best = keras.saving.load_model(paths["best_model"], compile=False)
    val_preds = predict_sentences(best, val_data, cfg)
    threshold, _ = tune_threshold(np.asarray(val_data["importance"]), val_preds["importance"])
    zone_thresholds = calibrate_zones(val_data, val_preds, threshold, cfg)
    test_data = NotesSet.load(processed / "test")
    test_preds = predict_sentences(best, test_data, cfg)
    write_jsonl(
        paths["results"] / "test_predictions.jsonl",
        prediction_records(test_data, test_preds, zone_thresholds),
    )
    metrics = {
        "seed": seed,
        "params": best.count_params(),
        "threshold": threshold,
        "zone_thresholds": zone_thresholds,
        "validation": sentence_metrics(val_data, val_preds, threshold, zone_thresholds),
        "test": sentence_metrics(test_data, test_preds, threshold, zone_thresholds),
    }
    write_json(paths["results"] / "metrics.json", metrics)
    manifest.update(status="complete", seconds=round(time.time() - start, 1))
    write_json(paths["results"] / "manifest.json", manifest)
    log.info("%s seed %d done: %s", cfg["name"], seed, metrics["validation"]["all"])
    return manifest


def calibrate_zones(
    val_data: NotesSet, val_preds: dict[str, np.ndarray], threshold: float, cfg: dict[str, Any]
) -> dict[str, float] | None:
    """Zone thresholds tuned on validation. ``inference.zone_calibration``:

    - ``ami``: AMI validation only (finds rare real decisions, over-labels clean meetings);
    - ``all``: every validation sentence pooled (synthetic dominates; AMI decisions vanish);
    - ``balanced``: the mean of AMI F1 and synthetic F1, so both kinds of meeting count;
    - ``null``: argmax zones.
    """
    mode = cfg["inference"].get("zone_calibration")
    if mode is None:
        return None
    if mode not in ("ami", "all", "balanced"):
        raise ValueError(f"Unknown zone_calibration: {mode!r}")
    lengths = np.asarray(val_data["note_length"])
    ami = np.repeat(np.asarray(val_data["source"]) == AMI, lengths)
    rows = ami if mode == "ami" else np.ones(len(ami), dtype=bool)
    groups = [ami[rows], ~ami[rows]] if mode == "balanced" else None
    grid = np.round(np.arange(*cfg["inference"]["zone_threshold_grid"]), 3)
    return tune_zone_thresholds(
        np.asarray(val_data["importance"])[rows],
        np.asarray(val_data["zone"])[rows],
        val_preds["importance"][rows] >= threshold,
        val_preds["zone"][rows],
        grid,
        groups,
    )


def prediction_records(
    dataset: NotesSet,
    preds: dict[str, np.ndarray],
    zone_thresholds: dict[str, float] | None = None,
):
    """One record per meeting: sentence IDs, importance probabilities and zone predictions."""
    starts, lengths = np.asarray(dataset["note_start"]), np.asarray(dataset["note_length"])
    for note_id, s, n in zip(dataset.note_ids, starts, lengths, strict=True):
        rows = slice(int(s), int(s) + int(n))
        yield {
            "note_id": note_id,
            "sentence_ids": np.asarray(dataset["sentence_id"][rows]).tolist(),
            "importance": [round(float(p), 6) for p in preds["importance"][rows]],
            "zone": [ZONES[z] for z in assign_zones(preds["zone"][rows], zone_thresholds)],
        }


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
