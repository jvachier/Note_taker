"""Zone calibration for Model 1 v3: per-zone probability thresholds for the rare zones.

Taking the most likely zone (argmax) almost never yields ``decision`` or ``action_item`` on real
meetings, where they are ~1.6% of sentences (AMI test, 2026-10-25: decision / action F1 = 0
for every run). Instead, a kept sentence is a decision when ``p(decision) >= t_decision`` (same
for action items), with ``t`` tuned on validation to maximise that zone's F1 among kept
sentences; if both pass, the one with the larger ``p / t`` wins; otherwise the argmax zone is
used. No ML imports.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.model1_notes.metrics import ZONES, f1_score

CALIBRATED = ("decision", "action_item")


def tune_zone_thresholds(
    importance: np.ndarray,
    zone: np.ndarray,
    kept: np.ndarray,
    zone_prob: np.ndarray,
    grid: Sequence[float],
    groups: Sequence[np.ndarray] | None = None,
) -> dict[str, float]:
    """Threshold per calibrated zone maximising F1 of ``kept & p(z) >= t`` against the gold
    ``importance & zone == z`` (all arrays per sentence; ``zone_prob`` is (N, 5)).

    With ``groups`` (boolean row masks, e.g. AMI and synthetic), the mean of the per-group
    F1 is maximised, so a small group (real meetings) weighs as much as a large one."""
    importance, kept = np.asarray(importance, dtype=bool), np.asarray(kept, dtype=bool)
    masks = list(groups) if groups else [np.ones_like(importance)]
    thresholds = {}
    for name in CALIBRATED:
        z = ZONES.index(name)
        truth = importance & (np.asarray(zone) == z)

        scores = [
            float(np.mean([f1_score(truth[m], (kept & (zone_prob[:, z] >= t))[m]) for m in masks]))
            for t in grid
        ]
        thresholds[name] = float(grid[int(np.argmax(scores))])
    return thresholds


def assign_zones(zone_prob: np.ndarray, thresholds: dict[str, float] | None) -> np.ndarray:
    """Zone index per sentence: argmax, overridden by calibrated zones above their threshold."""
    zones = np.asarray(zone_prob).argmax(-1)
    if not thresholds:
        return zones
    ratios = np.stack([zone_prob[:, ZONES.index(z)] / thresholds[z] for z in CALIBRATED], axis=-1)
    best = ratios.argmax(-1)
    passes = ratios.max(-1) >= 1.0
    calibrated = np.array([ZONES.index(z) for z in CALIBRATED])[best]
    return np.where(passes, calibrated, zones)


def calibrate_run(
    cfg: dict, seed: int, probs_dir: str | None = None
) -> dict:  # pragma: no cover - exercised end to end by the training tests
    """Apply zone calibration to a finished run: tune on validation, rewrite ``metrics.json``
    (``zone_thresholds`` + recomputed metrics) and the test predictions.

    Probabilities come from ``<probs_dir>/<name>_<split>.npz`` when present (``importance``,
    ``zone``), else from the run's ``best.keras``.
    """
    import json
    from pathlib import Path

    from src.data.io import write_jsonl
    from src.model1_notes.data import NotesSet
    from src.model1_notes.train import (
        calibrate_zones,
        predict_sentences,
        prediction_records,
        run_paths,
        sentence_metrics,
    )

    paths = run_paths(cfg, seed)
    metrics = json.loads((paths["results"] / "metrics.json").read_text())
    processed = Path(cfg["paths"]["processed_dir"])
    data, preds, model = {}, {}, None
    for split in ("validation", "test"):
        data[split] = NotesSet.load(processed / split)
        cached = (
            None if probs_dir is None else Path(probs_dir) / f"{cfg['name']}_s{seed}_{split}.npz"
        )
        if cached is not None and cached.exists():
            saved = np.load(cached)
            preds[split] = {"importance": saved["importance"], "zone": saved["zone"]}
        else:
            if model is None:
                import keras

                import src.model1_notes.losses  # noqa: F401
                import src.model1_notes.model  # noqa: F401

                model = keras.saving.load_model(paths["best_model"], compile=False)
            preds[split] = predict_sentences(model, data[split], cfg)
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cached, importance=preds[split]["importance"], zone=preds[split]["zone"])
    threshold = metrics["threshold"]
    zone_thresholds = calibrate_zones(data["validation"], preds["validation"], threshold, cfg)
    test = sentence_metrics(data["test"], preds["test"], threshold, zone_thresholds)
    if "spans" in metrics.get("test", {}):
        test["spans"] = metrics["test"]["spans"]  # span tags are not recomputed from the cache
    metrics.update(
        zone_thresholds=zone_thresholds,
        validation=sentence_metrics(
            data["validation"], preds["validation"], threshold, zone_thresholds
        ),
        test=test,
    )
    (paths["results"] / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_jsonl(
        paths["results"] / "test_predictions.jsonl",
        prediction_records(data["test"], preds["test"], zone_thresholds),
    )
    return metrics


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - thin CLI
    """``python -m src.model1_notes.calibrate --config configs/model1_v3_E4.yaml --seed 0``."""
    import argparse
    import json

    from src.config import load_config

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probs-dir", default=None, help="cached <name>_<split>.npz files")
    args = parser.parse_args(argv)
    metrics = calibrate_run(load_config(args.config), args.seed, args.probs_dir)
    print(json.dumps({k: metrics[k] for k in ("threshold", "zone_thresholds")}, indent=2))
    for layer in ("ami", "synthetic"):
        m = metrics["test"].get(layer, {})
        print(layer, {k: m.get(k) for k in ("importance_f1", "decision_f1", "action_f1")})


if __name__ == "__main__":
    main()
