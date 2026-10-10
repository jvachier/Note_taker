"""Model 2 evaluation: word-level punctuation / casing metrics and the rule baseline.

Usage (after training)::

    uv run python -m src.model2_cleanup.evaluate --config configs/model2.yaml --seed 0

Scores every ``test_<dataset>_noise<NN>`` set with the best checkpoint of that seed and with
the rule baseline, and writes ``results/<name>/seed_<n>/metrics.json``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import keras
import numpy as np
from tokenizers import Tokenizer

from src.config import load_config
from src.eval.metrics import classification_report
from src.layers.losses import IGNORE
from src.model2_cleanup.build_data import test_set_name
from src.model2_cleanup.dataset import ArraySet
from src.model2_cleanup.labels import CAPITALISED, CASING_CLASSES, PERIOD, PUNCT_CLASSES
from src.tokenizer.tokenizer import load_tokenizer
from src.training import write_json

log = logging.getLogger(__name__)
PUNCT_MACRO = ("comma", "period", "question")
CASING_MACRO = ("capitalised", "upper")


def predict(model: keras.Model, token_ids: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    """Argmax class per position for both heads."""
    out = model.predict(np.asarray(token_ids), batch_size=batch_size, verbose=0)
    return {k: np.argmax(v, axis=-1) for k, v in out.items()}


def score(arrays: ArraySet, pred: dict[str, np.ndarray]) -> dict[str, Any]:
    """Word-level metrics; only labelled first-subword positions count."""
    report = {}
    for head, names, macro in (
        ("punct", PUNCT_CLASSES, PUNCT_MACRO),
        ("casing", CASING_CLASSES, CASING_MACRO),
    ):
        true = np.asarray(getattr(arrays, head))
        valid = true != IGNORE
        report[head] = classification_report(true[valid], pred[head][valid], names, macro)
    return report


def rule_baseline(arrays: ArraySet, tokenizer: Tokenizer) -> dict[str, np.ndarray]:
    """Capitalise the first word and the word "i" (incl. "i'm"), period after the last word."""
    ids = np.asarray(arrays.token_ids)
    words = np.asarray(arrays.punct) != IGNORE  # every kept word has a punct label
    i_id = tokenizer.token_to_id("i")
    is_continuation = np.zeros(tokenizer.get_vocab_size(), dtype=bool)
    for tok, idx in tokenizer.get_vocab().items():
        is_continuation[idx] = tok.startswith("##")
    next_ids = np.concatenate([ids[:, 1:], np.zeros_like(ids[:, :1])], axis=1)
    casing = np.zeros_like(ids)
    punct = np.zeros_like(ids)
    is_i = words & (ids == i_id) & ~is_continuation[next_ids]
    casing[is_i] = CAPITALISED
    rows = np.flatnonzero(words.any(axis=1))
    first = np.argmax(words[rows], axis=1)
    last = words.shape[1] - 1 - np.argmax(words[rows, ::-1], axis=1)
    casing[rows, first] = CAPITALISED
    punct[rows, last] = PERIOD
    return {"casing": casing, "punct": punct}


def evaluate_all(
    model: keras.Model | None, cfg: dict[str, Any], tokenizer: Tokenizer
) -> dict[str, Any]:
    """Scores per test set and noise level: ``{set: {"model": ..., "baseline": ...}}``."""
    processed = Path(cfg["paths"]["processed_dir"])
    batch_size = cfg["eval"]["batch_size"]
    results: dict[str, Any] = {}
    for source in cfg["data"]["test_sources"]:
        for level in cfg["eval"]["noise_levels"]:
            name = test_set_name(source, level)
            arrays = ArraySet.load(processed / name)
            entry = {"noise": level, "baseline": score(arrays, rule_baseline(arrays, tokenizer))}
            if model is not None:
                entry["model"] = score(arrays, predict(model, arrays.token_ids, batch_size))
            results[name] = entry
            log.info(
                "%s: model punct macro-F1 %s | baseline %.4f",
                name,
                f"{entry['model']['punct']['macro_f1']:.4f}" if model is not None else "-",
                entry["baseline"]["punct"]["macro_f1"],
            )
    return results


def main(argv: list[str] | None = None) -> None:
    """Evaluate the best checkpoint of one seed on all test sets."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/model2.yaml"))
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    from src.model2_cleanup.train import run_paths  # avoid a circular import at module load

    paths = run_paths(cfg, args.seed)
    model = keras.saving.load_model(paths["best_model"])
    tokenizer = load_tokenizer(cfg["paths"]["tokenizer_dir"])
    write_json(paths["results"] / "metrics.json", {"test": evaluate_all(model, cfg, tokenizer)})


if __name__ == "__main__":
    main()
