"""Stage 4a: build Model 2 arrays (train / validation / test at every noise level).

Usage::

    uv run python -m src.model2_cleanup.build_data --config configs/model2.yaml [--sets test]

Writes under ``paths.processed_dir``:
    ``<set>/token_ids.npy`` (int32), ``<set>/casing.npy``, ``<set>/punct.npy`` (int8, -1 = no label)
    ``<set>/meta.json``      counts, truncation, noise level
    ``noise_vocab.txt``      top-k train word forms used for substitution / insertion
    ``meta.json``            class counts and punctuation class weights (from train)
Set names: ``train``, ``validation_<dataset>``, ``test_<dataset>_noise<NN>``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_config
from src.layers.losses import inverse_sqrt_frequency_weights
from src.model2_cleanup.examples import (
    EncodedWindows,
    build_split,
    iter_units,
    label_counts,
    word_frequencies,
)
from src.model2_cleanup.labels import CASING_CLASSES, PUNCT_CLASSES
from src.model2_cleanup.noise import NoiseConfig
from src.seed import set_seed
from src.tokenizer.tokenizer import load_tokenizer

log = logging.getLogger(__name__)
SET_GROUPS = ("train", "validation", "test")


def noise_from_config(data_cfg: dict[str, Any]) -> NoiseConfig:
    """The default (training) noise config."""
    n = data_cfg["noise"]
    return NoiseConfig(n["substitution"], n["deletion"], n["insertion"])


def test_set_name(source: str, level: float) -> str:
    """``wikitext_test`` at 0.05 -> ``test_wikitext_noise05``."""
    return f"test_{source.split('_')[0]}_noise{round(level * 100):02d}"


def validation_set_name(source: str) -> str:
    """``dialogsum_validation`` -> ``validation_dialogsum``."""
    return f"validation_{source.split('_')[0]}"


def set_rng(seed: int, name: str) -> np.random.Generator:
    """Independent, reproducible generator per array set."""
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def save_set(out_dir: Path, arrays: EncodedWindows, meta: dict[str, Any]) -> None:
    """Save the three arrays and a meta.json for one set."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "token_ids.npy", arrays.token_ids)
    np.save(out_dir / "casing.npy", arrays.casing)
    np.save(out_dir / "punct.npy", arrays.punct)
    meta = {
        **meta,
        "windows": int(arrays.token_ids.shape[0]),
        "max_len": int(arrays.token_ids.shape[1]),
        "words": arrays.total_words,
        "truncated_words": arrays.truncated_words,
        "casing_counts": label_counts(arrays.casing, len(CASING_CLASSES)),
        "punct_counts": label_counts(arrays.punct, len(PUNCT_CLASSES)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info(
        "%s: %d windows, %d words (%d truncated)",
        out_dir.name,
        meta["windows"],
        meta["words"],
        meta["truncated_words"],
    )


def load_noise_vocab(processed_dir: Path) -> list[str]:
    """Read ``noise_vocab.txt`` (one word form per line, most frequent first)."""
    return (processed_dir / "noise_vocab.txt").read_text(encoding="utf-8").split("\n")[:-1]


def build_noise_vocab(cfg: dict[str, Any], raw_dir: Path, processed_dir: Path) -> list[str]:
    """Top-k word forms over the train sources (each counted once, not oversampled)."""
    data = cfg["data"]
    counts: Counter[str] = Counter()
    for source in data["train_sources"]:
        counts += word_frequencies(iter_units(raw_dir, source, data["max_units"]))
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab = [w for w, _ in ranked[: data["noise"]["substitution_vocab_size"]]]
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "noise_vocab.txt").write_text("".join(f"{w}\n" for w in vocab), "utf-8")
    log.info("noise vocab: %d forms (from %d distinct)", len(vocab), len(counts))
    return vocab


def main(argv: list[str] | None = None) -> None:
    """Entry point: build the requested set groups from a config file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/model2.yaml"))
    parser.add_argument("--sets", default=",".join(SET_GROUPS), help="comma-separated groups")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    data, paths = cfg["data"], cfg["paths"]
    groups = args.sets.split(",")
    if unknown := set(groups) - set(SET_GROUPS):
        raise ValueError(f"Unknown set groups: {sorted(unknown)}")
    set_seed(data["seed"])
    raw_dir, processed_dir = Path(paths["raw_dir"]), Path(paths["processed_dir"])
    tokenizer = load_tokenizer(paths["tokenizer_dir"])
    start = time.time()

    vocab = build_noise_vocab(cfg, raw_dir, processed_dir)
    noise = noise_from_config(data)
    common = {"window_words": data["window_words"], "max_len": data["max_len"], "vocab": vocab}

    def build(name: str, sources: dict[str, int], level: NoiseConfig) -> None:
        units = (
            (text, copies)
            for source, copies in sources.items()
            for text in iter_units(raw_dir, source, data["max_units"])
        )
        arrays = build_split(
            tokenizer, units, noise=level, rng=set_rng(data["seed"], name), **common
        )
        meta = {"sources": sources, "noise": level.__dict__, "noise_total": level.total}
        save_set(processed_dir / name, arrays, meta)

    if "train" in groups:
        build("train", dict(data["train_sources"]), noise)
        train_meta = json.loads((processed_dir / "train" / "meta.json").read_text())
        weights = inverse_sqrt_frequency_weights(train_meta["punct_counts"])
        summary = {
            "casing_classes": list(CASING_CLASSES),
            "punct_classes": list(PUNCT_CLASSES),
            "punct_class_weights": weights,
            "train_punct_counts": train_meta["punct_counts"],
            "train_casing_counts": train_meta["casing_counts"],
            "config": cfg,
        }
        (processed_dir / "meta.json").write_text(json.dumps(summary, indent=2) + "\n")
        log.info("punct class weights: %s", weights)
    if "validation" in groups:
        for source in data["validation_sources"]:
            build(validation_set_name(source), {source: 1}, noise)
    if "test" in groups:
        for source in data["test_sources"]:
            for level in cfg["eval"]["noise_levels"]:
                scaled = noise.scaled(level) if level else NoiseConfig(0.0, 0.0, 0.0)
                build(test_set_name(source, level), {source: 1}, scaled)
    log.info("Done in %.1f s", time.time() - start)


if __name__ == "__main__":
    main()
