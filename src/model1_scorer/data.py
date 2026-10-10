"""Stage 7a: Model 1 student arrays (one row per DialogSum dialogue) and their targets.

Usage::

    uv run python -m src.model1_scorer.data --config configs/model1_student_A.yaml

Writes ``<processed_dir>/<split>/`` for train / validation / test (shared by all students):
    ``token_ids.npy`` (N, max_len) int32, ``speaker_ids.npy`` (N, max_len) int8,
    ``cls_positions.npy`` (N, max_utterances) int32 (0 after the last kept utterance),
    ``n_utterances.npy`` (N,) int32 kept utterances, ``ids.json``, ``meta.json`` (truncation).
Dialogues are encoded with :func:`~src.model1_scorer.dialogue.encode_dialogue`; utterances
beyond ``data.max_utterances`` or beyond what fits in ``max_len`` are dropped and counted.

Targets (:func:`load_targets`) come from the label files of Stage 6 and are aligned to these
rows: per utterance a score in [0, 1] (teacher score or oracle 0/1) and a tag index, ``-1``
where there is no label. Dialogues without labels (teacher failure, train subset) are masked.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from src.config import load_config
from src.data.io import read_jsonl
from src.layers.losses import IGNORE
from src.model1_scorer.dialogue import encode_dialogue
from src.tokenizer.tokenizer import load_tokenizer

log = logging.getLogger(__name__)
SPLITS = ("train", "validation", "test")
TAGS = ("none", "key_point", "decision", "action_item")


def encode_split(
    tokenizer: Tokenizer, records: list[dict[str, Any]], max_len: int, max_utterances: int
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Arrays for one split and truncation counts."""
    n = len(records)
    arrays = {
        "token_ids": np.zeros((n, max_len), dtype=np.int32),
        "speaker_ids": np.zeros((n, max_len), dtype=np.int8),
        "cls_positions": np.zeros((n, max_utterances), dtype=np.int32),
        "n_utterances": np.zeros(n, dtype=np.int32),
    }
    counts = {
        "dialogues": n,
        "truncated_dialogues": 0,
        "truncated_utterances": 0,
        "dropped_utterances": 0,
    }
    for i, r in enumerate(records):
        speakers, utterances = r["speakers"][:max_utterances], r["utterances"][:max_utterances]
        enc = encode_dialogue(tokenizer, speakers, utterances, max_len)
        dropped = enc.dropped + len(r["utterances"]) - len(utterances)
        k = len(enc.cls_positions)
        arrays["token_ids"][i] = enc.token_ids
        arrays["speaker_ids"][i] = enc.speaker_ids
        arrays["cls_positions"][i, :k] = enc.cls_positions
        arrays["n_utterances"][i] = k
        counts["truncated_dialogues"] += bool(enc.truncated or dropped)
        counts["truncated_utterances"] += enc.truncated
        counts["dropped_utterances"] += dropped
    return arrays, counts


@dataclass
class StudentSet:
    """Encoded dialogues of one split (memory-mapped)."""

    ids: list[str]
    token_ids: np.ndarray
    speaker_ids: np.ndarray
    cls_positions: np.ndarray
    n_utterances: np.ndarray

    @classmethod
    def load(cls, directory: str | Path) -> StudentSet:
        d = Path(directory)
        arrays = {
            k: np.load(d / f"{k}.npy", mmap_mode="r")
            for k in ("token_ids", "speaker_ids", "cls_positions", "n_utterances")
        }
        return cls(ids=json.loads((d / "ids.json").read_text()), **arrays)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def utterance_mask(self) -> np.ndarray:
        """(N, max_utterances) bool, True for kept utterances."""
        return (
            np.arange(self.cls_positions.shape[1])[None, :] < np.asarray(self.n_utterances)[:, None]
        )


@dataclass
class Targets:
    """Per-utterance targets aligned to a :class:`StudentSet` (``-1`` = no label)."""

    scores: np.ndarray  # (N, U) float32 in [0, 1]
    tags: np.ndarray  # (N, U) int8 tag index
    labelled: np.ndarray  # (N,) bool: the dialogue has labels


def load_targets(labels_dir: str | Path, split: str, dataset: StudentSet, source: str) -> Targets:
    """Read ``<labels_dir>/<split>.jsonl`` (teacher or oracle) and align it to ``dataset``."""
    n, u = dataset.cls_positions.shape
    scores = np.full((n, u), IGNORE, dtype=np.float32)
    tags = np.full((n, u), IGNORE, dtype=np.int8)
    labelled = np.zeros(n, dtype=bool)
    row = {dialogue_id: i for i, dialogue_id in enumerate(dataset.ids)}
    unknown = 0
    for record in read_jsonl(Path(labels_dir) / f"{split}.jsonl"):
        i = row.get(record["id"])
        if i is None:  # e.g. the arrays were built with ``max_dialogues``
            unknown += 1
            continue
        k = int(dataset.n_utterances[i])
        if source == "oracle":
            scores[i, :k] = record["labels"][:k]
        else:
            scores[i, :k] = record["scores"][:k]
            tags[i, :k] = [TAGS.index(t) for t in record["tags"][:k]]
        labelled[i] = True
    if unknown:
        log.warning("%s/%s: %d labelled dialogues not in the arrays", labels_dir, split, unknown)
    return Targets(scores, tags, labelled)


def main(argv: list[str] | None = None) -> None:
    """Entry point: encode every split from a student config."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/model1_student_A.yaml"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    paths = cfg["paths"]
    tokenizer = load_tokenizer(paths["tokenizer_dir"])
    raw = Path(paths["raw_dir"]) / "dialogsum"
    for split in SPLITS:
        records = list(read_jsonl(raw / f"{split}.jsonl"))
        if cfg["data"]["max_dialogues"] is not None:
            records = records[: cfg["data"]["max_dialogues"]]
        arrays, counts = encode_split(
            tokenizer, records, cfg["model"]["max_len"], cfg["data"]["max_utterances"]
        )
        out = Path(paths["processed_dir"]) / split
        out.mkdir(parents=True, exist_ok=True)
        for name, array in arrays.items():
            np.save(out / f"{name}.npy", array)
        (out / "ids.json").write_text(json.dumps([r["id"] for r in records]))
        (out / "meta.json").write_text(json.dumps(counts, indent=2) + "\n")
        log.info("%s: %s", split, counts)


if __name__ == "__main__":
    main()
