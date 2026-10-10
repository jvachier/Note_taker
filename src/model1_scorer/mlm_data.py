"""Stage 5a: build the MLM pretraining arrays for the Model 1 encoder.

Usage::

    uv run python -m src.model1_scorer.mlm_data --config configs/mlm.yaml

Two kinds of rows, ``seq_len`` tokens each:
    WikiText   paragraphs (detokenised, headings skipped) each followed by ``[SEP]``, as one
               stream cut into chunks of ``seq_len - 1`` tokens, each chunk prefixed by
               ``[CLS]``; speaker ID 0.
    DialogSum  one dialogue per row in the Model 1 input format
               (``[CLS] utt [SEP] [CLS] utt [SEP] ...``, see :mod:`dialogue`) with its speaker
               IDs, so pretraining sees the same layout and speaker embeddings as the students.
Writes ``<processed_dir>/<set>/token_ids.npy`` (int32), ``speaker_ids.npy`` (int8) and
``meta.json`` for ``train`` and every ``eval.held_out`` source. ``data.corpus`` maps each train
source to a number of copies (DialogSum is oversampled; each copy gets different masks, since
masking happens at training time).
Masking happens at training time (:mod:`mlm_train`).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from src.config import load_config
from src.data.io import read_jsonl, read_lines
from src.data.wikitext import iter_articles
from src.model1_scorer.dialogue import encode_dialogue
from src.seed import set_seed
from src.tokenizer.tokenizer import CLS_ID, PAD_ID, SEP_ID, load_tokenizer

log = logging.getLogger(__name__)


def iter_paragraphs(raw_dir: Path, split: str, max_units: int | None) -> Iterator[str]:
    """Detokenised WikiText paragraphs of the first ``max_units`` articles."""
    articles = iter_articles(read_lines(raw_dir / "wikitext" / f"{split}.txt"))
    for paragraphs in islice(articles, max_units):
        yield from paragraphs


def pack_stream(
    tokenizer: Tokenizer, texts: Iterable[str], seq_len: int, batch: int = 10000
) -> np.ndarray:
    """``[CLS]`` + ``seq_len - 1`` tokens of the ``text [SEP] text [SEP] ...`` stream per row.

    The last row is padded with ``[PAD]``.
    """
    chunks: list[np.ndarray] = []
    pending: list[str] = []

    def flush() -> None:
        for enc in tokenizer.encode_batch(pending, add_special_tokens=False):
            chunks.append(np.asarray([*enc.ids, SEP_ID], dtype=np.int32))
        pending.clear()

    for text in texts:
        pending.append(text)
        if len(pending) == batch:
            flush()
    flush()
    stream = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int32)
    body = seq_len - 1
    n_rows = -(-len(stream) // body)
    rows = np.full((n_rows, seq_len), PAD_ID, dtype=np.int32)
    rows[:, 0] = CLS_ID
    padded = np.full(n_rows * body, PAD_ID, dtype=np.int32)
    padded[: len(stream)] = stream
    rows[:, 1:] = padded.reshape(n_rows, body)
    return rows


def encode_dialogues(
    tokenizer: Tokenizer, records: Iterable[dict[str, Any]], seq_len: int
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """One row per dialogue: token IDs, speaker IDs and truncation counts."""
    ids, speakers = [], []
    counts = {
        "dialogues": 0,
        "truncated_dialogues": 0,
        "truncated_utterances": 0,
        "dropped_utterances": 0,
    }
    for record in records:
        enc = encode_dialogue(tokenizer, record["speakers"], record["utterances"], seq_len)
        ids.append(enc.token_ids)
        speakers.append(enc.speaker_ids)
        counts["dialogues"] += 1
        counts["truncated_dialogues"] += bool(enc.truncated or enc.dropped)
        counts["truncated_utterances"] += enc.truncated
        counts["dropped_utterances"] += enc.dropped
    if not ids:
        return np.zeros((0, seq_len), np.int32), np.zeros((0, seq_len), np.int8), counts
    return np.stack(ids), np.stack(speakers), counts


def build_source(
    tokenizer: Tokenizer, raw_dir: Path, source: str, seq_len: int, max_units: int | None
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Rows of one source (``wikitext_<split>`` or ``dialogsum_<split>``)."""
    dataset, _, split = source.partition("_")
    if dataset == "wikitext":
        ids = pack_stream(tokenizer, iter_paragraphs(raw_dir, split, max_units), seq_len)
        return ids, np.zeros(ids.shape, dtype=np.int8), {"rows": len(ids)}
    if dataset == "dialogsum":
        records = islice(read_jsonl(raw_dir / "dialogsum" / f"{split}.jsonl"), max_units)
        ids, speakers, counts = encode_dialogues(tokenizer, records, seq_len)
        return ids, speakers, {"rows": len(ids), **counts}
    raise ValueError(f"Unknown source: {source!r}")


def save_set(out_dir: Path, ids: np.ndarray, speakers: np.ndarray, meta: dict[str, Any]) -> None:
    """Write one array set and its meta.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "token_ids.npy", ids)
    np.save(out_dir / "speaker_ids.npy", speakers)
    real = int((ids != PAD_ID).sum())
    meta = {**meta, "rows": int(ids.shape[0]), "seq_len": int(ids.shape[1]), "tokens": real}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("%s: %d rows, %d tokens", out_dir.name, meta["rows"], real)


def main(argv: list[str] | None = None) -> None:
    """Entry point: build ``train`` and the held-out sets from a config file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/mlm.yaml"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    data, paths = cfg["data"], cfg["paths"]
    set_seed(cfg["seeds"][0])
    raw_dir, out = Path(paths["raw_dir"]), Path(paths["processed_dir"])
    tokenizer = load_tokenizer(paths["tokenizer_dir"])
    seq_len, max_units = data["seq_len"], data["max_units"]
    start = time.time()

    ids, speakers, sources = [], [], {}
    for source, copies in data["corpus"].items():
        source_ids, source_speakers, meta = build_source(
            tokenizer, raw_dir, source, seq_len, max_units
        )
        ids.append(np.tile(source_ids, (copies, 1)))
        speakers.append(np.tile(source_speakers, (copies, 1)))
        sources[source] = {**meta, "copies": copies}
    save_set(out / "train", np.concatenate(ids), np.concatenate(speakers), {"sources": sources})
    for source in cfg["eval"]["held_out"]:
        ids, speakers, meta = build_source(tokenizer, raw_dir, source, seq_len, max_units)
        save_set(out / source, ids, speakers, {"sources": {source: meta}})
    log.info("Done in %.1f s", time.time() - start)


if __name__ == "__main__":
    main()
