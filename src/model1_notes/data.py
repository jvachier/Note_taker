"""Model 1 v3 arrays from the meeting-notes corpus (PLAN.md sections 10.2 and 10.4).

Usage::

    uv run python -m src.model1_notes.data --config configs/model1_v3.yaml

Reads the corpus ``meeting-notes-extractive-v1`` in its release layout, one Parquet table per
layer and split (``<corpus_dir>/data/{synthetic,ami}/sentences/<split>.parquet``, one row per
sentence; the ``synthetic`` layer holds the ``synthetic`` and ``synthetic_hn`` sources), and
writes ``<processed_dir>/<split>/``:

Per sentence (meetings stored one after another, sentences in ``sentence_id`` order):
    ``token_ids.npy``   (N, T) int32  ``[CLS] subwords [SEP]``, ``PAD_ID`` after, T = max tokens
    ``span_tags.npy``   (N, T) int8   BIO owner/deadline tags (:data:`SPAN_TAGS`), ``IGNORE``
                                       on ``[CLS]`` / ``[SEP]`` / padding
    ``importance.npy``  (N,) int8     gold 0/1
    ``confidence.npy``  (N,) float32  gold label confidence (loss weight)
    ``zone.npy``        (N,) int8     index in :data:`ZONES`
    ``speaker.npy``     (N,) int8     1..8, numbered by first appearance in the meeting
    ``sentence_id.npy`` (N,) int32    corpus ``sentence_id`` (to align teacher outputs)
    ``hard_negative.npy`` (N,) bool
Per meeting: ``note_ids.json``, ``note_start.npy`` / ``note_length.npy`` (M,) int64 / int32
(rows of the meeting in the sentence arrays), ``source.npy`` (M,) int8 (index in
:data:`SOURCES`).
Per window: ``window_note.npy``, ``window_offset.npy``, ``window_length.npy`` (W,) int32. A
meeting longer than ``max_sentences`` is covered by windows of ``max_sentences`` sentences that
overlap by ``window_overlap``; windows reference the sentence rows, nothing is duplicated.
``meta.json``: counts (meetings, sentences, windows, truncated sentences, speakers capped).

Sentences longer than ``max_sentence_tokens`` are truncated (counted). Meetings with more than
8 speakers map the extra speakers to speaker 8 (counted). Span tokens cut by truncation are lost.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from src.config import load_config
from src.layers.losses import IGNORE
from src.model1_notes.metrics import SPAN_TAGS, ZONES  # noqa: F401  (re-exported)
from src.model1_notes.teacher.windows import REASONS, window_starts  # noqa: F401  (re-exported)
from src.tokenizer.tokenizer import CLS_ID, NUM_SPEAKERS, PAD_ID, SEP_ID, load_tokenizer

log = logging.getLogger(__name__)
SPLITS = ("train", "validation", "test")
SOURCES = ("synthetic", "synthetic_hn", "ami")
COLUMNS = (
    "note_id",
    "source",
    "sentence_id",
    "speaker",
    "importance",
    "confidence",
    "zone",
    "hard_negative",
    "spans",
)


def layer_of(source: str) -> str:
    """Release layer holding a source: ``synthetic`` (incl. hard negatives) or ``ami``."""
    return "ami" if source == "ami" else "synthetic"


def corpus_split_paths(corpus_dir: str | Path, split: str, sources: Sequence[str]) -> list[Path]:
    """Sentence tables of ``split`` for the layers of ``sources`` (synthetic first, then AMI)."""
    unknown = set(sources) - set(SOURCES)
    if unknown:
        raise ValueError(f"Unknown sources {sorted(unknown)}; expected a subset of {SOURCES}")
    layers = sorted({layer_of(s) for s in sources}, key=("synthetic", "ami").index)
    paths = [
        Path(corpus_dir) / "data" / layer / "sentences" / f"{split}.parquet" for layer in layers
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Corpus split not found: {', '.join(map(str, missing))}")
    return paths


def read_corpus_split(
    corpus_dir: str | Path, split: str, text_field: str, sources: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    """Rows of one split grouped by meeting, sentences sorted by ``sentence_id``.

    Meetings keep the order of their first row (synthetic layer first). Rows whose ``source``
    is not in ``sources`` are dropped.
    """
    meetings: dict[str, list[dict[str, Any]]] = {}
    for path in corpus_split_paths(corpus_dir, split, sources):
        for row in pq.read_table(path, columns=[*COLUMNS, text_field]).to_pylist():
            if row["source"] in sources:
                row["text"] = row.pop(text_field)
                meetings.setdefault(row["note_id"], []).append(row)
    for rows in meetings.values():
        rows.sort(key=lambda r: r["sentence_id"])
    return meetings


def sentence_rows(dataset: NotesSet) -> dict[tuple[str, int], int]:
    """``(note_id, sentence_id) -> row`` of the sentence arrays (to align teacher outputs)."""
    rows: dict[tuple[str, int], int] = {}
    sentence_id = np.asarray(dataset["sentence_id"])
    starts, lengths = np.asarray(dataset["note_start"]), np.asarray(dataset["note_length"])
    for note_id, start, n in zip(dataset.note_ids, starts, lengths, strict=True):
        for row in range(int(start), int(start) + int(n)):
            rows[(note_id, int(sentence_id[row]))] = row
    return rows


def speaker_numbers(speakers: Sequence[str]) -> tuple[list[int], int]:
    """1-based speaker numbers by first appearance, capped at ``NUM_SPEAKERS``.

    Returns the numbers and how many distinct speakers were capped.
    """
    order: dict[str, int] = {}
    for s in speakers:
        order.setdefault(s, len(order) + 1)
    capped = max(0, len(order) - NUM_SPEAKERS)
    return [min(order[s], NUM_SPEAKERS) for s in speakers], capped


def span_bio(
    offsets: Sequence[tuple[int, int]], spans: Sequence[dict[str, Any]] | None
) -> list[int]:
    """BIO tag index (:data:`SPAN_TAGS`) for each subword, from character spans.

    A subword belongs to a span when its character range overlaps it; the first such subword
    is ``B-<role>``, the following ones ``I-<role>``.
    """
    tags = [0] * len(offsets)
    for span in spans or ():
        begin = SPAN_TAGS.index(f"B-{span['role']}")
        inside = False
        for i, (start, end) in enumerate(offsets):
            if start < span["end"] and end > span["start"]:
                tags[i] = begin + 1 if inside else begin
                inside = True
    return tags


def encode_sentence(
    tokenizer: Tokenizer, text: str, spans: Sequence[dict[str, Any]] | None, max_tokens: int
) -> tuple[np.ndarray, np.ndarray, bool]:
    """``[CLS] subwords [SEP]`` padded to ``max_tokens``, its span tags and a truncation flag."""
    enc = tokenizer.encode(text, add_special_tokens=False)
    budget = max_tokens - 2
    ids, tags = list(enc.ids), span_bio(enc.offsets, spans)
    truncated = len(ids) > budget
    ids, tags = ids[:budget], tags[:budget]
    token_ids = np.full(max_tokens, PAD_ID, dtype=np.int32)
    span_tags = np.full(max_tokens, IGNORE, dtype=np.int8)
    token_ids[: len(ids) + 2] = [CLS_ID, *ids, SEP_ID]
    span_tags[1 : len(ids) + 1] = tags
    return token_ids, span_tags, truncated


def encode_split(
    tokenizer: Tokenizer, meetings: dict[str, list[dict[str, Any]]], data_cfg: dict[str, Any]
) -> tuple[dict[str, np.ndarray], list[str], dict[str, int]]:
    """Arrays for one split (see module docstring), the meeting IDs and counts."""
    max_tokens, size = data_cfg["max_sentence_tokens"], data_cfg["max_sentences"]
    note_ids = list(meetings)
    n = sum(len(rows) for rows in meetings.values())
    a: dict[str, np.ndarray] = {
        "token_ids": np.zeros((n, max_tokens), dtype=np.int32),
        "span_tags": np.zeros((n, max_tokens), dtype=np.int8),
        "importance": np.zeros(n, dtype=np.int8),
        "confidence": np.zeros(n, dtype=np.float32),
        "zone": np.zeros(n, dtype=np.int8),
        "speaker": np.zeros(n, dtype=np.int8),
        "sentence_id": np.zeros(n, dtype=np.int32),
        "hard_negative": np.zeros(n, dtype=bool),
        "note_start": np.zeros(len(note_ids), dtype=np.int64),
        "note_length": np.zeros(len(note_ids), dtype=np.int32),
        "source": np.zeros(len(note_ids), dtype=np.int8),
    }
    windows: list[tuple[int, int, int]] = []
    counts = {"truncated_sentences": 0, "meetings_speakers_capped": 0}
    row = 0
    for m, note_id in enumerate(note_ids):
        rows = meetings[note_id]
        speakers, capped = speaker_numbers([r["speaker"] for r in rows])
        counts["meetings_speakers_capped"] += bool(capped)
        a["note_start"][m], a["note_length"][m] = row, len(rows)
        a["source"][m] = SOURCES.index(rows[0]["source"])
        for r, speaker in zip(rows, speakers, strict=True):
            ids, tags, truncated = encode_sentence(tokenizer, r["text"], r["spans"], max_tokens)
            a["token_ids"][row], a["span_tags"][row] = ids, tags
            a["importance"][row] = r["importance"]
            a["confidence"][row] = r["confidence"]
            a["zone"][row] = ZONES.index(r["zone"])
            a["speaker"][row] = speaker
            a["sentence_id"][row] = r["sentence_id"]
            a["hard_negative"][row] = r["hard_negative"]
            counts["truncated_sentences"] += truncated
            row += 1
        for start in window_starts(len(rows), size, data_cfg["window_overlap"]):
            windows.append((m, start, min(size, len(rows) - start)))
    w = np.asarray(windows, dtype=np.int32).reshape(-1, 3)
    a["window_note"], a["window_offset"], a["window_length"] = w[:, 0], w[:, 1], w[:, 2]
    counts = {"meetings": len(note_ids), "sentences": n, "windows": len(w), **counts}
    return a, note_ids, counts


@dataclass
class NotesSet:
    """Encoded split (memory-mapped arrays, see module docstring)."""

    note_ids: list[str]
    arrays: dict[str, np.ndarray]

    @classmethod
    def load(cls, directory: str | Path) -> NotesSet:
        d = Path(directory)
        arrays = {p.stem: np.load(p, mmap_mode="r") for p in sorted(d.glob("*.npy"))}
        return cls(json.loads((d / "note_ids.json").read_text()), arrays)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    @property
    def num_windows(self) -> int:
        return len(self.arrays["window_note"])


def bucket_for(length: int, buckets: Sequence[int]) -> int:
    """Smallest bucket that holds ``length`` sentences."""
    for b in sorted(buckets):
        if length <= b:
            return b
    raise ValueError(f"{length} sentences exceed the largest bucket {max(buckets)}")


def gather_windows(dataset: NotesSet, windows: Sequence[int], slots: int) -> dict[str, np.ndarray]:
    """Padded batch for the given window indices, ``slots`` sentences per window.

    Returns ``token_ids`` / ``span_tags`` (B, slots, T), per-sentence ``importance``,
    ``confidence``, ``zone``, ``speaker``, ``hard_negative`` (B, slots), ``sentence_mask``
    (B, slots) bool, ``rows`` (B, slots) int64 sentence rows (``-1`` on padding, to look up
    teacher arrays) and ``source`` (B,). Padded sentences have ``PAD_ID`` tokens, ``IGNORE``
    labels and speaker 0.
    """
    b = len(windows)
    rows = np.full((b, slots), -1, dtype=np.int64)
    for i, w in enumerate(windows):
        length = int(dataset["window_length"][w])
        if length > slots:
            raise ValueError(f"window {w} has {length} sentences > {slots} slots")
        note = int(dataset["window_note"][w])
        start = int(dataset["note_start"][note]) + int(dataset["window_offset"][w])
        rows[i, :length] = np.arange(start, start + length)
    mask = rows >= 0
    safe = np.where(mask, rows, 0)

    def take(name: str, fill: float | int) -> np.ndarray:
        values = np.asarray(dataset[name])[safe]
        pad = mask.reshape(mask.shape + (1,) * (values.ndim - 2))
        return np.where(pad, values, np.asarray(fill, dtype=values.dtype))

    return {
        "token_ids": take("token_ids", PAD_ID),
        "span_tags": take("span_tags", IGNORE),
        "importance": take("importance", IGNORE),
        "confidence": take("confidence", 0.0),
        "zone": take("zone", IGNORE),
        "speaker": take("speaker", 0),
        "hard_negative": take("hard_negative", False),
        "sentence_mask": mask,
        "rows": rows,
        "source": np.asarray(dataset["source"])[np.asarray(dataset["window_note"])[list(windows)]],
    }


def build_split(
    tokenizer: Tokenizer, cfg: dict[str, Any], split: str, out_dir: str | Path
) -> dict[str, int]:
    """Read, encode and save one split. Returns the counts written to ``meta.json``."""
    corpus = cfg["corpus"]
    meetings = read_corpus_split(
        cfg["paths"]["corpus_dir"], split, corpus["text_field"], corpus["sources"]
    )
    cap = cfg["data"]["max_meetings"]
    if cap is not None:
        meetings = dict(list(meetings.items())[:cap])
    arrays, note_ids, counts = encode_split(tokenizer, meetings, cfg["data"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(out / f"{name}.npy", array)
    (out / "note_ids.json").write_text(json.dumps(note_ids))
    (out / "meta.json").write_text(json.dumps(counts, indent=2) + "\n")
    return counts


def main(argv: list[str] | None = None) -> None:
    """Entry point: encode every split from a v3 config."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/model1_v3.yaml"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    tokenizer = load_tokenizer(cfg["paths"]["tokenizer_dir"])
    for split in SPLITS:
        counts = build_split(tokenizer, cfg, split, Path(cfg["paths"]["processed_dir"]) / split)
        log.info("%s: %s", split, counts)


if __name__ == "__main__":
    main()
