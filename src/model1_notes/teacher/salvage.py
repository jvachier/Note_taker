"""Rebuild the Qwen tables from saved raw answers, without running the LLM (PLAN.md 10.3).

Usage::

    uv run python -m src.model1_notes.teacher.salvage \\
        --raw results_qwen/data/labels/model1_v3/qwen --corpus data/raw/meeting_notes_v1 \\
        --out data/labels/model1_v3 --config configs/model1_v3.yaml

The first Qwen3-0.6B run (2026-10-21) wrote valid-looking answers in a format the strict
reader rejected (objects instead of lists, ``<|im_end|>`` left in, answers cut at the token
budget), so 99% of its meetings were discarded. Every answer is kept in
``shards/<k>/raw_<split>.jsonl`` (one record per window: ``note_id``, ``start``, ``answer``,
``retry_answer``). This module re-reads them with the lenient reader
(:func:`~src.model1_notes.teacher.answer.parse_entries`), keeps the better of the first answer and
the retry per window, merges each meeting per sentence
(:func:`~src.model1_notes.teacher.label.merge_meeting`) and writes ``<out>/qwen/<split>.parquet``,
``<out>/qwen/gate.json`` and ``<out>/qwen/salvage.json`` (coverage per split).

The raw records do not keep the token log-probabilities, so the soft score is the integer score
/ 3 (0, 1/3, 2/3, 1). No ML imports: it runs anywhere.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.model1_notes.teacher.answer import labels_from_answer, parse_entries
from src.model1_notes.teacher.gate import run_gate
from src.model1_notes.teacher.label import merge_meeting, read_meetings
from src.model1_notes.teacher.windows import REASONS

SPLITS = ("validation", "test", "train")


def window_labels(record: dict[str, Any], n: int) -> dict[int, Any] | None:
    """The better of a window's first answer and retry (more labelled sentences)."""
    best = None
    for text in (record.get("answer"), record.get("retry_answer")):
        if parse_entries(text, n, REASONS):
            labels = labels_from_answer(text, n, REASONS)
            if best is None or len(labels) > len(best):
                best = labels
    return best


def salvage_split(
    raw_dir: Path, corpus_dir: str | Path, split: str, size: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Per-sentence rows of one split from every shard's raw answers, and counts."""
    meetings = read_meetings(corpus_dir, split)
    windows: dict[str, dict[int, dict[str, Any]]] = {}
    for path in sorted(raw_dir.glob(f"shards/*/raw_{split}.jsonl")):
        for line in path.open():
            if line.strip():
                rec = json.loads(line)
                windows.setdefault(rec["note_id"], {})[int(rec["start"])] = rec
    rows, counts = [], {"meetings": 0, "meetings_labelled": 0, "sentences": 0, "labelled": 0}
    for note_id, by_start in windows.items():
        sentences = meetings.get(note_id)
        if not sentences:
            continue
        starts = sorted(by_start)
        labels = [window_labels(by_start[s], len(sentences[s : s + size])) for s in starts]
        merged = merge_meeting(sentences, starts, labels, size) or []
        counts["meetings"] += 1
        counts["meetings_labelled"] += bool(merged)
        counts["sentences"] += len(sentences)
        counts["labelled"] += len(merged)
        rows += [{"note_id": note_id, **s} for s in merged]
    return rows, counts


def write_table(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """The Qwen table read by the student (same schema as ``label.merge_shards``)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("note_id", pa.string()),
            ("sentence_id", pa.int32()),
            ("score", pa.float32()),
            ("raw_score", pa.int8()),
            ("reason", pa.string()),
            ("depends_on", pa.int32()),
        ]
    )
    rows = sorted(rows, key=lambda r: (r["note_id"], r["sentence_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(rows), schema=schema), path)


def salvage(
    raw_dir: str | Path,
    corpus_dir: str | Path,
    out_dir: str | Path,
    teacher_cfg: dict[str, Any],
    baselines: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild every split's table and the gate (see module docstring)."""
    raw_dir, out = Path(raw_dir), Path(out_dir)
    report = {}
    for split in SPLITS:
        rows, counts = salvage_split(raw_dir, corpus_dir, split, teacher_cfg["window_sentences"])
        write_table(out / "qwen" / f"{split}.parquet", rows)
        counts["coverage"] = round(counts["labelled"] / max(counts["sentences"], 1), 4)
        report[split] = counts
    report["gate"] = run_gate(corpus_dir, out, baselines, teacher_cfg["gate_split"])
    report["soft_scores"] = "integer score / 3 (raw answers keep no log-probabilities)"
    (out / "qwen" / "salvage.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    """Entry point (see module docstring)."""
    import yaml

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True, help="the qwen/ folder with shards/")
    parser.add_argument("--corpus", type=Path, default=Path("data/raw/meeting_notes_v1"))
    parser.add_argument("--out", type=Path, default=Path("data/labels/model1_v3"))
    parser.add_argument("--config", type=Path, default=Path("configs/model1_v3.yaml"))
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    report = salvage(args.raw, args.corpus, args.out, cfg["teacher"], cfg["baselines"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
