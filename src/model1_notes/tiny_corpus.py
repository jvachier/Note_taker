"""A tiny corpus in the ``meeting-notes-extractive-v1`` release layout, for tests and smoke runs.

Usage::

    uv run python -m src.model1_notes.tiny_corpus --out data/raw/meeting_notes_tiny

Writes ``<out>/data/{synthetic,ami}/{sentences,notes}/{train,validation,test}.parquet`` (the
release layout; AMI notes carry made-up summary sections). Each meeting repeats one six-sentence
exchange (context, discussion, decision, action item with owner/deadline spans, filler,
closing) 1-6 times, cycling through the three sources, so meetings of several lengths, all
zones, AMI zone restrictions and hard negatives all occur. It is not real data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SPLITS = ("train", "validation", "test")
SOURCES = ("synthetic", "synthetic_hn", "ami")
MEETING_SENTENCES = [
    ("Ana", "Thanks for joining, this is the weekly sync.", 0, "context", False),
    ("Ben", "The load tests failed again on Monday.", 1, "discussion", False),
    ("Ana", "Okay, we hold the release until they pass.", 1, "decision", False),
    ("Ben", "Ben will fix the test rig by Friday.", 1, "action_item", True),
    ("Cleo", "Sounds good.", 0, "discussion", False),
    ("Ana", "See you next week.", 0, "closing", False),
]


def _spans(text: str) -> list[dict[str, Any]]:
    """Owner ("Ben", at the start) and deadline ("Friday") spans of the action-item sentence."""
    day = text.index("Friday")
    return [
        {"start": 0, "end": 3, "role": "owner"},
        {"start": day, "end": day + len("Friday"), "role": "deadline"},
    ]


def tiny_corpus_rows(
    note_id: str, split: str, source: str = "synthetic", repeat: int = 1
) -> list[dict[str, Any]]:
    """Rows of one meeting in the corpus ``sentences`` schema (``repeat`` x 6 sentences)."""
    ami = source == "ami"
    rows = []
    for k in range(repeat):
        for j, (speaker, text, importance, zone, has_spans) in enumerate(MEETING_SENTENCES):
            if ami and zone in ("context", "closing"):
                zone = "discussion"  # AMI has no context / closing annotation
            rows.append(
                {
                    "note_id": note_id,
                    "source": source,
                    "meeting_type": "ami_design" if ami else "standup",
                    "domain": "product_design" if ami else "engineering",
                    "split": split,
                    "sentence_id": k * len(MEETING_SENTENCES) + j,
                    "speaker": speaker,
                    "sentence_text": text,
                    "sentence_text_asr": text.lower().replace(",", "").replace(".", ""),
                    "importance": importance,
                    "confidence": 0.8 if ami else 1.0,
                    "zone": zone,
                    "segment_id": None,
                    "family_id": None,
                    "hard_negative": source == "synthetic_hn",
                    "token_count": len(text.split()),
                    "spans": _spans(text) if has_spans and not ami else [],
                }
            )
    return rows


def tiny_note_row(note_id: str, split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The ``notes`` row of a meeting (AMI rows carry summary sections)."""
    ami = rows[0]["source"] == "ami"
    return {
        "note_id": note_id,
        "source": rows[0]["source"],
        "meeting_type": rows[0]["meeting_type"],
        "split": split,
        "n_sentences": len(rows),
        "n_words": sum(r["token_count"] for r in rows),
        "n_participants": len({r["speaker"] for r in rows}),
        "summary_abstract": "The team discussed the failing load tests." if ami else None,
        "summary_decisions": "The release is held until the tests pass." if ami else None,
        "summary_actions": "Ben fixes the test rig by Friday." if ami else None,
        "summary_problems": "The load tests fail." if ami else None,
    }


def write_tiny_corpus(corpus_dir: str | Path, meetings_per_split: int = 3) -> None:
    """Write the release layout; meeting ``i`` has ``6 * (1 + i % 6)`` sentences."""
    for split in SPLITS:
        layers: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {
            "synthetic": ([], []),
            "ami": ([], []),
        }
        for i in range(meetings_per_split):
            source = SOURCES[i % 3]
            rows = tiny_corpus_rows(f"{split}-{i}", split, source, repeat=1 + i % 6)
            sentences, notes = layers["ami" if source == "ami" else "synthetic"]
            sentences += rows
            notes.append(tiny_note_row(f"{split}-{i}", split, rows))
        for layer, (sentences, notes) in layers.items():
            for table, records in (("sentences", sentences), ("notes", notes)):
                out = Path(corpus_dir) / "data" / layer / table
                out.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist(records), out / f"{split}.parquet")


def main(argv: list[str] | None = None) -> None:
    """Entry point: write the tiny corpus."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/raw/meeting_notes_tiny"))
    parser.add_argument("--meetings", type=int, default=24, help="meetings per split")
    args = parser.parse_args(argv)
    write_tiny_corpus(args.out, args.meetings)


if __name__ == "__main__":
    main()
