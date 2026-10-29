"""Small helpers to read and write the raw text files under ``data/``."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Write one JSON object per line (UTF-8, non-ASCII kept). Returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield the JSON objects of a JSONL file."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_lines(path: str | Path, lines: Iterable[str]) -> int:
    """Write text lines, one per line, stripping any trailing newline first.

    Empty lines are kept, so line structure survives a round trip with :func:`read_lines`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")
            n += 1
    return n


def read_lines(path: str | Path) -> Iterator[str]:
    """Yield the lines of a text file without their trailing newline."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")
