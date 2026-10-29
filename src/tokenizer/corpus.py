"""Text corpora for tokenizer training and evaluation, read from the raw files of Stage 2.

Sources are named ``<dataset>_<split>``: ``wikitext_train``, ``dialogsum_test``, ...
WikiText yields detokenised paragraph lines (headings skipped); DialogSum yields one
utterance per item (speaker tags already removed in Stage 2).
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice
from pathlib import Path

from src.data.io import read_jsonl, read_lines
from src.data.wikitext import iter_clean_text


def iter_source(raw_dir: str | Path, source: str, max_items: int | None = None) -> Iterator[str]:
    """Yield the texts of one source, optionally only the first ``max_items``."""
    dataset, _, split = source.partition("_")
    raw_dir = Path(raw_dir)
    if dataset == "wikitext":
        texts = iter_clean_text(read_lines(raw_dir / "wikitext" / f"{split}.txt"))
    elif dataset == "dialogsum":
        texts = (
            utt
            for record in read_jsonl(raw_dir / "dialogsum" / f"{split}.jsonl")
            for utt in record["utterances"]
        )
    else:
        raise ValueError(f"Unknown corpus source: {source!r}")
    return islice(texts, max_items)


def iter_corpus(
    raw_dir: str | Path, sources: list[str], max_items_per_source: int | None = None
) -> Iterator[str]:
    """Yield the texts of several sources, one after the other."""
    for source in sources:
        yield from iter_source(raw_dir, source, max_items_per_source)
