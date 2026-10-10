"""Stage 2: download WikiText-103 (raw) and DialogSum, check the splits, write raw files.

Usage::

    uv run python -m src.data.download --config configs/data.yaml

Outputs (under ``paths.raw_dir``):
    ``wikitext/{split}.txt``     original lines, one per line (blank lines kept)
    ``dialogsum/{split}.jsonl``  one :class:`~src.data.dialogsum.Dialogue` per line
    ``manifest.json``            dataset revisions, row counts, overlap report, timings
and ``paths.stats_path`` with descriptive statistics.

Official splits are kept, except that train items duplicated in validation/test are removed
(DialogSum: identical utterance text; WikiText: same article title). Validation and test are
never modified. Remaining validation/test overlap is only reported.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset

from src.config import load_config
from src.data.dialogsum import Dialogue, build_dialogue, count_empty_turns, group_test_rows
from src.data.io import write_jsonl, write_lines
from src.data.splits import assert_disjoint, find_overlaps
from src.data.stats import dialogsum_stats, wikitext_stats
from src.data.wikitext import article_titles, drop_articles
from src.seed import set_seed

log = logging.getLogger(__name__)


def dialogue_text_overlap(dialogues: dict[str, list[Dialogue]]) -> dict[str, list[list[str]]]:
    """Find dialogues whose utterance text (speaker tags ignored) occurs in two splits.

    Returns ``{"a/b": [[ids in a], [ids in b]], ...}`` per shared text, sorted by first ID.
    """
    by_text: dict[str, dict[str, list[str]]] = {}
    for split, items in dialogues.items():
        for d in items:
            by_text.setdefault(split, {}).setdefault("\n".join(d.utterances), []).append(d.id)
    report = {}
    for (a, b), texts in find_overlaps(by_text).items():
        report[f"{a}/{b}"] = sorted([by_text[a][t], by_text[b][t]] for t in texts)
    return report


def drop_train_duplicates(dialogues: dict[str, list[Dialogue]], against: list[str]) -> list[str]:
    """Remove (in place) train dialogues whose utterance text occurs in an ``against`` split.

    Returns the dropped train IDs, sorted.
    """
    eval_texts = {"\n".join(d.utterances) for split in against for d in dialogues[split]}
    kept, dropped = [], []
    for d in dialogues["train"]:
        (dropped if "\n".join(d.utterances) in eval_texts else kept).append(d)
    dialogues["train"] = kept
    return sorted(d.id for d in dropped)


def prepare_dialogsum(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Download DialogSum, parse utterances, merge test references, check splits, write JSONL."""
    ds = load_dataset(cfg["hf_path"], revision=cfg["revision"])
    max_speakers = cfg["max_speakers"]
    dialogues: dict[str, list[Dialogue]] = {}
    empty_turns: dict[str, int] = {}
    for split in cfg["splits"]:
        rows = list(ds[split])
        empty_turns[split] = sum(count_empty_turns(r["dialogue"]) for r in rows)
        if split == "test":
            dialogues[split] = group_test_rows(rows, max_speakers)
        else:
            dialogues[split] = [build_dialogue(r, split, max_speakers) for r in rows]
        log.info(
            "dialogsum/%s: %d rows -> %d dialogues (%d empty turns dropped)",
            split,
            len(rows),
            len(dialogues[split]),
            empty_turns[split],
        )

    assert_disjoint({s: [d.id for d in ds_] for s, ds_ in dialogues.items()}, "Dialogue ID")
    against = cfg["drop_train_duplicates_of"]
    dropped = drop_train_duplicates(dialogues, against)
    log.info("dialogsum/train: dropped %d duplicates of %s", len(dropped), against)
    text_overlap = dialogue_text_overlap(dialogues)
    if any("train" in pair.split("/") for pair in text_overlap):
        raise ValueError(f"Train still overlaps an evaluation split: {list(text_overlap)}")
    for pair, ids in text_overlap.items():
        log.warning("dialogsum %s share %d dialogue text(s), e.g. %s", pair, len(ids), ids[:3])
    for split, items in dialogues.items():
        write_jsonl(out_dir / f"{split}.jsonl", (d.to_dict() for d in items))
    return {
        "revision": cfg["revision"],
        "rows": {s: ds[s].num_rows for s in cfg["splits"]},
        "dialogues": {s: len(v) for s, v in dialogues.items()},
        "empty_turns_dropped": empty_turns,
        "train_duplicates_dropped": dropped,
        "text_overlap": text_overlap,
        "stats": {s: dialogsum_stats(v) for s, v in dialogues.items()},
    }


def prepare_wikitext(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Download WikiText-103 raw, drop train articles also in validation/test, write text."""
    ds = load_dataset(cfg["hf_path"], cfg["hf_config"], revision=cfg["revision"])
    lines_by_split = {split: ds[split]["text"] for split in cfg["splits"]}
    against = cfg["drop_train_articles_in"]
    eval_titles = {t for split in against for t in article_titles(lines_by_split[split])}
    lines_by_split["train"], n_dropped = drop_articles(lines_by_split["train"], eval_titles)
    log.info("wikitext/train: dropped %d article(s) whose title is in %s", n_dropped, against)

    titles: dict[str, list[str]] = {}
    stats: dict[str, Any] = {}
    for split, lines in lines_by_split.items():
        write_lines(out_dir / f"{split}.txt", lines)
        titles[split] = article_titles(lines)
        stats[split] = wikitext_stats(lines)
        log.info("wikitext/%s: %d lines, %d articles", split, len(lines), len(titles[split]))
    overlaps = {f"{a}/{b}": sorted(s) for (a, b), s in find_overlaps(titles).items()}
    for pair, shared in overlaps.items():
        log.warning("wikitext %s share %d article title(s): %s", pair, len(shared), shared[:5])
    if any("train" in pair.split("/") for pair in overlaps):
        raise ValueError(f"Train still shares article titles with an eval split: {overlaps}")
    return {
        "revision": cfg["revision"],
        "rows": {s: ds[s].num_rows for s in cfg["splits"]},
        "train_articles_dropped": n_dropped,
        "title_overlap": overlaps,
        "stats": stats,
    }


def main(argv: list[str] | None = None) -> None:
    """Entry point: run the whole stage from a config file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    raw_dir = Path(cfg["paths"]["raw_dir"])
    start = time.time()

    dialogsum = prepare_dialogsum(cfg["dialogsum"], raw_dir / "dialogsum")
    wikitext = prepare_wikitext(cfg["wikitext"], raw_dir / "wikitext")

    stats = {"dialogsum": dialogsum.pop("stats"), "wikitext": wikitext.pop("stats")}
    manifest = {
        "stage": "data",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": round(time.time() - start, 1),
        "config": cfg,
        "dialogsum": dialogsum,
        "wikitext": wikitext,
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stats_path = Path(cfg["paths"]["stats_path"])
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    log.info("Wrote %s and %s", raw_dir / "manifest.json", stats_path)


if __name__ == "__main__":
    main()
