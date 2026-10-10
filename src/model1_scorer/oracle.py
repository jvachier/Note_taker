"""Stage 6b: greedy ROUGE oracle labels (PLAN.md section 5.4).

Usage::

    uv run python -m src.model1_scorer.oracle --config configs/oracle.yaml

Per dialogue, repeatedly add the utterance that most increases ROUGE-1 F1 + ROUGE-2 F1 of the
selection (selected utterances in dialogue order) against the reference summary; stop when no
utterance gives a gain or ``max_utterances`` are selected. Selected = 1, others = 0, no tags.
With several references (test split) the objective is the mean over references.

ROUGE-N F1 is computed here from n-gram counts on ``rouge_score``'s own tokenizer (identical to
``RougeScorer``, checked in the tests) so each utterance is tokenised once.

Writes ``<output_dir>/<split>.jsonl`` (``labels``, ``selected``, ``objective``) and
``stats.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rouge_score.tokenizers import DefaultTokenizer

from src.config import load_config
from src.data.io import read_jsonl, write_jsonl

log = logging.getLogger(__name__)
ORDERS = {"rouge1": 1, "rouge2": 2}


def ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    """Counts of the n-grams of ``tokens``."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n_f1(prediction: Counter, reference: Counter) -> float:
    """ROUGE-N F1 from n-gram counts (clipped overlap), as in ``rouge_score``."""
    overlap = sum((prediction & reference).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(prediction.values())
    recall = overlap / sum(reference.values())
    return 2 * precision * recall / (precision + recall)


class Objective:
    """Mean over references of the summed ROUGE-N F1 of a selection of utterances."""

    def __init__(
        self,
        utterances: Sequence[str],
        references: Sequence[str],
        rouge_types: Sequence[str],
        tokenizer: DefaultTokenizer,
    ) -> None:
        self.orders = [ORDERS[t] for t in rouge_types]
        self.tokens = [tokenizer.tokenize(u) for u in utterances]
        refs = [tokenizer.tokenize(r) for r in references]
        self.references = [{n: ngrams(r, n) for n in self.orders} for r in refs]

    def __call__(self, selected: Sequence[int]) -> float:
        tokens = [t for i in sorted(selected) for t in self.tokens[i]]
        pred = {n: ngrams(tokens, n) for n in self.orders}
        per_ref = [sum(rouge_n_f1(pred[n], ref[n]) for n in self.orders) for ref in self.references]
        return sum(per_ref) / len(per_ref)


def greedy_oracle(
    objective: Objective, n_utterances: int, max_selected: int
) -> tuple[list[int], float]:
    """Greedy selection; returns (selected indices in dialogue order, final objective)."""
    selected: list[int] = []
    best = 0.0
    while len(selected) < max_selected:
        gains = [(objective([*selected, i]), i) for i in range(n_utterances) if i not in selected]
        if not gains:
            break
        score, choice = max(gains, key=lambda si: (si[0], -si[1]))  # ties -> earliest
        if score <= best:
            break
        selected.append(choice)
        best = score
    return sorted(selected), best


def label_record(record: dict[str, Any], cfg: dict[str, Any], tokenizer: DefaultTokenizer):
    """Oracle labels for one dialogue record."""
    objective = Objective(record["utterances"], record["summaries"], cfg["rouge_types"], tokenizer)
    n = len(record["utterances"])
    selected, score = greedy_oracle(objective, n, cfg["max_utterances"])
    labels = [int(i in selected) for i in range(n)]
    return {
        "id": record["id"],
        "labels": labels,
        "selected": selected,
        "objective": round(score, 6),
    }


def main(argv: list[str] | None = None) -> None:
    """Entry point: label every configured split."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/oracle.yaml"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    tokenizer = DefaultTokenizer(use_stemmer=cfg["use_stemmer"])
    raw = Path(cfg["paths"]["raw_dir"]) / "dialogsum"
    out = Path(cfg["paths"]["output_dir"])
    stats: dict[str, Any] = {"config": cfg, "splits": {}}
    for split in cfg["splits"]:
        start = time.time()
        records = [label_record(r, cfg, tokenizer) for r in read_jsonl(raw / f"{split}.jsonl")]
        write_jsonl(out / f"{split}.jsonl", records)
        n = max(len(records), 1)
        stats["splits"][split] = {
            "dialogues": len(records),
            "mean_selected": round(sum(len(r["selected"]) for r in records) / n, 3),
            "mean_objective": round(sum(r["objective"] for r in records) / n, 4),
            "none_selected": sum(not r["selected"] for r in records),
            "seconds": round(time.time() - start, 1),
        }
        log.info("oracle/%s: %s", split, stats["splits"][split])
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
