"""Quality gate for the Qwen teacher, before any distillation (PLAN.md 10.3).

On the validation split, Qwen's labels (soft score > 0.5 = keep) are compared with the gold
labels, next to the keyword-rule baseline on the same sentences:

- ``importance_f1``: Qwen vs gold; ``keyword_f1``: the keyword rule vs gold; ``lead_f1``: the
  best Lead-k on the same meetings (fraction tuned on them, so an optimistic baseline);
- ``hard_negative_fp``: share of gold-0 hard-negative sentences Qwen keeps (lower is better),
  and the same for the keyword rule;
- ``reason_zone_agreement``: share of gold kept decisions labelled ``agreed_outcome`` and of
  gold kept action items labelled ``commitment_with_owner``;
- ``coverage``: share of validation sentences Qwen labelled.

``passed`` is true when Qwen beats both baselines on importance F1 (the keyword rule alone is a
very low bar: about 0.13 F1 on synthetic validation). Notebook 2 refuses to
distil Qwen when it is false. No ML imports: runs in the teacher environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.model1_notes.baselines import KeywordRule, binary_f1, tune_lead

REASON_FOR_ZONE = {"decision": "agreed_outcome", "action_item": "commitment_with_owner"}


def gate_metrics(
    gold: list[dict[str, Any]],
    qwen: dict[tuple[str, int], dict[str, Any]],
    rule: KeywordRule,
    lead_fractions: list[float],
) -> dict[str, Any]:
    """Gate metrics on the gold sentences that Qwen labelled (see module docstring)."""
    rows = [
        (g, qwen[(g["note_id"], g["sentence_id"])])
        for g in gold
        if (g["note_id"], g["sentence_id"]) in qwen
    ]
    truth = [g["importance"] for g, _ in rows]
    pred = [int(q["score"] > 0.5) for _, q in rows]
    keyword = [rule.importance(g["sentence_text"]) for g, _ in rows]
    hn = [i for i, (g, _) in enumerate(rows) if g["hard_negative"] and not g["importance"]]
    agree = [
        q["reason"] == REASON_FOR_ZONE[g["zone"]]
        for g, q in rows
        if g["importance"] and g["zone"] in REASON_FOR_ZONE
    ]
    q_f1, k_f1 = binary_f1(truth, pred), binary_f1(truth, keyword)
    meetings: dict[str, list[tuple[int, int]]] = {}
    for g, _ in rows:
        meetings.setdefault(g["note_id"], []).append((g["sentence_id"], g["importance"]))
    ordered = [[imp for _, imp in sorted(m)] for m in meetings.values()]
    lead_frac, l_f1 = tune_lead(ordered, lead_fractions) if ordered else (None, 0.0)
    return {
        "sentences": len(rows),
        "coverage": round(len(rows) / max(len(gold), 1), 4),
        "importance_f1": round(q_f1, 4),
        "keyword_f1": round(k_f1, 4),
        "lead_f1": round(l_f1, 4),
        "lead_fraction": lead_frac,
        "hard_negative_fp": round(sum(pred[i] for i in hn) / max(len(hn), 1), 4),
        "keyword_hard_negative_fp": round(sum(keyword[i] for i in hn) / max(len(hn), 1), 4),
        "reason_zone_agreement": round(sum(agree) / max(len(agree), 1), 4),
        "passed": bool(rows) and q_f1 > max(k_f1, l_f1),
    }


def run_gate(
    corpus_dir: str | Path, out_dir: str | Path, baselines: dict[str, Any], split: str
) -> dict[str, Any]:
    """Compute the gate on ``split`` and write ``<out_dir>/qwen/gate.json``."""
    import pyarrow.parquet as pq

    columns = ["note_id", "sentence_id", "sentence_text", "importance", "zone", "hard_negative"]
    gold = []
    for layer in ("synthetic", "ami"):
        path = Path(corpus_dir) / "data" / layer / "sentences" / f"{split}.parquet"
        if path.exists():
            gold += pq.read_table(path, columns=columns).to_pylist()
    table = pq.read_table(Path(out_dir) / "qwen" / f"{split}.parquet").to_pylist()
    qwen = {(r["note_id"], r["sentence_id"]): r for r in table}
    rule = KeywordRule(baselines["keywords"])
    result = {"split": split, **gate_metrics(gold, qwen, rule, baselines["lead_fractions"])}
    (Path(out_dir) / "qwen" / "gate.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
