"""Summary quality against several references: ROUGE-1/2/L and BERTScore (evaluation only).

``rouge_score`` and ``bert_score`` are imported here and nowhere in training code. BERTScore
pulls in PyTorch, so it is an optional extra (``uv sync --extra eval``) imported lazily.
With several references, ROUGE takes the mean over references and BERTScore the best
reference (``bert_score``'s own multi-reference rule).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rouge_score.rouge_scorer import RougeScorer

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")


def rouge(
    predictions: Sequence[str], references: Sequence[Sequence[str]], use_stemmer: bool = True
) -> dict[str, float]:
    """Mean F1 per ROUGE type over dialogues; each dialogue averages over its references."""
    scorer = RougeScorer(list(ROUGE_TYPES), use_stemmer=use_stemmer)
    totals = {t: [] for t in ROUGE_TYPES}
    for pred, refs in zip(predictions, references, strict=True):
        scores = [scorer.score(ref, pred) for ref in refs]
        for t in ROUGE_TYPES:
            totals[t].append(np.mean([s[t].fmeasure for s in scores]))
    return {t: float(np.mean(v)) if v else 0.0 for t, v in totals.items()}


def bertscore(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    model_type: str = "roberta-large",
    batch_size: int = 32,
) -> float:
    """Mean BERTScore F1 (``bert_score``, best reference per dialogue, no baseline rescaling)."""
    from bert_score import score

    _, _, f1 = score(
        list(predictions),
        [list(r) for r in references],
        model_type=model_type,
        batch_size=batch_size,
        verbose=False,
    )
    return float(f1.mean())
