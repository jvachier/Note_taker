"""Descriptive statistics of the raw datasets (written to ``results/data/stats.json``)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from src.data.dialogsum import Dialogue
from src.data.wikitext import article_title


def _summary(values: Sequence[int]) -> dict[str, float]:
    arr = np.asarray(values)
    return {
        "mean": round(float(arr.mean()), 2),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(arr.max()),
    }


def dialogsum_stats(dialogues: Sequence[Dialogue]) -> dict[str, Any]:
    """Counts and length distributions (whitespace words) for one DialogSum split."""
    speakers = Counter(s for d in dialogues for s in d.speakers)
    return {
        "dialogues": len(dialogues),
        "utterances_per_dialogue": _summary([len(d.utterances) for d in dialogues]),
        "words_per_utterance": _summary([len(u.split()) for d in dialogues for u in d.utterances]),
        "words_per_dialogue": _summary(
            [sum(len(u.split()) for u in d.utterances) for d in dialogues]
        ),
        "speakers_per_dialogue": _summary([len(set(d.speakers)) for d in dialogues]),
        "utterances_by_speaker": {str(k): v for k, v in sorted(speakers.items())},
        "references_per_dialogue": _summary([len(d.summaries) for d in dialogues]),
    }


def wikitext_stats(lines: Iterable[str]) -> dict[str, Any]:
    """Line, article and whitespace-word counts for one WikiText split."""
    n_lines = n_nonempty = n_articles = n_words = 0
    for line in lines:
        n_lines += 1
        if line.strip():
            n_nonempty += 1
            n_words += len(line.split())
        if article_title(line) is not None:
            n_articles += 1
    return {
        "lines": n_lines,
        "nonempty_lines": n_nonempty,
        "articles": n_articles,
        "words": n_words,
    }
