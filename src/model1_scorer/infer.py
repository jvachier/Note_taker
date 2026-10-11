"""Model 1 inference on one raw dialogue: utterances in, scores, tags and notes out.

The dialogue is encoded exactly like the training arrays
(:func:`~src.model1_scorer.dialogue.encode_dialogue`), the student predicts one importance
score (sigmoid) and, if it has a tag head, one tag per utterance. Utterances that did not fit
in ``max_len`` get score 0 and tag ``none``. Notes follow the shared inference rule
(:mod:`~src.model1_scorer.notes`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import keras
import numpy as np
from tokenizers import Tokenizer

import src.model1_scorer.model  # noqa: F401  (registers the custom layers for loading)
from src.model1_scorer.data import TAGS
from src.model1_scorer.dialogue import encode_dialogue
from src.model1_scorer.notes import render_notes, select_top_k


@dataclass
class ScoredDialogue:
    """Per-utterance predictions for one dialogue.

    Attributes:
        scores: importance in [0, 1] per utterance.
        tags: predicted tag per utterance, ``None`` for a student without a tag head.
        selected: indices of the top-k utterances, in dialogue order.
        dropped: utterances that did not fit in ``max_len`` (scored 0).
    """

    scores: list[float]
    tags: list[str] | None
    selected: list[int]
    dropped: int


def score_dialogue(
    model: keras.Model,
    tokenizer: Tokenizer,
    speakers: Sequence[int],
    utterances: Sequence[str],
    max_len: int,
    top_k: int,
) -> ScoredDialogue:
    """Score every utterance of one dialogue and select the top-k (see module docstring)."""
    enc = encode_dialogue(tokenizer, speakers, utterances, max_len)
    k = len(enc.cls_positions)
    out = model.predict_on_batch(
        {
            "token_ids": enc.token_ids[None, :],
            "speaker_ids": enc.speaker_ids[None, :].astype(np.int32),
            "cls_positions": enc.cls_positions[None, :],
        }
    )
    logits = np.asarray(out["importance"], dtype=np.float64)[0, :k]
    scores = [float(s) for s in 1.0 / (1.0 + np.exp(-logits))] + [0.0] * enc.dropped
    tags = None
    if "tags" in out:
        tags = [TAGS[t] for t in np.argmax(np.asarray(out["tags"])[0, :k], axis=-1)]
        tags += ["none"] * enc.dropped
    return ScoredDialogue(scores, tags, select_top_k(scores, top_k), enc.dropped)


def dialogue_notes(
    scored: ScoredDialogue, speakers: Sequence[int], utterances: Sequence[str]
) -> str:
    """Markdown notes for a scored dialogue (top-k utterances grouped by tag)."""
    return render_notes(utterances, speakers, scored.selected, scored.tags)
