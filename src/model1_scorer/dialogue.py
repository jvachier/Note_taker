"""BertSum-style dialogue encoding for Model 1 (PLAN.md section 5.2).

A dialogue becomes ``[CLS] utt_1 [SEP] [CLS] utt_2 [SEP] ...``. Every token carries the
speaker ID of its utterance (1..8; 0 = padding), and the ``[CLS]`` positions index the
utterances. A dialogue longer than ``max_len`` is fitted by truncating the longest utterances
first (water-filling: every utterance is cut to a common cap, the largest cap that fits).
Utterances that do not even fit as ``[CLS] [SEP]`` pairs are dropped from the end.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from tokenizers import Tokenizer

from src.tokenizer.tokenizer import CLS_ID, PAD_ID, SEP_ID


@dataclass
class EncodedDialogue:
    """One dialogue as fixed-length arrays.

    Attributes:
        token_ids: (max_len,) int32, ``PAD_ID`` after the last ``[SEP]``.
        speaker_ids: (max_len,) int8, the speaker of each token's utterance, 0 on padding.
        cls_positions: (n_kept,) int32, position of each kept utterance's ``[CLS]``.
        truncated: number of utterances that lost tokens.
        dropped: number of utterances dropped from the end (no room for ``[CLS] [SEP]``).
    """

    token_ids: np.ndarray
    speaker_ids: np.ndarray
    cls_positions: np.ndarray
    truncated: int = 0
    dropped: int = 0


def length_cap(lengths: Sequence[int], budget: int) -> int:
    """Largest ``cap`` with ``sum(min(n, cap)) <= budget`` (``max(lengths)`` if all fit)."""
    if sum(lengths) <= budget:
        return max(lengths, default=0)
    lo, hi = 0, max(lengths)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(n, mid) for n in lengths) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def encode_dialogue(
    tokenizer: Tokenizer, speakers: Sequence[int], utterances: Sequence[str], max_len: int
) -> EncodedDialogue:
    """Encode one dialogue (see module docstring)."""
    if len(speakers) != len(utterances):
        raise ValueError("speakers and utterances must have the same length")
    encodings = tokenizer.encode_batch(list(utterances), add_special_tokens=False)
    pieces = [enc.ids for enc in encodings]
    keep = min(len(pieces), max_len // 2)
    cap = length_cap([len(p) for p in pieces[:keep]], max_len - 2 * keep)

    token_ids = np.full(max_len, PAD_ID, dtype=np.int32)
    speaker_ids = np.zeros(max_len, dtype=np.int8)
    cls_positions = np.zeros(keep, dtype=np.int32)
    pos = truncated = 0
    for i in range(keep):
        ids = pieces[i][:cap]
        truncated += len(ids) < len(pieces[i])
        n = len(ids) + 2
        cls_positions[i] = pos
        token_ids[pos : pos + n] = [CLS_ID, *ids, SEP_ID]
        speaker_ids[pos : pos + n] = speakers[i]
        pos += n
    return EncodedDialogue(token_ids, speaker_ids, cls_positions, truncated, len(pieces) - keep)
