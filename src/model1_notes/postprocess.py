"""Rule-based tidying of Model 1 v3 notes (no model, no change to which sentences are kept).

1. **Confirmations**: an unkept short reply ("Yes, I will book them today.", "Okay, will do.")
   is attached under the kept request or decision it answers: the sentence the model's
   ``depends_on`` head points to (only for models trained with Qwen's links:
   ``use_pointer``), else the nearest kept sentence among the ``lookback``
   previous ones (skipping unkept fragments in between); a confirmation ("Yes", "Okay",
   ...) after an attached reply joins the same request, a bare commitment ("I'll ...") does not.
   Only across speakers. A
   first-person reply supplies the request's missing owner (the replying speaker) and deadline.
2. **Owner from the speaker**: a kept first-person action item ("I'll …", "I will …") without a
   predicted owner gets its speaker as owner (the training data names owners, so the span head
   never tags "I").

A third rule, keeping unkept first-person commitments ("rescue"), was measured on test and
dropped: only 11% of such sentences are important in the gold labels, and it lowered AMI
action-item F1 from 0.111 to 0.08-0.09 (2026-10-25).

Patterns and limits come from the config (``inference.postprocess``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

ATTACH_TO = ("action_item", "decision")


def _matcher(patterns: Sequence[str]):
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    return lambda text: any(p.search(text) for p in compiled)


def postprocess(notes: list, depends: Sequence[int], cfg: dict[str, Any] | None) -> list:
    """Apply the rules above in place to ``notes`` (:class:`~src.model1_notes.notes.Note`)
    and return them. ``depends[i]`` is the sentence ``i`` depends on, or -1."""
    if not cfg:
        return notes
    is_commitment = _matcher(cfg["commitment_patterns"])
    is_confirmation = _matcher(cfg["confirmation_patterns"])
    # Rule 2 first, so a reply never takes the owner of the speaker's own commitment.
    for note in notes:
        if note.kept and note.zone == "action_item" and not note.owner and is_commitment(note.text):
            note.owner = note.speaker
    attached: dict[int, int] = {}  # reply index -> request index
    for note in notes:
        if note.kept or len(note.text.split()) > cfg["confirmation_max_words"]:
            continue
        confirms = is_confirmation(note.text)
        if not (confirms or is_commitment(note.text)):
            continue
        pointer = depends[note.index] if cfg.get("use_pointer", True) else -1
        target = pointer if 0 <= pointer < note.index else -1
        if target < 0:  # no pointer: the nearest kept (or attached) sentence just before
            back = range(note.index - 1, max(note.index - 1 - cfg["lookback"], -1), -1)
            target = next((j for j in back if notes[j].kept or j in attached), -1)
        if target < 0 or (target in attached and not confirms):
            continue  # a bare commitment ("I'll …") does not join another reply's chain
        request = notes[attached.get(target, target)]
        if not request.kept or request.zone not in ATTACH_TO or request.speaker == note.speaker:
            continue
        request.followups.append(note)
        attached[note.index] = request.index
        if request.zone == "action_item" and is_commitment(note.text):
            request.owner = request.owner or note.speaker
            request.deadline = request.deadline or note.deadline
    return notes
