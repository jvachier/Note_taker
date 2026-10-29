"""Qwen3-0.6B prompt for one meeting window (PLAN.md 10.3).

The teacher sees numbered sentences with speakers, never the gold labels. It thinks first (Qwen3
thinking mode, capped), then answers with a JSON array holding one compact entry per sentence:
``[id, score, "reason", depends_on]``. The format line shows **no example values** (in v1 an
example score became the weak teacher's answer for 98% of utterances).
"""

from __future__ import annotations

from collections.abc import Sequence

SYSTEM = (
    "You take notes in business meetings. You decide which sentences of a transcript belong in "
    "the notes and why. You answer with JSON only."
)
REASON_RULES = """- "reason": exactly one of
  - "agreed_outcome": something the group explicitly agrees or decides (not a suggestion),
  - "commitment_with_owner": a person commits to a task, usually with a deadline,
  - "risk_or_blocker": a problem, risk or blocker someone raises,
  - "key_fact": an important fact, number or status the notes need,
  - "status_update": routine progress that the notes can skip,
  - "rejected_proposal": a proposal that is turned down, or a decision later reversed,
  - "tentative": an idea, a maybe, a hypothetical, or a task without an owner,
  - "question": a question that gets no answer or commitment,
  - "filler": greetings, thanks, backchannels, small talk, logistics."""
INSTRUCTIONS = """Read the meeting below and decide, for every sentence, whether it belongs in \
the meeting notes.

For each sentence give, in this order:
- the sentence id,
- "score": an integer from 0 to 3:
  3 = a decision, an action item, or a key fact the notes must keep,
  2 = useful detail for the notes,
  1 = context the notes do not need,
  0 = filler, greetings, backchannels, or a bare yes / no / okay,
{reason_rules}
- "depends_on": the id of the earlier sentence that makes this one meaningful (for example the \
proposal an agreement refers to), or -1 if none.

Answer with one JSON array, one entry per sentence, in order, and nothing else:
[[<id>, <score>, "<reason>", <depends_on>], ...]

Meeting ({n} sentences):
{transcript}"""
RETRY = (
    "Your answer was not valid: {error}. Answer again with the JSON array only: exactly one entry "
    '[id, score, "reason", depends_on] per sentence id from 1 to {n}, score an integer 0-3, '
    "reason one of {reasons}, depends_on an earlier id or -1."
)


def transcript(speakers: Sequence[str], sentences: Sequence[str]) -> str:
    """``<id>. <speaker>: <sentence>`` lines, ids from 1."""
    return "\n".join(
        f"{i}. {s}: {t}" for i, (s, t) in enumerate(zip(speakers, sentences, strict=True), 1)
    )


def build_messages(speakers: Sequence[str], sentences: Sequence[str]) -> list[dict[str, str]]:
    """Chat messages (system + user) for one meeting window."""
    user = INSTRUCTIONS.format(
        reason_rules=REASON_RULES, n=len(sentences), transcript=transcript(speakers, sentences)
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def retry_message(error: str, n: int, reasons: Sequence[str]) -> dict[str, str]:
    """User message asking for a corrected answer."""
    names = ", ".join(f'"{r}"' for r in reasons)
    return {"role": "user", "content": RETRY.format(error=error, n=n, reasons=names)}
