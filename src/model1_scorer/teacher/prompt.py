"""Teacher prompt (PLAN.md section 5.3): numbered utterances in, JSON array out.

Version 2 (after the v1 labels were found to be weak): the output format is shown without
example values (v1's ``"score": 3`` became the 1.5B teacher's answer for 98% of utterances),
the scale is a short anchored 0-3 scale, the tag rules are stricter, and the reference
summary of the dialogue is given so the teacher rates how much each utterance supports it.
Without a reference (``summaries=None``) the teacher first writes a one-sentence summary
itself, then rates against it.

Pure Python (no vLLM): the same messages are used by every backend and tested locally.
"""

from __future__ import annotations

from collections.abc import Sequence

SYSTEM = (
    "You write meeting notes from conversations. You rate how much each utterance contributes "
    "to the notes and tag what kind of content it is. You answer with JSON only."
)

SCALE = """- "score": an integer from {lo} to {hi}:
  {hi} = states a main fact, problem, decision or plan of the {target},
  {mid_hi} = adds a useful detail to it,
  {mid_lo} = context the {target} does not need,
  {lo} = greetings, thanks, goodbyes, filler, or a bare yes / no / okay,"""

TAG_RULES = """- "tag": one of {tags}:
  - "decision": something explicitly agreed or decided (not a suggestion),
  - "action_item": a concrete task, request or commitment that someone will carry out,
  - "key_point": any other important fact, problem or opinion,
  - "none": anything else (most utterances scored {lo} or {mid_lo})."""

WITH_REFERENCE = """Rate every utterance of the conversation below by how much it contributes \
to the reference summary.

For each utterance give:
- "id": the utterance number,
{scale}
{tag_rules}

Answer with a JSON array only, one object per utterance, in order, no other text:
[{{"id": <int>, "score": <int>, "tag": <string>}}, ...]

Reference summary:
{summary}

Conversation ({n} utterances):
{conversation}"""

WITHOUT_REFERENCE = """Rate every utterance of the conversation below.

First write one line "Summary: <one sentence summarising the conversation>". Then rate every \
utterance by how much it contributes to that summary. For each utterance give:
- "id": the utterance number,
{scale}
{tag_rules}

After the summary line, answer with a JSON array, one object per utterance, in order, no other \
text:
[{{"id": <int>, "score": <int>, "tag": <string>}}, ...]

Conversation ({n} utterances):
{conversation}"""

RETRY = (
    "Your answer was not valid: {error}. Answer again with the JSON array only: exactly one "
    'object per utterance id from 1 to {n}, each with "id", an integer "score" from {lo} to '
    '{hi} and a "tag" from {tags}.'
)


def format_conversation(speakers: Sequence[int], utterances: Sequence[str]) -> str:
    """``1. #Person1#: text`` lines, numbered from 1."""
    return "\n".join(
        f"{i}. #Person{s}#: {u}"
        for i, (s, u) in enumerate(zip(speakers, utterances, strict=True), 1)
    )


def format_summaries(summaries: Sequence[str]) -> str:
    """One reference as is; several (DialogSum test) as ``- `` bullet lines."""
    if len(summaries) == 1:
        return summaries[0]
    return "\n".join(f"- {s}" for s in summaries)


def _tags(tags: Sequence[str]) -> str:
    return ", ".join(f'"{t}"' for t in tags)


def build_messages(
    speakers: Sequence[int],
    utterances: Sequence[str],
    score_range: Sequence[int],
    tags: Sequence[str],
    summaries: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """Chat messages for one dialogue (with its reference summaries if given)."""
    lo, hi = score_range
    if hi - lo != 3:
        raise ValueError(f"the v2 prompt describes a 4-point scale, got {score_range}")
    target = "summary" if summaries else "conversation"
    fields = {
        "scale": SCALE.format(lo=lo, hi=hi, mid_lo=lo + 1, mid_hi=hi - 1, target=target),
        "tag_rules": TAG_RULES.format(tags=_tags(tags), lo=lo, mid_lo=lo + 1),
        "n": len(utterances),
        "conversation": format_conversation(speakers, utterances),
    }
    if summaries:
        user = WITH_REFERENCE.format(summary=format_summaries(summaries), **fields)
    else:
        user = WITHOUT_REFERENCE.format(**fields)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def retry_messages(
    messages: list[dict[str, str]],
    answer: str,
    error: str,
    n: int,
    score_range: Sequence[int],
    tags: Sequence[str],
) -> list[dict[str, str]]:
    """The conversation extended with the invalid answer and a correction request.

    The retry is decoded greedily, so it would repeat the same answer without the correction.
    """
    lo, hi = score_range
    retry = RETRY.format(error=error, n=n, lo=lo, hi=hi, tags=_tags(tags))
    return [*messages, {"role": "assistant", "content": answer}, {"role": "user", "content": retry}]
