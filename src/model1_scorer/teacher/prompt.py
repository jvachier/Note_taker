"""Teacher prompt (PLAN.md section 5.3): numbered utterances in, JSON array out.

Pure Python (no vLLM): the same messages are used by every backend and tested locally.
"""

from __future__ import annotations

from collections.abc import Sequence

SYSTEM = (
    "You write meeting notes from conversations. You rate how important each utterance is "
    "for the notes and tag what kind of content it is. You answer with JSON only."
)

INSTRUCTIONS = """Rate every utterance of the conversation below.

For each utterance give:
- "id": the utterance number,
- "score": an integer from {lo} to {hi}, how important the utterance is for notes
  summarising the conversation ({lo} = irrelevant, e.g. greetings or filler; {hi} = essential),
- "tag": one of {tags}:
  - "key_point": an important fact, problem, opinion or piece of information,
  - "decision": something agreed or decided,
  - "action_item": a task, request or commitment someone will carry out,
  - "none": anything else.

Answer with a JSON array only, one object per utterance, in order, no other text:
[{{"id": 1, "score": 3, "tag": "none"}}, ...]

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


def _tags(tags: Sequence[str]) -> str:
    return ", ".join(f'"{t}"' for t in tags)


def build_messages(
    speakers: Sequence[int],
    utterances: Sequence[str],
    score_range: Sequence[int],
    tags: Sequence[str],
) -> list[dict[str, str]]:
    """Chat messages for one dialogue."""
    lo, hi = score_range
    user = INSTRUCTIONS.format(
        lo=lo,
        hi=hi,
        tags=_tags(tags),
        n=len(utterances),
        conversation=format_conversation(speakers, utterances),
    )
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

    Decoding is greedy, so repeating the same prompt would repeat the same answer.
    """
    lo, hi = score_range
    retry = RETRY.format(error=error, n=n, lo=lo, hi=hi, tags=_tags(tags))
    return [*messages, {"role": "assistant", "content": answer},
            {"role": "user", "content": retry}]  # fmt: skip
