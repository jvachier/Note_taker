"""Teacher labelling without a GPU: prompt, JSON validation, self-consistency, retry, resume."""

import json

import pytest

from src.config import load_config
from src.data.io import read_jsonl, write_jsonl
from src.model1_scorer.teacher import label
from src.model1_scorer.teacher.prompt import build_messages, format_conversation, retry_messages
from src.model1_scorer.teacher.validate import (
    InvalidAnswer,
    TeacherLabels,
    combine_samples,
    validate_answer,
)

TAGS = ["none", "key_point", "decision", "action_item"]
RANGE = [0, 3]


def _answer(n, score=2, tag="none"):
    return json.dumps([{"id": i, "score": score, "tag": tag} for i in range(1, n + 1)])


def test_prompt_numbers_utterances_with_speakers():
    assert (
        format_conversation([1, 2], ["Hi.", "Hello."]) == "1. #Person1#: Hi.\n2. #Person2#: Hello."
    )
    msgs = build_messages([1, 2], ["Hi.", "Hello."], RANGE, TAGS)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "(2 utterances)" in msgs[1]["content"] and '"action_item"' in msgs[1]["content"]
    retry = retry_messages(msgs, "oops", "no JSON array found", 2, RANGE, TAGS)
    assert [m["role"] for m in retry] == ["system", "user", "assistant", "user"]
    assert "no JSON array found" in retry[-1]["content"]


def test_prompt_v2_has_no_example_values_and_uses_the_reference():
    with_ref = build_messages([1, 2], ["Hi.", "Hello."], RANGE, TAGS, ["#Person1# greets."])
    user = with_ref[1]["content"]
    assert '"score": <int>' in user and '"score": 3' not in user  # v1 anchor removed
    assert "Reference summary:\n#Person1# greets." in user and "Summary:" not in user
    assert user.rstrip().endswith("2. #Person2#: Hello.")  # conversation last (fake teachers)
    several = build_messages([1], ["Hi."], RANGE, TAGS, ["A.", "B."])[1]["content"]
    assert "Reference summary:\n- A.\n- B." in several
    without = build_messages([1], ["Hi."], RANGE, TAGS)[1]["content"]
    assert "Reference summary" not in without and '"Summary: <one sentence' in without
    with pytest.raises(ValueError, match="4-point"):
        build_messages([1], ["Hi."], [0, 10], TAGS)


def test_valid_answer_is_normalised():
    text = (
        'Here you go:\n```json\n[{"id": 2, "score": 3, "tag": "decision"},\n'
        ' {"id": 1, "score": 0.0, "tag": "none"}]\n```'
    )
    labels = validate_answer(text, 2, RANGE, TAGS)
    assert labels.raw_scores == [0, 3] and labels.scores == [0.0, 1.0]
    assert labels.tags == ["none", "decision"] and labels.samples == 1


def test_summary_line_before_the_array_is_tolerated():
    text = "Summary: they plan a trip [to Paris].\n" + _answer(2, score=1)
    assert validate_answer(text, 2, RANGE, TAGS).raw_scores == [1, 1]


def test_combine_samples_mean_score_and_majority_tag():
    a = TeacherLabels([3, 0], [1.0, 0.0], ["decision", "none"])
    b = TeacherLabels([2, 0], [2 / 3, 0.0], ["key_point", "none"])
    c = TeacherLabels([2, 1], [2 / 3, 1 / 3], ["key_point", "key_point"])
    merged = combine_samples([a, b, c], RANGE)
    assert merged.raw_scores == [round(7 / 3, 4), round(1 / 3, 4)]
    assert merged.scores == [round(7 / 9, 4), round(1 / 9, 4)]
    assert merged.tags == ["key_point", "none"] and merged.samples == 3
    # a 1-1 tie goes to the more specific tag
    assert combine_samples([a, b], RANGE).tags == ["decision", "none"]
    with pytest.raises(ValueError):
        combine_samples([], RANGE)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("I think utterance 1 matters.", "no JSON array"),
        ('[{"id": 1, "score": 2, "tag": "none"}', "invalid JSON"),
        ('{"id": 1}', "no JSON array"),
        ('[{"id": 1, "score": 2, "tag": "none"}]', "missing ids [2]"),
        (
            '[{"id": 1, "score": 2, "tag": "none"}, {"id": 1, "score": 2, "tag": "none"}]',
            "more than once",
        ),
        (
            '[{"id": 1, "score": 4, "tag": "none"}, {"id": 2, "score": 2, "tag": "none"}]',
            "score 4",
        ),
        (
            '[{"id": 1, "score": 1.5, "tag": "none"}, {"id": 2, "score": 2, "tag": "none"}]',
            "score 1.5",
        ),
        (
            '[{"id": 1, "score": 2, "tag": "todo"}, {"id": 2, "score": 2, "tag": "none"}]',
            "tag 'todo'",
        ),
        ('[{"id": 3, "score": 2, "tag": "none"}]', "id 3"),
        ('[{"id": 1, "score": 2, "tag": "none"}, 2]', "must be an object"),
        ("[1, 2]", "no JSON array"),  # not an array of objects
    ],
)
def test_invalid_answers_are_rejected(text, reason):
    with pytest.raises(InvalidAnswer, match=reason.replace("[", r"\[").replace("]", r"\]")):
        validate_answer(text, 2, RANGE, TAGS)


DIALOGUES = [
    ["Hi, Mr. Smith. How are you today?", "I'm fine, thanks. And you?", "Not bad."],
    ["Did you see the game last night?", "Yes, it was great!", "I can't believe they won."],
]


@pytest.fixture
def raw(tmp_path):
    """DialogSum-style raw files with unique dialogue IDs."""
    for split in ("validation", "test"):
        records = [
            {
                "id": f"{split}_{i}",
                "speakers": [1, 2, 1],
                "utterances": DIALOGUES[i % 2],
                "summaries": ["#Person1# and #Person2# chat."],
            }
            for i in range(6)
        ]
        write_jsonl(tmp_path / "raw" / "dialogsum" / f"{split}.jsonl", records)
    return tmp_path / "raw"


def _cfg(tmp_path, configs_dir, raw, **extra):
    cfg = load_config(configs_dir / "teacher.yaml")
    cfg["paths"] = {"raw_dir": str(raw), "output_dir": str(tmp_path / "labels")}
    cfg["batch_dialogues"] = 4
    cfg.update(extra)
    return cfg


class FakeTeacher:
    """``samples`` answers per sampled call (one per greedy call). Every 3rd dialogue of the
    first call gets only invalid samples (fixed on retry); 'bad' dialogues never validate;
    with ``one_bad_sample`` the first sample of every dialogue is invalid."""

    def __init__(self, bad_ids=(), too_long_ids=(), samples=3, one_bad_sample=False):
        self.calls, self.bad, self.too_long = [], set(bad_ids), set(too_long_ids)
        self.samples, self.one_bad_sample = samples, one_bad_sample

    def __call__(self, conversations, sample):
        self.calls.append((len(conversations), sample))
        out = []
        for conv in conversations:
            user = conv[1]["content"]
            n = int(user.split("Conversation (")[1].split(" ")[0])
            key = conv[1]["content"].rsplit("\n", 1)[-1]  # last utterance line identifies it
            k = self.samples if sample else 1
            first_call_rule = not self.one_bad_sample and len(self.calls) == 1 and sample
            if key in self.too_long:
                out.append(None)
            elif key in self.bad or (first_call_rule and len(out) % 3 == 0):
                out.append(["not json"] * k)
            else:
                good = [_answer(n, score=s % 4, tag="key_point") for s in range(2, 2 + k)]
                if self.one_bad_sample and k > 1:
                    good[0] = "not json"
                out.append(good)
        return out


def test_label_dialogues_retry_failure_and_resume(tmp_path, configs_dir, raw):
    cfg = _cfg(tmp_path, configs_dir, raw, splits=["validation"])
    records = list(read_jsonl(raw / "dialogsum" / "validation.jsonl"))
    teacher = FakeTeacher(bad_ids={"3. #Person1#: I can't believe they won."})
    labels, failed = tmp_path / "l.jsonl", tmp_path / "f.jsonl"
    counts = label.label_dialogues(teacher, records, labels, failed, cfg)
    good, bad = list(read_jsonl(labels)), list(read_jsonl(failed))
    # The 'game' dialogue never validates; every other one ends up labelled.
    assert {b["id"] for b in bad} == {r["id"] for r in records if "game" in r["utterances"][0]}
    assert len(good) + len(bad) == len(records) and counts["retried"] >= len(bad)
    retried = [g for g in good if g["retried"]]
    sampled = [g for g in good if not g["retried"]]
    # 3 samples with scores 2, 3, 0 -> mean 5/3; the greedy retry gives one answer (score 2)
    assert sampled and all(g["raw_scores"] == [round(5 / 3, 4)] * g["n"] for g in sampled)
    assert all(g["samples"] == 3 and g["tags"][0] == "key_point" for g in sampled)
    assert retried and all(g["samples"] == 1 and g["raw_scores"] == [2] * g["n"] for g in retried)
    assert all(len(b["answers"]) == 4 and "retry" in b["error"] for b in bad)
    assert teacher.calls[0][1] is True and teacher.calls[1][1] is False  # retry is greedy

    again = FakeTeacher()
    counts = label.label_dialogues(again, records, labels, failed, cfg)
    assert again.calls == [] and counts["skipped_done"] == len(records)


def test_too_long_prompts_are_not_retried(tmp_path, configs_dir, raw):
    cfg = _cfg(tmp_path, configs_dir, raw)
    records = list(read_jsonl(raw / "dialogsum" / "validation.jsonl"))[:2]
    teacher = FakeTeacher(too_long_ids={"3. #Person1#: Not bad."})
    counts = label.label_dialogues(
        teacher, records, tmp_path / "l.jsonl", tmp_path / "f.jsonl", cfg
    )
    assert counts["too_long"] == 1 and teacher.calls == [(2, True)]  # no retry call


def test_invalid_samples_are_dropped_not_retried(tmp_path, configs_dir, raw):
    cfg = _cfg(tmp_path, configs_dir, raw, batch_dialogues=8)
    records = list(read_jsonl(raw / "dialogsum" / "validation.jsonl"))
    teacher = FakeTeacher(one_bad_sample=True)
    counts = label.label_dialogues(
        teacher, records, tmp_path / "l.jsonl", tmp_path / "f.jsonl", cfg
    )
    good = list(read_jsonl(tmp_path / "l.jsonl"))
    assert counts["retried"] == 0 and len(good) == len(records)
    # valid samples have scores 3 and 0 -> mean 1.5
    assert all(g["samples"] == 2 and g["raw_scores"] == [1.5] * g["n"] for g in good)
    assert counts["samples_generated"] == 3 * len(records)
    assert counts["samples_valid"] == 2 * len(records)


def test_prompt_includes_the_reference_only_when_configured(configs_dir):
    cfg = load_config(configs_dir / "teacher.yaml")
    record = {"speakers": [1], "utterances": ["Hi."], "summaries": ["#Person1# greets."]}
    assert "Reference summary" in label.build_dialogue_messages(record, cfg)[1]["content"]
    cfg["use_reference"] = False
    assert "Reference summary" not in label.build_dialogue_messages(record, cfg)[1]["content"]


def test_train_subset_is_fixed_and_shared(tmp_path):
    records = [{"id": f"train_{i}"} for i in range(50)]
    a = label.select_dialogues(records, "train", 10, seed=0)
    assert len(a) == 10 and a == label.select_dialogues(records, "train", 10, seed=0)
    assert [r["id"] for r in a] == sorted((r["id"] for r in a), key=lambda s: int(s.split("_")[1]))
    assert label.select_dialogues(records, "test", 10, seed=0) == records
    assert label.select_dialogues(records, "train", None, seed=0) == records


def test_run_writes_stats(tmp_path, configs_dir, raw):
    cfg = _cfg(tmp_path, configs_dir, raw, splits=["validation", "test"])
    stats = label.run("qwen1_5b", cfg, FakeTeacher(), {"backend": "fake", "model": "fake"})
    assert set(stats["splits"]) == {"validation", "test"} and stats["backend"] == "fake"
    saved = json.loads((tmp_path / "labels" / "qwen1_5b" / "stats.json").read_text())
    assert saved["splits"]["test"]["failed"] == 0
    assert saved["splits"]["test"]["labelled"] == saved["splits"]["test"]["dialogues"]
    with pytest.raises(ValueError, match="Unknown teacher"):
        label.run("gpt", cfg, FakeTeacher())
