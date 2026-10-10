"""Teacher labelling without a GPU: prompt, JSON validation, retry, resume, subset."""

import json

import pytest

from src.config import load_config
from src.data.io import read_jsonl, write_jsonl
from src.model1_scorer.teacher import label
from src.model1_scorer.teacher.prompt import build_messages, format_conversation, retry_messages
from src.model1_scorer.teacher.validate import InvalidAnswer, validate_answer

TAGS = ["none", "key_point", "decision", "action_item"]


def _answer(n, score=5, tag="none"):
    return json.dumps([{"id": i, "score": score, "tag": tag} for i in range(1, n + 1)])


def test_prompt_numbers_utterances_with_speakers():
    assert (
        format_conversation([1, 2], ["Hi.", "Hello."]) == "1. #Person1#: Hi.\n2. #Person2#: Hello."
    )
    msgs = build_messages([1, 2], ["Hi.", "Hello."], [0, 10], TAGS)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "(2 utterances)" in msgs[1]["content"] and '"action_item"' in msgs[1]["content"]
    retry = retry_messages(msgs, "oops", "no JSON array found", 2, [0, 10], TAGS)
    assert [m["role"] for m in retry] == ["system", "user", "assistant", "user"]
    assert "no JSON array found" in retry[-1]["content"]


def test_valid_answer_is_normalised():
    text = (
        'Here you go:\n```json\n[{"id": 2, "score": 10, "tag": "decision"},\n'
        ' {"id": 1, "score": 0.0, "tag": "none"}]\n```'
    )
    labels = validate_answer(text, 2, [0, 10], TAGS)
    assert labels.raw_scores == [0, 10] and labels.scores == [0.0, 1.0]
    assert labels.tags == ["none", "decision"]


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("I think utterance 1 matters.", "no JSON array"),
        ('[{"id": 1, "score": 5, "tag": "none"}', "invalid JSON"),
        ('{"id": 1}', "no JSON array"),
        ('[{"id": 1, "score": 5, "tag": "none"}]', "missing ids [2]"),
        (
            '[{"id": 1, "score": 5, "tag": "none"}, {"id": 1, "score": 5, "tag": "none"}]',
            "more than once",
        ),
        (
            '[{"id": 1, "score": 11, "tag": "none"}, {"id": 2, "score": 5, "tag": "none"}]',
            "score 11",
        ),
        (
            '[{"id": 1, "score": 5.5, "tag": "none"}, {"id": 2, "score": 5, "tag": "none"}]',
            "score 5.5",
        ),
        (
            '[{"id": 1, "score": 5, "tag": "todo"}, {"id": 2, "score": 5, "tag": "none"}]',
            "tag 'todo'",
        ),
        ('[{"id": 3, "score": 5, "tag": "none"}]', "id 3"),
        ("[1, 2]", "must be an object"),
    ],
)
def test_invalid_answers_are_rejected(text, reason):
    with pytest.raises(InvalidAnswer, match=reason.replace("[", r"\[").replace("]", r"\]")):
        validate_answer(text, 2, [0, 10], TAGS)


DIALOGUES = [
    ["Hi, Mr. Smith. How are you today?", "I'm fine, thanks. And you?", "Not bad."],
    ["Did you see the game last night?", "Yes, it was great!", "I can't believe they won."],
]


@pytest.fixture
def raw(tmp_path):
    """DialogSum-style raw files with unique dialogue IDs."""
    for split in ("validation", "test"):
        records = [
            {"id": f"{split}_{i}", "speakers": [1, 2, 1], "utterances": DIALOGUES[i % 2]}
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
    """Invalid first answer for every 3rd dialogue (fixed on retry), never valid for 'bad'."""

    def __init__(self, bad_ids=(), too_long_ids=()):
        self.calls, self.bad, self.too_long = [], set(bad_ids), set(too_long_ids)

    def __call__(self, conversations):
        self.calls.append(len(conversations))
        out = []
        for conv in conversations:
            user = conv[1]["content"]
            n = int(user.split("Conversation (")[1].split(" ")[0])
            key = conv[1]["content"].rsplit("\n", 1)[-1]  # last utterance line identifies it
            retry = len(conv) > 2
            if key in self.too_long:
                out.append(None)
            elif key in self.bad or (len(self.calls) == 1 and not retry and len(out) % 3 == 0):
                out.append("not json")
            else:
                out.append(_answer(n, score=7, tag="key_point"))
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
    assert all(g["scores"] == [0.7] * g["n"] and g["tags"][0] == "key_point" for g in good)
    assert all(len(b["answers"]) == 2 and "retry" in b["error"] for b in bad)

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
    assert counts["too_long"] == 1 and teacher.calls == [2]  # no retry call


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
