"""Model 1 v3 teachers (PLAN.md 10.3): Qwen prompt / answers / labelling with a fake backend,
the quality gate, the untrained baselines and the encoder teachers with a stand-in backbone."""

import json
import math

import keras
import numpy as np
import pyarrow.parquet as pq
import pytest

from src.config import load_config
from src.model1_notes import encoder_teachers as enc
from src.model1_notes.baselines import KeywordRule, binary_f1, lead_fraction, tune_lead
from src.model1_notes.teacher import label
from src.model1_notes.teacher.answer import (
    InvalidAnswer,
    digit_distribution,
    labels_from_answer,
    parse_answer,
    soft_scores,
)
from src.model1_notes.teacher.gate import gate_metrics, run_gate
from src.model1_notes.teacher.label import Generation
from src.model1_notes.teacher.prompt import build_messages, retry_message
from src.model1_notes.teacher.windows import REASONS, best_window, window_starts
from src.model1_notes.tiny_corpus import write_tiny_corpus

CFG = load_config("configs/model1_v3.yaml")


def test_prompt_numbers_sentences_without_example_values():
    msgs = build_messages(["Ana", "Ben"], ["Hello.", "We decided."])
    user = msgs[1]["content"]
    assert msgs[0]["role"] == "system" and "1. Ana: Hello.\n2. Ben: We decided." in user
    assert '[[<id>, <score>, "<reason>", <depends_on>], ...]' in user  # placeholders only
    assert all(r in user for r in REASONS)
    assert "between" not in retry_message("bad", 2, REASONS)["content"]


@pytest.mark.parametrize(
    ("answer", "error"),
    [
        ("no json here", "no JSON array"),
        ('[[1, 0, "filler", -1]]', "missing ids"),
        ('[[1, 0, "filler", -1], [1, 0, "filler", -1]]', "more than once"),
        ('[[1, 4, "filler", -1], [2, 0, "filler", -1]]', "score"),
        ('[[1, 0, "nope", -1], [2, 0, "filler", -1]]', "reason"),
        ('[[1, 0, "filler", 2], [2, 0, "filler", -1]]', "depends_on"),
        ('[[1, 0, "filler"], [2, 0, "filler", -1]]', "every entry"),
        ('{"a": 1}', "no JSON array"),
    ],
)
def test_parse_answer_rejects(answer, error):
    with pytest.raises(InvalidAnswer, match=error):
        parse_answer(answer, 2, REASONS)


def test_parse_answer_accepts_fenced_and_prose():
    text = 'Sure.\n```json\n[[1, 3, "agreed_outcome", -1], [2, 0, "filler", 1]]\n```'
    assert parse_answer(text, 2, REASONS) == [(3, "agreed_outcome", -1), (0, "filler", 1)]


def test_soft_scores_from_logprobs():
    text = '[[1, 3, "key_fact", -1], [2, 0, "filler", -1]]'
    i = text.index("3")
    tokens = [(text[:i], {}), ("3", {"3": math.log(0.6), "2": math.log(0.2), "x": 0.0})]
    tokens.append((text[i + 1 :], {}))
    soft = soft_scores(text, [3, 0], tokens)
    assert soft[0] == pytest.approx((3 * 0.75 + 2 * 0.25) / 3, abs=1e-6)
    assert soft[1] == 0.0  # the second digit is inside a token without alternatives
    assert soft_scores(text, [3, 0], None) == [1.0, 0.0]
    assert digit_distribution({"a": -1.0}) is None
    labels = labels_from_answer(text, 2, REASONS, tokens)
    assert labels[1].score == 3 and labels[1].soft == pytest.approx(soft[0])  # ids from 1


def test_best_window_prefers_the_centre():
    starts = window_starts(20, 8, 2)  # [0, 6, 12]
    best = best_window(20, 8, 2)
    assert best[0] == 0 and best[19] == 2 and best[10] == 1
    for i, w in enumerate(best):
        assert starts[w] <= i < starts[w] + 8


def _run_cfg(tmp_path, **teacher):
    write_tiny_corpus(tmp_path / "corpus", meetings_per_split=6)
    cfg = {**CFG["teacher"], "window_sentences": 8, "window_overlap": 2, "batch_meetings": 2}
    cfg.update(splits=["validation", "train"], max_train_meetings=2, **teacher)
    return {
        "teacher": cfg,
        "seed": 0,
        "corpus_dir": str(tmp_path / "corpus"),
        "out_dir": str(tmp_path / "out"),
    }


def test_fake_labelling_two_shards_merge_and_resume(tmp_path):
    run_cfg = _run_cfg(tmp_path)
    for shard in range(2):
        assert label.run(run_cfg, "fake", math.inf, shard, 2) == 0
    counts = label.merge_shards(run_cfg["out_dir"], 2, ["validation", "train"])
    meetings = label.read_meetings(run_cfg["corpus_dir"], "validation")
    assert counts["validation"] == sum(len(m) for m in meetings.values())
    table = pq.read_table(tmp_path / "out" / "qwen" / "validation.parquet").to_pylist()
    assert {r["reason"] for r in table} <= set(REASONS)
    assert all(0.0 <= r["score"] <= 1.0 for r in table)
    decided = [r for r in table if r["reason"] == "agreed_outcome"]
    assert decided and all(r["depends_on"] == r["sentence_id"] - 1 for r in decided)
    assert any(0 < r["score"] < 1 for r in table)  # soft scores came from the log-probs
    train_ids = {
        r["note_id"] for r in pq.read_table(tmp_path / "out/qwen/train.parquet").to_pylist()
    }
    meetings_tr = label.read_meetings(run_cfg["corpus_dir"], "train")
    ami = {m for m, rows in meetings_tr.items() if rows[0]["source"] == "ami"}
    assert ami <= train_ids and len(train_ids - ami) == 2  # synthetic train capped at 2
    stats = json.loads((tmp_path / "out/qwen/shards/0/stats.json").read_text())
    assert stats["complete"] and stats["splits"]["validation"]["failure_rate"] == 0
    raw = (tmp_path / "out/qwen/shards/0/raw_validation.jsonl").read_text().splitlines()
    assert json.loads(raw[0])["thinking"]
    before = (tmp_path / "out/qwen/shards/0/validation.jsonl").read_text()
    label.run(run_cfg, "fake", math.inf, 0, 2)  # resume: nothing new
    assert (tmp_path / "out/qwen/shards/0/validation.jsonl").read_text() == before


def test_deadline_stops_early(tmp_path):
    run_cfg = _run_cfg(tmp_path)
    assert label.run(run_cfg, "fake", 0.0, 0, 1) == 0
    stats = json.loads((tmp_path / "out/qwen/shards/0/stats.json").read_text())
    assert not stats["complete"] and stats["splits"]["validation"]["stopped_early"]


def test_retry_then_failure():
    rows = [{"speaker": "A", "sentence_text": "Hi."}, {"speaker": "B", "sentence_text": "Bye."}]
    good = '[[1, 0, "filler", -1], [2, 0, "filler", -1]]'
    calls = []

    def flaky(convs, think):
        calls.append(think)
        return [Generation("t", "oops" if think else good) for _ in convs]

    labels, raw = label.label_windows(flaky, [rows])
    assert calls == [True, False] and labels[0] is not None and raw[0]["retried"]
    labels, raw = label.label_windows(lambda c, t: [Generation("", "oops")] * len(c), [rows])
    assert labels[0] is None and "only 0 of 2" in raw[0]["error"]
    too_long, _ = label.label_windows(lambda c, t: [Generation("", None)] * len(c), [rows])
    assert too_long[0] is None
    assert label.merge_meeting(rows, [0], [None], 8) is None


def test_merge_meeting_maps_depends_to_sentence_ids():
    rows = [{"sentence_id": 10 + i} for i in range(10)]
    w0 = labels_from_answer(json.dumps([[i, 1, "key_fact", -1] for i in range(1, 9)]), 8, REASONS)
    w1 = labels_from_answer(
        json.dumps([[i, 2, "key_fact", i - 1 if i > 1 else -1] for i in range(1, 9)]), 8, REASONS
    )
    merged = label.merge_meeting(rows, [0, 2], [w0, w1], 8)
    assert [m["sentence_id"] for m in merged] == list(range(10, 20))
    assert merged[-1]["depends_on"] == 18 and merged[0]["depends_on"] == -1


def test_gate(tmp_path):
    run_cfg = _run_cfg(tmp_path)
    label.run(run_cfg, "fake", math.inf, 0, 1)
    label.merge_shards(run_cfg["out_dir"], 1, ["validation"])
    result = run_gate(run_cfg["corpus_dir"], run_cfg["out_dir"], CFG["baselines"], "validation")
    assert result["coverage"] == 1.0 and isinstance(result["passed"], bool)
    assert json.loads((tmp_path / "out/qwen/gate.json").read_text()) == result
    base = {"note_id": "m", "sentence_text": "Sounds good.", "hard_negative": False}
    gold = [
        {**base, "sentence_id": 0, "importance": 0, "zone": "discussion"},
        {**base, "sentence_id": 1, "importance": 1, "zone": "decision"},
    ]
    rule = KeywordRule(CFG["baselines"]["keywords"])
    fractions = CFG["baselines"]["lead_fractions"]
    qwen = {
        ("m", 0): {"score": 0.1, "reason": "filler"},
        ("m", 1): {"score": 0.9, "reason": "agreed_outcome"},
    }
    good = gate_metrics(gold, qwen, rule, fractions)
    assert good["passed"] and good["reason_zone_agreement"] == 1.0 and good["lead_f1"] == 0.0
    assert not gate_metrics(gold, {}, rule, fractions)["passed"]
    first = [{**g, "importance": 1 - g["importance"]} for g in gold]  # Lead-1 is perfect here
    assert gate_metrics(first, qwen, rule, fractions)["lead_f1"] == 1.0
    assert not gate_metrics(first, qwen, rule, fractions)["passed"]


def test_baselines():
    assert lead_fraction(10, 0.25) == [1, 1, 1] + [0] * 7 and lead_fraction(3, 0.01) == [1, 0, 0]
    fraction, f1 = tune_lead([[1, 1, 0, 0], [1, 0, 0, 0]], [0.25, 0.5, 1.0])
    assert fraction == 0.5 and f1 == pytest.approx(
        binary_f1([1, 1, 0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0, 0])
    )
    rule = KeywordRule(CFG["baselines"]["keywords"])
    assert rule.zone("We have decided to wait.") == "decision"
    assert rule.zone("Ben will fix the rig by Friday.") == "action_item"
    assert rule.importance("Sounds good.") == 0


def _toy_tokenizer():
    vocab: dict[str, int] = {}

    def encode(text):
        return [4 + vocab.setdefault(w, len(vocab) % 90) for w in text.lower().split()]

    return encode, enc.Specials(bos=1, sep=2, pad=0)


def _toy_backbone(max_tokens):
    from src.layers.transformer import TransformerEncoder

    token_ids = keras.Input((max_tokens,), dtype="int32", name="token_ids")
    padding_mask = keras.Input((max_tokens,), dtype="bool", name="padding_mask")
    hidden = TransformerEncoder(100, max_tokens, 16, 1, 2, 32, 0.0, name="encoder")(token_ids)
    return keras.Model({"token_ids": token_ids, "padding_mask": padding_mask}, hidden)


def test_pack_meetings_covers_every_sentence_once():
    encode, specials = _toy_tokenizer()
    rows = [
        {"sentence_id": i, "text": f"word{i} " * (1 + i % 4), "importance": i % 2}
        for i in range(30)
    ]
    packed = enc.pack_meetings({"m": rows, "n": rows[:3]}, encode, specials, 24, 3, 5)
    keys = [k for w in packed.keys for k in w]
    assert len(keys) == 33 and len(set(keys)) == 33
    assert (packed.labels >= 0).sum(1).max() <= 5
    for w, slots in enumerate(packed.keys):
        assert packed.token_ids[w, 0] == specials.bos
        assert set(packed.sentence_index[w][packed.sentence_index[w] >= 0]) == set(
            range(len(slots))
        )
        assert packed.padding_mask[w].sum() == (packed.token_ids[w] != specials.pad).sum()


def test_sentence_mean_pool():
    hidden = np.arange(12, dtype=np.float32).reshape(1, 4, 3)
    index = np.array([[0, 0, 1, -1]])
    out = np.asarray(enc.SentenceMeanPool(3)(hidden, index))
    np.testing.assert_allclose(out[0, 0], hidden[0, :2].mean(0))
    np.testing.assert_allclose(out[0, 1], hidden[0, 2])
    np.testing.assert_allclose(out[0, 2], 0)


def test_run_encoder_end_to_end(tmp_path):
    write_tiny_corpus(tmp_path / "corpus", meetings_per_split=3)
    encode, specials = _toy_tokenizer()
    cfg = {
        **CFG["encoders"],
        "max_tokens": 32,
        "max_sentence_tokens": 8,
        "max_sentences": 6,
        "epochs": 1,
        "batch_size": 4,
        "mixed_precision": "float32",
    }
    result = enc.run_encoder(
        "roberta",
        _toy_backbone(32),
        encode,
        specials,
        cfg,
        tmp_path / "corpus",
        tmp_path / "out",
        seed=0,
    )
    assert result["sentences"] == {"train": 36, "validation": 36, "test": 36}
    table = pq.read_table(tmp_path / "out" / "roberta" / "test.parquet").to_pylist()
    assert len(table) == 36 and np.isfinite([r["logit"] for r in table]).all()
    assert "loss" in result["history"]


def test_select_meetings_caps_synthetic_and_ami_train():
    meetings = {f"s{i}": [{"source": "synthetic"}] for i in range(5)}
    meetings |= {f"a{i}": [{"source": "ami"}] for i in range(4)}
    cfg = {"max_train_meetings": 2, "max_ami_train_meetings": 1}
    picked = label.select_meetings(meetings, "train", cfg, seed=0)
    assert (
        sum(m.startswith("s") for m in picked) == 2 and sum(m.startswith("a") for m in picked) == 1
    )
    assert picked == label.select_meetings(meetings, "train", cfg, seed=0)  # seeded
    assert len(label.select_meetings(meetings, "validation", cfg, seed=0)) == 9
    assert len(label.select_meetings(meetings, "train", {}, seed=0)) == 9


def test_find_local_preset(tmp_path):
    target = tmp_path / "roberta" / "keras" / "roberta_large_en" / "3"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")
    other = tmp_path / "distil_bert" / "keras" / "distil_bert_base_en_uncased" / "2"
    other.mkdir(parents=True)
    (other / "config.json").write_text("{}")
    assert enc.find_local_preset(tmp_path, "roberta_large_en") == target
    assert enc.find_local_preset(tmp_path, "distil_bert_base_en_uncased") == other
    assert enc.find_local_preset(tmp_path, "bert_base_en") is None
    assert enc.find_local_preset(tmp_path / "missing", "roberta_large_en") is None


def test_parse_entries_reads_what_qwen_0_6b_actually_wrote():
    from src.model1_notes.teacher.answer import parse_entries

    real = (
        '[\n  [\n    { "id": 1, "score": 0, "reason": "filler", "depends_on": -1 },\n'
        '    { "id": 2, "score": 3, "reason": "context", "depends_on": -1 },\n'
        '    { "id": 3, "score": 3, "reason": "agreed_outcome", "depends_on": 3 },\n'
        '    { "id": 4, "score": 2, "reason": "key_f'  # cut off by the token budget
    )
    assert parse_entries(real + "<|im_end|>", 4, REASONS) == {
        1: (0, "filler", -1),
        2: (3, None, -1),  # unknown reason: only the reason is lost
        3: (3, "agreed_outcome", -1),  # depends on itself: link dropped
    }
    lists = '```json\n[[1, 3, "key_fact", -1], [2, 9, "filler", -1], [3, 1, "question", 1]]\n```'
    assert parse_entries(lists, 3, REASONS) == {1: (3, "key_fact", -1), 3: (1, "question", 1)}
    assert parse_entries(None, 3, REASONS) == {} and parse_entries("no idea", 3, REASONS) == {}


def test_labels_from_answer_partial_and_min_coverage():
    text = '[{"id": 1, "score": 3, "reason": "key_fact", "depends_on": -1}]'
    labels = labels_from_answer(text, 4, REASONS)
    assert set(labels) == {1} and labels[1].soft == 1.0
    with pytest.raises(InvalidAnswer, match="only 1 of 4"):
        labels_from_answer(text, 4, REASONS, min_coverage=0.5)


def test_merge_meeting_keeps_partial_windows():
    rows = [{"sentence_id": 10 + i} for i in range(10)]
    w0 = labels_from_answer(json.dumps([[i, 1, "key_fact", -1] for i in (1, 2, 5)]), 8, REASONS)
    w1 = labels_from_answer(json.dumps([[i, 2, "key_fact", i - 1] for i in (2, 8)]), 8, REASONS)
    merged = {m["sentence_id"]: m for m in label.merge_meeting(rows, [0, 2], [w0, w1], 8)}
    assert set(merged) == {10, 11, 14, 13, 19}  # 13 = window 1's id 2; 14 = window 0's id 5
    assert merged[19]["depends_on"] == 18 and merged[11]["raw_score"] == 1
    assert label.merge_meeting(rows, [0], [None], 8) is None


def test_salvage_rebuilds_tables_from_raw_answers(tmp_path):
    from src.model1_notes.teacher import salvage

    write_tiny_corpus(tmp_path / "corpus", meetings_per_split=3)
    raw = tmp_path / "raw"
    for split in ("validation", "test", "train"):
        meetings = label.read_meetings(tmp_path / "corpus", split)
        out = raw / "shards" / "0"
        out.mkdir(parents=True, exist_ok=True)
        with (out / f"raw_{split}.jsonl").open("w") as f:
            for note_id, rows in meetings.items():
                objs = [
                    {"id": i, "score": i % 4, "reason": "key_fact", "depends_on": -1}
                    for i in range(1, min(len(rows), 8) + 1)
                ]
                bad = json.dumps([objs])[:-30] + "<|im_end|>"  # nested, cut off, special token
                f.write(
                    json.dumps(
                        {"note_id": note_id, "start": 0, "answer": bad, "retry_answer": None}
                    )
                    + "\n"
                )
    cfg = {**CFG["teacher"], "window_sentences": 8}
    report = salvage.salvage(raw, tmp_path / "corpus", tmp_path / "out", cfg, CFG["baselines"])
    assert report["validation"]["meetings_labelled"] == 3
    assert 0 < report["validation"]["coverage"] < 1 and "passed" in report["gate"]
    table = pq.read_table(tmp_path / "out" / "qwen" / "validation.parquet").to_pylist()
    assert table and {r["reason"] for r in table} == {"key_fact"}
