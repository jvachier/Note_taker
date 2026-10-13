"""Greedy ROUGE oracle: exact ROUGE-N, greedy selection and stopping, labels on disk."""

import json

import pytest
from rouge_score.rouge_scorer import RougeScorer
from rouge_score.tokenizers import DefaultTokenizer

from src.data.io import read_jsonl
from src.model1_scorer import oracle

TOK = DefaultTokenizer(use_stemmer=True)


@pytest.mark.parametrize(
    ("pred", "ref"),
    [
        ("The cat sat on the mat.", "A cat was sitting on the mat today."),
        ("#Person1# asks for the keys.", "#Person1#'s looking for a set of keys."),
        ("nothing in common", "completely different words"),
        ("the the the", "the cat"),
    ],
)
def test_rouge_matches_rouge_score(pred, ref):
    expected = RougeScorer(["rouge1", "rouge2"], use_stemmer=True).score(ref, pred)
    p, r = TOK.tokenize(pred), TOK.tokenize(ref)
    for name, n in oracle.ORDERS.items():
        ours = oracle.rouge_n_f1(oracle.ngrams(p, n), oracle.ngrams(r, n))
        assert ours == pytest.approx(expected[name].fmeasure, abs=1e-12)


def test_greedy_selects_matching_utterances_in_order():
    utts = [
        "Hello there.",
        "We will ship the order on Monday.",
        "Nice weather.",
        "The invoice is attached to the email.",
    ]
    ref = ["They will ship the order on Monday and the invoice is in the email."]
    obj = oracle.Objective(utts, ref, ["rouge1", "rouge2"], TOK)
    selected, score = oracle.greedy_oracle(obj, len(utts), 3)
    assert selected == [1, 3] and score == pytest.approx(obj([1, 3]))
    assert score > obj([1]) > 0


def test_greedy_stops_without_gain_and_at_max():
    obj = oracle.Objective(["abc", "def"], ["xyz"], ["rouge1", "rouge2"], TOK)
    assert oracle.greedy_oracle(obj, 2, 3) == ([], 0.0)
    utts = [f"word{i}" for i in range(6)]
    obj = oracle.Objective(utts, [" ".join(utts)], ["rouge1"], TOK)
    assert len(oracle.greedy_oracle(obj, 6, 3)[0]) == 3


def test_multiple_references_are_averaged():
    obj = oracle.Objective(["red apple"], ["red apple", "green pear"], ["rouge1"], TOK)
    assert obj([0]) == pytest.approx(0.5)


def test_main_writes_labels(tmp_path, tiny_assets):
    cfg = tmp_path / "oracle.yaml"
    cfg.write_text(
        "name: o\nseed: 0\nrouge_types: [rouge1, rouge2]\nmax_utterances: 3\nuse_stemmer: true\n"
        f"splits: [validation]\npaths:\n  raw_dir: {tiny_assets['raw']}\n"
        f"  output_dir: {tmp_path / 'oracle'}\n"
    )
    oracle.main(["--config", str(cfg)])
    rows = list(read_jsonl(tmp_path / "oracle" / "validation.jsonl"))
    raw = list(read_jsonl(tiny_assets["raw"] / "dialogsum" / "validation.jsonl"))
    assert [r["id"] for r in rows] == [r["id"] for r in raw]
    assert all(len(r["labels"]) == 3 and sum(r["labels"]) == len(r["selected"]) for r in rows)
    assert all(len(r["soft"]) == 3 and all(0 <= x <= 1 for x in r["soft"]) for r in rows)
    assert all(r["soft"][r["selected"][0]] > 0 for r in rows if r["selected"])
    stats = json.loads((tmp_path / "oracle" / "stats.json").read_text())
    assert stats["splits"]["validation"]["dialogues"] == len(rows)


def test_soft_labels_are_relative_single_utterance_scores():
    utts = ["Hello there.", "We will ship the order on Monday.", "The order ships Monday."]
    obj = oracle.Objective(utts, ["They will ship the order on Monday."], ["rouge1"], TOK)
    soft = oracle.soft_labels(obj, 3)
    assert soft[1] == 1.0 and soft[0] == 0.0 and 0 < soft[2] < 1
    assert soft[2] == pytest.approx(round(obj([2]) / obj([1]), 4))
    none = oracle.Objective(["abc"], ["xyz"], ["rouge1"], TOK)
    assert oracle.soft_labels(none, 1) == [0.0]
