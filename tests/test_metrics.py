"""Hand-checked tests for the shared classification metrics."""

import numpy as np
import pytest

from src.eval.metrics import classification_report, confusion_matrix


def test_confusion_matrix():
    conf = confusion_matrix(np.array([0, 0, 1, 2, 2]), np.array([0, 1, 1, 2, 0]), 3)
    np.testing.assert_array_equal(conf, [[1, 1, 0], [0, 1, 0], [1, 0, 1]])


def test_classification_report_hand_checked():
    y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 0, 2, 2, 0])
    rep = classification_report(y_true, y_pred, ["none", "a", "b"], macro_over=["a", "b"])
    # class a: tp=1, fp=1, fn=1 -> p=r=f1=0.5; class b: tp=2, fp=0, fn=1 -> p=1, r=2/3, f1=0.8
    assert rep["per_class"]["a"] == pytest.approx(
        {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}
    )
    assert rep["per_class"]["b"]["f1"] == pytest.approx(0.8)
    assert rep["macro_f1"] == pytest.approx(0.65)
    assert rep["accuracy"] == pytest.approx(5 / 8)
    assert rep["n"] == 8


def test_unpredicted_class_scores_zero():
    rep = classification_report(np.array([1, 1]), np.array([0, 0]), ["x", "y"], ["y"])
    assert rep["per_class"]["y"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 2}


def test_macro_skips_classes_never_present_nor_predicted():
    names = ["none", "a", "b"]
    rep = classification_report(np.array([0, 1, 1]), np.array([0, 1, 1]), names, ["a", "b"])
    assert rep["macro_f1"] == 1.0
    rep = classification_report(np.array([0, 1, 1]), np.array([0, 1, 2]), names, ["a", "b"])
    assert rep["macro_f1"] == pytest.approx((2 / 3 + 0.0) / 2)


def test_spearman_hand_checked():
    from src.eval.metrics import jaccard, mean_spearman, spearman

    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([3, 2, 1], [10, 20, 30]) == pytest.approx(-1.0)
    # ranks [1,2,3,4] vs [1,3,2,4]: 1 - 6*2/(4*15) = 0.8
    assert spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    assert spearman([1, 1, 1], [1, 2, 3]) is None and spearman([5], [5]) is None
    assert mean_spearman([[1, 2, 3], [1, 1, 1]], [[1, 2, 3], [3, 2, 1]]) == pytest.approx(1.0)
    assert jaccard([0, 1, 2], [1, 2, 3]) == pytest.approx(0.5) and jaccard([], []) == 1.0


def test_rouge_multi_reference_is_the_mean():
    from rouge_score.rouge_scorer import RougeScorer

    from src.eval.summary import rouge

    pred, refs = "the cat sat on the mat", ["the cat sat on the mat", "a dog ran"]
    single = RougeScorer(["rouge1"], use_stemmer=True).score(refs[1], pred)["rouge1"].fmeasure
    assert rouge([pred], [refs])["rouge1"] == pytest.approx((1.0 + single) / 2)
    assert rouge([pred, "x"], [[refs[0]], ["y"]])["rougeL"] == pytest.approx(0.5)
