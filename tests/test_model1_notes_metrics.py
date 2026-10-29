"""Model 1 v3 sentence metrics (PLAN.md 10.7)."""

import numpy as np
import pytest

from src.model1_notes.metrics import (
    SPAN_TAGS,
    ZONES,
    average_precision,
    selection_metrics,
    span_entities,
    span_f1,
    tune_threshold,
    zone_macro_f1,
)

B_OWN, I_OWN, B_DL, I_DL = (SPAN_TAGS.index(t) for t in SPAN_TAGS[1:])


def test_average_precision():
    assert average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([1, 1], [0.2, 0.9]) == 1.0
    assert average_precision([0, 0], [0.2, 0.9]) == 0.0


def test_tune_threshold_and_zone_f1():
    threshold, f1 = tune_threshold(np.array([1, 1, 0, 0]), np.array([0.9, 0.6, 0.4, 0.1]))
    assert f1 == 1.0 and 0.4 < threshold <= 0.6
    assert zone_macro_f1(np.array([1, 2, 2]), np.array([1, 2, 2])) == 1.0
    assert zone_macro_f1(np.array([]), np.array([])) == 0.0


def test_span_entities_and_f1():
    tags = [0, B_OWN, I_OWN, 0, B_DL, I_OWN, I_DL]
    assert span_entities(tags) == {
        (1, 3, "owner"),
        (4, 5, "deadline"),
        (5, 6, "owner"),
        (6, 7, "deadline"),
    }
    gold = np.array([[-1, B_OWN, 0, B_DL, I_DL, -1]])
    exact = span_f1(gold, gold.clip(0))
    assert exact["span_f1"] == 1.0 and exact["gold_spans"] == 2
    partial = span_f1(gold, np.array([[0, B_OWN, 0, B_DL, 0, 0]]))
    assert partial["owner_f1"] == 1.0 and partial["deadline_f1"] == 0.0


def test_selection_metrics():
    dec, act, disc = ZONES.index("decision"), ZONES.index("action_item"), ZONES.index("discussion")
    truth = np.array([1, 1, 0, 0])
    zone_true = np.array([dec, act, disc, disc])
    picked = np.array([1, 1, 1, 0])
    zone_pred = np.array([dec, disc, disc, disc])
    out = selection_metrics(
        truth, zone_true, picked, np.array([0.9, 0.8, 0.7, 0.1]), zone_pred, np.array([0, 0, 1, 1])
    )
    assert out["importance_f1"] == pytest.approx(0.8)
    assert out["decision_f1"] == 1.0 and out["action_f1"] == 0.0
    assert out["hard_negative_fp"] == 0.5 and out["pr_auc"] == 1.0
    bare = selection_metrics(truth, zone_true, picked)
    assert bare["zone_macro_f1"] is None and bare["pr_auc"] is None


def test_zone_calibration_finds_rare_zones():
    from src.model1_notes.calibrate import assign_zones, tune_zone_thresholds

    dec, act, disc = ZONES.index("decision"), ZONES.index("action_item"), ZONES.index("discussion")
    probs = np.full((6, 5), 0.05)
    probs[:, disc] = 0.8
    probs[0, dec], probs[1, act], probs[2, dec] = 0.15, 0.12, 0.03  # rare zones never win argmax
    importance = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    zone = np.array([dec, act, disc, disc, disc, disc])
    grid = np.round(np.arange(0.02, 0.5, 0.01), 2)
    th = tune_zone_thresholds(importance, zone, np.ones(6, dtype=bool), probs, grid)
    assert 0.03 < th["decision"] <= 0.15 and 0.05 < th["action_item"] <= 0.12
    assert assign_zones(probs, th).tolist() == [dec, act, disc, disc, disc, disc]
    assert (assign_zones(probs, None) == disc).all()


def test_balanced_calibration_weighs_small_group_equally():
    from src.model1_notes.calibrate import tune_zone_thresholds

    dec = ZONES.index("decision")
    # group A (large): decisions have p = 0.6; group B (small, "real"): decisions have p = 0.08
    n_a, n_b = 40, 4
    probs = np.full((n_a + n_b, 5), 0.01)
    importance = np.zeros(n_a + n_b, dtype=bool)
    zone = np.full(n_a + n_b, ZONES.index("discussion"))
    for i in range(0, n_a, 2):  # half of A are decisions
        probs[i, dec], importance[i], zone[i] = 0.6, True, dec
    for i in range(1, n_a, 2):  # the other half are kept non-decisions at p = 0.09
        probs[i, dec], importance[i] = 0.09, True
    probs[n_a, dec], importance[n_a], zone[n_a] = 0.08, True, dec  # one real decision in B
    grid = np.round(np.arange(0.05, 0.7, 0.01), 2)
    groups = [np.arange(n_a + n_b) >= n_a, np.arange(n_a + n_b) < n_a]
    pooled = tune_zone_thresholds(importance, zone, np.ones(n_a + n_b, bool), probs, grid)
    balanced = tune_zone_thresholds(importance, zone, np.ones(n_a + n_b, bool), probs, grid, groups)
    assert pooled["decision"] > 0.09  # the large group dominates: B's decision is missed
    assert balanced["decision"] <= 0.08  # each group counts: B's decision is found
