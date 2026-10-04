"""Tests for src/evaluation/metrics.py (pure maths, no TensorFlow)."""
import math

import numpy as np

from src.evaluation import metrics as M


def test_metrics_at_known_confusion():
    y = np.array([1, 1, 0, 0]); p = np.array([0.9, 0.4, 0.6, 0.1])
    m = M.metrics_at(y, p, 0.5)                       # predictions: 1 0 1 0
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)
    assert m["accuracy"] == m["precision"] == m["sensitivity"] == m["specificity"] == 0.5
    assert m["recall"] == m["sensitivity"] and abs(m["f1"] - 0.5) < 1e-12


def test_threshold_is_inclusive():
    assert M.confusion_at([1], [0.5], 0.5)[0] == 1    # prob == threshold counts as positive


def test_undefined_ratios_are_nan_not_zero():
    m = M.metrics_at(np.array([1, 0]), np.array([0.1, 0.2]), 0.9)   # nothing predicted positive
    assert math.isnan(m["precision"]) and math.isnan(m["f1"]) and m["sensitivity"] == 0.0 and m["specificity"] == 1.0


def test_threshold_table_covers_the_required_grid():
    t = M.threshold_table(np.array([1, 0, 1, 0]), np.array([0.8, 0.2, 0.6, 0.4]), M.REPORT_THRESHOLDS)
    assert list(t["threshold"]) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert {"precision", "recall", "sensitivity", "specificity", "f1"} <= set(t.columns)


def test_ranking_metrics_perfect_and_single_class():
    y = np.array([0, 0, 1, 1]); r = M.ranking_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert r["roc_auc"] == 1.0 and r["pr_auc"] == 1.0
    assert math.isnan(M.ranking_metrics(np.array([1, 1]), np.array([0.2, 0.9]))["roc_auc"])


def test_select_threshold_picks_highest_specificity_meeting_target():
    y = np.array([1] * 20 + [0] * 20)
    p = np.concatenate([np.linspace(0.30, 0.95, 20), np.linspace(0.02, 0.60, 20)])
    s = M.select_threshold(y, p)
    assert not s["fallback_max_f1"] and s["val_sensitivity"] >= 0.95
    higher = M.metrics_at(y, p, s["threshold"] + 0.01)           # any higher threshold must break the target
    assert higher["sensitivity"] < 0.95


def test_select_threshold_fallback_when_target_unreachable():
    y = np.array([1, 1, 0, 0]); p = np.array([0.2, 0.1, 0.9, 0.8])   # model is backwards
    s = M.select_threshold(y, p, target=1.01)
    assert s["fallback_max_f1"] is True and 0 < s["threshold"] < 1


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    rng = np.random.default_rng(0)
    pid = np.repeat(np.arange(30), 40); y = rng.integers(0, 2, pid.size)
    p = np.clip(0.5 + (y - 0.5) * 0.6 + rng.normal(0, 0.25, pid.size), 0, 1)
    a = M.bootstrap_by_patient(y, p, pid, 0.5, n_boot=200, seed=1)
    b = M.bootstrap_by_patient(y, p, pid, 0.5, n_boot=200, seed=1)
    assert a == b and a["n_patients"] == 30 and a["n_valid_resamples"] == 200
    est = M.ranking_metrics(y, p)["roc_auc"]
    assert a["roc_auc"]["lo"] <= est <= a["roc_auc"]["hi"]


def test_bootstrap_resamples_whole_patients():
    pid = np.array([1, 1, 2, 2, 3, 3]); y = np.array([0, 1, 0, 1, 0, 1]); p = np.array([.1, .9, .2, .8, .3, .7])
    out = M.bootstrap_by_patient(y, p, pid, 0.5, n_boot=50, seed=3)
    assert out["n_patients"] == 3 and out["roc_auc"]["lo"] == 1.0
