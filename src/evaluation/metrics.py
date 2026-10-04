"""Pure evaluation maths (NumPy/pandas/scikit-learn only; no TensorFlow). Positive class = Parasitized, label 1.
A sample is predicted positive when prob >= threshold. Undefined ratios (e.g. precision with no predicted positives) are NaN.

PRE-REGISTERED RULES (fixed before any model was scored on the validation set):
  * Threshold rule: among thresholds 0.01..0.99 (step 0.01), keep those with validation sensitivity >= 0.95 and pick the one
    with the highest validation specificity (ties: higher F1, then higher threshold). If no threshold reaches 0.95 sensitivity,
    fall back to the threshold with the highest F1 and flag it. Thresholds are chosen on VALIDATION data only.
  * PR-AUC is reported as average precision (sklearn); it can differ slightly from Keras' logged val_pr_auc.
  * Confidence intervals: patient-level bootstrap (resample whole patients), 95% percentile interval.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPORT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FINE_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
TARGET_SENSITIVITY = 0.95
RULE_TEXT = ("highest validation specificity among thresholds with validation sensitivity >= 0.95 "
             "(fallback: highest F1 if the target is unreachable)")
NAN = float("nan")


def _div(a, b):
    return a / b if b else NAN


def confusion_at(y, prob, thr):
    y, pred = np.asarray(y).astype(int), (np.asarray(prob, float) >= thr).astype(int)
    return (int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()),
            int(((pred == 0) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum()))   # tp, fp, tn, fn


def metrics_at(y, prob, thr):
    tp, fp, tn, fn = confusion_at(y, prob, thr)
    prec, sens, spec = _div(tp, tp + fp), _div(tp, tp + fn), _div(tn, tn + fp)
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else NAN       # NaN propagates if prec/sens undefined
    return {"threshold": float(thr), "accuracy": _div(tp + tn, tp + fp + tn + fn), "precision": prec, "recall": sens,
            "sensitivity": sens, "specificity": spec, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def threshold_table(y, prob, thresholds):
    return pd.DataFrame([metrics_at(y, prob, t) for t in thresholds])


def ranking_metrics(y, prob):
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return {"roc_auc": NAN, "pr_auc": NAN}
    return {"roc_auc": float(roc_auc_score(y, prob)), "pr_auc": float(average_precision_score(y, prob))}


def select_threshold(y, prob, target=TARGET_SENSITIVITY, grid=FINE_GRID):
    t = threshold_table(y, prob, grid)
    ok = t[t["sensitivity"] >= target]
    if len(ok):
        best = ok.sort_values(["specificity", "f1", "threshold"], ascending=[False, False, False]).iloc[0]
        fallback = False
    else:
        best = t.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]
        fallback = True
    return {"threshold": float(best["threshold"]), "rule": RULE_TEXT, "target_sensitivity": target,
            "fallback_max_f1": fallback, "val_sensitivity": float(best["sensitivity"]),
            "val_specificity": float(best["specificity"]), "val_f1": float(best["f1"])}


def bootstrap_by_patient(y, prob, patient_ids, thr, n_boot=1000, seed=42):
    """95% percentile intervals from resampling whole patients (images of one patient are not independent)."""
    y, prob, pid = np.asarray(y).astype(int), np.asarray(prob, float), np.asarray(patient_ids)
    groups = [np.flatnonzero(pid == p) for p in np.unique(pid)]
    rng = np.random.default_rng(seed)
    keys = ["roc_auc", "pr_auc", "accuracy", "sensitivity", "specificity", "precision", "f1"]
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        yy, pp = y[idx], prob[idx]
        if yy.min() == yy.max():
            continue                                   # a resample with one class only has no AUC; skip it
        m, r = metrics_at(yy, pp, thr), ranking_metrics(yy, pp)
        for k in keys:
            vals[k].append(r[k] if k in r else m[k])
    out = {k: {"lo": float(np.nanpercentile(v, 2.5)), "hi": float(np.nanpercentile(v, 97.5))} if v else {"lo": NAN, "hi": NAN}
           for k, v in vals.items()}
    out["n_valid_resamples"] = len(vals["roc_auc"])
    out["n_patients"] = len(groups)
    return out
