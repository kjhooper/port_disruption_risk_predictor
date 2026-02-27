"""
metrics.py — Pure evaluation functions for M1, M2, M3.

No side effects, no Streamlit imports. All functions return plain dicts or arrays.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.preprocessing import label_binarize


# ── Multi-class classifier (M1) ──────────────────────────────────────────────

def eval_classifier(y_true, y_pred, y_prob=None) -> dict:
    """
    Evaluate multi-class classifier (M1).

    y_true / y_pred: array-like of class label strings
    y_prob: (n_samples, n_classes) probability array, optional

    Returns: macro_f1, per_class_f1, roc_auc (OvR), pr_auc, confusion_matrix,
             and ECE if y_prob is provided.
    """
    classes = sorted(set(y_true))

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_f1 = dict(zip(
        classes,
        f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0).tolist(),
    ))
    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    result: dict = {
        "macro_f1":        round(float(macro_f1), 4),
        "per_class_f1":    {k: round(v, 4) for k, v in per_class_f1.items()},
        "confusion_matrix": cm,
        "classes":          classes,
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            result["roc_auc"] = round(float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro", labels=classes,
            )), 4)
        except Exception:
            result["roc_auc"] = None

        try:
            y_bin = label_binarize(y_true, classes=classes)
            # label_binarize returns (n, 1) for binary case; expand to (n, 2)
            if y_bin.shape[1] == 1:
                y_bin = np.hstack([1 - y_bin, y_bin])
            result["pr_auc"] = round(float(
                average_precision_score(y_bin, y_prob, average="macro")
            ), 4)
        except Exception:
            result["pr_auc"] = None

        result["ece"] = round(float(_ece_multiclass(y_true, y_prob, classes)), 4)

    return result


# ── Binary detector (M2) ─────────────────────────────────────────────────────

def eval_binary(y_true, y_pred, y_prob) -> dict:
    """
    Evaluate binary event detector (M2).

    y_true / y_pred: array-like of 0/1 integers
    y_prob: 1-D array of positive-class probabilities

    Returns: roc_auc, pr_auc, f1, precision, recall, ECE.
    """
    y_prob = np.asarray(y_prob)
    return {
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc":    round(float(average_precision_score(y_true, y_prob)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "ece":       round(float(_ece_binary(y_true, y_prob)), 4),
    }


# ── Variable forecaster (M3) ─────────────────────────────────────────────────

def eval_forecaster(y_true, y_pred) -> dict:
    """
    Evaluate variable forecaster (M3).

    Returns: rmse, mae, r2, bias (mean signed error).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {"rmse": None, "mae": None, "r2": None, "bias": None}
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
        "mae":  round(float(mean_absolute_error(yt, yp)), 4),
        "r2":   round(float(r2_score(yt, yp)), 4),
        "bias": round(float(np.mean(yp - yt)), 4),
    }


# ── Effect size ───────────────────────────────────────────────────────────────

def cohen_d(a: pd.Series, b: pd.Series) -> float:
    """
    (mean_a - mean_b) / pooled_std

    Used in review.py Tab 7 to measure effect size between event and clear groups.
    Returns NaN when either group has < 2 observations or zero variance.
    """
    a = a.dropna()
    b = b.dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_var = (
        (len(a) - 1) * float(a.var()) + (len(b) - 1) * float(b.var())
    ) / (len(a) + len(b) - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled_std)


# ── Calibration ───────────────────────────────────────────────────────────────

def calibration_curve_data(
    y_true, y_prob, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    fraction_of_positives, mean_predicted_value — for calibration plot.

    Returns (fraction_of_positives, mean_predicted_value) as 1-D numpy arrays,
    with NaN bins removed.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    fractions = np.full(n_bins, np.nan)
    means = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() > 0:
            fractions[b] = y_true[mask].mean()
            means[b] = y_prob[mask].mean()
    valid = ~np.isnan(fractions)
    return fractions[valid], means[valid]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ece_binary(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected calibration error for binary classifier."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += mask.sum() * abs(acc - conf)
    return ece / max(n, 1)


def _ece_multiclass(y_true, y_prob: np.ndarray, classes: list, n_bins: int = 10) -> float:
    """Expected calibration error for multi-class classifier."""
    y_prob = np.asarray(y_prob)
    confidence = y_prob.max(axis=1)
    predicted_idx = y_prob.argmax(axis=1)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx.get(y, -1) for y in y_true])
    correct = (predicted_idx == y_idx).astype(float)

    bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    bin_ids = np.digitize(confidence, bins) - 1
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() > 0:
            acc = correct[mask].mean()
            conf = confidence[mask].mean()
            ece += mask.sum() * abs(acc - conf)
    return ece / max(n, 1)
