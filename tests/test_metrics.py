"""Tests for metrics.py — pure evaluation functions, no I/O."""

import numpy as np
import pandas as pd
import pytest

from metrics import (
    calibration_curve_data,
    cohen_d,
    eval_binary,
    eval_classifier,
    eval_forecaster,
)


# ── eval_binary ───────────────────────────────────────────────────────────────

def test_eval_binary_perfect():
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 1]
    y_prob = [1.0, 0.0, 1.0, 0.0, 1.0]
    result = eval_binary(y_true, y_pred, y_prob)
    assert result["roc_auc"] == 1.0
    assert result["f1"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["ece"] == 0.0


def test_eval_binary_all_keys_present():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 1]
    y_prob = [0.9, 0.1, 0.4, 0.6]
    result = eval_binary(y_true, y_pred, y_prob)
    for key in ("roc_auc", "pr_auc", "f1", "precision", "recall", "ece"):
        assert key in result


def test_eval_binary_scores_in_range():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 100).tolist()
    y_prob = rng.random(100).tolist()
    y_pred = [int(p > 0.5) for p in y_prob]
    result = eval_binary(y_true, y_pred, y_prob)
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert 0.0 <= result["f1"] <= 1.0
    assert 0.0 <= result["ece"] <= 1.0


# ── eval_forecaster ───────────────────────────────────────────────────────────

def test_eval_forecaster_perfect():
    y = [1.0, 2.0, 3.0, 4.0]
    result = eval_forecaster(y, y)
    assert result["rmse"] == 0.0
    assert result["r2"] == 1.0
    assert result["bias"] == 0.0
    assert result["mae"] == 0.0


def test_eval_forecaster_positive_bias():
    y_true = [0.0, 1.0, 2.0]
    y_pred = [1.0, 2.0, 3.0]  # always 1 too high
    result = eval_forecaster(y_true, y_pred)
    assert abs(result["bias"] - 1.0) < 1e-6


def test_eval_forecaster_nan_rows_dropped():
    y_true = [1.0, float("nan"), 3.0]
    y_pred = [1.0, 2.0, 3.0]
    result = eval_forecaster(y_true, y_pred)
    assert result["rmse"] == 0.0  # only valid rows compared


def test_eval_forecaster_all_nan():
    result = eval_forecaster([float("nan")], [float("nan")])
    assert result["rmse"] is None
    assert result["r2"] is None


# ── cohen_d ───────────────────────────────────────────────────────────────────

def test_cohen_d_large_separation():
    rng = np.random.default_rng(0)
    # Groups well-separated and with variance so pooled_std > 0
    a = pd.Series(rng.normal(loc=10.0, scale=1.0, size=50))
    b = pd.Series(rng.normal(loc=0.0, scale=1.0, size=50))
    d = cohen_d(a, b)
    assert d > 5.0  # effect size should be huge given 10-unit separation with sd=1


def test_cohen_d_zero_variance():
    a = pd.Series([5.0] * 20)
    b = pd.Series([5.0] * 20)
    assert np.isnan(cohen_d(a, b))


def test_cohen_d_too_few_samples():
    assert np.isnan(cohen_d(pd.Series([1.0]), pd.Series([2.0])))


def test_cohen_d_symmetric_sign():
    a = pd.Series(np.arange(10, dtype=float))
    b = pd.Series(np.arange(20, 30, dtype=float))
    assert cohen_d(a, b) < 0  # b > a so d should be negative
    assert cohen_d(b, a) > 0


# ── calibration_curve_data ───────────────────────────────────────────────────

def test_calibration_curve_lengths_match():
    y_true = np.array([1, 0, 1, 0] * 25)
    y_prob = np.array([0.9, 0.1, 0.8, 0.2] * 25)
    fracs, means = calibration_curve_data(y_true, y_prob)
    assert len(fracs) == len(means)


def test_calibration_curve_values_bounded():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_prob = rng.random(200)
    fracs, means = calibration_curve_data(y_true, y_prob)
    assert all(0.0 <= f <= 1.0 for f in fracs)
    assert all(0.0 <= m <= 1.0 for m in means)


def test_calibration_curve_perfect_model():
    # Perfect model: prob=1 for true=1, prob=0 for true=0
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([1.0, 1.0, 0.0, 0.0])
    fracs, means = calibration_curve_data(y_true, y_prob, n_bins=2)
    # Bin 0 (prob~0): fraction=0; bin 1 (prob~1): fraction=1
    assert 0.0 in fracs or 1.0 in fracs


# ── eval_classifier ───────────────────────────────────────────────────────────

def test_eval_classifier_perfect():
    y_true = ["clear", "fog", "clear", "thunderstorm"]
    y_pred = ["clear", "fog", "clear", "thunderstorm"]
    result = eval_classifier(y_true, y_pred)
    assert result["macro_f1"] == 1.0


def test_eval_classifier_with_probs():
    y_true = ["clear", "fog", "thunderstorm", "clear"]
    y_pred = ["clear", "fog", "thunderstorm", "clear"]
    y_prob = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.9, 0.05, 0.05],
    ])
    result = eval_classifier(y_true, y_pred, y_prob)
    assert "roc_auc" in result
    assert "ece" in result
    assert result["roc_auc"] == 1.0


def test_eval_classifier_classes_key():
    y_true = ["a", "b", "a", "b"]
    y_pred = ["a", "b", "b", "a"]
    result = eval_classifier(y_true, y_pred)
    assert "classes" in result
    assert set(result["classes"]) == {"a", "b"}
