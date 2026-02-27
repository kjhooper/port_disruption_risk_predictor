"""Tests for quality.py — data quality checks."""

import numpy as np
import pandas as pd
import pytest

from quality import (
    check_completeness,
    check_freshness,
    check_outliers,
    check_temporal_gaps,
    quality_summary_df,
    run_all_checks,
)


def _fresh_df(n=100):
    """Complete hourly DataFrame ending near now."""
    end = pd.Timestamp.utcnow().tz_localize(None)
    idx = pd.date_range(end=end, periods=n, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "wind_speed_10m": rng.uniform(0, 15, n),
        "pressure_msl":   rng.uniform(990, 1020, n),
    }, index=idx)


def _stale_df(n=50):
    """Complete DataFrame ending years ago."""
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame({"wind_speed_10m": np.ones(n)}, index=idx)


# ── check_completeness ────────────────────────────────────────────────────────

def test_completeness_full_data():
    df = _fresh_df()
    result = check_completeness(df)
    assert result["status"] == "ok"
    assert result["overall_score"] == 1.0


def test_completeness_partial_missing():
    df = _fresh_df()
    df.loc[df.index[:50], "wind_speed_10m"] = np.nan  # 50% missing in one col
    result = check_completeness(df)
    assert result["overall_score"] < 1.0


def test_completeness_returns_per_column():
    df = _fresh_df()
    result = check_completeness(df)
    assert "per_column" in result
    assert "wind_speed_10m" in result["per_column"]
    assert "pressure_msl" in result["per_column"]


def test_completeness_empty_col_scores_zero():
    df = _fresh_df()
    df["wind_speed_10m"] = np.nan
    result = check_completeness(df)
    assert result["per_column"]["wind_speed_10m"] == 0.0


# ── check_temporal_gaps ───────────────────────────────────────────────────────

def test_temporal_gaps_no_gap():
    df = _fresh_df()
    result = check_temporal_gaps(df)
    assert result["status"] == "ok"
    assert result["n_gaps"] == 0


def test_temporal_gaps_detects_large_gap():
    idx1 = pd.date_range("2024-01-01 00:00", periods=24, freq="h")
    idx2 = pd.date_range("2024-01-03 00:00", periods=24, freq="h")  # 24h gap
    idx = idx1.append(idx2)
    df = pd.DataFrame({"x": np.ones(len(idx))}, index=idx)
    result = check_temporal_gaps(df)
    assert result["max_gap_hours"] >= 24
    assert result["status"] == "fail"


def test_temporal_gaps_small_gap_warns():
    # Build a gap just over 3h but under 12h (warn threshold)
    idx1 = pd.date_range("2024-01-01 00:00", periods=12, freq="h")
    idx2 = pd.date_range("2024-01-01 18:00", periods=12, freq="h")  # 6h gap
    idx = idx1.append(idx2)
    df = pd.DataFrame({"x": np.ones(len(idx))}, index=idx)
    result = check_temporal_gaps(df)
    assert result["status"] == "warn"


def test_temporal_gaps_no_datetime_index():
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = check_temporal_gaps(df)
    assert result["status"] == "skip"


def test_temporal_gaps_gap_starts_list():
    idx1 = pd.date_range("2024-01-01", periods=4, freq="h")
    idx2 = pd.date_range("2024-01-02", periods=4, freq="h")
    idx = idx1.append(idx2)
    df = pd.DataFrame({"x": np.ones(8)}, index=idx)
    result = check_temporal_gaps(df)
    assert "gap_starts" in result
    assert len(result["gap_starts"]) >= 1


# ── check_freshness ───────────────────────────────────────────────────────────

def test_freshness_recent_data():
    df = _fresh_df()
    result = check_freshness(df)
    assert result["status"] == "ok"
    assert result["age_hours"] < 2.0


def test_freshness_stale_data():
    df = _stale_df()
    result = check_freshness(df)
    assert result["status"] == "fail"
    assert result["age_hours"] > 100


def test_freshness_no_datetime_index():
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = check_freshness(df)
    assert result["status"] == "skip"


def test_freshness_returns_latest_record():
    df = _fresh_df()
    result = check_freshness(df)
    assert "latest_record" in result


# ── check_outliers ────────────────────────────────────────────────────────────

def test_outliers_in_bounds():
    df = pd.DataFrame({"wind_speed_10m": [5.0, 10.0, 15.0]})
    result = check_outliers(df)
    assert result["per_column"]["wind_speed_10m"]["n_out_of_bounds"] == 0


def test_outliers_wind_above_max():
    df = pd.DataFrame({"wind_speed_10m": [5.0, 200.0]})  # 200 > 80 m/s limit
    result = check_outliers(df)
    assert result["per_column"]["wind_speed_10m"]["n_out_of_bounds"] == 1


def test_outliers_pressure_below_min():
    df = pd.DataFrame({"pressure_msl": [1010.0, 500.0]})  # 500 < 870 hPa limit
    result = check_outliers(df)
    assert result["per_column"]["pressure_msl"]["n_out_of_bounds"] == 1


def test_outliers_unknown_column_skips_bounds():
    # Columns not in VARIABLE_BOUNDS should still get IQR check but no bounds check
    df = pd.DataFrame({"unknown_feature": [1.0, 2.0, 3.0]})
    result = check_outliers(df)
    assert "n_out_of_bounds" not in result["per_column"]["unknown_feature"]
    assert "n_iqr_outliers" in result["per_column"]["unknown_feature"]


def test_outliers_status_keys():
    df = _fresh_df()
    result = check_outliers(df)
    assert "status" in result
    assert "total_out_of_bounds" in result
    assert result["status"] in ("ok", "warn", "fail")


# ── run_all_checks ────────────────────────────────────────────────────────────

def test_run_all_checks_structure():
    df = _fresh_df()
    report = run_all_checks(df)
    assert set(report["checks"].keys()) == {
        "completeness", "temporal_gaps", "freshness", "outliers"
    }
    assert 0.0 <= report["overall_score"] <= 1.0
    assert report["overall_status"] in ("ok", "warn", "fail")


def test_run_all_checks_row_col_counts():
    df = _fresh_df(n=48)
    report = run_all_checks(df)
    assert report["n_rows"] == 48
    assert report["n_cols"] == 2


def test_run_all_checks_good_data_scores_high():
    df = _fresh_df()
    report = run_all_checks(df)
    assert report["overall_score"] >= 0.5  # at least warn-level quality


# ── quality_summary_df ────────────────────────────────────────────────────────

def test_quality_summary_df_shape():
    df = _fresh_df()
    report = run_all_checks(df)
    summary = quality_summary_df(report)
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 4  # one row per check


def test_quality_summary_df_columns():
    df = _fresh_df()
    report = run_all_checks(df)
    summary = quality_summary_df(report)
    assert "Check" in summary.columns
    assert "Status" in summary.columns
    assert "Detail" in summary.columns


def test_quality_summary_df_statuses_valid():
    df = _fresh_df()
    report = run_all_checks(df)
    summary = quality_summary_df(report)
    valid = {"OK", "WARN", "FAIL", "SKIP"}
    assert set(summary["Status"].unique()).issubset(valid)
