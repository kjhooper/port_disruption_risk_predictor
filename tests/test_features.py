"""Tests for features.py — feature engineering on synthetic weather data."""

import numpy as np
import pandas as pd
import pytest

from features import (
    compute_all_features,
    compute_fog_risk,
    compute_rolling_stats,
    compute_storm_approach_index,
    compute_wind_components,
)


def _weather_df(n=100, seed=42):
    """Minimal hourly DataFrame with plausible weather values."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "wind_speed_10m":       rng.uniform(0, 20, n),
        "wind_direction_10m":   rng.uniform(0, 360, n),
        "temperature_2m":       rng.uniform(5, 25, n),
        "dew_point_2m":         rng.uniform(0, 20, n),
        "pressure_msl":         rng.uniform(990, 1020, n),
        "cape":                 rng.uniform(0, 1000, n),
        "lifted_index":         rng.uniform(-5, 5, n),
        "precipitation":        rng.uniform(0, 5, n),
        "wave_height":          rng.uniform(0, 3, n),
        "weather_code":         rng.integers(0, 4, n).astype(float),
    }, index=idx)


# ── compute_wind_components ───────────────────────────────────────────────────

def test_wind_components_columns_created():
    df = _weather_df()
    result = compute_wind_components(df, "rotterdam")
    assert "onshore_wind" in result.columns
    assert "cross_wind" in result.columns
    assert "wind_onshore_flag" in result.columns


def test_wind_onshore_flag_is_binary():
    df = _weather_df()
    result = compute_wind_components(df, "rotterdam")
    assert result["wind_onshore_flag"].isin([0, 1]).all()


def test_wind_components_direction_270_is_fully_onshore():
    # Rotterdam sea_bearing = 270°. Wind FROM 270° = directly onshore.
    idx = pd.date_range("2024-01-01", periods=1, freq="h")
    df = pd.DataFrame(
        {"wind_speed_10m": [10.0], "wind_direction_10m": [270.0]}, index=idx
    )
    result = compute_wind_components(df, "rotterdam")
    # cos(270 - 270) = cos(0) = 1 → onshore_wind ≈ 10.0
    assert abs(result["onshore_wind"].iloc[0] - 10.0) < 0.01
    assert result["wind_onshore_flag"].iloc[0] == 1


def test_wind_components_opposite_direction_is_offshore():
    # Wind FROM 90° is directly offshore for Rotterdam (sea_bearing=270).
    idx = pd.date_range("2024-01-01", periods=1, freq="h")
    df = pd.DataFrame(
        {"wind_speed_10m": [10.0], "wind_direction_10m": [90.0]}, index=idx
    )
    result = compute_wind_components(df, "rotterdam")
    # cos(90 - 270) = cos(-180) = -1 → onshore_wind ≈ -10.0
    assert result["onshore_wind"].iloc[0] < 0
    assert result["wind_onshore_flag"].iloc[0] == 0


def test_wind_components_missing_input_skipped():
    df = pd.DataFrame({"unrelated": [1.0, 2.0]})
    result = compute_wind_components(df, "rotterdam")
    assert "onshore_wind" not in result.columns


def test_wind_components_preserves_existing_columns():
    df = _weather_df()
    original_cols = set(df.columns)
    result = compute_wind_components(df, "rotterdam")
    assert original_cols.issubset(set(result.columns))


# ── compute_fog_risk ──────────────────────────────────────────────────────────

def test_fog_risk_columns_created():
    df = _weather_df()
    result = compute_fog_risk(df)
    for col in ("td_spread", "fog_risk_score", "fog_flag"):
        assert col in result.columns


def test_fog_risk_td_spread_formula():
    idx = pd.date_range("2024-01-01", periods=3, freq="h")
    df = pd.DataFrame(
        {"temperature_2m": [10.0, 15.0, 20.0], "dew_point_2m": [8.0, 14.0, 18.0]},
        index=idx,
    )
    result = compute_fog_risk(df)
    expected = [2.0, 1.0, 2.0]
    for got, exp in zip(result["td_spread"].tolist(), expected):
        assert abs(got - exp) < 1e-9


def test_fog_risk_score_bounded_0_1():
    df = _weather_df()
    result = compute_fog_risk(df)
    assert result["fog_risk_score"].between(0, 1).all()


def test_fog_flag_below_threshold():
    idx = pd.date_range("2024-01-01", periods=3, freq="h")
    df = pd.DataFrame(
        {"temperature_2m": [10.0, 10.0, 10.0], "dew_point_2m": [9.0, 8.5, 5.0]},
        index=idx,
    )
    result = compute_fog_risk(df)
    # spread = 1.0, 1.5, 5.0 → fog, fog, no fog
    assert result["fog_flag"].tolist() == [1, 1, 0]


def test_fog_risk_missing_columns_skipped():
    df = pd.DataFrame({"wind_speed_10m": [5.0, 10.0]})
    result = compute_fog_risk(df)
    assert "td_spread" not in result.columns


# ── compute_storm_approach_index ──────────────────────────────────────────────

def test_storm_approach_index_created():
    df = _weather_df()
    df = compute_wind_components(df, "rotterdam")
    result = compute_storm_approach_index(df, port="rotterdam")
    assert "storm_approach_index" in result.columns


def test_storm_approach_index_bounded_0_1():
    df = _weather_df()
    df = compute_wind_components(df, "rotterdam")
    result = compute_storm_approach_index(df, port="rotterdam")
    assert result["storm_approach_index"].between(0, 1).all()


def test_storm_approach_index_no_port():
    # Should still work without port argument
    df = _weather_df()
    df = compute_wind_components(df, "rotterdam")
    result = compute_storm_approach_index(df)
    assert "storm_approach_index" in result.columns


def test_storm_approach_index_extreme_values():
    # Maximum conditions → score close to 1
    idx = pd.date_range("2024-01-01", periods=1, freq="h")
    df = pd.DataFrame({
        "wind_speed_10m":     [30.0],
        "wind_direction_10m": [270.0],  # fully onshore for Rotterdam
        "cape":               [3000.0],
        "lifted_index":       [-10.0],
    }, index=idx)
    df = compute_wind_components(df, "rotterdam")
    result = compute_storm_approach_index(df)
    assert result["storm_approach_index"].iloc[0] > 0.7


# ── compute_rolling_stats ─────────────────────────────────────────────────────

def test_rolling_stats_columns_created():
    df = _weather_df()
    df = compute_wind_components(df, "rotterdam")
    result = compute_rolling_stats(df)
    assert "wind_speed_10m_mean_6h" in result.columns
    assert "wind_speed_10m_max_24h" in result.columns
    assert "pressure_drop_6h" in result.columns
    assert "humidity_mean_6h" not in result.columns  # relative_humidity_2m not in fixture


def test_rolling_stats_mean_le_max():
    df = _weather_df()
    df = compute_wind_components(df, "rotterdam")
    result = compute_rolling_stats(df)
    assert (result["wind_speed_10m_mean_6h"] <= result["wind_speed_10m_max_6h"] + 1e-9).all()


def test_rolling_stats_pressure_drop_first_rows_nan():
    df = _weather_df(n=24)
    result = compute_rolling_stats(df)
    # diff(6) produces 6 NaN rows at the start
    assert result["pressure_drop_6h"].iloc[:6].isna().all()


# ── compute_all_features (integration) ───────────────────────────────────────

def test_all_features_returns_dataframe():
    df = _weather_df()
    result = compute_all_features(df, "rotterdam")
    assert isinstance(result, pd.DataFrame)


def test_all_features_preserves_row_count():
    df = _weather_df(n=50)
    result = compute_all_features(df, "rotterdam")
    assert len(result) == 50


def test_all_features_adds_columns():
    df = _weather_df()
    result = compute_all_features(df, "rotterdam")
    assert len(result.columns) > len(df.columns)


def test_all_features_core_columns_present():
    df = _weather_df()
    result = compute_all_features(df, "rotterdam")
    for col in ("onshore_wind", "td_spread", "fog_flag", "storm_approach_index",
                "pressure_drop_6h", "wind_speed_10m_mean_6h"):
        assert col in result.columns, f"Expected column '{col}' missing"


def test_all_features_does_not_mutate_input():
    df = _weather_df()
    original_cols = list(df.columns)
    compute_all_features(df, "rotterdam")
    assert list(df.columns) == original_cols
