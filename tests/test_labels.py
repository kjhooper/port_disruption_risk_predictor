"""Tests for labels.py — WMO group mapping and disruption label logic."""

import numpy as np
import pandas as pd
import pytest

from labels import (
    GROUP_ORDER,
    SEVERE_WMO_CODES,
    WMO_GROUPS,
    label_stats,
    make_binary_label,
    make_composite_disruption_label,
    make_weather_code_label,
)


def _df(*codes, **extra_cols):
    """Build a minimal hourly DataFrame with given weather codes (and optional extras)."""
    n = len(codes)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    data = {"weather_code": list(codes)}
    data.update(extra_cols)
    return pd.DataFrame(data, index=idx)


def _df_n(n, **cols):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(cols, index=idx)


# ── WMO group structure ───────────────────────────────────────────────────────

def test_wmo_groups_disjoint():
    all_codes = [code for rng in WMO_GROUPS.values() for code in rng]
    assert len(all_codes) == len(set(all_codes)), "WMO groups must not overlap"


def test_group_order_covers_all_groups():
    assert set(GROUP_ORDER) == set(WMO_GROUPS.keys())


# ── make_weather_code_label ───────────────────────────────────────────────────

@pytest.mark.parametrize("code,expected_group", [
    (0, "clear"),
    (1, "clear"),
    (3, "clear"),
    (45, "fog"),
    (48, "fog"),
    (51, "rain_snow"),
    (61, "rain_snow"),
    (71, "rain_snow"),
    (80, "showers"),
    (85, "showers"),
    (95, "thunderstorm"),
    (99, "thunderstorm"),
])
def test_weather_code_label_known_codes(code, expected_group):
    df = _df(code)
    label = make_weather_code_label(df)
    assert label.iloc[0] == expected_group


def test_weather_code_label_unknown_returns_nan():
    df = _df(100)  # not a valid WMO code
    label = make_weather_code_label(df)
    assert label.isna().all()


def test_weather_code_label_missing_column():
    df = pd.DataFrame({"other": [1, 2, 3]})
    label = make_weather_code_label(df)
    assert label.isna().all()


def test_weather_code_label_mixed():
    df = _df(0, 45, 95)
    labels = make_weather_code_label(df).tolist()
    assert labels == ["clear", "fog", "thunderstorm"]


# ── make_binary_label ─────────────────────────────────────────────────────────

def test_binary_label_clear_codes_are_zero():
    df = _df(0, 1, 2, 3)
    assert make_binary_label(df).tolist() == [0, 0, 0, 0]


def test_binary_label_non_clear_codes_are_one():
    df = _df(45, 61, 95)
    assert make_binary_label(df).tolist() == [1, 1, 1]


def test_binary_label_mixed():
    df = _df(0, 45, 1, 95)
    assert make_binary_label(df).tolist() == [0, 1, 0, 1]


def test_binary_label_missing_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert make_binary_label(df).tolist() == [0, 0, 0]


# ── make_composite_disruption_label ──────────────────────────────────────────

def test_composite_high_wind_triggers():
    df = _df_n(2, wind_speed_10m=[5.0, 20.0])  # 5 = calm, 20 > 15 threshold
    labels = make_composite_disruption_label(df, "rotterdam")
    assert labels.tolist() == [0, 1]


def test_composite_high_gusts_triggers():
    df = _df_n(2, wind_gusts_10m=[10.0, 25.0])  # 10 = ok, 25 > 22 threshold
    labels = make_composite_disruption_label(df, "rotterdam")
    assert labels.tolist() == [0, 1]


def test_composite_high_wave_triggers():
    df = _df_n(2, wave_height=[1.0, 3.0])  # 3.0 > 2.5 threshold
    labels = make_composite_disruption_label(df, "rotterdam")
    assert labels.tolist() == [0, 1]


def test_composite_fog_triggers():
    df = _df_n(2, td_spread=[5.0, 1.0])  # 1.0 < 2.0 fog threshold
    labels = make_composite_disruption_label(df, "rotterdam")
    assert labels.tolist() == [0, 1]


def test_composite_severe_wmo_triggers_regardless_of_wind():
    # WMO 95 (thunderstorm) should trigger even with zero wind
    df = _df_n(2, weather_code=[0.0, 95.0], wind_speed_10m=[0.0, 0.0])
    labels = make_composite_disruption_label(df, "rotterdam")
    assert labels.tolist() == [0, 1]


def test_composite_uses_port_thresholds():
    # Hong Kong has wave threshold 2.0 m vs Rotterdam's 2.5 m
    df = _df_n(1, wave_height=[2.2])  # above HK threshold, below Rotterdam threshold
    hk = make_composite_disruption_label(df, "hong_kong")
    rot = make_composite_disruption_label(df, "rotterdam")
    assert hk.iloc[0] == 1
    assert rot.iloc[0] == 0


def test_composite_empty_df():
    df = pd.DataFrame({"wind_speed_10m": []}, index=pd.DatetimeIndex([]))
    labels = make_composite_disruption_label(df, "rotterdam")
    assert len(labels) == 0


def test_composite_unknown_port_uses_defaults():
    # Should not raise — falls back to _DEFAULT_THRESHOLDS
    df = _df_n(2, wind_speed_10m=[5.0, 20.0])
    labels = make_composite_disruption_label(df, "unknown_port")
    assert labels.tolist() == [0, 1]


def test_composite_output_dtype():
    df = _df_n(3, wind_speed_10m=[5.0, 5.0, 20.0])
    labels = make_composite_disruption_label(df, "rotterdam")
    assert labels.dtype == "int8"


# ── label_stats ───────────────────────────────────────────────────────────────

def test_label_stats_keys():
    df = _df(0, 0, 45, 95)
    stats = label_stats(df)
    assert "group_counts" in stats
    assert "event_rate" in stats
    assert "n_total" in stats
    assert "class_weights" in stats


def test_label_stats_event_rate():
    df = _df(0, 0, 45, 95)  # 2 clear, 2 non-clear → 50% event rate
    stats = label_stats(df)
    assert abs(stats["event_rate"] - 0.5) < 1e-6


def test_label_stats_n_total():
    df = _df(0, 1, 2, 3, 45)
    stats = label_stats(df)
    assert stats["n_total"] == 5


# ── SEVERE_WMO_CODES ──────────────────────────────────────────────────────────

def test_severe_codes_include_thunderstorm():
    assert 95 in SEVERE_WMO_CODES
    assert 99 in SEVERE_WMO_CODES


def test_light_drizzle_not_severe():
    # 51-55 are light drizzle — should not be in severe set
    assert 51 not in SEVERE_WMO_CODES
    assert 53 not in SEVERE_WMO_CODES
