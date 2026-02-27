"""
features.py — Derived and spatial features for port disruption modelling.

Builds on raw weather from fetch.py to create features that capture:
  - Directional risk: is wind pushing weather toward the port from the sea?
  - Fog likelihood: temperature/dewpoint spread
  - Atmospheric instability: CAPE + lifted index combined signal
  - Rolling trends: evolving wind, pressure, precipitation over multiple windows
  - Upstream zone features: pressure gradients and storm energy in the approach corridor
"""

import numpy as np
import pandas as pd
import holidays as _holidays_lib
from fetch import PORTS, zone_points


# ── Holiday calendars ─────────────────────────────────────────────────────────

PORT_HOLIDAY_CALENDARS = {
    "rotterdam": lambda y: _holidays_lib.NL(years=y),
    "houston":   lambda y: _holidays_lib.US(state="TX", years=y),
    "hong_kong": lambda y: _holidays_lib.HK(years=y),
    "kaohsiung": lambda y: _holidays_lib.TW(years=y),
}


def make_holiday_features(index: pd.DatetimeIndex, port: str) -> pd.DataFrame:
    """
    Return a DataFrame aligned to index with three columns:
      is_holiday      (int8)  — 1 if the calendar date is a public holiday
      days_to_holiday (int8)  — signed distance to nearest holiday, clipped ±14
                                (negative = days before holiday, 0 = on holiday,
                                positive = days after holiday)
      holiday_name    (str)   — name of the holiday, "" if none

    Uses the public holiday calendar for the given port country/region.
    """
    if port not in PORT_HOLIDAY_CALENDARS:
        n = len(index)
        return pd.DataFrame({
            "is_holiday":      np.zeros(n, dtype="int8"),
            "days_to_holiday": np.zeros(n, dtype="int8"),
            "holiday_name":    [""] * n,
        }, index=index)

    # Unique calendar dates to avoid per-hour API calls
    unique_dates = pd.Series(index.date).unique()
    years = {d.year for d in unique_dates}

    # Build holiday dict: date → name
    hday_dict: dict = {}
    for y in years:
        hday_dict.update(PORT_HOLIDAY_CALENDARS[port](y))

    # Sorted list of holiday ordinals for searchsorted
    hday_dates = sorted(hday_dict.keys())
    if hday_dates:
        hday_ords = np.array([d.toordinal() for d in hday_dates])
    else:
        hday_ords = np.array([], dtype=int)

    # Compute per-unique-date features
    date_is_holiday:      dict = {}
    date_days_to_holiday: dict = {}
    date_holiday_name:    dict = {}

    for d in unique_dates:
        name = hday_dict.get(d, "")
        date_is_holiday[d]   = int(bool(name))
        date_holiday_name[d] = name

        if hday_ords.size == 0:
            date_days_to_holiday[d] = 0
        elif bool(name):
            date_days_to_holiday[d] = 0
        else:
            ord_d = d.toordinal()
            pos   = np.searchsorted(hday_ords, ord_d)
            # Nearest future holiday
            if pos < len(hday_ords):
                days_fwd = hday_ords[pos] - ord_d
            else:
                days_fwd = 9999
            # Nearest past holiday
            if pos > 0:
                days_back = ord_d - hday_ords[pos - 1]
            else:
                days_back = 9999
            # Positive = after holiday (days_back); negative = before holiday (-days_fwd)
            if days_fwd <= days_back:
                delta = -days_fwd
            else:
                delta = days_back
            date_days_to_holiday[d] = int(np.clip(delta, -14, 14))

    # Map back to full hourly index
    dates_ser = pd.Series(index.date, index=index)
    return pd.DataFrame({
        "is_holiday":      dates_ser.map(date_is_holiday).astype("int8"),
        "days_to_holiday": dates_ser.map(date_days_to_holiday).astype("int8"),
        "holiday_name":    dates_ser.map(date_holiday_name).fillna(""),
    }, index=index)


# ── Directional / spatial features ───────────────────────────────────────────

def _angular_diff(a: float, b: float) -> float:
    """Signed angular difference a - b, normalised to [-180, 180]."""
    d = (a - b) % 360
    return d - 360 if d > 180 else d


def compute_wind_components(df: pd.DataFrame, port: str) -> pd.DataFrame:
    """
    Decompose wind into onshore / cross-shore components relative to the port's
    primary weather exposure direction (sea_bearing in PORTS).

    sea_bearing is the compass direction wind must come FROM to push weather
    toward the port — e.g. 270° for Rotterdam (North Sea to the west).
    When sea_bearing is a list (multi-bearing port), uses bearings[0].

    onshore_wind > 0  →  wind pushing storms in from the sea
    cross_wind        →  wind parallel to the coastline (signed)
    wind_onshore_flag →  1 when onshore_wind is positive

    Requires: wind_speed_10m, wind_direction_10m
    """
    if "wind_speed_10m" not in df.columns or "wind_direction_10m" not in df.columns:
        return df

    sea_bearing = PORTS[port]["sea_bearing"]
    if isinstance(sea_bearing, list):
        sea_bearing = sea_bearing[0]

    diff_deg = df["wind_direction_10m"].apply(lambda d: _angular_diff(d, sea_bearing))
    diff_rad = np.radians(diff_deg)

    df["onshore_wind"]      = df["wind_speed_10m"] * np.cos(diff_rad)
    df["cross_wind"]        = df["wind_speed_10m"] * np.sin(diff_rad)
    df["wind_onshore_flag"] = (df["onshore_wind"] > 0).astype(int)

    return df


# ── Zone wind components ──────────────────────────────────────────────────────

def compute_zone_wind_components(df: pd.DataFrame, port: str) -> pd.DataFrame:
    """
    Compute onshore wind component for each upstream zone point.
    Uses each zone's own bearing for the angular difference calculation.

    Requires: {prefix}_wind_speed_10m and {prefix}_wind_direction_10m columns
    (populated by fetch_port_all with zone data).
    """
    for z in zone_points(port):
        prefix   = z["prefix"]
        spd_col  = f"{prefix}_wind_speed_10m"
        dir_col  = f"{prefix}_wind_direction_10m"

        if spd_col not in df.columns or dir_col not in df.columns:
            continue

        diff_deg = df[dir_col].apply(lambda d: _angular_diff(d, z["bearing"]))
        diff_rad = np.radians(diff_deg)
        df[f"{prefix}_onshore_wind"] = df[spd_col] * np.cos(diff_rad)

    return df


# ── Fog risk ─────────────────────────────────────────────────────────────────

def compute_fog_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fog forms when temperature and dew point converge.
    td_spread = temperature_2m - dew_point_2m (°C)
      < 2°C   →  fog likely     (fog_flag = 1)
      < 0.5°C →  dense fog likely

    fog_risk_score: 0–1, higher = foggier (5°C spread mapped to 0).

    Requires: temperature_2m, dew_point_2m
    """
    if "temperature_2m" not in df.columns or "dew_point_2m" not in df.columns:
        return df

    df["td_spread"]      = df["temperature_2m"] - df["dew_point_2m"]
    df["fog_risk_score"] = 1 - (df["td_spread"].clip(0, 5) / 5)   # 0–1
    df["fog_flag"]       = (df["td_spread"] < 2.0).astype(int)

    return df


# ── Upstream approach features ────────────────────────────────────────────────

def compute_upstream_approach(df: pd.DataFrame, port: str) -> pd.DataFrame:
    """
    Compute features that measure storm energy and pressure gradients in the
    upstream approach corridor relative to the port.

    For each zone:
      {prefix}_pressure_gradient — pressure at port minus zone pressure (positive = lower pressure offshore)
      {prefix}_cape_excess       — CAPE energy above port level (storm energy in pipeline)
      {prefix}_wind_delta        — zone onshore wind minus port onshore wind (wind accelerating toward port)

    Requires: compute_wind_components and compute_zone_wind_components to have run first.
    """
    for z in zone_points(port):
        prefix = z["prefix"]

        if "pressure_msl" in df.columns and f"{prefix}_pressure_msl" in df.columns:
            df[f"{prefix}_pressure_gradient"] = df["pressure_msl"] - df[f"{prefix}_pressure_msl"]

        if "cape" in df.columns and f"{prefix}_cape" in df.columns:
            df[f"{prefix}_cape_excess"] = (df[f"{prefix}_cape"] - df["cape"]).clip(lower=0)

        if "onshore_wind" in df.columns and f"{prefix}_onshore_wind" in df.columns:
            df[f"{prefix}_wind_delta"] = df[f"{prefix}_onshore_wind"] - df["onshore_wind"]

    return df


# ── Storm approach index ──────────────────────────────────────────────────────

def compute_storm_approach_index(df: pd.DataFrame, port: str = None) -> pd.DataFrame:
    """
    Composite 0–1 score for a storm being pushed toward the port from the sea.

    Base components (averaged equally):
      cape_score    — convective energy (0 = calm, 1 = ≥3000 J/kg)
      li_score      — lifted index instability (0 = stable LI≥6, 1 = extreme LI≤-10)
      onshore_score — onshore wind fraction of 30 m/s (0 = calm/offshore, 1 = 30 m/s onshore)

    Optional upstream components (when port is provided, using farthest zone):
      upstream_cape     — farthest zone cape_excess normalised over 2000
      upstream_pressure — farthest zone pressure_gradient normalised over 20 hPa

    High score = high-energy atmosphere + unstable + wind pushing from the sea.
    """
    components = pd.DataFrame(index=df.index)

    if "cape" in df.columns:
        components["cape_score"] = df["cape"].clip(0, 3000) / 3000

    if "lifted_index" in df.columns:
        # Map LI from [6, -10] → [0, 1]; negative LI = unstable
        components["li_score"] = (df["lifted_index"].clip(-10, 6) * -1 + 6) / 16

    if "onshore_wind" in df.columns:
        components["onshore_score"] = df["onshore_wind"].clip(0, 30) / 30

    # Upstream zone contributions (farthest zone only)
    if port is not None:
        zones = zone_points(port)
        if zones:
            far_prefix = zones[-1]["prefix"]

            cape_exc_col = f"{far_prefix}_cape_excess"
            if cape_exc_col in df.columns:
                components["upstream_cape"] = df[cape_exc_col].clip(0, 2000) / 2000

            pgrad_col = f"{far_prefix}_pressure_gradient"
            if pgrad_col in df.columns:
                components["upstream_pressure"] = df[pgrad_col].clip(0, 20) / 20

    if not components.empty:
        df["storm_approach_index"] = components.mean(axis=1)

    return df


# ── Rolling trend features ────────────────────────────────────────────────────

def compute_rolling_stats(df: pd.DataFrame,
                          windows_hours: list = [3, 6, 12, 24]) -> pd.DataFrame:
    """
    Rolling mean and max over multiple time windows for key disruption variables.
    Captures building trends that point-in-time values miss.

    Also computes:
      pressure_drop_6h  — 6-hour pressure change (negative = rapidly falling)
      humidity_mean_6h  — smoothed humidity trend
    """
    roll_cols = [c for c in [
        "wind_speed_10m", "onshore_wind", "precipitation",
        "cape", "storm_approach_index",
    ] if c in df.columns]

    for col in roll_cols:
        for h in windows_hours:
            df[f"{col}_mean_{h}h"] = df[col].rolling(h, min_periods=1).mean()
            df[f"{col}_max_{h}h"]  = df[col].rolling(h, min_periods=1).max()

    if "pressure_msl" in df.columns:
        df["pressure_drop_6h"] = df["pressure_msl"].diff(6)   # negative = falling

    if "relative_humidity_2m" in df.columns:
        df["humidity_mean_6h"] = df["relative_humidity_2m"].rolling(6, min_periods=1).mean()

    return df


# ── Convenience wrapper ───────────────────────────────────────────────────────

def compute_all_features(df: pd.DataFrame, port: str) -> pd.DataFrame:
    """
    Run all feature engineering steps in order. Returns a copy with new columns appended.
    Call this on the historical_wide DataFrame after fetching.
    """
    df = df.copy()
    df = compute_wind_components(df, port)
    df = compute_zone_wind_components(df, port)
    df = compute_fog_risk(df)
    df = compute_upstream_approach(df, port)
    df = compute_storm_approach_index(df, port)
    df = compute_rolling_stats(df)
    if port in PORT_HOLIDAY_CALENDARS:
        hol = make_holiday_features(df.index, port)
        df["is_holiday"]      = hol["is_holiday"].values
        df["days_to_holiday"] = hol["days_to_holiday"].values
    return df
