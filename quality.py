"""
quality.py — Data quality checks and metrics.
Keeps things simple: completeness, outliers, temporal gaps, freshness.
Each check returns a dict so results are easy to display in a dashboard.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ── Thresholds (tune these as you learn your data) ───────────────────────────

QUALITY_THRESHOLDS = {
    "completeness_warn":  0.90,   # flag if <90% complete
    "completeness_fail":  0.75,   # fail if <75% complete
    "gap_warn_hours":     3,      # warn if any gap > 3 hours
    "gap_fail_hours":     12,     # fail if any gap > 12 hours
    "freshness_warn_hours": 2,    # warn if last record > 2h old
    "freshness_fail_hours": 6,    # fail if last record > 6h old
}

# Reasonable physical bounds for each variable
VARIABLE_BOUNDS = {
    "wind_speed_10m":    (0,   80),    # m/s
    "wind_gusts_10m":    (0,  100),    # m/s
    "precipitation":     (0,  200),    # mm/hr
    "pressure_msl":      (870, 1084),  # hPa
    "visibility":        (0,  24000),  # metres
    "weather_code":      (0,  99),
    "wave_height":       (0,  20),     # metres
    "wave_period":       (0,  30),     # seconds
    "wind_wave_height":  (0,  20),
    "wind_direction_10m": (0,  360),   # degrees
    "relative_humidity_2m": (0, 100), # %
    "dew_point_2m":      (-60, 40),   # °C
    "temperature_2m":    (-50, 60),   # °C
    "cape":              (0, 10000),  # J/kg
    "lifted_index":      (-20, 20),   # dimensionless
    "cloud_cover":       (0,  100),   # %
    "dust":              (0, 5000),   # μg/m³
    "WDIR":              (0,  360),    # buoy wind direction
    "WSPD":              (0,  80),     # buoy wind speed m/s
    "GST":               (0,  100),    # buoy gust
    "WVHT":              (0,  20),     # buoy wave height
    "ATMP":              (-50, 60),    # air temp C
    "WTMP":              (-2,  40),    # water temp C
}


def check_completeness(df: pd.DataFrame) -> dict:
    """
    For each column, compute fraction of non-null values.
    Returns per-column scores and an overall score.
    """
    col_scores = (df.notna().sum() / len(df)).to_dict()
    overall = float(np.mean(list(col_scores.values())))

    status = "ok"
    if overall < QUALITY_THRESHOLDS["completeness_fail"]:
        status = "fail"
    elif overall < QUALITY_THRESHOLDS["completeness_warn"]:
        status = "warn"

    return {
        "check": "completeness",
        "status": status,
        "overall_score": round(overall, 4),
        "per_column": {k: round(v, 4) for k, v in col_scores.items()},
    }


def check_temporal_gaps(df: pd.DataFrame, expected_freq: str = "1h") -> dict:
    """
    Detect gaps in the time index larger than expected_freq.
    Returns the largest gap and a list of gap start times.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return {"check": "temporal_gaps", "status": "skip", "reason": "No datetime index"}

    expected_delta = pd.tseries.frequencies.to_offset(expected_freq)
    diffs = df.index.to_series().diff().dropna()
    gaps = diffs[diffs > expected_delta * 1.5]  # 50% tolerance

    max_gap_hours = float(gaps.max().total_seconds() / 3600) if len(gaps) > 0 else 0.0

    status = "ok"
    if max_gap_hours > QUALITY_THRESHOLDS["gap_fail_hours"]:
        status = "fail"
    elif max_gap_hours > QUALITY_THRESHOLDS["gap_warn_hours"]:
        status = "warn"

    return {
        "check": "temporal_gaps",
        "status": status,
        "n_gaps": len(gaps),
        "max_gap_hours": round(max_gap_hours, 2),
        "gap_starts": [str(t) for t in gaps.index[:10]],  # first 10
    }


def check_freshness(df: pd.DataFrame) -> dict:
    """
    How old is the most recent record?
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return {"check": "freshness", "status": "skip", "reason": "No datetime index"}

    latest = df.index.max()
    age_hours = float((pd.Timestamp.utcnow().tz_localize(None) - latest).total_seconds() / 3600)

    status = "ok"
    if age_hours > QUALITY_THRESHOLDS["freshness_fail_hours"]:
        status = "fail"
    elif age_hours > QUALITY_THRESHOLDS["freshness_warn_hours"]:
        status = "warn"

    return {
        "check": "freshness",
        "status": status,
        "latest_record": str(latest),
        "age_hours": round(age_hours, 2),
    }


def check_outliers(df: pd.DataFrame) -> dict:
    """
    Flag values outside known physical bounds and IQR-based outliers.
    Returns per-column outlier counts.
    """
    results = {}

    for col in df.select_dtypes(include=[np.number]).columns:
        col_result = {"n_total": int(df[col].notna().sum())}

        # Physical bounds check
        if col in VARIABLE_BOUNDS:
            lo, hi = VARIABLE_BOUNDS[col]
            n_out_of_bounds = int(((df[col] < lo) | (df[col] > hi)).sum())
            col_result["n_out_of_bounds"] = n_out_of_bounds
            col_result["bounds"] = [lo, hi]

        # IQR check
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        n_iqr_outliers = int(((df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)).sum())
        col_result["n_iqr_outliers"] = n_iqr_outliers

        results[col] = col_result

    total_out_of_bounds = sum(v.get("n_out_of_bounds", 0) for v in results.values())
    status = "fail" if total_out_of_bounds > 10 else ("warn" if total_out_of_bounds > 0 else "ok")

    return {
        "check": "outliers",
        "status": status,
        "total_out_of_bounds": total_out_of_bounds,
        "per_column": results,
    }


def run_all_checks(df: pd.DataFrame, expected_freq: str = "1h") -> dict:
    """
    Run all quality checks and return a summary report.
    Also computes an overall quality score (0–1).
    """
    checks = {
        "completeness":   check_completeness(df),
        "temporal_gaps":  check_temporal_gaps(df, expected_freq),
        "freshness":      check_freshness(df),
        "outliers":       check_outliers(df),
    }

    # Simple scoring: ok=1, warn=0.5, fail=0
    status_scores = {"ok": 1.0, "warn": 0.5, "fail": 0.0, "skip": 1.0}
    scores = [status_scores[c["status"]] for c in checks.values()]
    overall_score = float(np.mean(scores))

    overall_status = "ok"
    if overall_score < 0.5:
        overall_status = "fail"
    elif overall_score < 1.0:
        overall_status = "warn"

    return {
        "overall_score": round(overall_score, 4),
        "overall_status": overall_status,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "checks": checks,
    }


def quality_summary_df(report: dict) -> pd.DataFrame:
    """
    Convert a quality report into a simple summary DataFrame for display.
    """
    rows = []
    for name, check in report["checks"].items():
        rows.append({
            "Check": name,
            "Status": check["status"].upper(),
            "Detail": _check_detail(check),
        })
    return pd.DataFrame(rows)


def _check_detail(check: dict) -> str:
    name = check["check"]
    if name == "completeness":
        return f"Overall: {check['overall_score']:.1%}"
    elif name == "temporal_gaps":
        return f"{check['n_gaps']} gaps, max {check['max_gap_hours']}h"
    elif name == "freshness":
        return f"Last record {check['age_hours']}h ago"
    elif name == "outliers":
        return f"{check['total_out_of_bounds']} out-of-bounds values"
    return ""
