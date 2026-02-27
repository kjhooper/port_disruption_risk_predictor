"""
labels.py — WMO weather code group mapping and disruption labels.

Open-Meteo uses a sparse subset of WMO codes; ranges are used for compactness.
Valid codes: 0-3, 45, 48, 51-57, 61-67, 71-77, 80-82, 85-86, 95, 96, 99.
Unrecognised codes return NaN from make_weather_code_label.
"""

import pandas as pd

# Ranges for readability; not all integers in each range are valid WMO codes —
# make_weather_code_label handles unknown codes gracefully (returns NaN)
WMO_GROUPS = {
    "clear":        range(0, 4),     # 0,1,2,3
    "fog":          range(45, 49),   # 45,48
    "rain_snow":    range(51, 78),   # 51-57 drizzle, 61-67 rain, 71-77 snow
    "showers":      range(80, 87),   # 80-82 rain showers, 85-86 snow showers
    "thunderstorm": range(95, 100),  # 95, 96, 99
}

_CODE_TO_GROUP: dict[int, str] = {
    code: grp for grp, rng in WMO_GROUPS.items() for code in rng
}

# Display order for charts
GROUP_ORDER = ["clear", "fog", "rain_snow", "showers", "thunderstorm"]

# Human-readable descriptions for UI
GROUP_DESCRIPTIONS = {
    "clear":        "0–3 · Clear sky through overcast — no precipitation",
    "fog":          "45–48 · Fog or depositing rime fog — visibility hazard",
    "rain_snow":    "51–77 · Drizzle, rain, freezing rain, snow, snow grains",
    "showers":      "80–86 · Rain or snow showers — convective bursts",
    "thunderstorm": "95–99 · Thunderstorm with or without hail",
}


def make_weather_code_label(df: pd.DataFrame) -> pd.Series:
    """Map weather_code column to group string. NaN for unrecognised/missing codes."""
    if "weather_code" not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="object")
    return df["weather_code"].map(_CODE_TO_GROUP)


def make_binary_label(df: pd.DataFrame) -> pd.Series:
    """1 if any non-clear event (weather_code > 3), else 0. 0 when column is absent."""
    if "weather_code" not in df.columns:
        return pd.Series(0, index=df.index, dtype="int8")
    return (df["weather_code"] > 3).astype("int8")


# ── Composite physics-based disruption label ──────────────────────────────────

# Operational thresholds by port (tunable).
# Wind and waves are the primary drivers; fog secondary; severe WMO tertiary.
DISRUPTION_THRESHOLDS: dict[str, dict] = {
    "rotterdam": dict(wind_speed=15.0, wind_gusts=22.0, wave_height=2.5, td_spread=2.0),
    "houston":   dict(wind_speed=15.0, wind_gusts=22.0, wave_height=2.5, td_spread=2.0),
    # Typhoon-belt ports: lower wave threshold (South China Sea / Philippine Sea swell);
    # higher td_spread (tropical humidity means fog forms at smaller T/Td gaps)
    "hong_kong": dict(wind_speed=15.0, wind_gusts=22.0, wave_height=2.0, td_spread=3.0),
    "kaohsiung": dict(wind_speed=15.0, wind_gusts=22.0, wave_height=2.0, td_spread=3.0),
}
_DEFAULT_THRESHOLDS = dict(wind_speed=15.0, wind_gusts=22.0, wave_height=2.5, td_spread=2.0)

# WMO codes severe enough to disrupt port operations regardless of wind speed
# (heavy precipitation / thunderstorm — excludes light drizzle 51-55)
_severe: set = set()
_severe |= set(range(65, 68))   # heavy rain
_severe |= set(range(75, 78))   # heavy snow
_severe |= set(range(80, 83))   # heavy rain showers
_severe |= {85, 86}             # heavy snow showers
_severe |= set(range(95, 100))  # thunderstorm
SEVERE_WMO_CODES: frozenset = frozenset(_severe)


def make_composite_disruption_label(
    df: pd.DataFrame, port: str = "default"
) -> pd.Series:
    """
    Physics-based binary disruption label for any port.

    More operationally valid than WMO classification alone because:
    - Wind (the primary port disruption driver) is orthogonal to WMO codes —
      a gale under clear sky is WMO 0 but fully disruptive.
    - Fog is detected via td_spread < threshold, which is more sensitive
      than waiting for WMO 45/48 to appear.

    Use this for Rotterdam where PortWatch traffic is too stable for traffic-drop labels.
    For Houston, Hong Kong, and Kaohsiung, prefer make_portwatch_disruption_label().
    """
    t = DISRUPTION_THRESHOLDS.get(port, _DEFAULT_THRESHOLDS)
    label = pd.Series(False, index=df.index)

    if "wind_speed_10m" in df.columns:
        label |= df["wind_speed_10m"] > t["wind_speed"]
    if "wind_gusts_10m" in df.columns:
        label |= df["wind_gusts_10m"] > t["wind_gusts"]
    if "wave_height" in df.columns:
        label |= df["wave_height"] > t["wave_height"]
    if "td_spread" in df.columns:
        label |= df["td_spread"] < t["td_spread"]
    if "weather_code" in df.columns:
        label |= df["weather_code"].isin(SEVERE_WMO_CODES)

    return label.astype("int8")


def label_stats(df: pd.DataFrame) -> dict:
    """Return group counts, event rate, and class weights dict."""
    groups = make_weather_code_label(df)
    binary = make_binary_label(df)

    group_counts = groups.value_counts().to_dict()
    n_total = len(df)
    event_rate = float(binary.mean()) if n_total > 0 else 0.0

    # Class weights: inverse frequency, normalised
    n_classes = max(len(group_counts), 1)
    weights = {
        grp: n_total / (n_classes * count)
        for grp, count in group_counts.items()
        if count > 0
    }

    return {
        "group_counts": group_counts,
        "event_rate":   event_rate,
        "n_total":      n_total,
        "class_weights": weights,
    }
