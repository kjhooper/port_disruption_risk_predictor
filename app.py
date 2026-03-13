"""
app.py — Harbinger Live Simulator
Animated pydeck map with ship simulation + weather heatmap + disruption risk forecast.
Run with: conda run -n personal streamlit run app.py
"""

import sys
import math
import random
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta

from fetch import fetch_openmeteo_historical, fetch_openmeteo_forecast, update_or_fetch, PORTS
from quality import run_all_checks, quality_summary_df
from features import compute_all_features
from model import (
    build_feature_matrix, build_lag_features, build_alert_features,
    add_time_features,
    NUMERIC_FORECAST_VARS, NUMERIC_HORIZONS, LAG_HOURS, WEATHER_COLORS,
)
from labels import (
    make_composite_disruption_label,
    make_disruption_window_label,
    make_weather_code_label,
)

try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except ImportError:
    PYDECK_AVAILABLE = False

# ── Page config ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Harbinger",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }

    .metric-card {
        background: #0f1923;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .risk-card {
        border-radius: 10px;
        padding: 28px 24px;
        margin-bottom: 20px;
        text-align: center;
    }
    .risk-low  { background: #0a2218; border: 1px solid #00d4aa; }
    .risk-med  { background: #2a1e00; border: 1px solid #f5a623; }
    .risk-high { background: #2a0808; border: 1px solid #e84545; }

    .status-ok   { color: #00d4aa; font-weight: 600; }
    .status-warn { color: #f5a623; font-weight: 600; }
    .status-fail { color: #e84545; font-weight: 600; }

    .stSelectbox label, .stSlider label { font-family: 'Space Mono', monospace; font-size: 0.8rem; }
    [data-testid="stSidebar"] { background: #0a1118; border-right: 1px solid #1e3a5f; }
</style>
""", unsafe_allow_html=True)

# ── Ship simulation constants ───────────────────────────────────────────────────

PORT_SIM = {
    "rotterdam": {"bbox": (3.5, 51.7, 5.0, 52.2), "n_ships": 20, "zoom": 8,  "pitch": 30},
    "houston":   {"bbox": (-96.0, 29.3, -94.5, 29.9), "n_ships": 15, "zoom": 9, "pitch": 30},
    "hong_kong": {"bbox": (113.7, 22.0, 114.6, 22.5), "n_ships": 18, "zoom": 9, "pitch": 30},
    "kaohsiung": {"bbox": (120.1, 22.3, 120.6, 22.8), "n_ships": 14, "zoom": 10, "pitch": 30},
}

VESSEL_NAMES = [
    "MSC GÜLSÜN", "EVER GIVEN", "STENA IMMACULATE", "NORDIC BOTHNIA",
    "CMA CGM MARCO POLO", "MAERSK MC-KINNEY", "COSCO SHIPPING UNIVERSE",
    "OOCL HONG KONG", "HYUNDAI PRIDE", "ATLANTIC NAVIGATOR",
    "ROTTERDAM EXPRESS", "HOUSTON STAR", "PEARL RIVER", "JADE SEA",
    "NORTH SEA GIANT", "GULF PIONEER", "PACIFIC TRADER", "EASTERN GRACE",
    "AMBER WIND", "SILVER HORIZON", "DELTA QUEEN", "ARCTIC WOLF",
]

VESSEL_TYPES = ["container", "tanker", "bulk", "tug"]
VESSEL_BASE_SPEEDS = {"container": 12, "tanker": 10, "bulk": 9, "tug": 6}

KM_PER_DEGREE_LAT = 111.0

RELEASE_BASE_URL = "https://github.com/kjhooper/port_disruption_risk_predictor/releases/download/v0.0.1"

# ── Data loading ────────────────────────────────────────────────────────────────

def _load_parquet_from_release(filename: str) -> pd.DataFrame:
    local_path = Path("data") / filename
    local_path.parent.mkdir(exist_ok=True)
    if not local_path.exists():
        url = f"{RELEASE_BASE_URL}/{filename}"
        r = requests.get(url)
        r.raise_for_status()
        local_path.write_bytes(r.content)
    return pd.read_parquet(local_path)


@st.cache_data(ttl=3600, show_spinner="Loading weather data...")
def load_data(port: str, days_back: int):
    try:
        cutoff = datetime.utcnow() - timedelta(days=days_back)
        hist = _load_parquet_from_release(f"{port}_historical_wide.parquet")
        hist = hist[hist.index >= cutoff]
        fore = _load_parquet_from_release(f"{port}_forecast.parquet")
    except Exception:
        hist = fetch_openmeteo_historical(port, days_back=days_back)
        fore = fetch_openmeteo_forecast(port, days_ahead=7)
    return hist, fore


@st.cache_data(ttl=3600, show_spinner=False)
def load_full_hist(port: str) -> pd.DataFrame:
    try:
        return _load_parquet_from_release(f"{port}_historical_wide.parquet")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_portwatch_activity(port: str) -> pd.DataFrame:
    from features import make_holiday_features, PORT_HOLIDAY_CALENDARS
    try:
        df = _load_parquet_from_release(f"{port}_portwatch_activity.parquet")
    except Exception:
        return pd.DataFrame()

    if "portcalls" not in df.columns:
        return pd.DataFrame()
    portcalls = df["portcalls"].astype(float)
    rolling_median = portcalls.rolling(window=28, min_periods=7).median()
    ratio = portcalls / rolling_median.replace(0, float("nan"))
    disrupted = ((ratio < 0.30) & (rolling_median >= 1.0)).astype("int8")
    df = df.copy()
    df["rolling_median"] = rolling_median
    df["ratio"] = ratio
    df["disrupted"] = disrupted
    if port in PORT_HOLIDAY_CALENDARS:
        dt_idx = pd.DatetimeIndex([pd.Timestamp(d) for d in df.index])
        hol = make_holiday_features(dt_idx, port)
        df["is_holiday"]   = hol["is_holiday"].values
        df["holiday_name"] = hol["holiday_name"].values
    else:
        df["is_holiday"]   = 0
        df["holiday_name"] = ""
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def compute_alert_hindcast(port: str, _alert_model_24h, days_back: int) -> pd.Series:
    try:
        df = _load_parquet_from_release(f"{port}_historical_wide.parquet")
    except Exception:
        return pd.Series(dtype=float)

    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days_back)
    df = df[df.index >= cutoff]
    try:
        feat_df = compute_all_features(df.copy(), port)
        X = build_alert_features(feat_df, port)
        for col in _alert_model_24h.feature_names_in_:
            if col not in X.columns:
                X[col] = 0.0
        X = X[list(_alert_model_24h.feature_names_in_)]
        probs = _alert_model_24h.predict_proba(X)[:, 1]
        return pd.Series(probs, index=X.index, name="alert_prob")
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def compute_alert_test_preds(port: str, _clf, window_h: int) -> pd.DataFrame:
    """
    Re-run DisruptionAlert on the held-out test split (last 365 days of training data)
    for the given window and return a DataFrame with columns y_prob and y_true.
    """
    try:
        df = _load_parquet_from_release(f"{port}_historical_wide.parquet")
    except Exception:
        return pd.DataFrame()
    try:
        feat_df = compute_all_features(df.copy(), port)
        X = build_alert_features(feat_df, port)
        # Same time split used in training
        cutoff = X.index.max() - pd.Timedelta(days=365)
        X_test = X[X.index > cutoff]
        for col in _clf.feature_names_in_:
            if col not in X_test.columns:
                X_test = X_test.copy()
                X_test[col] = 0.0
        X_test = X_test[list(_clf.feature_names_in_)]
        y_prob = _clf.predict_proba(X_test)[:, 1]
        base_label = make_composite_disruption_label(feat_df, port)
        y_true = make_disruption_window_label(base_label, window_h).reindex(X_test.index).fillna(0)
        return pd.DataFrame({"y_prob": y_prob, "y_true": y_true.values}, index=X_test.index)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def compute_numerics_test_preds(port: str, _numerics_models: dict) -> dict:
    """
    Re-run WeatherNumericsForecaster T+24h on the test split for all 6 variables.
    Returns dict {var: DataFrame(y_true, y_pred, index=timestamps)}.
    """
    try:
        df = _load_parquet_from_release(f"{port}_historical_wide.parquet")
    except Exception:
        return {}
    try:
        feat_df = compute_all_features(df.copy(), port)
        df_lag = feat_df.copy()
        for var in NUMERIC_FORECAST_VARS:
            if var in df_lag.columns:
                df_lag = build_lag_features(df_lag, var, LAG_HOURS)
        cutoff = df_lag.index.max() - pd.Timedelta(days=365)
        result = {}
        for var in NUMERIC_FORECAST_VARS:
            if (var, 24) not in _numerics_models or var not in df_lag.columns:
                continue
            reg, _ = _numerics_models[(var, 24)]
            y_all = df_lag[var].shift(-24)
            valid = y_all.notna()
            X_all = df_lag[valid].copy()
            y_all = y_all[valid]
            X_test = X_all[X_all.index > cutoff]
            y_test = y_all[y_all.index > cutoff]
            if X_test.empty:
                continue
            X_aligned = X_test.copy()
            for col in reg.feature_names_in_:
                if col not in X_aligned.columns:
                    X_aligned[col] = 0.0
            X_aligned = X_aligned[list(reg.feature_names_in_)]
            y_pred = reg.predict(X_aligned)
            result[var] = pd.DataFrame(
                {"y_true": y_test.values, "y_pred": y_pred},
                index=X_test.index,
            )
        return result
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def compute_wcode_test_preds(port: str, _clf_24h) -> pd.DataFrame:
    """
    Re-run WeatherCodePredictor T+24h on the test split.
    Returns DataFrame with columns y_true (WMO group string) and y_pred.
    """
    try:
        df = _load_parquet_from_release(f"{port}_historical_wide.parquet")
    except Exception:
        return pd.DataFrame()
    try:
        feat_df = compute_all_features(df.copy(), port)
        df_lag = feat_df.copy()
        for var in NUMERIC_FORECAST_VARS:
            if var in df_lag.columns:
                df_lag = build_lag_features(df_lag, var, LAG_HOURS)
        if "weather_code" in df_lag.columns:
            df_lag = build_lag_features(df_lag, "weather_code", LAG_HOURS)
        cutoff = df_lag.index.max() - pd.Timedelta(days=365)
        y_all = make_weather_code_label(df_lag).shift(-24)
        valid = y_all.notna()
        X_all = df_lag[valid].copy()
        y_all = y_all[valid]
        X_test = X_all[X_all.index > cutoff]
        y_test = y_all[y_all.index > cutoff]
        if X_test.empty:
            return pd.DataFrame()
        X_aligned = X_test.copy()
        for col in _clf_24h.feature_names_in_:
            if col not in X_aligned.columns:
                X_aligned[col] = 0.0
        X_aligned = X_aligned[list(_clf_24h.feature_names_in_)]
        int_to_label = getattr(_clf_24h, "_int_to_label", {})
        y_prob = _clf_24h.predict_proba(X_aligned)
        y_pred_int = y_prob.argmax(axis=1)
        y_pred = pd.Series(y_pred_int).map(int_to_label).values
        return pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred}, index=X_test.index)
    except Exception:
        return pd.DataFrame()


# Known model names — must match what save_models() writes in model.py
_MODEL_NAMES = ["wcode_predictors", "weather_numerics", "disruption_alerts"]


def _load_model_from_release(port: str, name: str):
    """
    Download a single model file from GitHub Releases if not already cached locally.

    Both local files and release assets use the same name: {port}_{name}.joblib
    in the flat  models/  directory.
    """
    import joblib
    local_path = Path("models") / f"{port}_{name}.joblib"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not local_path.exists():
        url = f"{RELEASE_BASE_URL}/{port}_{name}.joblib"
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        local_path.write_bytes(r.content)
    return joblib.load(local_path)


@st.cache_resource
def load_port_models(port: str) -> dict:
    """
    Load all three models for a port.

    Resolution order:
      1. Local  models/{port}_{name}.joblib  (fast path — after training or prior download)
      2. GitHub Release asset  {port}_{name}.joblib  (downloaded once, then cached locally)

    Returns a dict keyed by model name (e.g. "wcode_predictors").
    Missing models are silently skipped so the app degrades gracefully.
    """
    import joblib
    result = {}
    for name in _MODEL_NAMES:
        local_path = Path("models") / f"{port}_{name}.joblib"
        try:
            if local_path.exists():
                result[name] = joblib.load(local_path)
            else:
                result[name] = _load_model_from_release(port, name)
        except Exception as e:
            st.warning(f"Model `{name}` not available for {port}: {e}")
    return result


# ── Ship simulation helpers ──────────────────────────────────────────────────────

def _init_ships(port: str) -> pd.DataFrame:
    cfg = PORT_SIM.get(port, PORT_SIM["rotterdam"])
    lon_min, lat_min, lon_max, lat_max = cfg["bbox"]
    n = cfg["n_ships"]
    port_lat = PORTS[port]["lat"]
    port_lon = PORTS[port]["lon"]

    rng = random.Random(hash(port))
    ships = []
    names_pool = VESSEL_NAMES[:]
    rng.shuffle(names_pool)

    for i in range(n):
        vtype = rng.choice(VESSEL_TYPES)
        # Assign role
        role_r = rng.random()
        if role_r < 0.40:
            role = "inbound"
        elif role_r < 0.80:
            role = "outbound"
        else:
            role = "anchored"

        lat = rng.uniform(lat_min, lat_max)
        lon = rng.uniform(lon_min, lon_max)

        # Heading toward/away from port centre
        dlat = port_lat - lat
        dlon = port_lon - lon
        bearing_to_port = math.degrees(math.atan2(dlon, dlat)) % 360
        if role == "inbound":
            heading = bearing_to_port + rng.uniform(-20, 20)
        elif role == "outbound":
            heading = (bearing_to_port + 180 + rng.uniform(-20, 20)) % 360
        else:
            heading = rng.uniform(0, 360)

        ships.append({
            "id":          i,
            "name":        names_pool[i % len(names_pool)],
            "type":        vtype,
            "lat":         lat,
            "lon":         lon,
            "heading_deg": heading % 360,
            "speed_knots": VESSEL_BASE_SPEEDS[vtype],
            "status":      role,
            "color":       [0, 212, 170],
        })

    return pd.DataFrame(ships)


def _update_ships(ships_df: pd.DataFrame, risk: float, port: str, dt_hours: float = 0.1) -> pd.DataFrame:
    cfg = PORT_SIM.get(port, PORT_SIM["rotterdam"])
    lon_min, lat_min, lon_max, lat_max = cfg["bbox"]
    speed_factor = 1.0 - min(risk * 0.9, 0.85)

    rows = ships_df.to_dict("records")
    for s in rows:
        if s["status"] == "anchored":
            s["color"] = [150, 150, 150]
            continue
        # Chance to anchor when risk is high
        if risk > 0.40 and random.random() < 0.02:
            s["status"] = "anchored"
            s["color"] = [150, 150, 150]
            continue
        # If anchored ships should resume (low risk)
        if risk < 0.15 and s["status"] == "anchored" and random.random() < 0.01:
            s["status"] = "inbound"

        # Colour by risk
        if risk >= 0.40:
            s["color"] = [232, 69, 69]
        elif risk >= 0.15:
            s["color"] = [245, 166, 35]
        else:
            s["color"] = [0, 212, 170]

        # Move
        heading_rad = math.radians(s["heading_deg"])
        km_per_hour = s["speed_knots"] * 1.852
        dist_km = km_per_hour * speed_factor * dt_hours

        km_per_degree_lon = KM_PER_DEGREE_LAT * math.cos(math.radians(s["lat"]))
        delta_lat = (math.cos(heading_rad) * dist_km) / KM_PER_DEGREE_LAT
        delta_lon = (math.sin(heading_rad) * dist_km) / km_per_degree_lon

        new_lat = s["lat"] + delta_lat
        new_lon = s["lon"] + delta_lon

        # Wrap at bbox edges
        if new_lat < lat_min:
            new_lat = lat_max - (lat_min - new_lat)
        elif new_lat > lat_max:
            new_lat = lat_min + (new_lat - lat_max)
        if new_lon < lon_min:
            new_lon = lon_max - (lon_min - new_lon)
        elif new_lon > lon_max:
            new_lon = lon_min + (new_lon - lon_max)

        s["lat"] = new_lat
        s["lon"] = new_lon

    return pd.DataFrame(rows)


def _build_heatmap_grid(port: str, fore_df: pd.DataFrame, hour_idx: int = 0) -> pd.DataFrame:
    """Build a 7x7 IDW-interpolated intensity grid around the port centre."""
    pcfg = PORTS[port]
    center_lat, center_lon = pcfg["lat"], pcfg["lon"]
    span = 0.6

    lats = np.linspace(center_lat - span, center_lat + span, 7)
    lons = np.linspace(center_lon - span, center_lon + span, 7)

    # Get port centre values from forecast
    if not fore_df.empty and hour_idx < len(fore_df):
        row = fore_df.iloc[hour_idx]
        _wind_raw = row.get("wind_speed_10m")
        wind = float(_wind_raw) if pd.notna(_wind_raw) else 5.0
        _wave_raw = row.get("wave_height") if "wave_height" in fore_df.columns else None
        wave = float(_wave_raw) if pd.notna(_wave_raw) else 0.5
    else:
        wind = 5.0
        wave = 0.5

    centre_intensity = wind + wave * 2.0

    points = []
    for la in lats:
        for lo in lons:
            # IDW from port centre (simple approximation with distance falloff)
            d = math.sqrt((la - center_lat) ** 2 + (lo - center_lon) ** 2) + 1e-6
            weight = 1.0 / (d ** 1.5)
            noise = random.uniform(0.85, 1.15)
            intensity = centre_intensity * weight * noise
            points.append({"lat": la, "lon": lo, "weight": float(intensity)})

    df = pd.DataFrame(points)
    if df["weight"].max() > 0:
        df["weight"] = df["weight"] / df["weight"].max()
    return df


def _port_polygon(port: str, risk: float) -> list:
    """Return a hexagonal polygon ~15km radius around the port centre for the risk overlay."""
    pcfg = PORTS[port]
    clat, clon = pcfg["lat"], pcfg["lon"]
    radius_deg_lat = 0.14
    radius_deg_lon = radius_deg_lat / math.cos(math.radians(clat))
    sides = 6
    coords = []
    for i in range(sides + 1):
        angle = math.radians(i * 360 / sides)
        coords.append([
            clon + radius_deg_lon * math.sin(angle),
            clat + radius_deg_lat * math.cos(angle),
        ])
    # Colour: interpolate green→amber→red
    if risk >= 0.40:
        r, g, b, a = 232, 69, 69, 80
    elif risk >= 0.15:
        r, g, b, a = 245, 166, 35, 60
    else:
        r, g, b, a = 0, 212, 170, 40
    return [{"polygon": coords, "r": r, "g": g, "b": b, "a": a}]




# ── Inference helpers ───────────────────────────────────────────────────────────

def _disruption_alert_probs(full_hist: pd.DataFrame, port: str, models: dict) -> dict:
    alert_models = models.get("disruption_alerts")
    if not alert_models or full_hist.empty:
        return {}
    # 100 rows: covers the max 72h lag + buffer for clean feature computation
    recent = full_hist.tail(100).copy()
    try:
        feat_df = compute_all_features(recent.copy(), port)
        X = build_alert_features(feat_df, port)
    except Exception:
        return {}
    if X.empty:
        return {}
    current_idx = recent.index[-1]
    if current_idx not in X.index:
        return {}
    X_now = X.loc[[current_idx]]
    results = {}
    for window_h, (clf, _) in sorted(alert_models.items()):
        try:
            X_aligned = X_now.copy()
            for col in clf.feature_names_in_:
                if col not in X_aligned.columns:
                    X_aligned[col] = 0.0
            X_aligned = X_aligned[list(clf.feature_names_in_)]
            results[window_h] = float(clf.predict_proba(X_aligned)[0, 1])
        except Exception:
            pass
    return results


def _wcode_forecast_timeline(full_hist: pd.DataFrame, port: str, models: dict) -> pd.Series:
    wcode_preds = models.get("wcode_predictors")
    if not wcode_preds or full_hist.empty:
        return pd.Series(dtype=str)
    # 200 rows: covers 72h lags + enough history for clean feature computation
    recent = full_hist.tail(200).copy()
    df_lag = recent.copy()
    for var in NUMERIC_FORECAST_VARS:
        if var in df_lag.columns:
            df_lag = build_lag_features(df_lag, var, LAG_HOURS)
    if "weather_code" in df_lag.columns:
        df_lag = build_lag_features(df_lag, "weather_code", LAG_HOURS)
    df_lag = add_time_features(df_lag)
    extra_features = [
        c for c in ["wind_direction_10m", "relative_humidity_2m", "cape",
                    "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
        if c in df_lag.columns
    ]
    lag_cols = [f"{var}_lag_{h}h" for var in NUMERIC_FORECAST_VARS for h in LAG_HOURS
                if f"{var}_lag_{h}h" in df_lag.columns]
    wcode_lag_cols = [f"weather_code_lag_{h}h" for h in LAG_HOURS
                      if f"weather_code_lag_{h}h" in df_lag.columns]
    feature_cols = lag_cols + wcode_lag_cols + extra_features
    X_all = df_lag[feature_cols].ffill().fillna(0)
    current_idx = recent.index[-1]
    if current_idx not in X_all.index:
        return pd.Series(dtype=str)
    X_now = X_all.loc[[current_idx]]
    results = {}
    for h, (clf, _) in sorted(wcode_preds.items()):
        try:
            X_aligned = X_now.copy()
            for col in clf.feature_names_in_:
                if col not in X_aligned.columns:
                    X_aligned[col] = 0.0
            X_aligned = X_aligned[list(clf.feature_names_in_)]
            int_to_label = getattr(clf, "_int_to_label", {})
            y_prob = clf.predict_proba(X_aligned)
            y_pred_int = int(y_prob.argmax(axis=1)[0])
            results[current_idx + pd.Timedelta(hours=h)] = int_to_label.get(y_pred_int, "clear")
        except Exception:
            pass
    if not results:
        return pd.Series(dtype=str)
    return pd.Series(results, name="weather_group")


def _numerics_predictions(full_hist: pd.DataFrame, port: str, models: dict) -> dict:
    numerics = models.get("weather_numerics")
    if not numerics or full_hist.empty:
        return {}
    # 200 rows: 72h lags (72 rows) + residual lookback (72+ rows) + buffer
    recent = full_hist.tail(200).copy()
    df_lag = recent.copy()
    for var in NUMERIC_FORECAST_VARS:
        if var in df_lag.columns:
            df_lag = build_lag_features(df_lag, var, LAG_HOURS)
    df_lag = add_time_features(df_lag)
    extra_cols = [
        c for c in ["wind_direction_10m", "relative_humidity_2m", "cape",
                    "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
        if c in df_lag.columns
    ]
    lag_cols = [f"{var}_lag_{h}h" for var in NUMERIC_FORECAST_VARS for h in LAG_HOURS
                if f"{var}_lag_{h}h" in df_lag.columns]
    feature_cols = lag_cols + extra_cols
    X_all = df_lag[feature_cols].ffill().fillna(0)
    current_idx = recent.index[-1]
    if current_idx not in X_all.index:
        return {}
    X_now = X_all.loc[[current_idx]]
    results = {}
    for (var, h), (reg, _) in numerics.items():
        try:
            X_aligned = X_now.copy()
            for col in reg.feature_names_in_:
                if col not in X_aligned.columns:
                    X_aligned[col] = 0.0
            X_aligned = X_aligned[list(reg.feature_names_in_)]

            # Compute recent residuals for ARMA correction.
            # For horizon h, actual y(t+h) is knowable for rows where t+h <= now,
            # i.e. the rows at least h hours before the end of the window.
            recent_residuals = None
            if hasattr(reg, "predict_base") and var in df_lag.columns:
                y_actual = df_lag[var].shift(-h)
                valid_mask = y_actual.notna()
                X_valid = X_all[valid_mask]
                y_valid = y_actual[valid_mask]
                if len(X_valid) >= 5:
                    X_hist = X_valid.tail(5)
                    y_hist = y_valid.tail(5)
                    X_hist_a = X_hist.copy()
                    for col in reg.feature_names_in_:
                        if col not in X_hist_a.columns:
                            X_hist_a[col] = 0.0
                    X_hist_a = X_hist_a[list(reg.feature_names_in_)]
                    y_pred_hist = reg.predict_base(X_hist_a)
                    recent_residuals = y_hist.values - y_pred_hist

            results[(var, h)] = float(reg.predict(X_aligned, recent_residuals=recent_residuals)[0])
        except Exception:
            pass
    return results


def _risk_color(p: float) -> str:
    if p >= 0.40:
        return "#e84545"
    if p >= 0.15:
        return "#f5a623"
    return "#00d4aa"


def _risk_label(p: float) -> str:
    if p >= 0.40:
        return "HIGH"
    if p >= 0.15:
        return "ELEVATED"
    return "LOW"


def _risk_card_class(p: float) -> str:
    if p >= 0.40:
        return "risk-high"
    if p >= 0.15:
        return "risk-med"
    return "risk-low"


# ── Sidebar ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🚢 Harbinger")
    st.divider()

    port = st.selectbox(
        "SELECT PORT",
        options=list(PORTS.keys()),
        format_func=lambda k: PORTS[k]["label"],
    )

    st.divider()
    st.markdown("**SIMULATION MODE**")
    sim_mode = st.radio(
        "Mode",
        options=["Live", "Playback"],
        horizontal=True,
        label_visibility="collapsed",
    )

    playback_h = 0
    if sim_mode == "Playback":
        playback_h = st.slider("Forecast hour", 0, 71, 0, key="playback_h")

    st.divider()
    st.markdown("**ANIMATION**")
    anim_speed = st.select_slider(
        "Speed",
        options=[1, 2, 4],
        value=2,
        format_func=lambda x: f"{x}s tick",
        label_visibility="collapsed",
    )

    st.divider()
    days_back = st.slider("HISTORY (days)", min_value=7, max_value=365, value=30, step=7)
    fetch_btn = st.button("↻ Fetch / Refresh Data", use_container_width=True, type="primary")

    st.divider()
    st.markdown(f"**Port:** {PORTS[port]['label']}")
    st.markdown(f"**Lat/Lon:** {PORTS[port]['lat']}°, {PORTS[port]['lon']}°")
    st.markdown(f"**Last run:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")

# ── Data loading ─────────────────────────────────────────────────────────────────

if fetch_btn:
    with st.spinner("Updating data..."):
        update_or_fetch(port, save_dir="data")
    st.cache_data.clear()

hist_df, fore_df = load_data(port, days_back)
full_hist_df = load_full_hist(port)
models = load_port_models(port)

# ── Header ───────────────────────────────────────────────────────────────────────

st.markdown(f"# {PORTS[port]['label']}")
st.markdown("Real-time port simulator · weather risk · vessel traffic")
st.divider()

# ── Compute risk for current state ───────────────────────────────────────────────

alert_entry = models.get("disruption_alerts")
wcode_entry = models.get("wcode_predictors")

alert_probs: dict = {}
if alert_entry is not None and not full_hist_df.empty:
    with st.spinner("Running DisruptionAlert inference..."):
        alert_probs = _disruption_alert_probs(full_hist_df, port, models)

# In playback mode, approximate risk from wind forecast
if sim_mode == "Playback" and not fore_df.empty:
    h = min(playback_h, len(fore_df) - 1)
    wind_val = float(fore_df["wind_speed_10m"].iloc[h]) if "wind_speed_10m" in fore_df.columns else 5.0
    # Simple proxy: scale wind to 0-1 (30 m/s = max)
    playback_risk = min(wind_val / 30.0, 1.0)
    current_risk = playback_risk
else:
    # Use the best validated window for the map risk; fall back to wind proxy if all gates fail
    _raw_p24 = alert_probs.get(24)
    _raw_p48 = alert_probs.get(48)
    if _raw_p24 is not None:
        current_risk = _raw_p24
    elif _raw_p48 is not None:
        current_risk = _raw_p48
    elif not full_hist_df.empty and "wind_speed_10m" in full_hist_df.columns:
        current_risk = min(float(full_hist_df["wind_speed_10m"].iloc[-1]) / 30.0, 1.0)
    else:
        current_risk = 0.0

# ── Session state: ships ─────────────────────────────────────────────────────────

if "ships_port" not in st.session_state or st.session_state["ships_port"] != port:
    st.session_state["ships"] = _init_ships(port)
    st.session_state["ships_port"] = port

# ── Map fragment (animated) ──────────────────────────────────────────────────────

@st.fragment(run_every=anim_speed)
def _map_fragment():
    if not PYDECK_AVAILABLE:
        st.warning(
            "pydeck is not installed. Run `pip install pydeck>=0.8` to enable the map. "
            "Showing risk cards only."
        )
        return

    # Update ship positions
    st.session_state["ships"] = _update_ships(
        st.session_state["ships"],
        risk=current_risk,
        port=port,
        dt_hours=anim_speed / 3600.0 * 5,  # scale so ships move visibly at each tick
    )
    ships = st.session_state["ships"]

    pcfg = PORT_SIM.get(port, PORT_SIM["rotterdam"])
    center_lat = PORTS[port]["lat"]
    center_lon = PORTS[port]["lon"]

    # Heatmap data
    heatmap_df = _build_heatmap_grid(port, fore_df, hour_idx=playback_h if sim_mode == "Playback" else 0)

    # Port risk polygon
    polygon_data = _port_polygon(port, current_risk)

    # Layers
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=heatmap_df,
        get_position=["lon", "lat"],
        get_weight="weight",
        radiusPixels=80,
        intensity=1.5,
        threshold=0.05,
        opacity=0.5,
        color_range=[
            [0, 128, 60, 100],
            [120, 220, 0, 140],
            [255, 200, 0, 160],
            [255, 100, 0, 180],
            [232, 69, 69, 200],
        ],
    )

    polygon_layer = pdk.Layer(
        "PolygonLayer",
        data=polygon_data,
        get_polygon="polygon",
        get_fill_color=["r", "g", "b", "a"],
        get_line_color=[255, 255, 255, 60],
        line_width_min_pixels=1,
        filled=True,
        stroked=True,
        pickable=False,
    )

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=ships,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius=600,
        radius_min_pixels=4,
        radius_max_pixels=12,
        pickable=True,
    )

    text_layer = pdk.Layer(
        "TextLayer",
        data=ships[ships["status"] != "anchored"].head(10),  # show labels for moving ships
        get_position=["lon", "lat"],
        get_text="name",
        get_size=11,
        get_color=[220, 220, 220, 200],
        get_pixel_offset=[0, -18],
        font_family="monospace",
    )

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=pcfg["zoom"],
        pitch=pcfg["pitch"],
        bearing=0,
    )

    deck = pdk.Deck(
        layers=[heatmap_layer, polygon_layer, scatter_layer, text_layer],
        initial_view_state=view_state,
        map_style="dark",
        tooltip={"text": "{name}\nType: {type}\nStatus: {status}"},
    )

    st.pydeck_chart(deck, use_container_width=True, height=520)

    # Legend
    status_color = _risk_color(current_risk)
    mode_label = f"Playback · hour T+{playback_h}" if sim_mode == "Playback" else "Live"
    st.caption(
        f"**{mode_label}** · Risk: "
        f"<span style='color:{status_color};font-weight:700'>{_risk_label(current_risk)}</span> "
        f"({current_risk:.0%}) · "
        f"{(ships['status'] != 'anchored').sum()} moving · "
        f"{(ships['status'] == 'anchored').sum()} anchored",
        unsafe_allow_html=True,
    )


_map_fragment()

st.divider()

# ── 72h WeatherCodePredictor colour bar ──────────────────────────────────────────

if wcode_entry is not None and not full_hist_df.empty:
    with st.spinner("Running WeatherCodePredictor timeline..."):
        weather_72h = _wcode_forecast_timeline(full_hist_df, port, models)

    if not weather_72h.empty:
        w72 = weather_72h.iloc[:72]
        colors_72 = [WEATHER_COLORS.get(w, "#888888") for w in w72]

        fig_wt = go.Figure()
        # Highlight selected hour in playback mode
        highlight_colors = list(colors_72)
        if sim_mode == "Playback" and playback_h < len(highlight_colors):
            # Desaturate others slightly, keep selected bright
            highlight_colors = [
                c if i == playback_h else c + "99"
                for i, c in enumerate(colors_72)
            ]

        fig_wt.add_trace(go.Bar(
            x=w72.index,
            y=[1] * len(w72),
            marker_color=colors_72,
            hovertemplate="%{x|%a %d %b %H:%M}<br>%{customdata}<extra></extra>",
            customdata=w72.values,
            showlegend=False,
        ))
        if sim_mode == "Playback" and playback_h < len(w72):
            fig_wt.add_vline(
                x=w72.index[playback_h],
                line_color="white",
                line_width=2,
                opacity=0.9,
                annotation_text=f"T+{playback_h}h",
                annotation_position="top",
                annotation_font_color="white",
            )
        for wtype, wcolor in WEATHER_COLORS.items():
            if wtype in w72.values:
                fig_wt.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode="markers",
                    marker=dict(color=wcolor, size=10, symbol="square"),
                    name=wtype,
                ))
        fig_wt.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a1118",
            plot_bgcolor="#0f1923",
            height=175,
            margin=dict(l=40, r=20, t=12, b=72),
            xaxis=dict(
                showticklabels=True,
                tickformat="%d %b\n%H:%M",
                dtick=6 * 3600000,  # tick every 6 hours (ms)
                tickangle=0,
                tickfont=dict(size=10, color="#888"),
                showgrid=False,
            ),
            yaxis=dict(showticklabels=False, showgrid=False, range=[0, 1.2]),
            legend=dict(orientation="h", y=-0.6, x=0, font_size=12),
        )
        st.markdown(
            "**72h Weather Type Forecast** "
            "<span style='color:#888; font-size:0.8rem; font-family:monospace;'>"
            "WeatherCodePredictor</span>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_wt, use_container_width=True)
        st.divider()

# ── Disruption Risk cards ─────────────────────────────────────────────────────────

st.markdown("### Disruption Risk")

if alert_entry is None:
    st.info(
        f"No trained models found for **{PORTS[port]['label']}**. "
        f"Run `conda run -n personal python train.py --port {port}` to train."
    )
else:
    # Houston uses a relaxed ROC threshold (0.60) because:
    #   1. ERA5 reanalysis smooths over convective weather — Gulf Coast thunderstorms
    #      appear as WMO 0 (clear sky) in the training features, capping discriminative skill.
    #   2. The NCEI+physics label is very sparse (~1.6% positive rate) so 48h/72h ROC
    #      naturally sits in the 0.62–0.72 range even for a well-calibrated model.
    # All three windows show with a LOW CONFIDENCE badge so users understand the limitation.
    _roc_threshold = 0.60 if port == "houston" else 0.75

    def _gate(window_h: int) -> bool:
        """Return True if this window's model passes the port-specific ROC-AUC threshold
        AND ECE < 0.15. ECE threshold is 0.15 (not 0.05) because sparse disruption labels
        (< 2% base rate) make sub-5% calibration error unreachable without a large
        calibration dataset. The ROC-AUC gate remains the primary quality check."""
        _, met = alert_entry.get(window_h, (None, {}))
        if not isinstance(met, dict):
            return False
        roc = met.get("roc_auc")
        ece = met.get("ece")
        return bool(roc and roc > _roc_threshold and ece is not None and ece < 0.15)

    def _is_low_confidence(window_h: int) -> bool:
        """True when Houston passes the relaxed gate but would fail the strict 0.75
        gate used for all other ports. Cards in this state show the probability with a
        LOW CONFIDENCE badge rather than being fully suppressed."""
        if port != "houston":
            return False
        _, met = alert_entry.get(window_h, (None, {}))
        if not isinstance(met, dict):
            return False
        roc = met.get("roc_auc")
        return bool(roc and roc <= 0.75)

    gate24, gate48, gate72 = _gate(24), _gate(48), _gate(72)
    any_pass = gate24 or gate48 or gate72

    p24 = alert_probs.get(24, 0.0) if gate24 else None
    p48 = alert_probs.get(48, 0.0) if gate48 else None
    p72 = alert_probs.get(72, 0.0) if gate72 else None

    def _alert_card(label: str, window_h: int, prob, gate_ok: bool) -> str:
        if gate_ok and prob is not None:
            low_conf = _is_low_confidence(window_h)
            badge = (
                '<div style="font-size:0.7rem; color:#f5a623; margin-top:2px;">'
                '⚠ LOW CONFIDENCE</div>'
                '<div style="font-size:0.65rem; color:#666; margin-top:1px;">'
                'ERA5 misses convective events</div>'
            ) if low_conf else ""
            return f"""
            <div class="metric-card risk-card {_risk_card_class(prob)}">
              <div style="font-size:0.75rem; color:#888; font-family:'Space Mono',monospace;">{label}</div>
              <div style="font-size:2.2rem; font-weight:700; color:{_risk_color(prob)}; font-family:'Space Mono',monospace;">
                {prob:.0%}
              </div>
              <div style="font-size:0.8rem; color:{_risk_color(prob)};">{_risk_label(prob)}</div>
              {badge}
            </div>"""
        _, met = alert_entry.get(window_h, (None, {}))
        roc = met.get("roc_auc", 0) if isinstance(met, dict) else 0
        return f"""
            <div class="metric-card risk-card" style="background:#111820; border:1px solid #2a3a4a; opacity:0.6;">
              <div style="font-size:0.75rem; color:#888; font-family:'Space Mono',monospace;">{label}</div>
              <div style="font-size:1.4rem; font-weight:700; color:#4a6070; font-family:'Space Mono',monospace; margin-top:4px;">
                —
              </div>
              <div style="font-size:0.75rem; color:#f5a623; margin-top:4px;">⚠ UNVALIDATED</div>
              <div style="font-size:0.7rem; color:#666; margin-top:2px;">ROC-AUC {roc:.3f} &lt; {_roc_threshold:.2f}</div>
            </div>"""

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(_alert_card("P(DISRUPTION IN 24H)", 24, p24, gate24), unsafe_allow_html=True)
    with c2:
        st.markdown(_alert_card("P(DISRUPTION IN 48H)", 48, p48, gate48), unsafe_allow_html=True)
    with c3:
        st.markdown(_alert_card("P(DISRUPTION IN 72H)", 72, p72, gate72), unsafe_allow_html=True)

    # Aggregate gate summary in the 4th card
    _, met72 = alert_entry.get(72, (None, {}))
    roc_72 = met72.get("roc_auc") if isinstance(met72, dict) else None
    any_low_conf = any(_is_low_confidence(h) for h in [24, 48, 72])
    if any_pass and any_low_conf and port == "houston":
        summary_color = "#f5a623"
        summary_label = "⚠️ LOW CONFIDENCE"
        status_note = '<div style="font-size:0.65rem; color:#666; margin-top:2px;">ERA5 misses convective events</div>'
    elif any_pass:
        summary_color = "#00d4aa"
        summary_label = "✅ PASS"
        status_note = ""
    else:
        summary_color = "#f5a623"
        summary_label = "⚠️ UNVALIDATED"
        status_note = ""
    with c4:
        st.markdown(f"""
        <div class="metric-card risk-card risk-low">
          <div style="font-size:0.75rem; color:#888; font-family:'Space Mono',monospace;">MODEL STATUS</div>
          <div style="font-size:1.1rem; font-weight:700; color:{summary_color}; font-family:'Space Mono',monospace; margin-top:8px;">
            {summary_label}
          </div>
          <div style="font-size:0.7rem; color:#888; margin-top:6px; line-height:1.6;">
            24h {'✅' if gate24 else '✗'} &nbsp; 48h {'✅' if gate48 else '✗'} &nbsp; 72h {'✅' if gate72 else '✗'}
          </div>
          <div style="font-size:0.7rem; color:#666; margin-top:4px;">
            72h ROC: {f"{roc_72:.3f}" if roc_72 else "N/A"} (thr {_roc_threshold:.2f})
          </div>
          {status_note}
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "DisruptionAlert answers: 'Will a disruptive event occur SOMETIME in the next N hours?' — "
        "more operationally useful than a point-in-time detector."
    )

    # Rotterdam physics-label disclosure
    if port == "rotterdam":
        st.info(
            "**Rotterdam uses a physics-based disruption label** — PortWatch vessel traffic is too "
            "stable to detect disruptions via traffic drops. Disruption is flagged when: "
            "wind > 15 m/s (Beaufort 7 / PIANC threshold) · "
            "gusts > 22 m/s (STS crane suspension band) · "
            "wave height > 2.5 m (Europoort/Maasvlakte limit) · "
            "td_spread < 2°C (WMO fog-risk) · "
            "severe WMO codes (heavy rain, snow, thunderstorm)."
        )

st.divider()

# ── Weather tabs ────────────────────────────────────────────────────────────────

st.markdown("### Weather Overview")
_tab_labels = ["💨 Wind", "🌧️ Precipitation & Pressure", "🔮 Numerics Forecast", "📊 Raw Data"]
tab1, tab2, tab3, tab4 = st.tabs(_tab_labels)

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df.index, y=hist_df["wind_speed_10m"],
        name="Wind Speed (m/s)", line=dict(color="#00d4aa", width=1.5),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.08)"
    ))
    if "wind_gusts_10m" in hist_df.columns:
        fig.add_trace(go.Scatter(
            x=hist_df.index, y=hist_df["wind_gusts_10m"],
            name="Gusts (m/s)", line=dict(color="#f5a623", width=1, dash="dot")
        ))
    fig.add_trace(go.Scatter(
        x=fore_df.index, y=fore_df["wind_speed_10m"],
        name="Forecast Wind", line=dict(color="#4a9eff", width=1.5, dash="dash")
    ))
    fig.add_vline(x=datetime.utcnow(), line_dash="dash", line_color="white", opacity=0.3)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a1118", plot_bgcolor="#0f1923",
        height=350, margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", y=-0.15),
        xaxis_title="Date (UTC)", yaxis_title="m/s"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=hist_df.index, y=hist_df["precipitation"],
        name="Precipitation (mm)", marker_color="#4a9eff", opacity=0.7, yaxis="y1"
    ))
    if "pressure_msl" in hist_df.columns:
        fig2.add_trace(go.Scatter(
            x=hist_df.index, y=hist_df["pressure_msl"],
            name="Pressure (hPa)", line=dict(color="#f5a623", width=1.5), yaxis="y2"
        ))
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="#0a1118", plot_bgcolor="#0f1923",
        height=350, margin=dict(l=40, r=60, t=20, b=40),
        yaxis=dict(title="mm/hr"),
        yaxis2=dict(title="hPa", overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    numerics_entry = models.get("weather_numerics")
    if not numerics_entry:
        st.info(f"Run `python train.py --port {port}` to enable numerics forecast.")
    else:
        with st.spinner("Running WeatherNumericsForecaster..."):
            num_preds = _numerics_predictions(full_hist_df, port, models)

        if not num_preds:
            st.warning("Numerics predictions could not be computed — check that historical data is loaded.")
        else:
            now_utc = datetime.utcnow()
            var_labels = {
                "wind_speed_10m": "Wind Speed (m/s)",
                "wind_gusts_10m": "Wind Gusts (m/s)",
                "temperature_2m": "Temperature (°C)",
                "precipitation":  "Precipitation (mm/h)",
                "pressure_msl":   "Pressure (hPa)",
                "wave_height":    "Wave Height (m)",
            }
            var_colors = {
                "wind_speed_10m": "#00d4aa",
                "wind_gusts_10m": "#f5a623",
                "temperature_2m": "#ff8c69",
                "precipitation":  "#4a9eff",
                "pressure_msl":   "#c77dff",
                "wave_height":    "#7b5eff",
            }
            by_var: dict = {}
            for (var, h), val in num_preds.items():
                by_var.setdefault(var, {})[h] = val

            st.markdown("**WeatherNumericsForecaster — 11-horizon trajectory from current state**")
            st.caption(
                "Each curve shows predicted values at T+1h through T+72h (11 horizons). "
                "Trained on lag patterns; interpret alongside Open-Meteo NWP for best judgment."
            )
            var_list = [v for v in NUMERIC_FORECAST_VARS if v in by_var]
            for i in range(0, len(var_list), 2):
                cols = st.columns(2)
                for j, var in enumerate(var_list[i:i+2]):
                    with cols[j]:
                        horizons_sorted = sorted(by_var[var].keys())
                        vals = [by_var[var][h] for h in horizons_sorted]
                        times = [now_utc + timedelta(hours=h) for h in horizons_sorted]
                        fig_num = go.Figure()
                        fig_num.add_trace(go.Scatter(
                            x=times, y=vals,
                            mode="lines+markers",
                            line=dict(color=var_colors.get(var, "#00d4aa"), width=2),
                            marker=dict(size=5),
                            name=var_labels.get(var, var),
                            hovertemplate="%{x|%a %d %b %H:%M}<br>%{y:.2f}<extra></extra>",
                        ))
                        if not fore_df.empty and var in fore_df.columns:
                            fig_num.add_trace(go.Scatter(
                                x=fore_df.index[:72],
                                y=fore_df[var].iloc[:72],
                                mode="lines",
                                line=dict(color="#888888", width=1, dash="dot"),
                                name="Open-Meteo",
                                hovertemplate="%{x|%a %d %b %H:%M}<br>OME: %{y:.2f}<extra></extra>",
                            ))
                        fig_num.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#0a1118",
                            plot_bgcolor="#0f1923",
                            height=220,
                            margin=dict(l=40, r=10, t=30, b=30),
                            showlegend=True,
                            legend=dict(orientation="h", y=-0.25, font_size=10),
                            title=dict(text=var_labels.get(var, var),
                                       font_color="#aaa", font_size=12),
                        )
                        st.plotly_chart(fig_num, use_container_width=True)

with tab4:
    st.markdown("**Historical data (last 50 rows)**")
    st.dataframe(
        hist_df.tail(50).reset_index().rename(columns={"time": "datetime (UTC)"}),
        use_container_width=True, height=300,
    )
    st.markdown("**Forecast data (next 7 days)**")
    st.dataframe(
        fore_df.reset_index().rename(columns={"time": "datetime (UTC)"}),
        use_container_width=True, height=300,
    )

st.divider()

# ── Traffic vs Predictions ──────────────────────────────────────────────────────

st.markdown("### 📡 Traffic vs Predictions")
pw_df = load_portwatch_activity(port)

if pw_df.empty:
    st.info(
        "No PortWatch data available for this port. "
        "Run `conda run -n personal python fetch_portwatch.py` to download traffic data."
    )
else:
    pw_dt = pw_df.copy()
    pw_dt.index = pd.to_datetime(pw_df.index)
    # Show the full PortWatch history (2019–present) regardless of the weather
    # days_back slider — the two datasets have very different ranges and clipping
    # PortWatch to 365 days hides almost all disruption events.
    has_alert = alert_entry is not None

    if has_alert:
        alert_clf_24, _ = alert_entry.get(24, (None, None))
        if alert_clf_24 is not None:
            with st.spinner("Computing DisruptionAlert hindcast..."):
                # Use full weather history (1095 days) so the hindcast covers
                # as much of the PortWatch range as weather data allows.
                hindcast = compute_alert_hindcast(port, alert_clf_24, 1095)
            if not hindcast.empty:
                daily_prob = hindcast.resample("D").mean().rename("alert_prob")
                combined = pw_dt[["ratio", "disrupted"]].join(daily_prob, how="left")
            else:
                combined = pw_dt
        else:
            combined = pw_dt
    else:
        combined = pw_dt

    if combined.empty or "ratio" not in combined.columns:
        st.info("Insufficient data to display chart.")
    else:
        fig_tv = go.Figure()
        fig_tv.add_trace(go.Bar(
            x=combined.index,
            y=combined["ratio"].clip(0, 2),
            name="Portcall ratio",
            marker_color="#4a9eff",
            opacity=0.7,
            yaxis="y1",
            hovertemplate="%{x|%Y-%m-%d}<br>Ratio: %{y:.2f}<extra></extra>",
        ))
        if has_alert and "alert_prob" in combined.columns:
            fig_tv.add_trace(go.Scatter(
                x=combined.index,
                y=combined["alert_prob"],
                name="Alert P(24h)",
                line=dict(color="#e84545", width=2),
                yaxis="y2",
                hovertemplate="%{x|%Y-%m-%d}<br>P(disruption in 24h): %{y:.1%}<extra></extra>",
            ))
        _has_disruption_bands = False
        if "disrupted" in combined.columns:
            dis = combined[combined["disrupted"] == 1]
            if not dis.empty:
                _has_disruption_bands = True
                blocks = (dis.index.to_series().diff() > pd.Timedelta(days=2)).cumsum()
                for _, grp in dis.groupby(blocks):
                    fig_tv.add_vrect(
                        x0=grp.index[0] - pd.Timedelta(hours=12),
                        x1=grp.index[-1] + pd.Timedelta(hours=12),
                        fillcolor="rgba(232,69,69,0.15)",
                        layer="below", line_width=0,
                    )
        _has_holiday_bands = False
        if "is_holiday" in combined.columns:
            hols_only = combined[combined["is_holiday"] == 1]
            if not hols_only.empty:
                _has_holiday_bands = True
                hol_blocks = (hols_only.index.to_series().diff() > pd.Timedelta(days=1)).cumsum()
                for _, grp in hols_only.groupby(hol_blocks):
                    name = grp["holiday_name"].iloc[0] if "holiday_name" in grp.columns else "Holiday"
                    fig_tv.add_vrect(
                        x0=grp.index[0] - pd.Timedelta(hours=12),
                        x1=grp.index[-1] + pd.Timedelta(hours=12),
                        fillcolor="rgba(245,166,35,0.18)",
                        layer="below", line_width=0,
                        annotation_text=name[:22],
                        annotation_position="top left",
                        annotation_font_size=9,
                        annotation_font_color="#f5a623",
                    )

        # Dummy traces for legend entries (vrect/hrect don't appear in legend)
        if _has_disruption_bands:
            fig_tv.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(color="rgba(232,69,69,0.4)", size=12, symbol="square"),
                name="PortWatch disruption",
                showlegend=True,
            ))
        if _has_holiday_bands:
            fig_tv.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(color="rgba(245,166,35,0.5)", size=12, symbol="square"),
                name="Public holiday",
                showlegend=True,
            ))

        fig_tv.add_hline(
            y=0.30, line_dash="dot", line_color="#e84545",
            annotation_text="Disruption threshold (30%)", annotation_position="top right",
            annotation_font_color="#e84545",
        )
        ratio_max = float(combined["ratio"].max()) if not combined["ratio"].isna().all() else 2.0
        y2_config = (
            dict(title="P(disruption in 24h)", overlaying="y", side="right",
                 range=[0, 1], tickformat=".0%", showgrid=False)
            if has_alert and "alert_prob" in combined.columns
            else dict(overlaying="y", side="right", showgrid=False, visible=False)
        )
        fig_tv.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a1118",
            plot_bgcolor="#0f1923",
            height=320,
            margin=dict(l=40, r=60, t=30, b=40),
            xaxis_title="Date",
            yaxis=dict(title="Portcall ratio", range=[0, max(2.0, ratio_max * 1.1)]),
            yaxis2=y2_config,
            legend=dict(orientation="h", y=-0.2),
            title=dict(
                text=(
                    "PortWatch Traffic vs DisruptionAlert (24h window)"
                    if has_alert and "alert_prob" in combined.columns
                    else "PortWatch Daily Portcall Ratio"
                ),
                font_color="#aaa", font_size=13,
            ),
        )
        st.plotly_chart(fig_tv, use_container_width=True)

        # Date range caption — explicit about the two data sources
        pw_start = pw_dt.index.min()
        pw_end   = pw_dt.index.max()
        pw_range = (
            f"PortWatch: {pw_start.strftime('%Y-%m-%d')} → {pw_end.strftime('%Y-%m-%d')} (daily)"
        )
        if has_alert and "alert_prob" in combined.columns:
            alert_start = daily_prob.first_valid_index()
            alert_end   = daily_prob.last_valid_index()
            hindcast_range = (
                f"  ·  DisruptionAlert hindcast: {alert_start.strftime('%Y-%m-%d')} → {alert_end.strftime('%Y-%m-%d')}"
                if alert_start is not None else ""
            )
        else:
            hindcast_range = ""

        if port == "rotterdam":
            st.caption(
                f"{pw_range}{hindcast_range}  ·  "
                "Rotterdam is a stable port — DisruptionAlert trained on composite weather physics "
                "(PortWatch traffic shows no significant disruptions)."
            )
        elif has_alert and "alert_prob" in combined.columns:
            st.caption(
                f"{pw_range}{hindcast_range}  ·  "
                "Gold bands = public holidays · Pink bands = PortWatch-confirmed disruption days."
            )
        else:
            st.caption(f"{pw_range}  ·  Drops below 30% of 28-day median indicate traffic disruptions.")

        # ── Vessel type breakdown ──────────────────────────────────────────────
        vessel_cols = {
            "portcalls_container":    "Container",
            "portcalls_tanker":       "Tanker",
            "portcalls_dry_bulk":     "Dry Bulk",
            "portcalls_general_cargo":"General Cargo",
            "portcalls_roro":         "RoRo",
        }
        avail_vcols = {col: label for col, label in vessel_cols.items() if col in pw_df.columns}
        if avail_vcols:
            st.markdown("**Vessel type breakdown (daily portcalls)**")
            vtype_colors = {
                "Container":    "#4a9eff",
                "Tanker":       "#e84545",
                "Dry Bulk":     "#f5a623",
                "General Cargo":"#00d4aa",
                "RoRo":         "#a78bfa",
            }
            fig_vt = go.Figure()
            for col, label in avail_vcols.items():
                fig_vt.add_trace(go.Bar(
                    x=pw_dt.index,
                    y=pw_dt[col].clip(lower=0),
                    name=label,
                    marker_color=vtype_colors.get(label, "#888"),
                    hovertemplate=f"%{{x|%Y-%m-%d}}<br>{label}: %{{y:.0f}}<extra></extra>",
                ))
            fig_vt.update_layout(
                barmode="stack",
                template="plotly_dark",
                paper_bgcolor="#0a1118",
                plot_bgcolor="#0f1923",
                height=220,
                margin=dict(l=40, r=20, t=30, b=40),
                xaxis_title="Date",
                yaxis_title="Daily portcalls",
                legend=dict(orientation="h", y=-0.25, font_size=10),
                title=dict(text="Vessel Type Composition", font_color="#aaa", font_size=13),
            )
            st.plotly_chart(fig_vt, use_container_width=True)
            st.caption(
                f"{pw_range}  ·  "
                "Stacked daily portcalls by vessel type — useful for identifying which trade segment "
                "is driving disruptions (e.g. container vs tanker vs bulk)."
            )

st.divider()

# ── Data quality ────────────────────────────────────────────────────────────────

with st.expander("📋 Data Quality", expanded=False):
    quality_report = run_all_checks(hist_df)
    summary_df     = quality_summary_df(quality_report)

    q_cols = st.columns(4)
    STATUS_EMOJI = {"ok": "✅", "warn": "⚠️", "fail": "❌", "skip": "—"}
    for i, row in summary_df.iterrows():
        with q_cols[i]:
            emoji = STATUS_EMOJI.get(row["Status"].lower(), "—")
            st.metric(
                label=row["Check"].replace("_", " ").title(),
                value=f"{emoji} {row['Status']}",
                delta=row["Detail"],
            )

    overall_score = quality_report["overall_score"]
    score_color   = "#00d4aa" if overall_score >= 0.9 else ("#f5a623" if overall_score >= 0.5 else "#e84545")
    st.markdown(f"""
    <div class="metric-card">
      <b>Overall Quality Score:</b>
      <span style="font-size:1.4rem; color:{score_color}; font-family:'Space Mono',monospace;">
        {overall_score:.0%}
      </span>
      &nbsp;·&nbsp; {quality_report['n_rows']} rows &nbsp;·&nbsp; {quality_report['n_cols']} columns
    </div>
    """, unsafe_allow_html=True)

    comp    = quality_report["checks"]["completeness"]["per_column"]
    comp_df = pd.DataFrame({"Column": list(comp.keys()), "Completeness": list(comp.values())})
    comp_df = comp_df.sort_values("Completeness")
    fig3 = px.bar(
        comp_df, x="Completeness", y="Column", orientation="h",
        color="Completeness", color_continuous_scale=["#e84545", "#f5a623", "#00d4aa"],
        range_color=[0, 1], template="plotly_dark",
    )
    fig3.update_layout(
        paper_bgcolor="#0a1118", plot_bgcolor="#0f1923",
        height=250, margin=dict(l=20, r=20, t=10, b=30),
        coloraxis_showscale=False,
    )
    fig3.update_traces(marker_line_width=0)
    st.plotly_chart(fig3, use_container_width=True)

# ── Model Performance ────────────────────────────────────────────────────────────

with st.expander("📋 Model Performance", expanded=False):
    alert_entry_perf = models.get("disruption_alerts")
    st.markdown("**DisruptionAlert — Window Disruption Classifiers (24h / 48h / 72h)**")
    if alert_entry_perf is None:
        st.info(f"No DisruptionAlert models trained for **{PORTS[port]['label']}**. Run `python train.py --port {port}`.")
    else:
        alert_rows = []
        for window_h in [24, 48, 72]:
            if window_h not in alert_entry_perf:
                continue
            _, met = alert_entry_perf[window_h]
            roc  = met.get("roc_auc")
            pr   = met.get("pr_auc")
            f1   = met.get("f1")
            ece  = met.get("ece")
            # ECE < 0.15 is the operational gate (0.05 is unachievable at < 2% base rate)
            gate = "✅ PASS" if (roc and roc > 0.75 and ece is not None and ece < 0.15) else "⚠️ FAIL"
            prec = met.get("precision")
            rec  = met.get("recall")
            spec = met.get("specificity")
            mcc  = met.get("mcc")
            brier = met.get("brier_score")
            tp   = met.get("tp")
            fp   = met.get("fp")
            tn   = met.get("tn")
            fn   = met.get("fn")
            alert_rows.append({
                "Window":      f"T+{window_h}h",
                "ROC-AUC":     f"{roc:.3f}"   if roc   is not None else "N/A",
                "PR-AUC":      f"{pr:.3f}"    if pr    is not None else "N/A",
                "F1":          f"{f1:.3f}"    if f1    is not None else "N/A",
                "Precision":   f"{prec:.3f}"  if prec  is not None else "N/A",
                "Recall":      f"{rec:.3f}"   if rec   is not None else "N/A",
                "Specificity": f"{spec:.3f}"  if spec  is not None else "N/A",
                "MCC":         f"{mcc:.3f}"   if mcc   is not None else "N/A",
                "Brier":       f"{brier:.4f}" if brier is not None else "N/A",
                "ECE":         f"{ece:.4f}"   if ece   is not None else "N/A",
                "TP/FP/TN/FN": f"{tp}/{fp}/{tn}/{fn}" if tp is not None else "N/A",
                "Gate":        gate,
            })
        if alert_rows:
            st.dataframe(pd.DataFrame(alert_rows), hide_index=True, use_container_width=True)
            st.caption(
                "Gate: ROC-AUC > 0.75 AND ECE < 0.15  ·  "
                "MCC = Matthews Correlation Coefficient (gold standard for imbalanced labels)  ·  "
                "Brier score = probability MSE (lower is better)  ·  "
                "Specificity = TNR (fraction of calm days correctly called calm)"
            )

        # ── Test-set charts: one per window (24h / 48h / 72h) ──────────────────
        st.markdown("**Test-set predictions vs actuals — DisruptionAlert (last 365 days of training data)**")
        _alert_cols = st.columns(3)
        for _j, _wh in enumerate([24, 48, 72]):
            _clf_w, _ = alert_entry_perf.get(_wh, (None, None))
            if _clf_w is None:
                continue
            with st.spinner(f"Computing T+{_wh}h test predictions..."):
                _adf = compute_alert_test_preds(port, _clf_w, _wh)
            with _alert_cols[_j]:
                if _adf.empty:
                    st.caption(f"T+{_wh}h — no data")
                    continue
                _fig_a = go.Figure()
                # Actual disruption bands
                _dm = _adf["y_true"] == 1
                _db = (_dm.astype(int).diff().fillna(0) != 0).cumsum()[_dm]
                for _, _gi in _adf[_dm].groupby(_db):
                    _fig_a.add_vrect(
                        x0=_gi.index[0], x1=_gi.index[-1],
                        fillcolor="rgba(232,69,69,0.22)", layer="below", line_width=0,
                    )
                _fig_a.add_trace(go.Scatter(
                    x=_adf.index, y=_adf["y_prob"],
                    fill="tozeroy", fillcolor="rgba(232,69,69,0.10)",
                    line=dict(color="#e84545", width=1.2),
                    name=f"P(T+{_wh}h)",
                    hovertemplate="%{x|%d %b %H:%M}<br>P: %{y:.1%}<extra></extra>",
                    showlegend=False,
                ))
                _fig_a.add_hline(y=0.5, line_dash="dot", line_color="#f5a623", opacity=0.7)
                # Dummy legend
                _fig_a.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(color="rgba(232,69,69,0.4)", size=10, symbol="square"),
                    name="Actual", showlegend=True,
                ))
                _fig_a.update_layout(
                    template="plotly_dark", paper_bgcolor="#0a1118", plot_bgcolor="#0f1923",
                    height=220, margin=dict(l=44, r=8, t=28, b=36),
                    title=dict(text=f"T+{_wh}h", font_color="#aaa", font_size=12),
                    xaxis=dict(tickformat="%b %y", tickfont_size=9),
                    yaxis=dict(tickformat=".0%", range=[0, 1], tickfont_size=9),
                    legend=dict(orientation="h", y=-0.28, font_size=9),
                )
                st.plotly_chart(_fig_a, use_container_width=True)
        st.caption(
            "Red shading = actual disruption hours (composite physics label for Rotterdam; "
            "PortWatch / NCEI label for other ports).  "
            "Red line = model predicted probability.  0.5 dotted line = classification threshold."
        )

    st.divider()
    wcode_entry_perf = models.get("wcode_predictors")
    st.markdown("**WeatherCodePredictor — WMO Group Classifiers (T+1h … T+72h)**")
    if wcode_entry_perf is None:
        st.info(f"No WeatherCodePredictor trained for **{PORTS[port]['label']}**. Run `python train.py --port {port}`.")
    else:
        wcode_rows = []
        for h in [1, 6, 12, 24, 48, 72]:
            if h not in wcode_entry_perf:
                continue
            _, met = wcode_entry_perf[h]
            f1     = met.get("macro_f1")
            roc    = met.get("roc_auc")
            prec_w = met.get("macro_precision")
            rec_w  = met.get("macro_recall")
            wcode_rows.append({
                "Horizon":    f"T+{h}h",
                "Macro F1":   f"{f1:.3f}"    if f1    is not None else "N/A",
                "Precision":  f"{prec_w:.3f}" if prec_w is not None else "N/A",
                "Recall":     f"{rec_w:.3f}"  if rec_w  is not None else "N/A",
                "ROC-AUC (OvR)": f"{roc:.3f}" if roc is not None else "N/A",
            })
        if wcode_rows:
            st.dataframe(pd.DataFrame(wcode_rows), hide_index=True, use_container_width=True)
            st.caption(
                f"Showing 6 representative horizons of {len(wcode_entry_perf)} total classifiers.  ·  "
                "Macro-averaged across all 5 WMO groups (clear / fog / rain_snow / showers / thunderstorm)."
            )

        # ── Test-set chart: confusion matrix for T+24h ────────────────────────
        _wcode_clf_test, _ = wcode_entry_perf.get(24, (None, None)) if wcode_entry_perf else (None, None)
        if _wcode_clf_test is not None:
            st.markdown("**Test-set predictions vs actuals — WeatherCodePredictor T+24h**")
            with st.spinner("Computing test-set predictions..."):
                wcode_test_df = compute_wcode_test_preds(port, _wcode_clf_test)
            if not wcode_test_df.empty:
                from sklearn.metrics import confusion_matrix as sk_cm
                _groups = ["clear", "fog", "rain_snow", "showers", "thunderstorm"]
                _present = sorted(wcode_test_df["y_true"].dropna().unique())
                _labels = [g for g in _groups if g in _present]
                cm_arr = sk_cm(
                    wcode_test_df["y_true"].dropna(),
                    wcode_test_df["y_pred"].reindex(wcode_test_df["y_true"].dropna().index),
                    labels=_labels,
                )
                # Normalise by row (actual) so colour = recall per class
                cm_norm = cm_arr.astype(float) / np.maximum(cm_arr.sum(axis=1, keepdims=True), 1)
                fig_cm = go.Figure(go.Heatmap(
                    z=cm_norm,
                    x=[f"pred: {l}" for l in _labels],
                    y=[f"actual: {l}" for l in _labels],
                    colorscale=[[0, "#0f1923"], [1, "#4a9eff"]],
                    zmin=0, zmax=1,
                    text=[[f"{cm_arr[r][c]}<br>({cm_norm[r][c]:.0%})" for c in range(len(_labels))]
                          for r in range(len(_labels))],
                    texttemplate="%{text}",
                    textfont=dict(size=11),
                    hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
                ))
                fig_cm.update_layout(
                    template="plotly_dark", paper_bgcolor="#0a1118", plot_bgcolor="#0f1923",
                    height=320, margin=dict(l=20, r=20, t=20, b=60),
                    xaxis=dict(side="bottom"),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                st.caption(
                    "Colour intensity = recall (row fraction correctly predicted).  "
                    "Numbers show raw count and row-normalised fraction.  "
                    "Diagonal = correct predictions; off-diagonal = misclassifications."
                )

                # ── Dual colour-bar timeline: actual vs predicted ────────────
                st.markdown("**Actual vs predicted weather class — T+24h test set (daily mode)**")
                # Resample to daily mode so colour bars are readable (365 bars vs 8760)
                def _resample_mode(s):
                    return s.resample("D").agg(
                        lambda x: x.mode().iloc[0] if not x.empty and not x.mode().empty else None
                    )
                _daily_true = _resample_mode(wcode_test_df["y_true"].dropna())
                _daily_pred = _resample_mode(
                    wcode_test_df["y_pred"].reindex(wcode_test_df["y_true"].dropna().index)
                )
                _shared_idx = _daily_true.index.intersection(_daily_pred.index)
                _daily_true = _daily_true.loc[_shared_idx]
                _daily_pred = _daily_pred.loc[_shared_idx]

                from plotly.subplots import make_subplots
                _fig_cb = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.5, 0.5],
                    vertical_spacing=0.06,
                    subplot_titles=["Actual", "Predicted"],
                )
                for _row, _series, _label in [
                    (1, _daily_true, "Actual"),
                    (2, _daily_pred, "Predicted"),
                ]:
                    _fig_cb.add_trace(go.Bar(
                        x=_series.index,
                        y=[1] * len(_series),
                        marker_color=[WEATHER_COLORS.get(v, "#555") for v in _series],
                        hovertemplate="%{x|%d %b %Y}<br>" + _label + ": %{customdata}<extra></extra>",
                        customdata=_series.values,
                        showlegend=False,
                    ), row=_row, col=1)
                # Legend traces (one per present group)
                for _wt, _wc in WEATHER_COLORS.items():
                    if _wt in _daily_true.values or _wt in _daily_pred.values:
                        _fig_cb.add_trace(go.Scatter(
                            x=[None], y=[None], mode="markers",
                            marker=dict(color=_wc, size=10, symbol="square"),
                            name=_wt,
                        ), row=1, col=1)
                _fig_cb.update_layout(
                    template="plotly_dark", paper_bgcolor="#0a1118", plot_bgcolor="#0f1923",
                    height=220, margin=dict(l=50, r=20, t=30, b=10),
                    legend=dict(orientation="h", y=-0.08, x=0, font_size=11),
                    bargap=0,
                )
                for _r in [1, 2]:
                    _fig_cb.update_yaxes(showticklabels=False, showgrid=False, row=_r, col=1)
                st.plotly_chart(_fig_cb, use_container_width=True)
                st.caption(
                    "Each column = one day (daily mode of hourly predictions).  "
                    "Rows that share the same colour = correct day; mismatched colour = wrong class.  "
                    "Misclassifications are easiest to spot at class boundaries (e.g. clear → showers)."
                )

    st.divider()
    numerics_entry_perf = models.get("weather_numerics")
    st.markdown("**WeatherNumericsForecaster — Variable Forecasters (6 vars × 11 horizons)**")
    if not numerics_entry_perf:
        st.info(f"No WeatherNumericsForecaster trained for **{PORTS[port]['label']}**. Run `python train.py --port {port}`.")
    else:
        num_var_labels = {
            "wind_speed_10m": "Wind Speed",
            "wind_gusts_10m": "Wind Gusts",
            "temperature_2m": "Temperature",
            "precipitation":  "Precipitation",
            "pressure_msl":   "Pressure",
            "wave_height":    "Wave Height",
        }
        num_rows: dict[str, dict[str, str]] = {}
        for (var, h), (_, met) in numerics_entry_perf.items():
            if h not in [24, 48, 72]:
                continue
            var_name = num_var_labels.get(var, var)
            rmse  = met.get("rmse")
            r2    = met.get("r2")
            nrmse = met.get("nrmse")
            p90   = met.get("p90_abs_err")
            def _fmt(v, spec):
                return format(v, spec) if v is not None else "N/A"
            cell  = (
                f"{_fmt(rmse,'0.3f')} / R²={_fmt(r2,'0.3f')} / NRMSE={_fmt(nrmse,'0.3f')} / P90={_fmt(p90,'0.3f')}"
                if rmse is not None else "N/A"
            )
            if var_name not in num_rows:
                num_rows[var_name] = {}
            num_rows[var_name][f"T+{h}h"] = cell
        if num_rows:
            num_table = pd.DataFrame(num_rows).T
            num_table.index.name = "Variable"
            col_order = [c for c in ["T+24h", "T+48h", "T+72h"] if c in num_table.columns]
            st.dataframe(
                num_table[col_order] if col_order else num_table,
                use_container_width=True,
            )
            st.caption(
                f"Cells: RMSE / R² / NRMSE / P90 abs error at T+24/48/72h  ·  "
                f"{len(numerics_entry_perf)} total models (6 vars × 11 horizons)  ·  "
                "NRMSE < 1 means model beats predicting climatological mean  ·  "
                "P90 = 90th-percentile absolute error (worst-case tail risk)"
            )

        # ── Test-set charts: predicted vs actual scatter per variable ─────────
        st.markdown("**Test-set predictions vs actuals — WeatherNumericsForecaster T+24h**")
        with st.spinner("Computing test-set predictions..."):
            num_test = compute_numerics_test_preds(port, numerics_entry_perf)
        if num_test:
            _var_labels = {
                "wind_speed_10m": "Wind Speed (m/s)",
                "wind_gusts_10m": "Wind Gusts (m/s)",
                "temperature_2m": "Temperature (°C)",
                "precipitation":  "Precipitation (mm/h)",
                "pressure_msl":   "Pressure (hPa)",
                "wave_height":    "Wave Height (m)",
            }
            _var_colors = {
                "wind_speed_10m": "#4a9eff",
                "wind_gusts_10m": "#7b5eff",
                "temperature_2m": "#f5a623",
                "precipitation":  "#00d4aa",
                "pressure_msl":   "#e84545",
                "wave_height":    "#a78bfa",
            }
            vars_present = [v for v in NUMERIC_FORECAST_VARS if v in num_test]
            ncols = 3
            cols_num = st.columns(ncols)
            for i, var in enumerate(vars_present):
                vdf = num_test[var].dropna()
                if vdf.empty:
                    continue
                r2_val = float(np.corrcoef(vdf["y_true"], vdf["y_pred"])[0, 1] ** 2)
                vmin = float(min(vdf["y_true"].min(), vdf["y_pred"].min()))
                vmax = float(max(vdf["y_true"].max(), vdf["y_pred"].max()))
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=vdf["y_true"], y=vdf["y_pred"],
                    mode="markers",
                    marker=dict(color=_var_colors.get(var, "#4a9eff"), size=3, opacity=0.4),
                    hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
                ))
                # Perfect prediction line
                fig_sc.add_trace(go.Scatter(
                    x=[vmin, vmax], y=[vmin, vmax],
                    mode="lines",
                    line=dict(color="#888", width=1, dash="dot"),
                    name="Perfect",
                    showlegend=False,
                ))
                fig_sc.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0a1118",
                    plot_bgcolor="#0f1923",
                    height=220,
                    margin=dict(l=40, r=10, t=40, b=40),
                    title=dict(
                        text=f"{_var_labels.get(var, var)}  R²={r2_val:.3f}",
                        font_color="#aaa", font_size=11,
                    ),
                    xaxis_title="Actual",
                    yaxis_title="Predicted",
                    showlegend=False,
                )
                with cols_num[i % ncols]:
                    st.plotly_chart(fig_sc, use_container_width=True)
            st.caption(
                "Each point = one test-set hour.  Dotted diagonal = perfect prediction.  "
                "Tight clustering along the diagonal = good forecast skill.  "
                "Test split: last 365 days of available historical data."
            )

# ── Footer ───────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Data: Open-Meteo (archive + forecast) · "
    "Models: DisruptionAlert (24/48/72h windows) · WeatherCodePredictor (72h WMO) · WeatherNumericsForecaster (6 vars × 11 horizons) · "
    "Ground truth: NCEI Storm Events (Houston) · PortWatch portcalls (Houston, Hong Kong, Kaohsiung) · Composite physics label (Rotterdam)"
)
