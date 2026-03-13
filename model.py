"""
model.py — Three forward-looking weather models.

WeatherCodePredictor    — 72 XGBClassifiers → WMO group at each of T+1..T+72h
WeatherNumericsForecaster — 66 XGBRegressors → continuous vars at T+1..T+72h (11 horizons)
DisruptionAlert         — 3 XGBClassifiers → P(any disruption in next 24/48/72h)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from fetch import PORTS, zone_points
from labels import (
    make_weather_code_label,
    make_composite_disruption_label,
    make_disruption_window_label,
)
from metrics import eval_classifier, eval_binary, eval_forecaster


# ── Constants ────────────────────────────────────────────────────────────────

# Instantaneous features only — no rolling stats to avoid look-ahead at inference
CLASSIFIER_FEATURES = [
    "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m",
    "precipitation", "pressure_msl", "temperature_2m", "dew_point_2m",
    "relative_humidity_2m", "cloud_cover", "visibility", "cape", "lifted_index",
    "wave_height", "wave_period",
    "onshore_wind", "cross_wind", "fog_risk_score", "td_spread",
    "storm_approach_index",
    "is_holiday",
    "days_to_holiday",
    # Cyclical time encodings — capture diurnal and seasonal patterns
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]

WEATHER_COLORS = {
    "clear":        "#00d4aa",
    "fog":          "#8899aa",
    "rain_snow":    "#4a9eff",
    "showers":      "#7b5eff",
    "thunderstorm": "#e84545",
}

NUMERIC_FORECAST_VARS = [
    "wind_speed_10m", "wind_gusts_10m", "temperature_2m",
    "precipitation", "pressure_msl", "wave_height",
]
NUMERIC_HORIZONS        = [1, 2, 3, 6, 12, 18, 24, 36, 48, 60, 72]
CODE_PREDICTOR_HORIZONS = list(range(1, 73))   # 1..72
ALERT_WINDOWS           = [24, 48, 72]
LAG_VARS_ALERT          = ["wind_speed_10m", "pressure_msl", "wave_height"]
LAG_HOURS               = [1, 3, 6, 12, 24, 48, 72]

MODEL_DIR = Path("models")


# ── Time features ────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical hour-of-day and day-of-year encodings.
    Returns a copy — safe to call on slices.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    df = df.copy()
    hours = df.index.hour + df.index.minute / 60.0
    doy   = df.index.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    df["doy_sin"]  = np.sin(2 * np.pi * doy  / 365.25)
    df["doy_cos"]  = np.cos(2 * np.pi * doy  / 365.25)
    return df


# ── ARMA residual correction ─────────────────────────────────────────────────

class ARMAResidualCorrector:
    """
    AR(2) model fitted on XGBoost training residuals to reduce autocorrelation.

    At inference, call predict_correction(recent_residuals) where recent_residuals
    is a small array of (actual - predicted) values from the last few observations.
    """

    def __init__(self, order: int = 2):
        self.order = order
        self.ar_params: np.ndarray = np.zeros(order)
        self.mean_residual: float = 0.0

    def fit(self, residuals: np.ndarray) -> "ARMAResidualCorrector":
        r = np.asarray(residuals, dtype=float)
        self.mean_residual = float(r.mean())
        r = r - self.mean_residual
        n = len(r)
        if n < self.order + 20:
            return self
        Y = r[self.order:]
        X = np.column_stack(
            [r[self.order - i - 1 : n - i - 1] for i in range(self.order)]
        )
        try:
            params, *_ = np.linalg.lstsq(X, Y, rcond=None)
            # Clip to ensure stationarity (|φ| < 0.95)
            self.ar_params = np.clip(params, -0.95, 0.95)
        except Exception:
            pass
        return self

    def predict_correction(self, recent_residuals: np.ndarray) -> float:
        if len(recent_residuals) < self.order:
            return 0.0
        tail = (np.asarray(recent_residuals, dtype=float) - self.mean_residual)
        tail = tail[-self.order:][::-1]  # most-recent first
        return float(np.dot(self.ar_params, tail))


class ARMACorrectedForecaster:
    """
    Wraps an XGBRegressor with an AR(2) residual correction layer.

    predict(X)                              — plain XGBoost (no correction)
    predict(X, recent_residuals=arr)        — XGBoost + AR correction
    predict_base(X)                         — always plain XGBoost (for residual recomputation)
    """

    def __init__(self, reg: XGBRegressor, corrector: ARMAResidualCorrector):
        self.reg = reg
        self.corrector = corrector
        self.feature_names_in_ = getattr(reg, "feature_names_in_", None)

    def predict_base(self, X) -> np.ndarray:
        """XGBoost prediction without AR correction."""
        return self.reg.predict(X)

    def predict(self, X, recent_residuals: np.ndarray | None = None) -> np.ndarray:
        base = self.reg.predict(X)
        if recent_residuals is not None and len(recent_residuals) >= self.corrector.order:
            correction = self.corrector.predict_correction(recent_residuals)
            base = base + correction
        return base


# ── Feature matrix ───────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame, port: str, include_zones: bool = True
) -> pd.DataFrame:
    """
    Select and align classifier feature columns, filling NaN with column median.
    Dynamically appends zone gradient columns when include_zones=True.
    Cyclical time features are computed here if not already present.
    """
    df = add_time_features(df)
    cols = [c for c in CLASSIFIER_FEATURES if c in df.columns]

    if include_zones:
        zone_suffixes = ("_pressure_gradient", "_cape_excess", "_wind_delta", "_onshore_wind")
        for z in zone_points(port):
            pfx = z["prefix"]
            for suf in zone_suffixes:
                col = f"{pfx}{suf}"
                if col in df.columns:
                    cols.append(col)

    X = df[cols].copy()
    for col in X.columns:
        med = X[col].median()
        X[col] = X[col].fillna(med if not np.isnan(med) else 0.0)
    return X


def build_alert_features(df: pd.DataFrame, port: str) -> pd.DataFrame:
    """
    Feature matrix for DisruptionAlert: CLASSIFIER_FEATURES + zone gradients
    + lags of LAG_VARS_ALERT at LAG_HOURS.
    """
    df_lag = df.copy()
    for var in LAG_VARS_ALERT:
        if var in df_lag.columns:
            df_lag = build_lag_features(df_lag, var, LAG_HOURS)

    X = build_feature_matrix(df_lag, port, include_zones=True)

    for var in LAG_VARS_ALERT:
        for h in LAG_HOURS:
            col = f"{var}_lag_{h}h"
            if col in df_lag.columns:
                series = df_lag[col]
                med = series.median()
                X[col] = series.fillna(med if pd.notna(med) else 0.0)

    return X


# ── Train / test split ───────────────────────────────────────────────────────

def _time_split(index: pd.DatetimeIndex, test_days: int = 365):
    """
    Return (train_idx, test_idx) DatetimeIndex objects based on the last test_days days.
    """
    cutoff = index.max() - pd.Timedelta(days=test_days)
    return index[index <= cutoff], index[index > cutoff]


# ── Lag features ─────────────────────────────────────────────────────────────

def build_lag_features(
    df: pd.DataFrame, target_col: str, lag_hours: list
) -> pd.DataFrame:
    """Add {target_col}_lag_{h}h columns to df."""
    for h in lag_hours:
        df[f"{target_col}_lag_{h}h"] = df[target_col].shift(h)
    return df


# ── WeatherCodePredictor ──────────────────────────────────────────────────────

def train_weather_code_predictor(df: pd.DataFrame, port: str) -> dict:
    """
    Train 72 XGBClassifiers to predict WMO weather group at T+h for h in 1..72.

    Features: lags of NUMERIC_FORECAST_VARS + weather_code lags at LAG_HOURS
              + wind_direction, humidity, cape context.
    Time-based split: last 365 days = test.

    Returns dict keyed by h → (XGBClassifier, metrics_dict).
    """
    df_lag = df.copy()
    for var in NUMERIC_FORECAST_VARS:
        if var in df_lag.columns:
            df_lag = build_lag_features(df_lag, var, LAG_HOURS)
    if "weather_code" in df_lag.columns:
        df_lag = build_lag_features(df_lag, "weather_code", LAG_HOURS)

    extra_features = [
        c for c in ["wind_direction_10m", "relative_humidity_2m", "cape"]
        if c in df_lag.columns
    ]
    lag_cols = [
        f"{var}_lag_{h}h"
        for var in NUMERIC_FORECAST_VARS
        for h in LAG_HOURS
        if f"{var}_lag_{h}h" in df_lag.columns
    ]
    wcode_lag_cols = [
        f"weather_code_lag_{h}h"
        for h in LAG_HOURS
        if f"weather_code_lag_{h}h" in df_lag.columns
    ]
    feature_cols = lag_cols + wcode_lag_cols + extra_features

    train_idx, test_idx = _time_split(df_lag.index)

    feature_df = df_lag[feature_cols].copy()
    for col in feature_df.columns:
        med = feature_df[col].median()
        feature_df[col] = feature_df[col].fillna(med if pd.notna(med) else 0.0)

    y_all_labels = make_weather_code_label(df_lag)

    models: dict = {}

    for h in CODE_PREDICTOR_HORIZONS:
        y = y_all_labels.shift(-h)
        valid = y.notna()
        X_all = feature_df.loc[valid]
        y_all = y[valid]

        X_tr = X_all.loc[X_all.index.isin(train_idx)]
        X_te = X_all.loc[X_all.index.isin(test_idx)]
        y_tr = y_all.loc[y_all.index.isin(train_idx)]
        y_te = y_all.loc[y_all.index.isin(test_idx)]

        if len(X_tr) < 10 or len(X_te) < 10:
            continue

        classes = sorted(y_tr.unique())
        label_to_int = {c: i for i, c in enumerate(classes)}
        int_to_label = {v: k for k, v in label_to_int.items()}

        y_tr_int = y_tr.map(label_to_int)
        y_te_int = y_te.map(label_to_int)

        sample_weights = compute_sample_weight("balanced", y_tr_int)

        clf = XGBClassifier(
            n_estimators=1000,
            early_stopping_rounds=20,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=len(classes),
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(
            X_tr, y_tr_int,
            sample_weight=sample_weights,
            eval_set=[(X_te, y_te_int)],
            verbose=False,
        )

        clf._label_to_int = label_to_int
        clf._int_to_label = int_to_label
        clf._classes = classes

        y_prob = clf.predict_proba(X_te)
        y_pred_int = y_prob.argmax(axis=1)
        y_pred = pd.Series(y_pred_int, index=y_te.index).map(int_to_label)

        met = eval_classifier(y_te.values, y_pred.values, y_prob)
        models[h] = (clf, met)

    return models


# ── WeatherNumericsForecaster ─────────────────────────────────────────────────

def train_weather_numerics_forecaster(df: pd.DataFrame, port: str) -> dict:
    """
    Train one XGBRegressor per (variable, horizon) for horizons in NUMERIC_HORIZONS.

    Variables: NUMERIC_FORECAST_VARS (6 vars × 11 horizons = 66 models).
    Features: lags of all 6 vars at LAG_HOURS + zone gradient columns
              + wind_direction, humidity, cape.
    Time-based split: last 365 days = test.

    Returns dict keyed by (var, h) → (XGBRegressor, metrics_dict).
    """
    df_lag = df.copy()
    for var in NUMERIC_FORECAST_VARS:
        if var in df_lag.columns:
            df_lag = build_lag_features(df_lag, var, LAG_HOURS)

    extra_features = [
        c for c in ["wind_direction_10m", "relative_humidity_2m", "cape"]
        if c in df_lag.columns
    ]
    lag_cols = [
        f"{var}_lag_{h}h"
        for var in NUMERIC_FORECAST_VARS
        for h in LAG_HOURS
        if f"{var}_lag_{h}h" in df_lag.columns
    ]

    zone_cols = []
    zone_suffixes = ("_pressure_gradient", "_wind_delta")
    for z in zone_points(port):
        pfx = z["prefix"]
        for suf in zone_suffixes:
            col = f"{pfx}{suf}"
            if col in df_lag.columns:
                zone_cols.append(col)

    feature_cols = lag_cols + zone_cols + extra_features

    train_idx, test_idx = _time_split(df_lag.index)

    feature_df = df_lag[feature_cols].copy()
    for col in feature_df.columns:
        med = feature_df[col].median()
        feature_df[col] = feature_df[col].fillna(med if pd.notna(med) else 0.0)

    models: dict = {}

    for var in NUMERIC_FORECAST_VARS:
        if var not in df_lag.columns:
            continue
        for h in NUMERIC_HORIZONS:
            y = df_lag[var].shift(-h)
            valid = y.notna()

            X_all = feature_df.loc[valid]
            y_all = y[valid]

            X_tr = X_all.loc[X_all.index.isin(train_idx)]
            X_te = X_all.loc[X_all.index.isin(test_idx)]
            y_tr = y_all.loc[y_all.index.isin(train_idx)]
            y_te = y_all.loc[y_all.index.isin(test_idx)]

            if len(X_tr) < 10 or len(X_te) < 10:
                continue

            reg = XGBRegressor(
                n_estimators=1000,
                early_stopping_rounds=20,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
            reg.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

            y_pred = reg.predict(X_te)
            met = eval_forecaster(y_te.values, y_pred)

            # Fit AR(2) corrector on training residuals to reduce autocorrelation
            y_tr_pred = reg.predict(X_tr)
            train_residuals = y_tr.values - y_tr_pred
            corrector = ARMAResidualCorrector(order=2).fit(train_residuals)
            models[(var, h)] = (ARMACorrectedForecaster(reg, corrector), met)

    return models


# ── Calibrated wrapper ───────────────────────────────────────────────────────

class CalibratedAlert:
    """
    Thin wrapper around a fitted XGBClassifier + IsotonicRegression calibrator.
    Exposes predict_proba() and feature_names_in_ so it slots in wherever a
    plain XGBClassifier was used.  Serialises cleanly with joblib.
    """

    def __init__(self, clf: XGBClassifier, calibrator: IsotonicRegression):
        self.clf = clf
        self.calibrator = calibrator
        self.feature_names_in_ = getattr(clf, "feature_names_in_", None)

    def predict_proba(self, X) -> np.ndarray:
        raw = self.clf.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ── DisruptionAlert ───────────────────────────────────────────────────────────

def train_disruption_alert(
    df: pd.DataFrame,
    port: str,
    y: pd.Series | None = None,
) -> dict:
    """
    Train 3 XGBClassifiers for disruption windows [24, 48, 72]h.

    y: optional per-hour binary disruption label (from NCEI / PortWatch / AIS).
       When None, falls back to make_composite_disruption_label().
    For each window w, the label y_w[t] = 1 if any disruption in (t, t+w].
    Acceptability gate: ROC-AUC > 0.75 AND ECE < 0.05 (checked in train.py).

    Returns dict keyed by window_h → (XGBClassifier, metrics_dict).
    """
    X = build_alert_features(df, port)

    base_label = y if y is not None else make_composite_disruption_label(df, port)

    # Three-way time split: train | calibrate (6 months) | test (12 months)
    # Using separate calibration and test sets prevents the isotonic calibration
    # from seeing the same data used to evaluate ECE/ROC-AUC.
    full_end  = X.index.max()
    test_cut  = full_end - pd.Timedelta(days=365)
    cal_cut   = test_cut - pd.Timedelta(days=180)

    models: dict = {}

    for window_h in ALERT_WINDOWS:
        y_window = make_disruption_window_label(base_label, window_h)

        valid = y_window.notna()
        X_v = X[valid]
        y_v = y_window[valid]

        X_tr  = X_v[X_v.index <= cal_cut]
        X_cal = X_v[(X_v.index > cal_cut) & (X_v.index <= test_cut)]
        X_te  = X_v[X_v.index > test_cut]
        y_tr  = y_v[y_v.index <= cal_cut]
        y_cal = y_v[(y_v.index > cal_cut) & (y_v.index <= test_cut)]
        y_te  = y_v[y_v.index > test_cut]

        if len(X_tr) < 50 or len(X_cal) < 20 or len(X_te) < 10:
            continue

        n_neg    = int((y_tr == 0).sum())
        n_pos    = int((y_tr == 1).sum())
        pos_rate = n_pos / max(n_pos + n_neg, 1)

        # Cap scale_pos_weight at 20 — extreme weighting causes overconfident
        # positive predictions that cannot be calibrated away.
        spw = min(n_neg / max(n_pos, 1), 20.0)

        # Hyperparameter profile by positive-class density:
        #   sparse  (< 2%)  — very shallow, heavy regularisation (Houston / NCEI labels)
        #   moderate (2–15%) — balanced (Rotterdam physics, HK/Kaohsiung PortWatch)
        #   dense   (> 15%) — deeper allowed, moderate regularisation
        if pos_rate < 0.02:
            depth, lr, mcw, reg_a, reg_l = 3, 0.02, 10, 0.5, 3.0
        elif pos_rate < 0.15:
            depth, lr, mcw, reg_a, reg_l = 4, 0.03, 5,  0.2, 2.0
        else:
            depth, lr, mcw, reg_a, reg_l = 5, 0.05, 3,  0.1, 1.5

        clf = XGBClassifier(
            n_estimators=1000,        # high ceiling; early stopping controls actual count
            max_depth=depth,
            learning_rate=lr,
            min_child_weight=mcw,
            subsample=0.8,
            colsample_bytree=0.7,
            scale_pos_weight=spw,
            reg_alpha=reg_a,
            reg_lambda=reg_l,
            max_delta_step=1,         # stabilises gradient updates for imbalanced labels
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=30, # stop when calibration-set loss plateaus
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)], verbose=False)

        # Isotonic calibration on the held-out calibration set.
        # This maps raw XGBoost logit scores to probabilities that match the
        # empirical base rate — the key fix for the "always near 100%" issue.
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(clf.predict_proba(X_cal)[:, 1], y_cal)
        cal_clf = CalibratedAlert(clf, ir)

        y_prob = cal_clf.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        met = eval_binary(y_te.values, y_pred, y_prob)
        models[window_h] = (cal_clf, met)

    return models


# ── Persistence ───────────────────────────────────────────────────────────────

def save_models(models: dict, port: str) -> None:
    """
    Save all models for a port to models/{port}/ using joblib.

    models: flat dict of str → any (model, tuple, etc.)
    """
    port_dir = MODEL_DIR / port
    port_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in models.items():
        joblib.dump(obj, port_dir / f"{name}.joblib")
    print(f"Saved {len(models)} model(s) to {port_dir}/")


def load_models(port: str) -> dict:
    """Load all saved models for a port from models/{port}/."""
    port_dir = MODEL_DIR / port
    if not port_dir.exists():
        raise FileNotFoundError(
            f"No models directory for port '{port}': {port_dir}"
        )
    return {path.stem: joblib.load(path) for path in sorted(port_dir.glob("*.joblib"))}
