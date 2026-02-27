"""
model.py — Three-model weather event system.

M1: Event Classifier   — multi-class XGBoost → WMO group
                         (clear / fog / rain_snow / showers / thunderstorm)
M2: Binary Detector    — binary XGBoost → is any weather event occurring?
M3: Variable Forecaster — per (target × horizon) XGBRegressor → T+24/48/72h
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from fetch import PORTS, zone_points
from labels import make_weather_code_label, make_binary_label, make_composite_disruption_label
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
]

FORECAST_TARGETS = ["wind_speed_10m", "precipitation", "pressure_msl", "wave_height"]
HORIZONS  = [24, 48, 72]   # hours ahead
LAG_HOURS = [1, 3, 6, 12, 24]

MODEL_DIR = Path("models")


# ── Feature matrix ───────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame, port: str, include_zones: bool = True
) -> pd.DataFrame:
    """
    Select and align classifier feature columns, filling NaN with column median.
    Dynamically appends zone gradient columns when include_zones=True.
    """
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


# ── Train / test split ───────────────────────────────────────────────────────

def _time_split(index: pd.DatetimeIndex, test_days: int = 365):
    """
    Return (train_idx, test_idx) DatetimeIndex objects based on the last test_days days.
    """
    cutoff = index.max() - pd.Timedelta(days=test_days)
    return index[index <= cutoff], index[index > cutoff]


# ── Classifiers (M1, M2) ─────────────────────────────────────────────────────

def train_classifier(
    df: pd.DataFrame,
    port: str,
    model_type: str = "multi",
    y: pd.Series | None = None,
) -> tuple[XGBClassifier, dict]:
    """
    Train M1 (multi-class) or M2 (binary) XGBoost classifier.

    model_type: "multi" → M1 (WMO group label)
                "binary" → M2 (composite physics disruption label, or custom y)
    y: optional pre-built label Series (e.g. AIS-derived for Houston).
       When None: M1 uses make_weather_code_label; M2 uses make_composite_disruption_label.
    Time-based split: last 365 days = test, rest = train.
    Returns (fitted_model, eval_metrics_dict).
    """
    X = build_feature_matrix(df, port, include_zones=True)

    if y is None:
        if model_type == "multi":
            y = make_weather_code_label(df)
        else:
            y = make_composite_disruption_label(df, port)

    # Drop rows with missing labels
    valid = y.notna()
    X, y = X[valid], y[valid]

    train_idx, test_idx = _time_split(X.index)
    X_tr = X.loc[X.index.isin(train_idx)]
    X_te = X.loc[X.index.isin(test_idx)]
    y_tr = y.loc[y.index.isin(train_idx)]
    y_te = y.loc[y.index.isin(test_idx)]

    if len(X_tr) < 10 or len(X_te) < 10:
        raise ValueError(
            f"Insufficient data: {len(X_tr)} train rows, {len(X_te)} test rows."
        )

    sample_weights = compute_sample_weight("balanced", y_tr)

    if model_type == "multi":
        classes = sorted(y_tr.unique())
        label_to_int = {c: i for i, c in enumerate(classes)}
        int_to_label = {v: k for k, v in label_to_int.items()}

        y_tr_int = y_tr.map(label_to_int)
        y_te_int = y_te.map(label_to_int)

        model = XGBClassifier(
            n_estimators=200,
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
        model.fit(
            X_tr, y_tr_int,
            sample_weight=sample_weights,
            eval_set=[(X_te, y_te_int)],
            verbose=False,
        )

        # predict_proba always returns (n, n_classes); argmax gives class indices
        y_prob = model.predict_proba(X_te)
        y_pred_int = y_prob.argmax(axis=1)
        y_pred = pd.Series(y_pred_int, index=y_te.index).map(int_to_label)

        # Attach label mapping as attributes for later inference
        model._label_to_int = label_to_int
        model._int_to_label = int_to_label
        model._classes = classes

        metrics = eval_classifier(y_te.values, y_pred.values, y_prob)

    else:  # binary
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_tr, y_tr,
            sample_weight=sample_weights,
            eval_set=[(X_te, y_te)],
            verbose=False,
        )

        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        metrics = eval_binary(y_te.values, y_pred, y_prob)

    return model, metrics


def train_binary(
    df: pd.DataFrame,
    port: str,
    y: pd.Series | None = None,
) -> tuple[XGBClassifier, dict]:
    """
    Train M2 binary disruption detector.

    If y is provided it is used directly as the label (e.g. AIS-derived label
    for Houston). Otherwise falls back to make_composite_disruption_label(),
    which uses physics-based wind / wave / fog thresholds.
    """
    return train_classifier(df, port, model_type="binary", y=y)


# ── Forecaster (M3) ──────────────────────────────────────────────────────────

def build_lag_features(
    df: pd.DataFrame, target_col: str, lag_hours: list
) -> pd.DataFrame:
    """Add {target_col}_lag_{h}h columns to df."""
    for h in lag_hours:
        df[f"{target_col}_lag_{h}h"] = df[target_col].shift(h)
    return df


def train_forecaster(
    df: pd.DataFrame,
    port: str,
    targets: list = None,
    horizons: list = None,
) -> dict:
    """
    Train one XGBRegressor per (target_var, horizon_hours).

    Direct multi-step approach — no recursive chaining.
    Features: lag columns for all FORECAST_TARGETS + wind_direction, humidity, cape.
    Time-based split: last 365 days = test.

    Returns dict keyed by (target_var, horizon_hours) → (fitted XGBRegressor, eval_metrics_dict).
    """
    if targets is None:
        targets = FORECAST_TARGETS
    if horizons is None:
        horizons = HORIZONS

    # Build lag features for all forecast targets
    df_lag = df.copy()
    for tgt in targets:
        if tgt in df_lag.columns:
            df_lag = build_lag_features(df_lag, tgt, LAG_HOURS)

    # Extra context features available at inference time
    extra_features = [
        c for c in ["wind_direction_10m", "relative_humidity_2m", "cape"]
        if c in df_lag.columns
    ]

    lag_cols = [
        f"{tgt}_lag_{h}h"
        for tgt in targets
        for h in LAG_HOURS
        if f"{tgt}_lag_{h}h" in df_lag.columns
    ]
    feature_cols = lag_cols + extra_features

    train_idx, test_idx = _time_split(df_lag.index)
    models: dict = {}

    # Fill NaN in feature columns with column median (handles all-NaN cols via 0 fallback)
    feature_df = df_lag[feature_cols].copy()
    for col in feature_df.columns:
        med = feature_df[col].median()
        feature_df[col] = feature_df[col].fillna(med if pd.notna(med) else 0.0)

    for target in targets:
        if target not in df_lag.columns:
            continue
        for horizon in horizons:
            # Future target: y[t] = target[t + horizon]
            y = df_lag[target].shift(-horizon)

            # Keep rows where target is available (features already filled)
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
                n_estimators=200,
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
            models[(target, horizon)] = (reg, met)

    return models


def predict_forecaster(
    df: pd.DataFrame, models: dict, horizon: int
) -> pd.DataFrame:
    """
    Given current df, return DataFrame of predicted values at T+horizon.

    models: dict keyed by (target_var, horizon_hours) as returned by train_forecaster.
    """
    # Build lag features for all FORECAST_TARGETS (must match training)
    df_lag = df.copy()
    for tgt in FORECAST_TARGETS:
        if tgt in df_lag.columns:
            df_lag = build_lag_features(df_lag, tgt, LAG_HOURS)

    extra_features = [
        c for c in ["wind_direction_10m", "relative_humidity_2m", "cape"]
        if c in df_lag.columns
    ]
    lag_cols = [
        f"{tgt}_lag_{h}h"
        for tgt in FORECAST_TARGETS
        for h in LAG_HOURS
        if f"{tgt}_lag_{h}h" in df_lag.columns
    ]
    feature_cols = lag_cols + extra_features

    X = df_lag[feature_cols].ffill().fillna(0)

    results: dict = {}
    for (target, h), (reg, _) in models.items():
        if h != horizon:
            continue
        results[target] = reg.predict(X)

    return pd.DataFrame(results, index=df.index)


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
