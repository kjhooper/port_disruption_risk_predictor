"""
train.py — CLI to train WeatherCodePredictor, WeatherNumericsForecaster,
           and DisruptionAlert for one or all ports.

Usage:
    conda run -n personal python train.py --port rotterdam
    conda run -n personal python train.py --port all
    conda run -n personal python train.py --port rotterdam --mlflow
"""

import argparse
from pathlib import Path

import pandas as pd

from fetch import PORTS, zone_points
from features import compute_all_features
from labels import make_composite_disruption_label
from model import (
    build_lag_features,
    NUMERIC_FORECAST_VARS,
    LAG_HOURS,
    train_weather_code_predictor,
    train_weather_numerics_forecaster,
    train_disruption_alert,
    save_models,
)


def _delete_old_models(port: str) -> None:
    """Remove any .joblib files that don't match the new naming pattern."""
    port_dir = Path("models") / port
    if not port_dir.exists():
        return
    new_names = {
        "wcode_predictors.joblib",
        "weather_numerics.joblib",
        "disruption_alerts.joblib",
    }
    for f in port_dir.glob("*.joblib"):
        if f.name not in new_names:
            f.unlink()
            print(f"  Deleted stale model: {f.name}")


def train_port(port: str, mlflow_enabled: bool = False) -> None:
    print(f"\n{'=' * 60}")
    print(f"Training models for: {PORTS[port]['label']}")
    print(f"{'=' * 60}")

    # ── Load data ──────────────────────────────────────────────────────────────
    wide_path = Path("data") / f"{port}_historical_wide.parquet"
    if not wide_path.exists():
        print(f"[ERROR] {wide_path} not found. Run fetch.py first.")
        return

    df = pd.read_parquet(wide_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns from {wide_path}")

    if "weather_code" not in df.columns:
        print("[WARN] weather_code column missing — WeatherCodePredictor labels will be empty. "
              "Re-fetch data to include it.")

    df = compute_all_features(df, port)
    print(f"After feature engineering: {len(df.columns)} columns")

    # ── Clean up stale model files ─────────────────────────────────────────────
    _delete_old_models(port)

    models_to_save: dict = {}

    # ── WeatherCodePredictor ───────────────────────────────────────────────────
    print("\n[WeatherCodePredictor] Training 72 classifiers (T+1h … T+72h, WMO group)...")
    try:
        wcode_models = train_weather_code_predictor(df, port)
        models_to_save["wcode_predictors"] = wcode_models
        # Report metrics at T+24h, T+48h, T+72h as representative samples
        for h in [24, 48, 72]:
            if h in wcode_models:
                _, met = wcode_models[h]
                f1  = met.get("macro_f1", "N/A")
                roc = met.get("roc_auc",  "N/A")
                print(f"  T+{h:2d}h  Macro F1={f1}  ROC-AUC={roc}")
        print(f"  Total classifiers trained: {len(wcode_models)}")
    except Exception as e:
        print(f"  [SKIP] WeatherCodePredictor failed: {e}")

    # ── WeatherNumericsForecaster ──────────────────────────────────────────────
    print("\n[WeatherNumericsForecaster] Training 66 regressors "
          "(6 vars × 11 horizons: T+1h … T+72h)...")
    numerics_models: dict = {}
    try:
        numerics_models = train_weather_numerics_forecaster(df, port)
        models_to_save["weather_numerics"] = numerics_models
        for var in NUMERIC_FORECAST_VARS:
            for h in [24, 48, 72]:
                if (var, h) in numerics_models:
                    _, met = numerics_models[(var, h)]
                    rmse = met.get("rmse", "N/A")
                    mae  = met.get("mae",  "N/A")
                    r2   = met.get("r2",   "N/A")
                    print(f"  {var:<26} T+{h:2d}h  RMSE={rmse}  MAE={mae}  R²={r2}")
        print(f"  Total regressors trained: {len(numerics_models)}")
    except Exception as e:
        print(f"  [SKIP] WeatherNumericsForecaster failed: {e}")

    # ── WeatherNumericsForecaster Residual Autocorrelation — Ljung-Box ─────────
    if numerics_models:
        print("\n[WeatherNumericsForecaster] Residual Autocorrelation — "
              "Ljung-Box at lags 6, 12, 24 (T+24h models)")
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox

            test_cutoff = df.index.max() - pd.Timedelta(days=365)
            df_test = df[df.index > test_cutoff].copy()

            for var in NUMERIC_FORECAST_VARS:
                if var in df_test.columns:
                    df_test = build_lag_features(df_test, var, LAG_HOURS)

            extra_cols = [c for c in ["wind_direction_10m", "relative_humidity_2m", "cape"]
                          if c in df_test.columns]
            lag_cols = [f"{var}_lag_{h}h" for var in NUMERIC_FORECAST_VARS
                        for h in LAG_HOURS if f"{var}_lag_{h}h" in df_test.columns]

            zone_cols = []
            zone_suffixes = ("_pressure_gradient", "_wind_delta")
            for z in zone_points(port):
                pfx = z["prefix"]
                for suf in zone_suffixes:
                    col = f"{pfx}{suf}"
                    if col in df_test.columns:
                        zone_cols.append(col)

            feature_cols = lag_cols + zone_cols + extra_cols
            feature_df_test = df_test[feature_cols].copy()
            for col in feature_df_test.columns:
                med = feature_df_test[col].median()
                feature_df_test[col] = feature_df_test[col].fillna(
                    med if pd.notna(med) else 0.0
                )

            for var in NUMERIC_FORECAST_VARS:
                if (var, 24) not in numerics_models or var not in df_test.columns:
                    continue
                reg, _ = numerics_models[(var, 24)]
                y_true = df_test[var].shift(-24).dropna()
                X_test = feature_df_test.loc[y_true.index]
                X_aligned = X_test.copy()
                for col in reg.feature_names_in_:
                    if col not in X_aligned.columns:
                        X_aligned[col] = 0.0
                X_aligned = X_aligned[list(reg.feature_names_in_)]
                y_pred = reg.predict(X_aligned)
                residuals = pd.Series(y_true.values - y_pred).dropna()

                lb = acorr_ljungbox(residuals, lags=[6, 12, 24], return_df=True)
                sig_lags = lb.index[lb["lb_pvalue"] < 0.05].tolist()
                if sig_lags:
                    status = f"Autocorrelated at lags {sig_lags} → consider LSTM / ARIMA residuals"
                else:
                    status = "OK — no significant residual autocorrelation"
                print(f"  {var:<26} T+24h  {status}")

        except Exception as e:
            print(f"  [SKIP] Residual ACF failed: {e}")

    # ── DisruptionAlert ────────────────────────────────────────────────────────
    # Label priority (all ports):
    #   1. NCEI Storm Events (Houston only) — USCG/NWS validated marine events
    #   2. IMF PortWatch daily portcalls — traffic-drop method, all ports
    #   3. AIS vessel activity (Houston only) — hourly traffic-drop (large download)
    #   4. Composite physics label — fallback (always valid, used for Rotterdam)
    alert_label = None
    alert_label_source = "composite physics label"

    storm_activity_path     = Path("data") / "houston_storm_activity.parquet"
    portwatch_activity_path = Path("data") / f"{port}_portwatch_activity.parquet"
    ais_activity_path       = Path("data") / "houston_ais_activity.parquet"

    if port == "houston" and storm_activity_path.exists():
        print(f"\n[DisruptionAlert] Loading NCEI Storm Events label from {storm_activity_path}")
        try:
            from fetch_disruptions import make_storm_disruption_label
            storm_activity = pd.read_parquet(storm_activity_path)
            alert_label = make_storm_disruption_label(storm_activity, df.index)
            n_ncei = int(alert_label.sum())
            # Supplement NCEI events with composite physics (OR logic) to increase
            # positive-class count.  NCEI gives ~165 high-quality hours; composite
            # adds physics-threshold crossings (wind > 15 m/s, gusts > 22 m/s,
            # wave > 2.5m, severe WMO).  Combining both gives a richer positive set
            # without introducing the daily-granularity noise of PortWatch traffic drops.
            composite_supplement = make_composite_disruption_label(df, port)
            alert_label = (
                alert_label.astype(bool) | composite_supplement.astype(bool)
            ).astype("int8")
            n_disrupted = int(alert_label.sum())
            n_added = n_disrupted - n_ncei
            alert_label_source = "NCEI Storm Events + composite physics supplement"
            print(f"  NCEI events:      {n_ncei:,} disrupted hours")
            print(f"  Physics added:    {n_added:,} additional hours "
                  f"(wind/gust/wave threshold crossings not in NCEI)")
            print(f"  Combined label:   {n_disrupted:,} disrupted hours "
                  f"({n_disrupted / len(alert_label) * 100:.2f}% of weather rows)")
        except Exception as exc:
            print(f"  [WARN] Storm Events label failed ({exc}), trying PortWatch ...")
            alert_label = None

    if alert_label is None and portwatch_activity_path.exists():
        print(f"\n[DisruptionAlert] Loading IMF PortWatch label from {portwatch_activity_path}")
        try:
            from fetch_portwatch import make_portwatch_disruption_label
            portwatch_activity = pd.read_parquet(portwatch_activity_path)
            label_raw = make_portwatch_disruption_label(portwatch_activity, df.index)
            alert_label = make_portwatch_disruption_label(
                portwatch_activity, df.index,
                exclude_holidays=True, port=port,
            )
            n_disrupted = int(alert_label.sum())
            n_suppressed = int(label_raw.sum()) - n_disrupted
            alert_label_source = "IMF PortWatch (daily portcalls, holidays excluded)"
            print(f"  PortWatch coverage: {n_disrupted:,} disrupted hours "
                  f"({n_disrupted / len(alert_label) * 100:.2f}% of weather rows)")
            print(f"  Holiday days suppressed: {n_suppressed:,} hours")
        except Exception as exc:
            print(f"  [WARN] PortWatch label failed ({exc}), trying AIS ...")
            alert_label = None

    if port == "houston" and alert_label is None and ais_activity_path.exists():
        print(f"\n[DisruptionAlert] Loading AIS-derived label from {ais_activity_path}")
        try:
            from fetch_ais import make_ais_disruption_label
            activity_df = pd.read_parquet(ais_activity_path)
            ais_label   = make_ais_disruption_label(activity_df)
            composite   = make_composite_disruption_label(df, port)
            alert_label = composite.copy()
            ais_aligned = ais_label.reindex(df.index)
            has_ais     = ais_aligned.notna()
            alert_label[has_ais] = ais_aligned[has_ais].astype("int8")
            alert_label_source = "AIS vessel activity"
            print(f"  AIS label coverage: {has_ais.mean() * 100:.1f}% of weather rows")
        except Exception as exc:
            print(f"  [WARN] AIS label failed ({exc}), falling back to composite label")

    # If label is all-zeros (e.g. Rotterdam PortWatch), fall back to composite physics
    if alert_label is not None and int(alert_label.sum()) == 0:
        print(f"\n[DisruptionAlert] Traffic label has 0 disruptions — "
              f"falling back to composite physics label for {port}.")
        alert_label = None
        alert_label_source = "composite physics label"


    if alert_label is None:
        hint = "" if port != "houston" else " (run fetch_disruptions.py for better labels)"
        print(f"\n[DisruptionAlert] No traffic-based label found — "
              f"using composite physics label{hint}.")

    print(f"\n[DisruptionAlert] Training 3 window classifiers "
          f"(24h / 48h / 72h, label: {alert_label_source})...")
    try:
        alert_models = train_disruption_alert(df, port, y=alert_label)
        models_to_save["disruption_alerts"] = alert_models
        for window_h, (clf, met) in sorted(alert_models.items()):
            roc = met.get("roc_auc", 0)
            ece = met.get("ece", 1)
            pr  = met.get("pr_auc", "N/A")
            f1  = met.get("f1", "N/A")
            gate = "PASS" if roc > 0.75 and ece < 0.15 else "FAIL"
            print(f"  Window {window_h:2d}h  ROC-AUC={roc:.4f}  PR-AUC={pr}  "
                  f"F1={f1}  ECE={ece}  gate={gate}")
    except Exception as e:
        print(f"  [SKIP] DisruptionAlert failed: {e}")

    # ── Save ───────────────────────────────────────────────────────────────────
    if models_to_save:
        save_models(models_to_save, port)
    else:
        print("\n[WARN] No models trained — nothing saved.")
        return

    # ── MLflow ─────────────────────────────────────────────────────────────────
    if mlflow_enabled:
        _log_mlflow(port, models_to_save)

    print(f"\nDone. {len(models_to_save)} model file(s) saved.\n")


def _log_mlflow(port: str, models_to_save: dict) -> None:
    try:
        import mlflow
        with mlflow.start_run(run_name=f"{port}_redesign"):
            mlflow.set_tag("port", port)
            for name, value in models_to_save.items():
                # value is a dict of sub-models keyed by horizon/window
                if isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        if isinstance(sub_val, tuple) and len(sub_val) == 2:
                            _, metrics = sub_val
                            if isinstance(metrics, dict):
                                prefix = f"{name}/{sub_key}"
                                for k, v in metrics.items():
                                    if isinstance(v, (int, float)):
                                        mlflow.log_metric(f"{prefix}/{k}", float(v))
        print("MLflow run logged.")
    except ImportError:
        print("[WARN] mlflow not installed; skipping MLflow logging.")
    except Exception as e:
        print(f"[WARN] MLflow logging failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train WeatherCodePredictor, WeatherNumericsForecaster, "
                    "and DisruptionAlert for port disruption risk."
    )
    parser.add_argument(
        "--port",
        required=True,
        choices=list(PORTS.keys()) + ["all"],
        help="Port key (rotterdam/houston/hong_kong/kaohsiung) or 'all'",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log metrics to local MLflow server",
    )
    args = parser.parse_args()

    ports = list(PORTS.keys()) if args.port == "all" else [args.port]
    for port in ports:
        train_port(port, mlflow_enabled=args.mlflow)


if __name__ == "__main__":
    main()
