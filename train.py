"""
train.py — CLI to train M1, M2, M3 for one or all ports.

Usage:
    conda run -n personal python train.py --port rotterdam
    conda run -n personal python train.py --port all
    conda run -n personal python train.py --port rotterdam --mlflow
"""

import argparse
from pathlib import Path

import pandas as pd

from fetch import PORTS
from features import compute_all_features
from labels import make_composite_disruption_label
from model import (
    train_classifier,
    train_binary,
    train_forecaster,
    train_forecast_classifier,
    save_models,
)


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
        print("[WARN] weather_code column missing — M1/M2 labels will be empty. "
              "Re-fetch data to include it.")

    df = compute_all_features(df, port)
    print(f"After feature engineering: {len(df.columns)} columns")

    models_to_save: dict = {}

    # ── M1: Event Classifier ───────────────────────────────────────────────────
    print("\n[M1] Training Event Classifier (multi-class WMO groups)...")
    try:
        m1_model, m1_metrics = train_classifier(df, port, model_type="multi")
        models_to_save["m1_classifier"] = (m1_model, m1_metrics)
        print(f"  Macro F1 : {m1_metrics.get('macro_f1', 'N/A')}")
        roc = m1_metrics.get("roc_auc")
        print(f"  ROC-AUC  : {roc if roc is not None else 'N/A'}")
        for cls, f1 in m1_metrics.get("per_class_f1", {}).items():
            print(f"    {cls:<16} F1 = {f1:.4f}")
    except Exception as e:
        print(f"  [SKIP] M1 failed: {e}")

    # ── M1-B: Forecast Classifier ──────────────────────────────────────────────
    print("\n[M1-B] Training Forecast Classifiers (T+24h / T+48h / T+72h)...")
    try:
        m1b_models = train_forecast_classifier(df, port)
        for horizon, (clf, met) in sorted(m1b_models.items()):
            key = f"m1_forecast_{horizon}h"
            models_to_save[key] = (clf, met)
            f1  = met.get("macro_f1", "N/A")
            roc = met.get("roc_auc",  "N/A")
            print(f"  T+{horizon:2d}h  Macro F1={f1}  ROC-AUC={roc}")
    except Exception as e:
        print(f"  [SKIP] M1-B failed: {e}")

    # ── M2: Binary Detector ────────────────────────────────────────────────────
    # Label priority (all ports):
    #   1. NCEI Storm Events (Houston only) — USCG/NWS validated marine events
    #   2. IMF PortWatch daily portcalls — traffic-drop method, all 3 ports
    #   3. AIS vessel activity (Houston only) — hourly traffic-drop (large download)
    #   4. Composite physics label — fallback
    m2_label = None
    m2_label_source = "composite physics label"

    storm_activity_path    = Path("data") / "houston_storm_activity.parquet"
    portwatch_activity_path = Path("data") / f"{port}_portwatch_activity.parquet"
    ais_activity_path      = Path("data") / "houston_ais_activity.parquet"

    if port == "houston" and storm_activity_path.exists():
        print(f"\n[M2] Loading NCEI Storm Events disruption label from {storm_activity_path}")
        try:
            from fetch_disruptions import make_storm_disruption_label
            storm_activity = pd.read_parquet(storm_activity_path)
            m2_label = make_storm_disruption_label(storm_activity, df.index)
            n_disrupted = int(m2_label.sum())
            m2_label_source = "NCEI Storm Events"
            print(f"  Storm events coverage: {n_disrupted:,} disrupted hours "
                  f"({n_disrupted / len(m2_label) * 100:.2f}% of weather rows)")
        except Exception as exc:
            print(f"  [WARN] Storm Events label failed ({exc}), trying PortWatch ...")
            m2_label = None

    if m2_label is None and portwatch_activity_path.exists():
        print(f"\n[M2] Loading IMF PortWatch disruption label from {portwatch_activity_path}")
        try:
            from fetch_portwatch import make_portwatch_disruption_label
            portwatch_activity = pd.read_parquet(portwatch_activity_path)
            # Count before holiday exclusion for comparison
            m2_label_raw = make_portwatch_disruption_label(portwatch_activity, df.index)
            m2_label = make_portwatch_disruption_label(
                portwatch_activity, df.index,
                exclude_holidays=True, port=port,
            )
            n_disrupted = int(m2_label.sum())
            n_suppressed = int(m2_label_raw.sum()) - n_disrupted
            m2_label_source = "IMF PortWatch (daily portcalls, holidays excluded)"
            print(f"  PortWatch coverage: {n_disrupted:,} disrupted hours "
                  f"({n_disrupted / len(m2_label) * 100:.2f}% of weather rows)")
            print(f"  Holiday days suppressed: {n_suppressed:,} hours")
        except Exception as exc:
            print(f"  [WARN] PortWatch label failed ({exc}), trying AIS ...")
            m2_label = None

    if port == "houston" and m2_label is None and ais_activity_path.exists():
        print(f"\n[M2] Loading AIS-derived disruption label from {ais_activity_path}")
        try:
            from fetch_ais import make_ais_disruption_label
            activity_df = pd.read_parquet(ais_activity_path)
            ais_label   = make_ais_disruption_label(activity_df)
            composite   = make_composite_disruption_label(df, port)
            m2_label    = composite.copy()
            ais_aligned = ais_label.reindex(df.index)
            has_ais     = ais_aligned.notna()
            m2_label[has_ais] = ais_aligned[has_ais].astype("int8")
            m2_label_source = "AIS vessel activity"
            print(f"  AIS label coverage: {has_ais.mean() * 100:.1f}% of weather rows")
        except Exception as exc:
            print(f"  [WARN] AIS label failed ({exc}), falling back to composite label")

    if m2_label is None:
        hint = "" if port != "houston" else " (run fetch_disruptions.py for better labels)"
        print(f"\n[M2] No traffic-based label found — using composite physics label{hint}.")

    # Resolve final label (None → composite physics fallback)
    resolved_label = m2_label  # may still be None → train_binary uses composite internally

    # Check for degenerate all-zero label: port is confirmed stable, skip M2.
    if resolved_label is not None and int(resolved_label.sum()) == 0:
        print(f"\n[M2] SKIP — PortWatch confirms {port} is a stable port (0 disrupted hours).")
        print("       No binary disruption model is meaningful for this port.")
    else:
        print(f"\n[M2] Training Binary Detector (label: {m2_label_source})...")
        try:
            m2_model, m2_metrics = train_binary(df, port, y=resolved_label)
            models_to_save["m2_binary"] = (m2_model, m2_metrics)
            roc = m2_metrics.get("roc_auc", 0)
            ece = m2_metrics.get("ece", 1)
            print(f"  ROC-AUC  : {roc}")
            print(f"  PR-AUC   : {m2_metrics.get('pr_auc', 'N/A')}")
            print(f"  F1       : {m2_metrics.get('f1', 'N/A')}")
            print(f"  Precision: {m2_metrics.get('precision', 'N/A')}")
            print(f"  Recall   : {m2_metrics.get('recall', 'N/A')}")
            print(f"  ECE      : {ece}")
            gate = "PASS" if roc > 0.75 and ece < 0.05 else "FAIL"
            print(f"  Acceptability gate (ROC-AUC > 0.75 AND ECE < 0.05): {gate}")
        except Exception as e:
            print(f"  [SKIP] M2 failed: {e}")

    # ── M3: Variable Forecaster ────────────────────────────────────────────────
    print("\n[M3] Training Variable Forecasters (T+24h / T+48h / T+72h)...")
    m3_models: dict = {}
    try:
        m3_models = train_forecaster(df, port)
        for (target, horizon), (reg, met) in sorted(
            m3_models.items(), key=lambda kv: (kv[0][0], kv[0][1])
        ):
            key = f"m3_{target}_{horizon}h"
            models_to_save[key] = (reg, met)
            rmse = met.get("rmse", "N/A")
            mae  = met.get("mae",  "N/A")
            r2   = met.get("r2",   "N/A")
            print(f"  {target:<26} T+{horizon:2d}h  RMSE={rmse}  MAE={mae}  R²={r2}")
    except Exception as e:
        print(f"  [SKIP] M3 failed: {e}")

    # ── M3 Residual Autocorrelation — Ljung-Box ────────────────────────────────
    if m3_models:
        print("\n[M3] Residual Autocorrelation — Ljung-Box test at lags 6, 12, 24 (T+24h models)")
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            from model import build_lag_features, LAG_HOURS, FORECAST_TARGETS

            test_cutoff = df.index.max() - pd.Timedelta(days=365)
            df_test = df[df.index > test_cutoff].copy()

            # Build lag features for all targets (match training feature set)
            for tgt in FORECAST_TARGETS:
                if tgt in df_test.columns:
                    df_test = build_lag_features(df_test, tgt, LAG_HOURS)

            extra_cols = [c for c in ["wind_direction_10m", "relative_humidity_2m", "cape"]
                          if c in df_test.columns]
            lag_cols = [f"{tgt}_lag_{h}h" for tgt in FORECAST_TARGETS
                        for h in LAG_HOURS if f"{tgt}_lag_{h}h" in df_test.columns]
            feature_cols = lag_cols + extra_cols

            feature_df_test = df_test[feature_cols].copy()
            for col in feature_df_test.columns:
                med = feature_df_test[col].median()
                feature_df_test[col] = feature_df_test[col].fillna(
                    med if pd.notna(med) else 0.0
                )

            for target in FORECAST_TARGETS:
                if (target, 24) not in m3_models or target not in df_test.columns:
                    continue
                reg, _ = m3_models[(target, 24)]
                y_true = df_test[target].shift(-24).dropna()
                X_test = feature_df_test.loc[y_true.index]
                y_pred = reg.predict(X_test)
                residuals = pd.Series(y_true.values - y_pred).dropna()

                lb = acorr_ljungbox(residuals, lags=[6, 12, 24], return_df=True)
                sig_lags = lb.index[lb["lb_pvalue"] < 0.05].tolist()
                if sig_lags:
                    status = f"Autocorrelated at lags {sig_lags} → consider LSTM / ARIMA residuals"
                else:
                    status = "OK — no significant residual autocorrelation"
                print(f"  {target:<26} T+24h  {status}")

        except Exception as e:
            print(f"  [SKIP] Residual ACF failed: {e}")

    # ── Save ───────────────────────────────────────────────────────────────────
    if models_to_save:
        save_models(models_to_save, port)
    else:
        print("\n[WARN] No models trained — nothing saved.")

    # ── MLflow ─────────────────────────────────────────────────────────────────
    if mlflow_enabled:
        _log_mlflow(port, models_to_save)

    print(f"\nDone. {len(models_to_save)} model(s) saved.\n")


def _log_mlflow(port: str, models_to_save: dict) -> None:
    try:
        import mlflow
        with mlflow.start_run(run_name=f"{port}_sprint3"):
            mlflow.set_tag("port", port)
            for name, value in models_to_save.items():
                # value is (model, metrics) tuple
                if isinstance(value, tuple) and len(value) == 2:
                    _, metrics = value
                    if isinstance(metrics, dict):
                        for k, v in metrics.items():
                            if isinstance(v, (int, float)):
                                mlflow.log_metric(f"{name}/{k}", float(v))
        print("MLflow run logged.")
    except ImportError:
        print("[WARN] mlflow not installed; skipping MLflow logging.")
    except Exception as e:
        print(f"[WARN] MLflow logging failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train M1/M2/M3 models for port disruption risk."
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
