# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
conda run -n personal pip install -r requirements.txt

# Fetch weather data for a port and save to disk
conda run -n personal python fetch.py

# Fetch NOAA AIS vessel movement data for Houston (resume-safe)
conda run -n personal python fetch_ais.py

# Train M1/M2/M3 models for one port or all ports
conda run -n personal python train.py --port rotterdam
conda run -n personal python train.py --port all
conda run -n personal python train.py --port houston --mlflow

# Run dashboards
conda run -n personal streamlit run app.py      # live risk dashboard
conda run -n personal streamlit run eda.py      # EDA (5 tabs)
conda run -n personal streamlit run review.py   # Statistical review (9 tabs)
```

## Architecture

Sprint-based ML project building a 72-hour weather-driven disruption risk scorer for shipping ports (Rotterdam, Houston, Singapore). All files live flat in the repo root.

**Data pipeline:**
- `fetch.py` — Open-Meteo (archive, forecast, marine) + NOAA NDBC buoys. Exports `PORTS` dict, `zone_points(port)`, `fetch_port_all()`, `update_or_fetch()`. No API keys.
- `fetch_ais.py` — NOAA MarineCadastre AIS for Houston Ship Channel only. Streams monthly Zone-15 ZIP files, filters to Houston bbox + commercial vessel types, saves `data/houston_ais_raw.parquet` + `data/houston_ais_activity.parquet`. Resume-safe.
- `features.py` — `compute_all_features(df, port)` → ~100 feature columns including wind components, fog risk, zone gradients, rolling stats, storm approach index.
- `quality.py` — Stateless quality checks; `run_all_checks()` → `overall_score` (0–1).

**Labels (labels.py):**
- `make_weather_code_label(df)` → WMO group string (clear/fog/rain_snow/showers/thunderstorm)
- `make_composite_disruption_label(df, port)` → physics-based binary label: wind > 15 m/s OR gusts > 22 m/s OR wave > 2.5m OR td_spread < 2°C OR severe WMO codes. **Primary M2 label for Rotterdam/Singapore.**
- `make_ais_disruption_label(activity_df)` in `fetch_ais.py` → AIS-derived binary: `n_vessels_moving < 30% of 28-day rolling median`. **Primary M2 label for Houston once AIS data is downloaded.**

**Three models (model.py + train.py):**
| ID | Name | Target | Features | Saved as |
|----|------|--------|----------|---------|
| M1 | Event Classifier | WMO group (5-class) | 19 instantaneous features + zone gradients | `models/{port}/m1_classifier.joblib` |
| M2 | Binary Detector | Is port disrupted? (composite label) | Same as M1 | `models/{port}/m2_binary.joblib` |
| M3 | Variable Forecaster | wind/precip/pressure/wave at T+24/48/72h | Lag features (1/3/6/12/24h) + direction/humidity/cape | `models/{port}/m3_{target}_{horizon}h.joblib` |

- Time-based train/test split: last 365 days = test, rest = train. No shuffle.
- **M2 acceptability gate:** ROC-AUC > 0.75 AND ECE < 0.05 before using in `app.py`.
- M3 residuals are autocorrelated at lags 6/12/24h (Ljung-Box) → LSTM/ARIMA residual correction is the next modelling step.

**Metrics (metrics.py):** Pure functions — `eval_classifier()`, `eval_binary()`, `eval_forecaster()`, `cohen_d()`, `calibration_curve_data()`. No Streamlit.

**Dashboards:**
- `app.py` — Live risk view. Prefers saved parquets, falls back to live fetch.
- `eda.py` — 5-tab EDA. Loads `{port}_historical_wide.parquet` → `compute_all_features()`.
- `review.py` — 9-tab statistical review: Glossary / Data Quality / Anomaly / STL / Distribution Shift / Cross-correlation / Ground Truth (WMO + Cohen's d + box plots + label drift + event duration) / ACF-PACF + ADF / Seasonality heatmaps.

**Zone features:** `zone_points(port)` returns list of upstream/side measurement points at 150km and 300km. Rotterdam/Houston have scalar sea_bearing; Singapore has list bearing `[45, 270]` → prefixes like `z150b45`, `z300b270`. Zone gradient columns: `{prefix}_pressure_gradient`, `{prefix}_cape_excess`, `{prefix}_wind_delta`, `{prefix}_onshore_wind`.

**Parquet files (data/):**
| File | Contents |
|------|----------|
| `{port}_historical_wide.parquet` | Atmospheric + marine + zone columns (primary training input) |
| `{port}_forecast.parquet` | 7-day forecast |
| `houston_ais_raw.parquet` | Raw filtered AIS positions |
| `houston_ais_activity.parquet` | Hourly: n_vessels_total, n_vessels_moving, mean_sog |

**Conda env:** Always use `conda run -n personal <cmd>`. Dependencies include xgboost, statsmodels ≥ 0.14, scipy ≥ 1.13, joblib, requests.
