# Harbinger

A simple, extensible ML project that produces a probabilistic weather-driven disruption risk score for major shipping ports. Built with rigor around data quality, model performance, and result acceptability metrics at every stage.

---

## Table of Contents

- [Project Goal](#project-goal)
- [Project Plan](#project-plan)
- [Metrics Philosophy](#metrics-philosophy)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Ports Available](#ports-available)
- [Setup](#setup)
- [Deploying to Streamlit Cloud](#deploying-to-streamlit-cloud)
- [Extension Ideas](#extension-ideas)

---

## Project Goal

Given real-time and forecasted weather at a major port, produce a **72-hour rolling disruption risk score** (0–1) with full transparency on:
- how clean the input data is
- how well the forecast model performs
- whether the risk score is calibrated and trustworthy enough to act on

The project is intentionally kept simple so it is easy to understand, extend, and explain.

---

## Project Plan

### Overview

| Sprint | Days  | Theme                        | Demo Deliverable                                      | Status |
|--------|-------|------------------------------|-------------------------------------------------------|--------|
| 1      | 1–3   | Data pipeline & quality      | Live weather dashboard with quality scorecard         | ✅ Built |
| 2      | 4–6   | EDA & anomaly detection      | Labelled event timeline, anomaly confidence scores    | 🔲 Next |
| 3      | 7–9   | Time series forecasting      | 72hr forecast with prediction intervals & metrics     | 🔲 Upcoming |
| 4      | 10–12 | Risk scoring model           | Trained classifier with full evaluation suite         | 🔲 Upcoming |
| 5      | 13–15 | Live risk dashboard          | End-to-end interactive app with all layers integrated | 🔲 Upcoming |

---

### Sprint 1 — Days 1–3: Data Pipeline & Quality
**Goal:** Establish a trustworthy, repeatable data feed before any modelling begins.

**Tasks:**
- Pull hourly historical weather from Open-Meteo archive API (90 days)
- Pull 7-day forecast from Open-Meteo forecast API
- Pull marine conditions (wave height, period) from Open-Meteo marine API
- Pull buoy observations from NOAA NDBC where available
- Run automated quality checks: completeness, temporal gaps, freshness, physical bounds
- Build Streamlit dashboard showing raw data + quality scorecard

**Files produced:**
- `src/fetch.py` — all data fetching logic
- `src/quality.py` — quality checks and scoring
- `dashboard/app.py` — Sprint 1 dashboard
- `notebooks/sprint1_pipeline.py` — exploratory notebook

**Quality metrics to track:**
- Completeness score per column (target: >90%)
- Largest temporal gap (target: <3 hours)
- Data freshness (target: <2 hours old)
- Out-of-bounds values per variable (target: 0)

**Demo:** Streamlit app showing live port weather, quality scorecard, and completeness heatmap.

---

### Sprint 2 — Days 4–6: EDA & Anomaly Detection
**Goal:** Understand the data distribution and create pseudo-labels for "disruption-grade" weather events to use as training signal in Sprint 4.

**Tasks:**
- Compute rolling statistics: 24h wind mean, 6h pressure drop, precipitation intensity
- Run STL decomposition on wind speed to separate trend, seasonal, and residual components
- Apply Isolation Forest to detect multivariate anomalies
- Cross-reference detected anomalies with known historical events (manual spot-check)
- Build an annotated event timeline

**Files produced:**
- `src/features.py` — feature engineering functions
- `notebooks/sprint2_eda.py` — EDA and anomaly analysis
- New dashboard tab: anomaly timeline

**Quality metrics to track:**
- Anomaly detection precision (spot-checked against 5+ known events)
- Feature correlation matrix — flag any >0.95 correlated pairs (redundant)
- Distribution shift check: compare recent 7 days vs historical baseline

**Demo:** Timeline of historical weather anomalies with confidence scores, plus feature correlation heatmap.

---

### Sprint 3 — Days 7–9: Time Series Forecasting
**Goal:** Forecast key weather variables 48–72 hours ahead with proper uncertainty quantification.

**Tasks:**
- Forecast `wind_speed_10m` and `pressure_msl` using Prophet (primary) and SARIMA (baseline)
- Generate prediction intervals (80% and 95% confidence bands)
- Evaluate against a naive baseline (last-value persistence)
- Backtest on held-out final 14 days of historical data

**Files produced:**
- `src/forecast.py` — forecasting logic
- `notebooks/sprint3_forecast.py` — model training and evaluation
- New dashboard tab: 72-hour forecast with uncertainty bands

**Performance metrics to track:**

| Metric | Variable | Acceptable threshold |
|--------|----------|----------------------|
| MAE    | Wind speed | < 2.0 m/s |
| RMSE   | Wind speed | < 3.0 m/s |
| MAPE   | Pressure   | < 1.5% |
| Coverage | 80% PI   | 75%–85% (well-calibrated) |

Beat the naive baseline on all metrics or explain why not.

**Demo:** Side-by-side forecast vs actuals chart with interval coverage statistics.

---

### Sprint 4 — Days 10–12: Risk Scoring Model
**Goal:** Train a classifier that outputs a calibrated disruption probability score.

**Tasks:**
- Define disruption label: wind > 15 m/s OR wave height > 3m OR pressure drop > 10 hPa in 6h (adjustable)
- Engineer lag features: 6h, 12h, 24h rolling stats
- Train Random Forest (primary) and XGBoost (comparison)
- Calibrate probabilities using Platt scaling or isotonic regression
- Backtest on held-out data

**Files produced:**
- `src/model.py` — training, calibration, prediction
- `src/metrics.py` — performance and acceptability metrics
- `notebooks/sprint4_model.py` — model training and evaluation
- New dashboard tab: risk score with confidence band

**Performance metrics to track:**

| Metric | Target | Notes |
|--------|--------|-------|
| ROC-AUC | > 0.75 | Overall discriminative ability |
| Precision @ 70% threshold | > 0.65 | Minimise false alarms |
| Recall @ 70% threshold | > 0.60 | Don't miss real disruptions |
| Brier Score | < 0.15 | Probabilistic accuracy |
| Calibration error (ECE) | < 0.05 | Is "70% risk" actually 70%? |

**Result acceptability rule:** Model is only deployed to dashboard if ROC-AUC > 0.75 AND ECE < 0.05. Otherwise, dashboard displays a warning and falls back to the anomaly detector from Sprint 2.

**Demo:** ROC curve, precision-recall curve, calibration curve, and feature importance chart.

---

### Sprint 5 — Days 13–15: Live Risk Dashboard
**Goal:** Wire all layers into a single coherent, explainable app.

**Tasks:**
- Integrate data pipeline → feature engineering → forecast → risk score into one flow
- Add data quality indicator to every panel (degraded data = degraded confidence)
- Add model confidence band to risk score display
- Add "what's driving this score" feature contribution breakdown
- Final review: can a non-technical user understand the output?

**Files produced:**
- `dashboard/app.py` — final integrated dashboard (all tabs)
- `notebooks/sprint5_review.py` — end-to-end pipeline smoke test

**Acceptability criteria for final dashboard:**
- Data quality score > 80% — otherwise show "data degraded" banner
- Forecast RMSE within acceptable threshold — otherwise show stale model warning
- Risk score calibration ECE < 0.05 — otherwise suppress score and show raw anomaly signal only
- Full pipeline runs end-to-end in < 60 seconds

**Demo:** Live, interactive Streamlit app showing current conditions, 72-hour forecast, risk score with confidence, and data quality status — all for a real port, with real data.

---

## Metrics Philosophy

This project treats metrics as **first-class citizens**, not an afterthought. Three layers of metrics are tracked throughout:

**1. Data quality metrics** — checked before any modelling. If the data is bad, all downstream results are flagged as unreliable.

**2. Model performance metrics** — tracked during development with a held-out test set. No model moves to the dashboard without meeting its acceptance threshold.

**3. Result acceptability metrics** — the dashboard itself surfaces calibration and confidence. A high-confidence score is only shown when the model has earned the right to show it.

---

## Data Sources

| Source | What it provides | API key needed? | Cost |
|--------|-----------------|-----------------|------|
| Open-Meteo Archive | Hourly historical weather (2 years) | No | Free |
| Open-Meteo Forecast | 7-day hourly forecast | No | Free |
| Open-Meteo Marine | Wave height, period | No | Free |
| NOAA NDBC | Real buoy observations near port | No | Free |

---

## Project Structure

```
port-risk/
├── data/                        # Raw and processed data (parquet, CSV)
├── notebooks/
│   ├── sprint1_pipeline.py      # Data fetch & quality exploration
│   ├── sprint2_eda.py           # EDA & anomaly detection
│   ├── sprint3_forecast.py      # Time series models
│   ├── sprint4_model.py         # Risk scoring model
│   └── sprint5_review.py        # End-to-end pipeline test
├── src/
│   ├── fetch.py                 # Data fetching (Open-Meteo, NOAA NDBC)
│   ├── quality.py               # Data quality checks and metrics
│   ├── features.py              # Feature engineering
│   ├── forecast.py              # Time series forecasting
│   ├── model.py                 # Risk scoring model
│   └── metrics.py               # Performance & acceptability metrics
├── dashboard/
│   └── app.py                   # Streamlit dashboard (all sprints)
├── requirements.txt
└── README.md
```

---

## Ports Available

| Port      | Lat     | Lon      | NOAA Buoy | Marine data |
|-----------|---------|----------|-----------|-------------|
| Rotterdam | 51.9500 | 4.1400   | 62081     | ✅ |
| Houston   | 29.7500 | -95.3500 | 42035     | ✅ |
| Singapore | 1.2900  | 103.8500 | —         | ✅ |

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py

# Or run the Sprint 1 notebook directly
python sprint1_pipeline.py
```

---

## Deploying to Streamlit Cloud

1. Connect the repo and set **Main file path** to `app.py`.
2. In **Advanced settings**, set **Python version** to **3.11** or **3.12** (not 3.13). This avoids long installs from missing wheels (e.g. scikit-learn building from source) and keeps `numpy<2` so dependency resolution stays consistent.
3. Use the repo `requirements.txt` as-is; do not add `great-expectations` to it (the app does not use it and it constrains numpy).

---

## Extension Ideas

Once the core project is complete, natural next steps include:

- **Add AIS vessel traffic data** — correlate vessel counts with disruption events for richer features
- **Multi-port model** — train a single model across all three ports to improve generalisation
- **Alerting** — send a notification when risk score exceeds a threshold
- **Longer forecast horizon** — extend from 72 hours to 5–7 days using ensemble methods
- **Containerise** — wrap in Docker for reproducible deployment
- **MLflow experiment tracking** — log all model runs for proper comparison
