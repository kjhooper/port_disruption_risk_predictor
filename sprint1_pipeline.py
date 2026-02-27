# Sprint 1 — Data Pipeline & Quality
# Run this notebook cell by cell to explore and validate your data

# ── Cell 1: Setup ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, ".")

import pandas as pd
import matplotlib.pyplot as plt
from fetch import update_or_fetch, PORTS
from quality import run_all_checks, quality_summary_df
from features import compute_all_features

print("Ports available:", list(PORTS.keys()))

# ── Cell 2: Fetch data ────────────────────────────────────────────────────────
# Change port to "houston" or "singapore" if preferred
data = update_or_fetch("rotterdam", save_dir="data")
hist = data["historical_wide"]
print(hist.shape, hist.columns.tolist())

# ── Cell 3: Quick look + feature engineering ──────────────────────────────────
print(hist.describe())
print("\nNull counts:\n", hist.isna().sum())

# Compute derived features
features_df = compute_all_features(hist, "rotterdam")
print("\nFeature columns added:")
print([c for c in features_df.columns if c not in hist.columns])

# ── Cell 4: Run quality checks ────────────────────────────────────────────────
report = run_all_checks(hist)
print("Overall score:", report["overall_score"])
print("Overall status:", report["overall_status"])

summary = quality_summary_df(report)
print(summary.to_string(index=False))

# ── Cell 5: Basic plots ───────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

hist["wind_speed_10m"].plot(ax=axes[0], title="Wind Speed (m/s)", color="teal")
hist["precipitation"].plot(ax=axes[1], title="Precipitation (mm/hr)", color="steelblue")
hist["pressure_msl"].plot(ax=axes[2], title="Pressure (hPa)", color="orange")

if "fog_risk_score" in features_df.columns:
    features_df["fog_risk_score"].plot(ax=axes[3], title="Fog Risk Score (0–1)", color="purple")
else:
    axes[3].set_title("Fog Risk Score (unavailable — missing temp/dewpoint)")

for ax in axes:
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("data/sprint1_overview.png", dpi=120)
plt.show()
print("Plot saved.")
