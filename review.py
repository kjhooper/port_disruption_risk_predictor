"""
review.py — Data Quality & Feature Engineering Review page.

In-depth statistical review: feature glossary, missing-data MNAR analysis,
IsolationForest anomaly detection, STL decomposition, distribution shift,
cross-correlation lag analysis, and ground-truth disruption study.

Run with: conda run -n personal streamlit run review.py
"""

import calendar
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as ps
from datetime import datetime, timedelta

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind, ks_2samp, pointbiserialr, chi2_contingency
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf as sm_acf, pacf as sm_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

from labels import make_weather_code_label, make_binary_label, GROUP_ORDER, GROUP_DESCRIPTIONS
from metrics import cohen_d
from fetch import (
    PORTS, zone_points, update_or_fetch, fetch_openmeteo_historical,
    OPENMETEO_VARIABLES, OPENMETEO_MARINE_VARIABLES, AIR_QUALITY_VARIABLES,
)
from features import compute_all_features
from quality import run_all_checks, quality_summary_df

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Feature Review",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────

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
    .status-ok   { color: #00d4aa; font-weight: 600; }
    .status-warn { color: #f5a623; font-weight: 600; }
    .status-fail { color: #e84545; font-weight: 600; }

    .stSelectbox label, .stSlider label { font-family: 'Space Mono', monospace; font-size: 0.8rem; }
    [data-testid="stSidebar"] { background: #0a1118; border-right: 1px solid #1e3a5f; }
</style>
""", unsafe_allow_html=True)

# ── Module-level helpers ──────────────────────────────────────────────────────

COLOR_CYCLE = ["#00d4aa", "#4a9eff", "#f5a623", "#e84545", "#a855f7", "#f97316"]

_ROLLING_PATTERN = re.compile(r"_(mean|max)_\d+h$")
_ROLLING_EXTRAS  = {"pressure_drop_6h", "humidity_mean_6h"}
_RAW_VARS        = set(OPENMETEO_VARIABLES + OPENMETEO_MARINE_VARIABLES + AIR_QUALITY_VARIABLES)


def classify_columns(df: pd.DataFrame, port: str) -> dict:
    """Partition all DataFrame columns into four disjoint groups."""
    zone_prefixes = tuple(f"{z['prefix']}_" for z in zone_points(port))
    groups: dict = {"zone": [], "rolling": [], "raw": [], "computed": []}

    for col in df.columns:
        if zone_prefixes and col.startswith(zone_prefixes):
            groups["zone"].append(col)
        elif _ROLLING_PATTERN.search(col) or col in _ROLLING_EXTRAS:
            groups["rolling"].append(col)
        elif col in _RAW_VARS:
            groups["raw"].append(col)
        else:
            groups["computed"].append(col)

    return groups


def make_layout(height: int = 400, title: str = None, **overrides) -> dict:
    """Shared Plotly layout dict for all charts."""
    layout = dict(
        template="plotly_dark",
        paper_bgcolor="#0a1118",
        plot_bgcolor="#0f1923",
        height=height,
        margin=dict(l=40, r=20, t=40 if title else 20, b=40),
        legend=dict(orientation="h", y=-0.15),
        font=dict(family="Inter, sans-serif"),
    )
    if title:
        layout["title"] = dict(text=title, font=dict(family="Space Mono, monospace", size=14))
    layout.update(overrides)
    return layout


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Computing features...")
def load_features(port: str, days_back: int) -> pd.DataFrame:
    wide_path = Path("data") / f"{port}_historical_wide.parquet"
    if wide_path.exists():
        cutoff = datetime.utcnow() - timedelta(days=days_back)
        df = pd.read_parquet(wide_path)
        df = df[df.index >= cutoff]
    else:
        df = fetch_openmeteo_historical(port, days_back=days_back)
    return compute_all_features(df, port)


# Weather event label: WMO-based (see labels.py)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 Feature Review")
    st.markdown("*Sprint 2 — Statistical Review*")
    st.divider()

    port = st.selectbox(
        "SELECT PORT",
        options=list(PORTS.keys()),
        format_func=lambda k: PORTS[k]["label"],
    )

    days_back = st.slider("HISTORY (days)", min_value=7, max_value=365, value=180, step=7)

    refresh_btn = st.button("↻ Refresh Data", width="stretch", type="primary")

    st.divider()
    st.markdown(f"**Port:** {PORTS[port]['label']}")
    st.markdown(f"**Lat/Lon:** {PORTS[port]['lat']}°, {PORTS[port]['lon']}°")
    st.markdown(f"**Last run:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")

if refresh_btn:
    with st.spinner("Updating data..."):
        update_or_fetch(port, save_dir="data")
    st.cache_data.clear()

# ── Load data ─────────────────────────────────────────────────────────────────

feat_df = load_features(port, days_back)

if feat_df.empty:
    st.error("No data available. Use the Refresh button to fetch data.")
    st.stop()

col_groups   = classify_columns(feat_df, port)
col_to_group = {col: grp for grp, cols in col_groups.items() for col in cols}
numeric_cols = feat_df.select_dtypes(include="number").columns.tolist()

# WMO-based event label for header display
_disrupt_default = make_binary_label(feat_df)
_n_events        = int(_disrupt_default.sum())
_event_rate      = _n_events / max(len(_disrupt_default), 1) * 100

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(f"# {PORTS[port]['label']} — Feature Review")
st.markdown(f"*{days_back} days · {len(feat_df):,} rows · {len(feat_df.columns)} columns*")

hc1, hc2, hc3 = st.columns(3)
hc1.metric("Weather event hours (WMO code > 3)", f"{_n_events:,}")
hc2.metric("Event rate", f"{_event_rate:.1f}%")
hc3.metric("Columns", len(feat_df.columns))
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_glossary, tab_quality, tab_anomaly, tab_stl, tab_shift, tab_xcorr, tab_gt, tab_acf, tab_season = st.tabs([
    "📖 Feature Glossary",
    "🔍 Data Quality",
    "🚨 Anomaly Detection",
    "📉 STL Decomposition",
    "📊 Distribution Shift",
    "🔗 Cross-correlation Lags",
    "🎯 Ground Truth",
    "📈 ACF / PACF",
    "🌡 Seasonality",
])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Feature Glossary
# ══════════════════════════════════════════════════════════════════════════════

with tab_glossary:
    st.markdown("### Feature Glossary")
    st.markdown("Formula, physical meaning, and disruption relevance for every feature group.")

    # ── Raw Open-Meteo variables ──────────────────────────────────────────────
    with st.expander("🌐 Raw Variables (Open-Meteo / NOAA)", expanded=False):
        raw_meta = [
            ("wind_speed_10m",       "m/s",   "0–80",    "HIGH",   "Primary disruption driver — threshold 15 m/s"),
            ("wind_gusts_10m",       "m/s",   "0–100",   "HIGH",   "Peak instantaneous wind — sudden hazard"),
            ("wind_direction_10m",   "°",     "0–360",   "MEDIUM", "Bearing for onshore/cross decomposition"),
            ("precipitation",        "mm/hr", "0–200",   "MEDIUM", "Heavy rain reduces visibility and adds load"),
            ("pressure_msl",         "hPa",   "870–1084","HIGH",   "Falling pressure = storm approaching"),
            ("temperature_2m",       "°C",    "-50–60",  "LOW",    "Combined with dew point for fog risk"),
            ("dew_point_2m",         "°C",    "-60–40",  "MEDIUM", "Near temp → fog likely"),
            ("relative_humidity_2m", "%",     "0–100",   "MEDIUM", "High humidity supports fog/precipitation"),
            ("cloud_cover",          "%",     "0–100",   "LOW",    "Visibility proxy"),
            ("visibility",           "m",     "0–24 000","HIGH",   "Direct safety constraint"),
            ("weather_code",         "WMO",   "0–99",    "MEDIUM", "Coded condition (storm, fog, etc.)"),
            ("cape",                 "J/kg",  "0–10 000","HIGH",   "Convective potential — thunderstorm fuel"),
            ("lifted_index",         "K",     "-20–20",  "HIGH",   "Negative = unstable; -4 or below = severe"),
            ("wave_height",          "m",     "0–20",    "HIGH",   "Threshold 3m triggers disruption label"),
            ("wave_period",          "s",     "0–30",    "MEDIUM", "Long period = swell energy"),
            ("wind_wave_height",     "m",     "0–20",    "HIGH",   "Wind-driven wave component"),
            ("dust",                 "μg/m³", "0–5 000", "LOW",    "Visibility reduction in arid corridors"),
        ]
        raw_df = pd.DataFrame(raw_meta, columns=["Variable", "Units", "Typical range", "Disruption relevance", "Notes"])
        st.dataframe(raw_df, hide_index=True, use_container_width=True)

    # ── Directional wind ──────────────────────────────────────────────────────
    with st.expander("🧭 Computed — Directional Wind", expanded=False):
        st.markdown("""
| Feature | Formula | Physical meaning | Risk interpretation |
|---|---|---|---|
| `onshore_wind` | `wind_speed × cos(wind_dir − sea_bearing)` | Component pushing weather from sea toward port | Positive = inbound weather system |
| `cross_wind` | `wind_speed × sin(wind_dir − sea_bearing)` | Component parallel to the coastline | High = beam-on exposure to vessels |
| `wind_onshore_flag` | `1 if onshore_wind > 0` | Binary indicator of inbound wind | Feature for classifier |

*`sea_bearing`* is the compass direction from which weather reaches the port (e.g. 270° for Rotterdam = North Sea to the west).
        """)

    # ── Fog risk ──────────────────────────────────────────────────────────────
    with st.expander("🌫 Computed — Fog Risk", expanded=False):
        st.markdown("""
| Feature | Formula | Physical meaning | Risk interpretation |
|---|---|---|---|
| `td_spread` | `temperature_2m − dew_point_2m` | Dew-point depression (°C) | <2°C = fog likely, <0.5°C = dense fog |
| `fog_risk_score` | `1 − clip(td_spread, 0, 5) / 5` | Continuous 0–1 fog probability | 0 = clear, 1 = dense fog |
| `fog_flag` | `1 if td_spread < 2°C` | Binary fog indicator | Used as event label proxy |
        """)

    # ── Zone spatial features ─────────────────────────────────────────────────
    with st.expander("🗺 Computed — Zone Spatial Features (per prefix)", expanded=False):
        st.markdown("""
Zone prefixes follow the pattern `z150`, `z300` (distance in km), or `z150b45` (distance + bearing) for ports with multiple approach corridors (e.g. Singapore).

| Feature | Formula | Physical meaning | Risk interpretation |
|---|---|---|---|
| `{prefix}_pressure_gradient` | `port_pressure − zone_pressure` | Pressure difference port vs upstream | Positive = low pressure offshore = storm inbound |
| `{prefix}_cape_excess` | `max(zone_CAPE − port_CAPE, 0)` | Convective energy surplus upstream | High = storm fuel building in approach corridor |
| `{prefix}_wind_delta` | `zone_onshore_wind − port_onshore_wind` | Wind acceleration from zone to port | Positive = wind strengthening toward port |
| `{prefix}_onshore_wind` | `zone_wind × cos(zone_dir − bearing)` | Zone onshore component | Measures whether upstream wind is heading portward |
        """)

    # ── Storm Approach Index ──────────────────────────────────────────────────
    with st.expander("⚡ Computed — Storm Approach Index (0–1)", expanded=False):
        st.markdown("""
Composite signal for a storm being pushed from the sea toward the port.
Mean of up to **5 normalised components** (only those with available data are included):

| Component | Formula | Source |
|---|---|---|
| `cape_score` | `clip(CAPE, 0, 3000) / 3000` | Port-level CAPE |
| `li_score` | `(clip(LI, −10, 6) × −1 + 6) / 16` | Lifted Index — negative = unstable |
| `onshore_score` | `clip(onshore_wind, 0, 30) / 30` | Port onshore wind fraction |
| `upstream_cape` | `clip(farthest_zone_cape_excess, 0, 2000) / 2000` | Storm energy in far zone |
| `upstream_pressure` | `clip(farthest_zone_pressure_gradient, 0, 20) / 20` | Pressure draw from far zone |

**Score ≥ 0.5 = elevated concern. Score ≥ 0.75 = high risk.**
        """)

    # ── Rolling stats ─────────────────────────────────────────────────────────
    with st.expander("📈 Computed — Rolling Statistics", expanded=False):
        rolling_meta = []
        roll_base_cols = ["wind_speed_10m", "onshore_wind", "precipitation", "cape", "storm_approach_index"]
        for col in roll_base_cols:
            for h in [3, 6, 12, 24]:
                rolling_meta.append((f"{col}_mean_{h}h", f"{h}h", f"Smoothed trend in {col} over {h}h — filters out noise"))
                rolling_meta.append((f"{col}_max_{h}h",  f"{h}h", f"Peak {col} in last {h}h — catches brief extremes"))
        rolling_meta.append(("pressure_drop_6h", "6h",  "6-hour pressure change — negative = rapidly falling (storm signal)"))
        rolling_meta.append(("humidity_mean_6h", "6h",  "Smoothed humidity — sustained high humidity precedes fog"))
        roll_df = pd.DataFrame(rolling_meta, columns=["Column", "Window", "What it captures"])
        st.dataframe(roll_df, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Data Quality & Missing Data Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab_quality:
    st.markdown("### Data Quality & Missing Data Analysis")

    # ── Section A: Quality scorecard ──────────────────────────────────────────
    st.markdown("#### Section A — Quality Scorecard")
    report = run_all_checks(feat_df)

    score_color = {"ok": "#00d4aa", "warn": "#f5a623", "fail": "#e84545"}.get(report["overall_status"], "white")
    st.markdown(
        f"**Overall quality score:** "
        f"<span style='color:{score_color}; font-size:1.3em; font-weight:700'>"
        f"{report['overall_score']:.0%}</span>",
        unsafe_allow_html=True,
    )

    qc1, qc2, qc3, qc4 = st.columns(4)
    check_cols = [qc1, qc2, qc3, qc4]
    status_emoji = {"ok": "✅", "warn": "⚠️", "fail": "❌", "skip": "⏭️"}
    for col_widget, (check_name, check_data) in zip(check_cols, report["checks"].items()):
        emoji = status_emoji.get(check_data["status"], "")
        detail = quality_summary_df(report)[quality_summary_df(report)["Check"] == check_name]["Detail"].values
        col_widget.metric(
            f"{emoji} {check_name.replace('_', ' ').title()}",
            check_data["status"].upper(),
            detail[0] if len(detail) else "",
        )

    st.divider()

    # ── Section B: Missing data T-tests (MNAR analysis) ───────────────────────
    st.markdown("#### Section B — Missing Data MNAR Analysis")
    st.markdown(
        "For each column with missing values, tests whether its missingness correlates with "
        "other variables using Welch's t-test. Bonferroni correction applied."
    )

    with st.spinner("Running t-tests for MNAR analysis..."):
        num_df = feat_df.select_dtypes(include="number")
        missing_cols = [c for c in num_df.columns if num_df[c].isna().any()]
        other_cols   = num_df.columns.tolist()

        mnar_rows   = []
        all_pvals   = {}   # (missing_col, other_col) → p_value

        for mc in missing_cols:
            is_missing = num_df[mc].isna()
            if is_missing.sum() < 5 or (~is_missing).sum() < 5:
                continue
            pct_missing = is_missing.mean() * 100
            sig_assocs  = 0
            best_p      = 1.0
            best_other  = ""
            for oc in other_cols:
                if oc == mc:
                    continue
                a = num_df.loc[~is_missing, oc].dropna()
                b = num_df.loc[is_missing,  oc].dropna()
                if len(a) < 5 or len(b) < 5:
                    continue
                try:
                    _, p = ttest_ind(a, b, equal_var=False)
                    all_pvals[(mc, oc)] = p
                    if p < best_p:
                        best_p     = p
                        best_other = oc
                except Exception:
                    pass

            n_tests = len([k for k in all_pvals if k[0] == mc])
            p_thresh = 0.05 / max(n_tests, 1)
            sig_assocs = sum(1 for k, v in all_pvals.items() if k[0] == mc and v < p_thresh)

            mnar_rows.append({
                "missing_column":            mc,
                "pct_missing":               round(pct_missing, 2),
                "n_significant_associations": sig_assocs,
                "top_associated_column":     best_other,
                "p_value":                   round(best_p, 6),
            })

    if mnar_rows:
        mnar_df = (
            pd.DataFrame(mnar_rows)
            .sort_values("n_significant_associations", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(
            mnar_df.style.format({"pct_missing": "{:.1f}%", "p_value": "{:.4f}"}),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Columns with many significant associations are **Missing Not At Random (MNAR)** — "
            "their absence may itself carry predictive signal and should not be blindly imputed. "
            "Columns with no significant associations are likely **MCAR** (safe to mean/median impute)."
        )

        if st.checkbox("Show full p-value heatmap (−log10 scale)"):
            if all_pvals:
                mc_list = sorted({k[0] for k in all_pvals})
                oc_list = sorted({k[1] for k in all_pvals})
                z_matrix = np.full((len(mc_list), len(oc_list)), np.nan)
                for (mc, oc), p in all_pvals.items():
                    if mc in mc_list and oc in oc_list:
                        i = mc_list.index(mc)
                        j = oc_list.index(oc)
                        z_matrix[i, j] = -np.log10(max(p, 1e-300))

                fig_heat = go.Figure(go.Heatmap(
                    z=z_matrix,
                    x=oc_list,
                    y=mc_list,
                    colorscale="YlOrRd",
                    colorbar=dict(title="−log10(p)"),
                ))
                fig_heat.update_layout(**make_layout(
                    height=max(300, len(mc_list) * 25),
                    title="MNAR p-value Heatmap (missing col × other col)",
                    xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
                    yaxis=dict(tickfont=dict(size=8)),
                ))
                st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No columns with sufficient missing data for MNAR analysis (need ≥5 missing rows).")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Anomaly Detection
# ══════════════════════════════════════════════════════════════════════════════

with tab_anomaly:
    st.markdown("### Anomaly Detection — IsolationForest")

    _numeric_set = set(numeric_cols)
    _default_anomaly_feats = [
        c for c in ["wind_speed_10m", "pressure_drop_6h", "onshore_wind",
                     "fog_risk_score", "storm_approach_index", "cape"]
        if c in _numeric_set
    ]

    ac1, ac2 = st.columns([1, 2])
    with ac1:
        contamination = st.slider(
            "Contamination (%)", min_value=1, max_value=20, value=5, step=1
        ) / 100.0
    with ac2:
        anomaly_feats = st.multiselect(
            "Features for IsolationForest",
            options=numeric_cols,
            default=_default_anomaly_feats,
        )

    if len(anomaly_feats) < 2:
        st.warning("Select at least 2 features to run IsolationForest.")
    else:
        with st.spinner("Running IsolationForest..."):
            anom_sub = feat_df[anomaly_feats].dropna()
            if len(anom_sub) < 20:
                st.error("Not enough rows after dropping NaN for anomaly detection.")
            else:
                scaler = StandardScaler()
                X = scaler.fit_transform(anom_sub)
                clf = IsolationForest(contamination=contamination, random_state=42)
                clf.fit(X)
                preds  = clf.predict(X)
                scores = clf.decision_function(X)  # higher = more normal

                anom_idx = anom_sub.index[preds == -1]
                is_anomaly = pd.Series(False, index=feat_df.index)
                is_anomaly[anom_idx] = True
                anomaly_score_full = pd.Series(np.nan, index=feat_df.index)
                anomaly_score_full[anom_sub.index] = scores

                # Metrics
                n_anomalies   = int(is_anomaly.sum())
                pct_anomalies = n_anomalies / max(len(anom_sub), 1) * 100
                disrupt_base  = make_binary_label(feat_df)
                overlap       = int((is_anomaly & disrupt_base.astype(bool)).sum())
                precision_approx = overlap / max(n_anomalies, 1) * 100

                am1, am2, am3 = st.columns(3)
                am1.metric("Anomalies detected", f"{n_anomalies:,}")
                am2.metric("% of rows", f"{pct_anomalies:.1f}%")
                am3.metric("Overlap with weather event label", f"{precision_approx:.1f}%")

                # Chart
                fig_anom = go.Figure()

                if "wind_speed_10m" in feat_df.columns:
                    fig_anom.add_trace(go.Scatter(
                        x=feat_df.index,
                        y=feat_df["wind_speed_10m"],
                        name="wind_speed_10m",
                        line=dict(color="#00d4aa", width=1.2),
                        yaxis="y1",
                    ))
                    # Anomaly markers
                    anom_wind = feat_df["wind_speed_10m"].reindex(anom_idx)
                    fig_anom.add_trace(go.Scatter(
                        x=anom_idx,
                        y=anom_wind,
                        mode="markers",
                        name="Anomaly",
                        marker=dict(color="#e84545", symbol="triangle-up", size=8),
                        yaxis="y1",
                    ))

                # Anomaly score on secondary y-axis
                fig_anom.add_trace(go.Scatter(
                    x=feat_df.index,
                    y=anomaly_score_full,
                    name="Anomaly score",
                    line=dict(color="grey", width=1, dash="dash"),
                    yaxis="y2",
                    opacity=0.6,
                ))

                fig_anom.update_layout(**make_layout(
                    height=420,
                    xaxis_title="Date (UTC)",
                    yaxis=dict(title="wind_speed_10m (m/s)"),
                    yaxis2=dict(title="Score (higher=normal)", overlaying="y", side="right",
                                showgrid=False),
                    legend=dict(orientation="h", y=-0.18),
                ))
                st.plotly_chart(fig_anom, use_container_width=True)

                # Top anomaly events table
                st.markdown("#### Top 20 Anomaly Events")
                top20_idx = anom_sub.index[np.argsort(scores)[:20]]
                top20_df  = feat_df.loc[top20_idx, anomaly_feats].copy()
                top20_df.insert(0, "anomaly_score", scores[np.argsort(scores)[:20]])
                top20_df.index.name = "timestamp"
                st.dataframe(top20_df.reset_index().style.format("{:.3f}", subset=anomaly_feats + ["anomaly_score"]),
                             hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — STL Decomposition
# ══════════════════════════════════════════════════════════════════════════════

with tab_stl:
    st.markdown("### STL Decomposition")
    st.markdown("Separate trend, seasonal, and residual components for any variable.")

    non_zone_non_rolling = [
        c for c in numeric_cols
        if col_to_group.get(c) not in ("zone", "rolling")
    ]
    default_var = "wind_speed_10m" if "wind_speed_10m" in non_zone_non_rolling else (non_zone_non_rolling[0] if non_zone_non_rolling else None)

    sc1, sc2 = st.columns([2, 1])
    with sc1:
        stl_var = st.selectbox(
            "Variable",
            options=non_zone_non_rolling if non_zone_non_rolling else numeric_cols,
            index=non_zone_non_rolling.index(default_var) if default_var in non_zone_non_rolling else 0,
        )
    with sc2:
        stl_period_days = st.slider("Seasonal period (days)", min_value=1, max_value=7, value=1)

    stl_period_hours = stl_period_days * 24

    if stl_var and stl_var in feat_df.columns:
        series = feat_df[stl_var].interpolate(method="linear", limit=6)
        series = series.dropna()

        if len(series) < stl_period_hours * 2:
            st.warning(f"Not enough data for period={stl_period_hours}h. Reduce period or increase history.")
        else:
            with st.spinner("Fitting STL..."):
                try:
                    stl_result = STL(series, period=stl_period_hours, robust=True).fit()

                    trend    = stl_result.trend
                    seasonal = stl_result.seasonal
                    resid    = stl_result.resid

                    fig_stl = ps.make_subplots(
                        rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=[
                            f"Original: {stl_var}",
                            "Trend",
                            f"Seasonal (period={stl_period_hours}h)",
                            "Residual",
                        ],
                        vertical_spacing=0.06,
                    )
                    for row_idx, (name, data, color) in enumerate([
                        ("Original", series,   COLOR_CYCLE[0]),
                        ("Trend",    trend,    COLOR_CYCLE[1]),
                        ("Seasonal", seasonal, COLOR_CYCLE[2]),
                        ("Residual", resid,    COLOR_CYCLE[3]),
                    ], start=1):
                        fig_stl.add_trace(
                            go.Scatter(x=data.index, y=data, name=name,
                                       line=dict(color=color, width=1.2)),
                            row=row_idx, col=1,
                        )

                    fig_stl.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0a1118",
                        plot_bgcolor="#0f1923",
                        height=700,
                        margin=dict(l=40, r=20, t=40, b=40),
                        showlegend=False,
                        font=dict(family="Inter, sans-serif"),
                    )
                    st.plotly_chart(fig_stl, use_container_width=True)

                    # Metrics
                    sm1, sm2, sm3 = st.columns(3)
                    sm1.metric("Trend range (max − min)", f"{float(trend.max() - trend.min()):.3f}")
                    sm2.metric("Seasonal amplitude",       f"{float(seasonal.max() - seasonal.min()):.3f}")
                    sm3.metric("Residual std",             f"{float(resid.std()):.3f}")

                except Exception as e:
                    st.error(f"STL failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Distribution Shift
# ══════════════════════════════════════════════════════════════════════════════

with tab_shift:
    st.markdown("### Distribution Shift")
    st.markdown("Compare the recent window vs the historical baseline using Kolmogorov–Smirnov test.")

    _default_shift_cols = [
        c for c in ["wind_speed_10m", "pressure_msl", "fog_risk_score", "storm_approach_index"]
        if c in _numeric_set
    ]

    dsc1, dsc2 = st.columns([1, 2])
    with dsc1:
        recent_window = st.slider("Recent window (days)", min_value=3, max_value=30, value=7)
    with dsc2:
        shift_cols = st.multiselect(
            "Columns to test",
            options=numeric_cols,
            default=_default_shift_cols,
        )

    if not shift_cols:
        st.warning("Select at least one column.")
    else:
        cutoff_recent = feat_df.index.max() - pd.Timedelta(days=recent_window)
        recent_df  = feat_df[feat_df.index > cutoff_recent]
        baseline_df = feat_df[feat_df.index <= cutoff_recent]

        shift_rows = []
        for col in shift_cols:
            base = baseline_df[col].dropna()
            rec  = recent_df[col].dropna()
            if len(base) < 5 or len(rec) < 5:
                continue
            ks_stat, p_val = ks_2samp(base, rec)
            shift_rows.append({
                "column":        col,
                "baseline_mean": round(float(base.mean()), 4),
                "recent_mean":   round(float(rec.mean()), 4),
                "Δmean":         round(float(rec.mean() - base.mean()), 4),
                "ks_stat":       round(ks_stat, 4),
                "p_value":       round(p_val, 6),
                "shifted":       p_val < 0.05,
            })

        if shift_rows:
            shift_df = pd.DataFrame(shift_rows).sort_values("ks_stat", ascending=False)

            def _highlight_shifted(row):
                color = "background-color: #3d2b00" if row["shifted"] else ""
                return [color] * len(row)

            styled_shift = (
                shift_df.style
                .apply(_highlight_shifted, axis=1)
                .format({
                    "baseline_mean": "{:.3f}", "recent_mean": "{:.3f}",
                    "Δmean": "{:.3f}", "ks_stat": "{:.4f}", "p_value": "{:.4f}",
                })
            )
            st.dataframe(styled_shift, hide_index=True, use_container_width=True)
            st.caption("Amber rows: p < 0.05 — distribution has shifted significantly in the recent window.")

            # Histogram overlay for most-shifted column
            most_shifted = shift_df.iloc[0]["column"]
            st.markdown(f"#### Distribution Overlay — `{most_shifted}` (most shifted)")

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=baseline_df[most_shifted].dropna(),
                name="Historical baseline",
                histnorm="probability density",
                marker_color="grey",
                opacity=0.3,
                nbinsx=50,
            ))
            fig_hist.add_trace(go.Histogram(
                x=recent_df[most_shifted].dropna(),
                name=f"Recent ({recent_window}d)",
                histnorm="probability density",
                marker_color="#00d4aa",
                opacity=0.6,
                nbinsx=50,
            ))
            fig_hist.update_layout(barmode="overlay", **make_layout(
                height=350,
                xaxis_title=most_shifted,
                yaxis_title="Density",
            ))
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Not enough data in both windows to run KS tests.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 6 — Cross-correlation Lags
# ══════════════════════════════════════════════════════════════════════════════

with tab_xcorr:
    st.markdown("### Cross-correlation Lags")
    st.markdown(
        "How far ahead do upstream zone variables lead port conditions? "
        "A peak at lag=N means the zone variable shifts N hours **before** the port responds."
    )

    # Zoneable vars: present at port level AND in at least one zone column
    zone_raw_vars: set = set()
    for z in zone_points(port):
        prefix = z["prefix"]
        for col in feat_df.columns:
            if col.startswith(f"{prefix}_"):
                base = col[len(f"{prefix}_"):]
                if base in feat_df.columns:
                    zone_raw_vars.add(base)
    zoneable_vars = sorted(zone_raw_vars)

    if not zoneable_vars:
        st.info("No zone columns available — ensure parquet contains zone-prefixed columns.")
    else:
        xc1, xc2 = st.columns([2, 1])
        with xc1:
            xcorr_var = st.selectbox(
                "Base variable (port-level)",
                options=zoneable_vars,
                index=zoneable_vars.index("pressure_msl") if "pressure_msl" in zoneable_vars else 0,
            )
        with xc2:
            max_lag = st.slider("Max lag (hours)", min_value=6, max_value=120, value=48, step=6)

        port_series = feat_df[xcorr_var].dropna()

        fig_xcorr = go.Figure()
        xcorr_results = []

        for i, z in enumerate(zone_points(port)):
            prefix    = z["prefix"]
            zone_col  = f"{prefix}_{xcorr_var}"
            if zone_col not in feat_df.columns:
                continue

            zone_series = feat_df[zone_col].dropna()
            common_idx  = port_series.index.intersection(zone_series.index)
            if len(common_idx) < max_lag + 10:
                continue

            ps_aligned = port_series.reindex(common_idx)
            zs_aligned = zone_series.reindex(common_idx)

            lags  = list(range(0, max_lag + 1))
            corrs = []
            for lag in lags:
                r = zs_aligned.shift(lag).corr(ps_aligned)
                corrs.append(r if not np.isnan(r) else 0.0)

            corrs_arr   = np.array(corrs)
            opt_idx     = int(np.nanargmax(np.abs(corrs_arr)))
            opt_lag     = lags[opt_idx]
            peak_corr   = corrs_arr[opt_idx]
            direction   = "Zone leads port" if opt_lag > 0 else "Simultaneous"

            xcorr_results.append({
                "zone":             prefix,
                "distance_km":      z["distance_km"],
                "optimal_lag_hours": opt_lag,
                "peak_correlation": round(float(peak_corr), 4),
                "direction":        direction,
            })

            fig_xcorr.add_trace(go.Scatter(
                x=lags,
                y=corrs,
                name=f"{prefix} ({z['distance_km']}km)",
                line=dict(color=COLOR_CYCLE[i % len(COLOR_CYCLE)], width=2),
            ))

        fig_xcorr.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.4)
        fig_xcorr.update_layout(**make_layout(
            height=380,
            title=f"Cross-correlation: zone vs port `{xcorr_var}`",
            xaxis_title="Lag (hours) — zone leads port →",
            yaxis_title="Pearson r",
        ))
        st.plotly_chart(fig_xcorr, use_container_width=True)

        if xcorr_results:
            xcorr_df = pd.DataFrame(xcorr_results).sort_values("distance_km")
            st.dataframe(xcorr_df, hide_index=True, use_container_width=True)

        st.caption(
            "A peak at lag=N means the zone variable tends to shift N hours *before* the port "
            "responds — this is the maximum predictive lead time available from that upstream point."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 7 — Ground Truth Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab_gt:
    st.markdown("### Ground Truth Analysis — WMO Weather Codes")
    st.markdown(
        "Observed weather codes from Open-Meteo are the ground-truth event labels. "
        "WMO codes tell us what the weather *actually was* at each hour."
    )

    if "weather_code" not in feat_df.columns:
        st.warning(
            "weather_code column not found in the loaded data. "
            "Use the Refresh button to re-fetch data with this variable included."
        )
        st.stop()

    wmo_labels   = make_weather_code_label(feat_df)
    binary_labels = make_binary_label(feat_df)
    n_events      = int(binary_labels.sum())
    event_rate    = n_events / max(len(binary_labels), 1) * 100
    n_groups      = int(wmo_labels.nunique())

    gm1, gm2, gm3 = st.columns(3)
    gm1.metric("Weather event hours", f"{n_events:,}")
    gm2.metric("Event rate",          f"{event_rate:.1f}%")
    gm3.metric("WMO groups present",  n_groups)

    st.divider()

    # ── Section A — WMO Group Distribution ───────────────────────────────────
    st.markdown("#### Section A — WMO Group Distribution")

    group_counts = (
        wmo_labels.value_counts()
        .reindex(GROUP_ORDER)
        .dropna()
    )

    fig_bar = go.Figure(go.Bar(
        x=group_counts.index.tolist(),
        y=group_counts.values,
        marker_color=COLOR_CYCLE[:len(group_counts)],
        text=[f"{int(v):,}" for v in group_counts.values],
        textposition="outside",
    ))
    fig_bar.update_layout(**make_layout(
        height=320,
        title="Hours in each WMO weather group (archive period)",
        xaxis_title="WMO group",
        yaxis_title="Hours",
    ))
    st.plotly_chart(fig_bar, use_container_width=True)

    desc_rows = [
        {
            "group":       g,
            "description": GROUP_DESCRIPTIONS.get(g, ""),
            "hours":       int(group_counts.get(g, 0)),
        }
        for g in GROUP_ORDER
        if g in group_counts.index
    ]
    st.dataframe(pd.DataFrame(desc_rows), hide_index=True, use_container_width=True)

    st.divider()

    # ── Section B — Feature–WMO Correlation Table ─────────────────────────────
    st.markdown("#### Section B — Feature–WMO Correlation")
    st.markdown(
        "Point-biserial r (binary: event vs clear) and Cohen's d "
        "(effect size: mean difference in standard deviation units) for each feature."
    )

    include_rolling_gt = st.checkbox("Include rolling features", value=False)
    analysis_cols = col_groups.get("raw", []) + col_groups.get("computed", [])
    if include_rolling_gt:
        analysis_cols += col_groups.get("rolling", [])
    analysis_cols = [c for c in analysis_cols if c in numeric_cols]

    event_mask = binary_labels == 1
    clear_mask = binary_labels == 0

    with st.spinner("Computing correlations..."):
        corr_rows = []
        for col in analysis_cols:
            feat_series = feat_df[col].fillna(feat_df[col].median())
            if feat_series.std() == 0:
                continue
            try:
                r_pb, p_pb = pointbiserialr(binary_labels, feat_series)
                d = cohen_d(feat_series[event_mask], feat_series[clear_mask])
                corr_rows.append({
                    "feature":         col,
                    "group":           col_to_group.get(col, "?"),
                    "r_pointbiserial": round(float(r_pb), 4),
                    "cohen_d":         round(float(d), 4) if not np.isnan(d) else None,
                    "p_value":         round(float(p_pb), 6),
                    "significant":     p_pb < 0.05,
                })
            except Exception:
                pass

    if corr_rows:
        corr_gt_df = (
            pd.DataFrame(corr_rows)
            .sort_values("r_pointbiserial", key=abs, ascending=False)
            .reset_index(drop=True)
        )

        def _style_significance(row):
            if row["significant"]:
                return ["color: #00d4aa"] * len(row)
            return ["opacity: 0.45"] * len(row)

        styled_corr = (
            corr_gt_df.style
            .apply(_style_significance, axis=1)
            .format(
                {"r_pointbiserial": "{:.4f}", "cohen_d": "{:.4f}", "p_value": "{:.4f}"},
                na_rep="—",
            )
        )
        st.dataframe(styled_corr, hide_index=True, use_container_width=True)
        st.caption("Teal = significant (p < 0.05). Sorted by |r_pointbiserial| descending.")

        st.divider()

        # ── Section C — Box Plot per WMO Group ────────────────────────────────
        st.markdown("#### Section C — Feature Distribution by WMO Group")
        st.markdown(
            "Box plots show how each feature is distributed within each WMO weather group. "
            "A shift across groups confirms the feature carries group-discriminating signal."
        )

        top10_sig = corr_gt_df[corr_gt_df["significant"]].head(10)["feature"].tolist()
        if not top10_sig:
            top10_sig = corr_gt_df.head(10)["feature"].tolist()

        sel_feat = st.selectbox("Feature to plot across WMO groups", options=top10_sig)

        if sel_feat and sel_feat in feat_df.columns:
            feat_s = feat_df[sel_feat].fillna(feat_df[sel_feat].median())
            plot_df = pd.DataFrame({"value": feat_s, "group": wmo_labels}).dropna()

            present_groups = [g for g in GROUP_ORDER if g in plot_df["group"].values]

            fig_box = go.Figure()
            for i, grp in enumerate(present_groups):
                subset = plot_df[plot_df["group"] == grp]["value"]
                fig_box.add_trace(go.Box(
                    y=subset,
                    name=grp,
                    marker_color=COLOR_CYCLE[i % len(COLOR_CYCLE)],
                    boxmean=True,
                ))
            fig_box.update_layout(**make_layout(
                height=400,
                title=f"{sel_feat} distribution by WMO weather group",
                xaxis_title="WMO group",
                yaxis_title=sel_feat,
                showlegend=False,
            ))
            st.plotly_chart(fig_box, use_container_width=True)

            # Group means table
            group_means = (
                plot_df.groupby("group")["value"]
                .agg(["mean", "median", "std", "count"])
                .reindex(present_groups)
                .round(4)
                .reset_index()
            )
            group_means.columns = ["group", "mean", "median", "std", "count"]
            st.dataframe(group_means, hide_index=True, use_container_width=True)

    else:
        st.info(
            "No correlation results — check that weather_code has non-zero event values "
            "(code > 3) in the loaded history window."
        )

    st.divider()

    # ── Section D — Label Drift Over Time ────────────────────────────────────
    st.markdown("#### Section D — Label Drift Over Time")
    st.markdown(
        "Monthly WMO group proportions — checks whether class balance is stable across the archive. "
        "A shift between the training period and test period biases M1/M2 evaluation numbers."
    )

    _drift_df = pd.DataFrame({
        "group": wmo_labels,
        "month": wmo_labels.index.to_period("M"),
    }).dropna()
    monthly_groups = (
        _drift_df.groupby("month")["group"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=GROUP_ORDER, fill_value=0)
    )
    monthly_groups.index = monthly_groups.index.to_timestamp()

    if len(monthly_groups) >= 2:
        fig_drift = go.Figure()
        for i, grp in enumerate(GROUP_ORDER):
            if grp not in monthly_groups.columns:
                continue
            fig_drift.add_trace(go.Scatter(
                x=monthly_groups.index,
                y=monthly_groups[grp],
                name=grp,
                stackgroup="one",
                mode="lines",
                line=dict(width=0.5, color=COLOR_CYCLE[i % len(COLOR_CYCLE)]),
                fillcolor=COLOR_CYCLE[i % len(COLOR_CYCLE)],
            ))
        fig_drift.update_layout(**make_layout(
            height=300,
            title="Monthly WMO group proportions (stacked)",
            xaxis_title="Month",
            yaxis=dict(title="Proportion", tickformat=".0%"),
        ))
        st.plotly_chart(fig_drift, use_container_width=True)

        # Chi-square: are train and test distributions the same?
        test_cutoff    = feat_df.index.max() - pd.Timedelta(days=365)
        train_counts   = wmo_labels[feat_df.index <= test_cutoff].value_counts()
        test_counts    = wmo_labels[feat_df.index > test_cutoff].value_counts()
        common_grps    = sorted(set(train_counts.index) & set(test_counts.index))

        if len(common_grps) >= 2:
            cont_table = np.array([
                [int(train_counts.get(g, 0)) for g in common_grps],
                [int(test_counts.get(g, 0))  for g in common_grps],
            ])
            chi2, p_chi, dof, _ = chi2_contingency(cont_table)
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("χ² statistic",    f"{chi2:.2f}")
            dc2.metric("p-value",          f"{p_chi:.4f}")
            dc3.metric("Drift detected",   "Yes ⚠" if p_chi < 0.05 else "No ✓")
            st.caption(
                "Chi-square test compares train vs test WMO group distributions. "
                "p < 0.05 means class balance has shifted — the time-based split may be biased."
            )
    else:
        st.info("Not enough monthly data points for drift analysis.")

    st.divider()

    # ── Section E — Event Duration & Inter-arrival Times ─────────────────────
    st.markdown("#### Section E — Event Duration & Inter-arrival Times")
    st.markdown(
        "How long do weather events last and how far apart are they? "
        "Long events mean consecutive training rows are correlated — "
        "standard CV metrics will be over-optimistic for M2."
    )

    run_id = (binary_labels != binary_labels.shift()).cumsum()
    run_df = (
        pd.DataFrame({"label": binary_labels, "run": run_id})
        .groupby("run")
        .agg(label=("label", "first"), duration_h=("label", "count"))
        .reset_index(drop=True)
    )
    event_durations = run_df.loc[run_df["label"] == 1, "duration_h"].values
    clear_durations  = run_df.loc[run_df["label"] == 0, "duration_h"].values

    if len(event_durations) > 0:
        med_dur = float(np.median(event_durations))
        med_gap = float(np.median(clear_durations)) if len(clear_durations) > 0 else 0.0

        em1, em2, em3, em4 = st.columns(4)
        em1.metric("Total events",         f"{len(event_durations):,}")
        em2.metric("Median duration (h)",  f"{med_dur:.1f}")
        em3.metric("Max duration (h)",     f"{int(event_durations.max())}")
        em4.metric("Median gap (h)",       f"{med_gap:.1f}")

        ec1, ec2 = st.columns(2)
        with ec1:
            fig_dur = go.Figure(go.Histogram(
                x=event_durations, nbinsx=40,
                marker_color=COLOR_CYCLE[3], opacity=0.85,
            ))
            fig_dur.update_layout(**make_layout(
                height=300,
                title="Event Duration",
                xaxis_title="Hours",
                yaxis_title="Count",
            ))
            st.plotly_chart(fig_dur, use_container_width=True)

        with ec2:
            fig_iat = go.Figure(go.Histogram(
                x=clear_durations, nbinsx=40,
                marker_color=COLOR_CYCLE[1], opacity=0.85,
            ))
            fig_iat.update_layout(**make_layout(
                height=300,
                title="Inter-arrival Time (clear gap between events)",
                xaxis_title="Hours",
                yaxis_title="Count",
            ))
            st.plotly_chart(fig_iat, use_container_width=True)

        if med_dur > 6:
            st.warning(
                f"Median event duration = **{med_dur:.0f}h**. "
                "Consecutive rows within the same event are highly correlated — "
                "standard train/test split metrics may be over-optimistic for M2. "
                "Consider event-grouped cross-validation for a more realistic estimate."
            )
    else:
        st.info("No weather events found in the loaded window.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 8 — ACF / PACF & Stationarity
# ══════════════════════════════════════════════════════════════════════════════

with tab_acf:
    st.markdown("### ACF / PACF & Stationarity Tests")
    st.markdown(
        "Autocorrelation (ACF) shows how much a variable correlates with its own past values. "
        "Partial autocorrelation (PACF) isolates direct lag effects, removing indirect ones. "
        "PACF spikes identify the exact lag depths worth including as features in M3."
    )

    _non_zone = [c for c in numeric_cols if col_to_group.get(c) not in ("zone", "rolling")]
    _default_acf = "wind_speed_10m" if "wind_speed_10m" in _non_zone else (_non_zone[0] if _non_zone else None)

    af1, af2 = st.columns([2, 1])
    with af1:
        acf_var = st.selectbox(
            "Variable",
            options=_non_zone if _non_zone else numeric_cols,
            index=_non_zone.index(_default_acf) if _default_acf in _non_zone else 0,
            key="acf_var",
        )
    with af2:
        acf_max_lags = st.slider(
            "Max lags (hours)", min_value=12, max_value=168, value=72, step=6, key="acf_lags"
        )

    if acf_var and acf_var in feat_df.columns:
        series_acf = feat_df[acf_var].interpolate(method="linear", limit=6).dropna()
        n_acf = len(series_acf)
        ci_val = 1.96 / np.sqrt(n_acf)

        if n_acf < acf_max_lags * 2:
            st.warning(
                f"Series too short ({n_acf} obs) for max_lags={acf_max_lags}. "
                "Reduce lags or increase history."
            )
        else:
            try:
                acf_vals  = sm_acf(series_acf,  nlags=acf_max_lags, fft=True)
                pacf_vals = sm_pacf(series_acf, nlags=acf_max_lags, method="yw")
                lags_arr  = np.arange(len(acf_vals))

                # ADF stationarity test
                adf_stat, adf_pval, adf_lags_used, _, adf_crit, _ = adfuller(series_acf, autolag="AIC")
                stationary = adf_pval < 0.05

                acf_sig_count  = int((np.abs(acf_vals[1:])  > ci_val).sum())
                pacf_sig_count = int((np.abs(pacf_vals[1:]) > ci_val).sum())

                a1, a2, a3, a4, a5 = st.columns(5)
                a1.metric("Observations",         f"{n_acf:,}")
                a2.metric("ADF p-value",           f"{adf_pval:.4f}")
                a3.metric("Stationary",            "Yes ✓" if stationary else "No ⚠")
                a4.metric("Sig. ACF lags",         str(acf_sig_count))
                a5.metric("Sig. PACF lags",        str(pacf_sig_count))

                # ACF + PACF bar charts
                fig_acf = ps.make_subplots(
                    rows=2, cols=1, vertical_spacing=0.14,
                    subplot_titles=[f"ACF — {acf_var}", f"PACF — {acf_var}"],
                )
                for row_i, (vals, label) in enumerate(
                    [(acf_vals, "ACF"), (pacf_vals, "PACF")], start=1
                ):
                    bar_colors = [
                        "#e84545" if abs(v) > ci_val else "#4a9eff"
                        for v in vals
                    ]
                    fig_acf.add_trace(
                        go.Bar(x=lags_arr, y=vals, marker_color=bar_colors,
                               name=label, showlegend=False),
                        row=row_i, col=1,
                    )
                    for sign in (1, -1):
                        fig_acf.add_hline(
                            y=sign * ci_val, line_dash="dash",
                            line_color="#f5a623", opacity=0.7,
                            row=row_i, col=1,
                        )

                fig_acf.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0a1118",
                    plot_bgcolor="#0f1923",
                    height=520,
                    margin=dict(l=40, r=20, t=50, b=40),
                    font=dict(family="Inter, sans-serif"),
                )
                fig_acf.update_xaxes(title_text="Lag (hours)")
                fig_acf.update_yaxes(title_text="Correlation")
                st.plotly_chart(fig_acf, use_container_width=True)
                st.caption(
                    f"Orange dashed lines = 95% CI (±{ci_val:.3f}). "
                    "Red bars exceed CI — statistically significant autocorrelation. "
                    "PACF spikes directly indicate which lags M3 should use as features."
                )

                if not stationary:
                    st.info(
                        f"ADF p = {adf_pval:.4f} → **non-stationary**. "
                        "Consider using the first-difference (Δ) of this variable as a model feature. "
                        f"ADF 5% critical value: {adf_crit['5%']:.4f}."
                    )

            except Exception as e:
                st.error(f"ACF/PACF computation failed: {e}")

    st.divider()
    st.markdown("#### Stationarity Summary — All Raw Variables")
    st.markdown(
        "ADF test for every raw weather variable. "
        "Non-stationary series may benefit from differencing (Δ) as model inputs."
    )

    if st.checkbox("Run ADF for all raw variables"):
        with st.spinner("Running ADF tests..."):
            adf_rows = []
            for col in col_groups.get("raw", []):
                if col not in feat_df.columns:
                    continue
                s = feat_df[col].interpolate(method="linear", limit=6).dropna()
                if len(s) < 30:
                    continue
                try:
                    stat, pval, nlags_used, _, crit, _ = adfuller(s, autolag="AIC")
                    adf_rows.append({
                        "variable":    col,
                        "adf_stat":    round(float(stat), 4),
                        "p_value":     round(float(pval), 6),
                        "lags_used":   int(nlags_used),
                        "5%_critical": round(float(crit["5%"]), 4),
                        "stationary":  pval < 0.05,
                    })
                except Exception:
                    pass

        if adf_rows:
            adf_df = (
                pd.DataFrame(adf_rows)
                .sort_values("p_value")
                .reset_index(drop=True)
            )

            def _style_adf(row):
                color = "#00d4aa" if row["stationary"] else "#e84545"
                return [f"color: {color}"] * len(row)

            st.dataframe(
                adf_df.style
                .apply(_style_adf, axis=1)
                .format({"adf_stat": "{:.4f}", "p_value": "{:.4f}", "5%_critical": "{:.4f}"}),
                hide_index=True,
                use_container_width=True,
            )
            n_nonstat = int((~adf_df["stationary"]).sum())
            st.caption(
                f"{n_nonstat} non-stationary variable(s) (p ≥ 0.05, shown in red). "
                "Consider adding Δ-differenced versions of these as extra M3 features."
            )
        else:
            st.info("No raw variables available for ADF testing.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 9 — Seasonality
# ══════════════════════════════════════════════════════════════════════════════

with tab_season:
    st.markdown("### Seasonality — Hour × Month Heatmaps")
    st.markdown(
        "Strong diurnal or seasonal patterns mean M1/M2/M3 could benefit from cyclical time "
        "features: `sin(2π × hour/24)`, `cos(2π × hour/24)`, `sin(2π × doy/365)`. "
        "A flat heatmap means adding them is unlikely to help."
    )

    _month_abbrs = [calendar.month_abbr[m] for m in range(1, 13)]
    _season_cols = [c for c in numeric_cols if col_to_group.get(c) not in ("zone", "rolling")]
    _default_sv  = "wind_speed_10m" if "wind_speed_10m" in _season_cols else (
        _season_cols[0] if _season_cols else None
    )

    season_var = st.selectbox(
        "Variable",
        options=_season_cols if _season_cols else numeric_cols,
        index=_season_cols.index(_default_sv) if _default_sv in _season_cols else 0,
        key="season_var",
    )

    if season_var and season_var in feat_df.columns:
        _df_s = pd.DataFrame({
            "value": feat_df[season_var],
            "hour":  feat_df.index.hour,
            "month": feat_df.index.month,
        })

        # Pivot: rows = hour (0–23), columns = month (1–12)
        pivot_mean = (
            _df_s.groupby(["hour", "month"])["value"]
            .mean()
            .unstack(level=1)
            .reindex(columns=range(1, 13), fill_value=np.nan)
        )

        sc1, sc2 = st.columns(2)

        with sc1:
            st.markdown(f"**{season_var} mean**")
            fig_hm = go.Figure(go.Heatmap(
                z=pivot_mean.values,
                x=_month_abbrs,
                y=list(range(24)),
                colorscale="RdBu_r",
                colorbar=dict(title=season_var, len=0.8),
            ))
            fig_hm.update_layout(**make_layout(
                height=420,
                xaxis_title="Month",
                yaxis=dict(title="Hour of day (UTC)", dtick=3, autorange="reversed"),
            ))
            st.plotly_chart(fig_hm, use_container_width=True)

        with sc2:
            st.markdown("**Weather event rate (WMO code > 3)**")
            if "weather_code" in feat_df.columns:
                _ev = pd.DataFrame({
                    "event": make_binary_label(feat_df),
                    "hour":  feat_df.index.hour,
                    "month": feat_df.index.month,
                })
                pivot_ev = (
                    _ev.groupby(["hour", "month"])["event"]
                    .mean()
                    .unstack(level=1)
                    .reindex(columns=range(1, 13), fill_value=np.nan)
                )
                fig_ev = go.Figure(go.Heatmap(
                    z=pivot_ev.values,
                    x=_month_abbrs,
                    y=list(range(24)),
                    colorscale="YlOrRd",
                    colorbar=dict(title="Event rate", len=0.8, tickformat=".0%"),
                ))
                fig_ev.update_layout(**make_layout(
                    height=420,
                    xaxis_title="Month",
                    yaxis=dict(title="Hour of day (UTC)", dtick=3, autorange="reversed"),
                ))
                st.plotly_chart(fig_ev, use_container_width=True)
            else:
                st.info("weather_code not available — re-fetch data for event rate heatmap.")

        # Seasonality strength summary
        overall_std = float(_df_s["value"].std())
        if overall_std > 0:
            diurnal_range  = float(_df_s.groupby("hour")["value"].mean().pipe(lambda s: s.max() - s.min()))
            seasonal_range = float(_df_s.groupby("month")["value"].mean().pipe(lambda s: s.max() - s.min()))
            diurnal_pct    = diurnal_range  / overall_std * 100
            seasonal_pct   = seasonal_range / overall_std * 100

            ss1, ss2, ss3, ss4 = st.columns(4)
            ss1.metric("Overall std",              f"{overall_std:.3f}")
            ss2.metric("Diurnal range",            f"{diurnal_range:.3f}")
            ss3.metric("Seasonal range",           f"{seasonal_range:.3f}")
            ss4.metric("Seasonal / std",           f"{seasonal_pct:.0f}%")
            st.caption(
                "**Diurnal range** = max(hourly mean) − min(hourly mean). "
                "**Seasonal range** = max(monthly mean) − min(monthly mean). "
                "If seasonal range / std > 20%, adding day-of-year sin/cos features to M1/M2/M3 is likely to help."
            )


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption("Sprint 3/5 · Data Quality & Feature Review · Data: Open-Meteo")
