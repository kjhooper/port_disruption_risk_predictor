"""
app.py — Port Disruption Risk Monitor
Live weather pipeline + ML disruption risk forecast (M2 + M3).
Run with: conda run -n personal streamlit run app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests

from fetch import fetch_openmeteo_historical, fetch_openmeteo_forecast, update_or_fetch, PORTS
from quality import run_all_checks, quality_summary_df
from features import compute_all_features
from model import build_feature_matrix, FORECAST_TARGETS, LAG_HOURS, build_lag_features, WEATHER_COLORS

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Port Risk Monitor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────

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
        padding: 20px 24px;
        margin-bottom: 8px;
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

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🚢 Port Risk Monitor")
    st.divider()

    port = st.selectbox(
        "SELECT PORT",
        options=list(PORTS.keys()),
        format_func=lambda k: PORTS[k]["label"],
    )

    days_back = st.slider("HISTORY (days)", min_value=7, max_value=365, value=30, step=7)

    fetch_btn = st.button("↻ Fetch / Refresh Data", use_container_width=True, type="primary")

    st.divider()
    st.markdown(f"**Port:** {PORTS[port]['label']}")
    st.markdown(f"**Lat/Lon:** {PORTS[port]['lat']}°, {PORTS[port]['lon']}°")
    st.markdown(f"**Last run:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")

# ── Data loading ───────────────────────────────────────────────────────────────

RELEASE_BASE_URL = "https://github.com/kjhooper/port_disruption_risk_predictor/releases/download/dashboard"


def _load_parquet_from_release(filename: str) -> pd.DataFrame:
    """Load a parquet file from GitHub releases, cache locally after first download."""
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
    try:
        df = _load_parquet_from_release(f"{port}_portwatch_activity.parquet")
    except Exception:
        return pd.DataFrame()
    
    # rest of your existing logic stays exactly the same...
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


@st.cache_resource
def load_port_models(port: str) -> dict:
    """Load trained M1/M2/M3 models for a port. Returns {} if not yet trained."""
    import joblib
    model_dir = Path("models") / port
    if not model_dir.exists():
        return {}
    return {p.stem: joblib.load(p) for p in sorted(model_dir.glob("*.joblib"))}


if fetch_btn:
    with st.spinner("Updating data..."):
        update_or_fetch(port, save_dir="data")
    st.cache_data.clear()

hist_df, fore_df = load_data(port, days_back)

# ── Inference helpers ──────────────────────────────────────────────────────────

def _m2_features(fore_df: pd.DataFrame, port: str, m2_model) -> pd.DataFrame | None:
    """
    Build the M2 classifier feature matrix from forecast data.

    Zone gradient features (z150_*, z300_*) are absent from the forecast parquet —
    they are set to 0, meaning "no upstream gradient" (conservative/neutral assumption).
    """
    try:
        feat_df = compute_all_features(fore_df.copy(), port)
        X = build_feature_matrix(feat_df, port, include_zones=False)

        # Add any training features missing from the forecast with value 0
        for col in m2_model.feature_names_in_:
            if col not in X.columns:
                X[col] = 0.0

        return X[list(m2_model.feature_names_in_)]
    except Exception:
        return None


def _m3_predictions(
    full_hist: pd.DataFrame,
    fore_df: pd.DataFrame,
    models: dict,
) -> dict[int, dict[str, float]]:
    """
    Run M3 forecasters from the most recent observed state.

    Returns {horizon_hours: {variable: predicted_value}} for horizons 24/48/72h.
    Lag features are built from the last 48h of historical observations so that
    all five lag windows (1h/3h/6h/12h/24h) are fully populated.
    """
    if full_hist.empty:
        return {}

    # Combine last 48h of history with forecast (for lag feature computation)
    recent = full_hist.tail(48).copy()
    combined = pd.concat([recent, fore_df])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    df_lag = combined.copy()
    for tgt in FORECAST_TARGETS:
        if tgt in df_lag.columns:
            df_lag = build_lag_features(df_lag, tgt, LAG_HOURS)

    extra_cols = [c for c in ["wind_direction_10m", "relative_humidity_2m", "cape"]
                  if c in df_lag.columns]
    lag_cols   = [f"{tgt}_lag_{h}h" for tgt in FORECAST_TARGETS for h in LAG_HOURS
                  if f"{tgt}_lag_{h}h" in df_lag.columns]
    feature_cols = lag_cols + extra_cols

    X_all = df_lag[feature_cols].ffill().fillna(0)

    # Predict from the most recent observation (last row of historical data)
    current_idx = recent.index[-1]
    if current_idx not in X_all.index:
        return {}
    X_now = X_all.loc[[current_idx]]

    results: dict[int, dict[str, float]] = {}
    for horizon in [24, 48, 72]:
        results[horizon] = {}
        for target in FORECAST_TARGETS:
            key = f"m3_{target}_{horizon}h"
            if key not in models:
                continue
            entry = models[key]
            reg   = entry[0] if isinstance(entry, tuple) else entry
            try:
                X_aligned = X_now.copy()
                for col in reg.feature_names_in_:
                    if col not in X_aligned.columns:
                        X_aligned[col] = 0.0
                X_aligned = X_aligned[list(reg.feature_names_in_)]
                results[horizon][target] = float(reg.predict(X_aligned)[0])
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


def _m1_forecast_timeline(fore_df: pd.DataFrame, port: str, m1_model) -> pd.Series:
    """
    Run existing M1 classifier on forecast rows.
    Returns pd.Series indexed by forecast timestamps with string weather group per hour.
    """
    try:
        feat_df = compute_all_features(fore_df.copy(), port)
        X = build_feature_matrix(feat_df, port, include_zones=False)
        for col in m1_model.feature_names_in_:
            if col not in X.columns:
                X[col] = 0.0
        X = X[list(m1_model.feature_names_in_)]
        y_prob = m1_model.predict_proba(X)
        y_pred_int = y_prob.argmax(axis=1)
        int_to_label = getattr(m1_model, "_int_to_label", {})
        return pd.Series(y_pred_int, index=fore_df.index).map(int_to_label)
    except Exception:
        return pd.Series(dtype=str)


def _m1b_predictions(full_hist: pd.DataFrame, models: dict) -> dict:
    """
    Run M1-B forecast classifiers from current lag state.
    Returns {24: "thunderstorm", 48: "showers", 72: "clear"}.
    """
    if full_hist.empty:
        return {}

    recent = full_hist.tail(48).copy()
    df_lag = recent.copy()
    for tgt in FORECAST_TARGETS:
        if tgt in df_lag.columns:
            df_lag = build_lag_features(df_lag, tgt, LAG_HOURS)
    if "weather_code" in df_lag.columns:
        df_lag = build_lag_features(df_lag, "weather_code", LAG_HOURS)

    extra_cols = [c for c in ["wind_direction_10m", "relative_humidity_2m", "cape"]
                  if c in df_lag.columns]
    lag_cols = [f"{tgt}_lag_{h}h" for tgt in FORECAST_TARGETS for h in LAG_HOURS
                if f"{tgt}_lag_{h}h" in df_lag.columns]
    weather_lag_cols = [f"weather_code_lag_{h}h" for h in LAG_HOURS
                        if f"weather_code_lag_{h}h" in df_lag.columns]
    feature_cols = lag_cols + weather_lag_cols + extra_cols

    X_all = df_lag[feature_cols].ffill().fillna(0)
    current_idx = recent.index[-1]
    if current_idx not in X_all.index:
        return {}
    X_now = X_all.loc[[current_idx]]

    results: dict = {}
    for horizon in [24, 48, 72]:
        key = f"m1_forecast_{horizon}h"
        if key not in models:
            continue
        entry = models[key]
        clf = entry[0] if isinstance(entry, tuple) else entry
        try:
            X_aligned = X_now.copy()
            for col in clf.feature_names_in_:
                if col not in X_aligned.columns:
                    X_aligned[col] = 0.0
            X_aligned = X_aligned[list(clf.feature_names_in_)]
            int_to_label = getattr(clf, "_int_to_label", {})
            y_prob = clf.predict_proba(X_aligned)
            y_pred_int = int(y_prob.argmax(axis=1)[0])
            results[horizon] = int_to_label.get(y_pred_int, str(y_pred_int))
        except Exception:
            pass

    return results


# ── Quality report ─────────────────────────────────────────────────────────────

quality_report = run_all_checks(hist_df)
summary_df     = quality_summary_df(quality_report)

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown(f"# {PORTS[port]['label']}")
st.markdown("Weather data pipeline + disruption risk forecast")
st.divider()

# ── Quality metrics row ────────────────────────────────────────────────────────

st.markdown("### Data Quality")

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
  &nbsp;·&nbsp; Status: <b>{quality_report['overall_status'].upper()}</b>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Disruption Risk Forecast ───────────────────────────────────────────────────

st.markdown("### Disruption Risk Forecast")

models    = load_port_models(port)
m2_entry  = models.get("m2_binary")

if m2_entry is None:
    st.info(
        f"No trained models found for **{PORTS[port]['label']}**. "
        f"Run `conda run -n personal python train.py --port {port}` to train."
    )
else:
    m2_model, m2_metrics = m2_entry

    with st.spinner("Running M2 inference on forecast..."):
        X_fore = _m2_features(fore_df, port, m2_model)

    if X_fore is None:
        st.warning("Feature engineering failed for forecast data.")
    else:
        probs      = m2_model.predict_proba(X_fore)[:, 1]
        risk_series = pd.Series(probs, index=fore_df.index, name="disruption_prob")

        # Summary metrics
        now_prob   = float(risk_series.iloc[0])
        peak_24    = float(risk_series.iloc[:24].max())
        peak_72    = float(risk_series.iloc[:72].max())
        peak_72_t  = risk_series.iloc[:72].idxmax()

        # Risk score cards
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="metric-card risk-card {_risk_card_class(now_prob)}">
              <div style="font-size:0.75rem; color:#888; font-family:'Space Mono',monospace;">NOW</div>
              <div style="font-size:2.2rem; font-weight:700; color:{_risk_color(now_prob)}; font-family:'Space Mono',monospace;">
                {now_prob:.0%}
              </div>
              <div style="font-size:0.8rem; color:{_risk_color(now_prob)};">{_risk_label(now_prob)}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card risk-card {_risk_card_class(peak_24)}">
              <div style="font-size:0.75rem; color:#888; font-family:'Space Mono',monospace;">PEAK — NEXT 24H</div>
              <div style="font-size:2.2rem; font-weight:700; color:{_risk_color(peak_24)}; font-family:'Space Mono',monospace;">
                {peak_24:.0%}
              </div>
              <div style="font-size:0.8rem; color:{_risk_color(peak_24)};">{_risk_label(peak_24)}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card risk-card {_risk_card_class(peak_72)}">
              <div style="font-size:0.75rem; color:#888; font-family:'Space Mono',monospace;">PEAK — NEXT 72H</div>
              <div style="font-size:2.2rem; font-weight:700; color:{_risk_color(peak_72)}; font-family:'Space Mono',monospace;">
                {peak_72:.0%}
              </div>
              <div style="font-size:0.8rem; color:{_risk_color(peak_72)};">{peak_72_t.strftime('%a %d %b %H:%M') if not pd.isna(peak_72_t) else '—'} UTC</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            roc  = m2_metrics.get("roc_auc", None) if isinstance(m2_metrics, dict) else None
            gate = "✅ PASS" if (roc and roc > 0.75) else "⚠️ UNVALIDATED"
            st.markdown(f"""
            <div class="metric-card risk-card risk-low">
              <div style="font-size:0.75rem; color:#888; font-family:'Space Mono',monospace;">MODEL STATUS</div>
              <div style="font-size:1.1rem; font-weight:700; color:#00d4aa; font-family:'Space Mono',monospace; margin-top:8px;">
                {gate}
              </div>
              <div style="font-size:0.75rem; color:#888; margin-top:4px;">
                ROC-AUC: {f"{roc:.3f}" if roc else "N/A"}
              </div>
            </div>
            """, unsafe_allow_html=True)

        # 72-hour probability timeline
        risk_72h = risk_series.iloc[:72]
        bar_colors = [_risk_color(p) for p in risk_72h.values]

        fig_risk = go.Figure()
        fig_risk.add_trace(go.Bar(
            x=risk_72h.index,
            y=risk_72h.values,
            marker_color=bar_colors,
            name="Disruption probability",
            hovertemplate="%{x|%a %d %b %H:%M}<br>P(disruption) = %{y:.1%}<extra></extra>",
        ))
        fig_risk.add_hline(
            y=0.40, line_dash="dash", line_color="#e84545",
            annotation_text="High threshold (40%)", annotation_position="top right",
            annotation_font_color="#e84545",
        )
        fig_risk.add_hline(
            y=0.15, line_dash="dot", line_color="#f5a623",
            annotation_text="Elevated threshold (15%)", annotation_position="top right",
            annotation_font_color="#f5a623",
        )
        fig_risk.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a1118",
            plot_bgcolor="#0f1923",
            height=260,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title="Forecast time (UTC)",
            yaxis_title="P(disruption)",
            yaxis=dict(tickformat=".0%", range=[0, max(1.0, risk_72h.max() * 1.1)]),
            showlegend=False,
            title=dict(text="72-hour Disruption Probability", font_color="#aaa", font_size=13),
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        # ── Option A: 72h weather type timeline (M1 on forecast rows) ─────────
        m1_entry = models.get("m1_classifier")
        if m1_entry is not None:
            m1_model_a, _ = m1_entry
            weather_72h = _m1_forecast_timeline(fore_df, port, m1_model_a)
            if not weather_72h.empty:
                w72 = weather_72h.iloc[:72]
                colors_72 = [WEATHER_COLORS.get(w, "#888888") for w in w72]

                fig_wt = go.Figure()
                fig_wt.add_trace(go.Bar(
                    x=w72.index,
                    y=[1] * len(w72),
                    marker_color=colors_72,
                    hovertemplate="%{x|%a %d %b %H:%M}<br>%{customdata}<extra></extra>",
                    customdata=w72.values,
                    showlegend=False,
                ))
                # Dummy scatter traces for legend labels (one per present weather type)
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
                    height=100,
                    margin=dict(l=40, r=20, t=30, b=10),
                    xaxis=dict(showticklabels=False),
                    yaxis=dict(showticklabels=False, showgrid=False, range=[0, 1.2]),
                    legend=dict(orientation="h", y=1.6, x=0),
                    title=dict(text="72h Weather Type Forecast (M1)", font_color="#aaa", font_size=13),
                )
                st.plotly_chart(fig_wt, use_container_width=True)
                st.caption(
                    "M1 predicted weather type per forecast hour. "
                    "Thunderstorm/shower hours explain elevated disruption probability."
                )

st.divider()

# ── Traffic vs Predictions ─────────────────────────────────────────────────────

st.markdown("### 📡 Traffic vs Predictions")

pw_df = load_portwatch_activity(port)

if pw_df.empty:
    st.info(
        "No PortWatch data available for this port. "
        "Run `conda run -n personal python fetch_portwatch.py` to download traffic data."
    )
else:
    # Convert date index (date objects) to datetime for slicing and charting
    pw_dt = pw_df.copy()
    pw_dt.index = pd.to_datetime(pw_df.index)
    cutoff_pw = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days_back)
    pw_dt = pw_dt[pw_dt.index >= cutoff_pw]

    # Rotterdam: portcall traffic is structurally stable — PortWatch never shows disruptions.
    # M2 was trained on composite weather physics, not traffic, so the comparison is not meaningful.
    has_m2 = m2_entry is not None and port != "rotterdam"

    if has_m2:
        m2_model_obj, _ = m2_entry
        with st.spinner("Computing M2 hindcast..."):
            hindcast = compute_m2_hindcast(port, m2_model_obj, days_back)
        if not hindcast.empty:
            daily_prob = hindcast.resample("D").mean().rename("m2_prob")
            combined = pw_dt[["ratio", "disrupted"]].join(daily_prob, how="inner")
        else:
            combined = pw_dt
    else:
        combined = pw_dt

    if combined.empty or "ratio" not in combined.columns:
        st.info("Insufficient data to display chart.")
    else:
        fig_tv = go.Figure()

        # Bar: portcall ratio (left axis)
        fig_tv.add_trace(go.Bar(
            x=combined.index,
            y=combined["ratio"].clip(0, 2),
            name="Portcall ratio",
            marker_color="#4a9eff",
            opacity=0.7,
            yaxis="y1",
            hovertemplate="%{x|%Y-%m-%d}<br>Ratio: %{y:.2f}<extra></extra>",
        ))

        # Line: M2 probability (right axis)
        if has_m2 and "m2_prob" in combined.columns:
            fig_tv.add_trace(go.Scatter(
                x=combined.index,
                y=combined["m2_prob"],
                name="M2 disruption prob",
                line=dict(color="#e84545", width=2),
                yaxis="y2",
                hovertemplate="%{x|%Y-%m-%d}<br>M2: %{y:.1%}<extra></extra>",
            ))

        # Shaded bands for PortWatch-confirmed disruption days
        if "disrupted" in combined.columns:
            dis = combined[combined["disrupted"] == 1]
            if not dis.empty:
                blocks = (dis.index.to_series().diff() > pd.Timedelta(days=2)).cumsum()
                for _, grp in dis.groupby(blocks):
                    fig_tv.add_vrect(
                        x0=grp.index[0] - pd.Timedelta(hours=12),
                        x1=grp.index[-1] + pd.Timedelta(hours=12),
                        fillcolor="rgba(232,69,69,0.15)",
                        layer="below", line_width=0,
                    )

        # Gold bands for public holidays
        if "is_holiday" in combined.columns:
            hols_only = combined[combined["is_holiday"] == 1]
            if not hols_only.empty:
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

        # Reference line at ratio=0.30
        fig_tv.add_hline(
            y=0.30, line_dash="dot", line_color="#e84545",
            annotation_text="Disruption threshold (30%)", annotation_position="top right",
            annotation_font_color="#e84545",
        )

        ratio_max = float(combined["ratio"].max()) if not combined["ratio"].isna().all() else 2.0
        y2_config = (
            dict(title="M2 probability", overlaying="y", side="right",
                 range=[0, 1], tickformat=".0%", showgrid=False)
            if has_m2 and "m2_prob" in combined.columns
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
                    "PortWatch Traffic vs M2 Weather Predictions"
                    if has_m2 and "m2_prob" in combined.columns
                    else "PortWatch Daily Portcall Ratio"
                ),
                font_color="#aaa", font_size=13,
            ),
        )
        st.plotly_chart(fig_tv, use_container_width=True)

        if port == "rotterdam":
            st.caption(
                "Rotterdam is a stable port — M2 trained on composite weather physics "
                "(PortWatch shows no significant traffic disruptions)."
            )
        elif has_m2 and "m2_prob" in combined.columns:
            st.caption(
                "PortWatch portcall ratio vs M2 weather-based disruption probability. "
                "Gold bands = public holidays (excluded from M2 training — traffic drops here are calendar-driven). "
                "Pink bands = PortWatch-confirmed disruption days. "
                "High red line without a gold band = weather-driven signal."
            )
        else:
            st.caption(
                "PortWatch daily portcall ratio relative to 28-day rolling median. "
                "Drops below 30% indicate traffic disruptions. "
                "Train M2 to overlay weather-based predictions."
            )

st.divider()

# ── Weather Overview + M3 forecast ────────────────────────────────────────────

st.markdown("### Weather Overview")

_tab_labels = ["💨 Wind", "🌧️ Precipitation & Pressure", "🔮 Model Forecast (M3)", "📊 Raw Data"]
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
    # M3 predictions vs Open-Meteo forecast
    if not models:
        st.info(f"Run `python train.py --port {port}` to enable M3 forecast comparison.")
    else:
        full_hist = load_full_hist(port)
        with st.spinner("Running M3 forecasters..."):
            m3_preds = _m3_predictions(full_hist, fore_df, models)

        if not m3_preds:
            st.warning("M3 predictions could not be computed — check that historical data is loaded.")
        else:
            now_utc = datetime.utcnow()

            # Build comparison table
            rows = []
            horizon_labels = {24: "T+24h", 48: "T+48h", 72: "T+72h"}
            var_labels = {
                "wind_speed_10m": "Wind Speed (m/s)",
                "precipitation":   "Precipitation (mm/h)",
                "pressure_msl":    "Pressure (hPa)",
                "wave_height":     "Wave Height (m)",
            }
            unit_fmt = {
                "wind_speed_10m": ".1f",
                "precipitation":  ".2f",
                "pressure_msl":   ".1f",
                "wave_height":    ".2f",
            }

            for horizon, preds in sorted(m3_preds.items()):
                target_time = now_utc + timedelta(hours=horizon)
                # Find closest forecast row to target time
                if len(fore_df):
                    pos = min(
                        fore_df.index.searchsorted(pd.Timestamp(target_time)),
                        len(fore_df) - 1
                    )
                    fore_idx = fore_df.index[pos]
                else:
                    fore_idx = None

                for tgt, m3_val in preds.items():
                    ome_val = float(fore_df.loc[fore_idx, tgt]) if (
                        fore_idx is not None and tgt in fore_df.columns
                    ) else None
                    delta = (m3_val - ome_val) if ome_val is not None else None
                    rows.append({
                        "Horizon":   horizon_labels[horizon],
                        "Target time (UTC)": target_time.strftime("%a %d %b %H:%M"),
                        "Variable":  var_labels.get(tgt, tgt),
                        "M3 Prediction": f"{m3_val:{unit_fmt.get(tgt, '.2f')}}",
                        "Open-Meteo":    f"{ome_val:{unit_fmt.get(tgt, '.2f')}}" if ome_val is not None else "—",
                        "Δ (M3 − OME)":  f"{delta:+.2f}" if delta is not None else "—",
                    })

            if rows:
                cmp_df = pd.DataFrame(rows)
                st.markdown("**M3 model predictions vs Open-Meteo NWP forecast**")
                st.caption(
                    "M3 is trained on 2+ years of lag patterns. "
                    "Large divergence from Open-Meteo highlights uncertainty — "
                    "consider both when assessing risk."
                )
                st.dataframe(cmp_df, hide_index=True, use_container_width=True)

                # Visual comparison for wind speed
                if any(r["Variable"] == "Wind Speed (m/s)" for r in rows):
                    wind_rows = [r for r in rows if r["Variable"] == "Wind Speed (m/s)"]
                    horizons_str  = [r["Horizon"] for r in wind_rows]
                    m3_vals       = [float(r["M3 Prediction"]) for r in wind_rows]
                    ome_vals      = [float(r["Open-Meteo"]) if r["Open-Meteo"] != "—" else None for r in wind_rows]

                    fig_m3 = go.Figure()
                    fig_m3.add_trace(go.Bar(
                        x=horizons_str, y=m3_vals,
                        name="M3", marker_color="#4a9eff", opacity=0.85,
                    ))
                    if any(v is not None for v in ome_vals):
                        fig_m3.add_trace(go.Bar(
                            x=horizons_str, y=[v for v in ome_vals if v is not None],
                            name="Open-Meteo", marker_color="#f5a623", opacity=0.7,
                        ))
                    fig_m3.update_layout(
                        template="plotly_dark", paper_bgcolor="#0a1118", plot_bgcolor="#0f1923",
                        height=280, margin=dict(l=40, r=20, t=30, b=40),
                        barmode="group",
                        title=dict(text="Wind Speed Forecast: M3 vs Open-Meteo (m/s)", font_color="#aaa", font_size=13),
                        legend=dict(orientation="h", y=-0.2),
                        yaxis_title="m/s",
                    )
                    st.plotly_chart(fig_m3, use_container_width=True)

        # ── Option B: M1-B weather type predictions from current lag features ──
        m1b_preds = _m1b_predictions(full_hist, models)
        if m1b_preds:
            st.markdown("**M1-B Predicted Weather Type (from current lag features)**")
            m1b_rows = []
            for h in [24, 48, 72]:
                if h in m1b_preds:
                    wtype = m1b_preds[h]
                    color = WEATHER_COLORS.get(wtype, "#888888")
                    m1b_rows.append({
                        "Horizon": f"T+{h}h",
                        "Predicted Weather Type": wtype,
                        "Color": color,
                    })
            if m1b_rows:
                m1b_df = pd.DataFrame(m1b_rows)

                def _color_wtype(val):
                    color = WEATHER_COLORS.get(val, "#888888")
                    return f"color: {color}; font-weight: 600"

                st.dataframe(
                    m1b_df[["Horizon", "Predicted Weather Type"]].style.applymap(
                        _color_wtype, subset=["Predicted Weather Type"]
                    ),
                    hide_index=True,
                    use_container_width=False,
                )
                st.caption(
                    "M1-B uses lag patterns to predict weather type at each horizon — "
                    "no NWP forecast needed."
                )

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

# ── Completeness heatmap ───────────────────────────────────────────────────────

st.divider()
st.markdown("### Completeness by Column")

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

# ── Model Performance ──────────────────────────────────────────────────────────

st.divider()

with st.expander("📋 Model Performance", expanded=False):

    # ── M2 ──
    m2_entry_perf = models.get("m2_binary")
    st.markdown("**M2 — Binary Disruption Detector**")
    if m2_entry_perf is None:
        if port == "rotterdam":
            st.info("M2 skipped — Rotterdam confirmed stable (0 disruption events in PortWatch).")
        else:
            st.info(
                f"No M2 model trained for **{PORTS[port]['label']}**. "
                f"Run `python train.py --port {port}`."
            )
    else:
        _, m2_met = m2_entry_perf
        roc   = m2_met.get("roc_auc")
        prauc = m2_met.get("pr_auc")
        f1    = m2_met.get("f1")
        ece   = m2_met.get("ece")
        gate_pass  = bool(roc and roc > 0.75 and ece is not None and ece < 0.05)
        gate_label = "✅ PASS" if gate_pass else "⚠️ FAIL"
        gate_color = "#00d4aa" if gate_pass else "#f5a623"

        m2c1, m2c2, m2c3, m2c4, m2c5 = st.columns(5)
        with m2c1:
            st.metric("ROC-AUC", f"{roc:.3f}" if roc is not None else "N/A")
        with m2c2:
            st.metric("PR-AUC", f"{prauc:.3f}" if prauc is not None else "N/A")
        with m2c3:
            st.metric("F1", f"{f1:.3f}" if f1 is not None else "N/A")
        with m2c4:
            st.metric("ECE", f"{ece:.4f}" if ece is not None else "N/A")
        with m2c5:
            st.markdown(
                f"<div style='padding-top:1.6rem; color:{gate_color}; font-weight:700;'>"
                f"{gate_label}</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── M1 ──
    m1_entry_perf = models.get("m1_classifier")
    st.markdown("**M1 — Weather Event Classifier**")
    if m1_entry_perf is None:
        st.info(
            f"No M1 model trained for **{PORTS[port]['label']}**. "
            f"Run `python train.py --port {port}`."
        )
    else:
        _, m1_met = m1_entry_perf
        m1_roc = m1_met.get("roc_auc")
        m1_f1  = m1_met.get("macro_f1")
        m1c1, m1c2 = st.columns(2)
        with m1c1:
            st.metric("Macro F1", f"{m1_f1:.3f}" if m1_f1 is not None else "N/A")
        with m1c2:
            st.metric("ROC-AUC (OvR)", f"{m1_roc:.3f}" if m1_roc is not None else "N/A")
        per_class = m1_met.get("per_class_f1", {})
        if per_class:
            pc_df = pd.DataFrame(
                [{"Weather Class": k, "F1 Score": round(v, 3)} for k, v in per_class.items()]
            )
            st.dataframe(pc_df, hide_index=True, use_container_width=True)

    st.divider()

    # ── M3 ──
    m3_keys = [k for k in models if k.startswith("m3_")]
    st.markdown("**M3 — Variable Forecasters**")
    if not m3_keys:
        st.info(
            f"No M3 models trained for **{PORTS[port]['label']}**. "
            f"Run `python train.py --port {port}`."
        )
    else:
        var_labels = {
            "wind_speed_10m": "Wind Speed",
            "precipitation":  "Precipitation",
            "pressure_msl":   "Pressure",
            "wave_height":    "Wave Height",
        }
        m3_rows: dict[str, dict[str, str]] = {}
        for target in FORECAST_TARGETS:
            var_name = var_labels.get(target, target)
            for horizon in [24, 48, 72]:
                key = f"m3_{target}_{horizon}h"
                if key not in models:
                    continue
                entry = models[key]
                _, m3_met = entry if isinstance(entry, tuple) else (entry, {})
                rmse = m3_met.get("rmse")
                r2   = m3_met.get("r2")
                cell = f"{rmse:.3f} / {r2:.3f}" if (rmse is not None and r2 is not None) else "N/A"
                if var_name not in m3_rows:
                    m3_rows[var_name] = {}
                m3_rows[var_name][f"T+{horizon}h"] = cell

        if m3_rows:
            m3_table = pd.DataFrame(m3_rows).T
            m3_table.index.name = "Variable"
            col_order = [c for c in ["T+24h", "T+48h", "T+72h"] if c in m3_table.columns]
            st.dataframe(
                m3_table[col_order] if col_order else m3_table,
                use_container_width=True,
            )
            st.caption("Cells show RMSE / R² for each forecast horizon.")

# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Data: Open-Meteo (archive + forecast) · "
    "Models: M2 binary disruption detector · M3 variable forecaster · "
    "Ground truth: NCEI Storm Events (Houston) · PortWatch portcalls (Houston, Hong Kong, Kaohsiung) · Composite physics label (Rotterdam)"
)
