"""
pages/3_🌊_History.py — Historical Explorer
Weather timelines, disruption events, and AIS vessel traffic (Houston only).
Run as part of multi-page app: conda run -n personal streamlit run app.py
"""

import sys
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from fetch import PORTS
from features import compute_all_features
from model import build_alert_features

# ── Page config ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Harbinger · History",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }
    [data-testid="stSidebar"] { background: #0a1118; border-right: 1px solid #1e3a5f; }
</style>
""", unsafe_allow_html=True)

RELEASE_BASE_URL = "https://github.com/youruser/port_disruption_risk_predictor/releases/download/v0.2.0-alpha"

# ── Data loading ─────────────────────────────────────────────────────────────────

def _load_parquet_from_release(filename: str) -> pd.DataFrame:
    local_path = Path("data") / filename
    local_path.parent.mkdir(exist_ok=True)
    if not local_path.exists():
        url = f"{RELEASE_BASE_URL}/{filename}"
        r = requests.get(url)
        r.raise_for_status()
        local_path.write_bytes(r.content)
    return pd.read_parquet(local_path)


@st.cache_data(ttl=3600, show_spinner="Loading historical data...")
def load_hist(port: str) -> pd.DataFrame:
    try:
        return _load_parquet_from_release(f"{port}_historical_wide.parquet")
    except Exception:
        return pd.DataFrame()


@st.cache_resource
def load_port_models(port: str) -> dict:
    import joblib
    model_dir = Path("models") / port
    if not model_dir.exists():
        return {}
    return {p.stem: joblib.load(p) for p in sorted(model_dir.glob("*.joblib"))}


@st.cache_data(ttl=3600, show_spinner="Computing disruption hindcast...")
def compute_hindcast(port: str, _alert_model_24h, days_back: int) -> pd.Series:
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
def load_ais_activity() -> pd.DataFrame:
    ais_path = Path("data/houston_ais_activity.parquet")
    if not ais_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(ais_path)


# ── Sidebar ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌊 Historical Explorer")
    st.divider()

    port = st.selectbox(
        "SELECT PORT",
        options=list(PORTS.keys()),
        format_func=lambda k: PORTS[k]["label"],
    )

    st.divider()
    st.markdown("**DATE RANGE**")
    days_back = st.slider("Days back", min_value=30, max_value=1095, value=180, step=30)

    st.divider()
    st.markdown("**WEATHER VARIABLE**")
    weather_var = st.selectbox(
        "Primary variable",
        options=["wind_speed_10m", "wind_gusts_10m", "precipitation", "pressure_msl", "wave_height", "temperature_2m"],
        format_func=lambda v: {
            "wind_speed_10m": "Wind Speed (m/s)",
            "wind_gusts_10m": "Wind Gusts (m/s)",
            "precipitation":  "Precipitation (mm/h)",
            "pressure_msl":   "Pressure (hPa)",
            "wave_height":    "Wave Height (m)",
            "temperature_2m": "Temperature (°C)",
        }.get(v, v),
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(f"**Port:** {PORTS[port]['label']}")
    st.markdown(f"**Period:** last {days_back} days")

# ── Header ────────────────────────────────────────────────────────────────────────

st.markdown(f"# {PORTS[port]['label']} · Historical Explorer")
st.markdown("Weather timelines · disruption events · vessel traffic")
st.divider()

# ── Load data ────────────────────────────────────────────────────────────────────

hist_df = load_hist(port)
models  = load_port_models(port)

if hist_df.empty:
    st.error(
        f"No historical data found for **{PORTS[port]['label']}**. "
        f"Run `conda run -n personal python fetch.py` to download data."
    )
    st.stop()

# Slice to selected date range
cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days_back)
hist_slice = hist_df[hist_df.index >= cutoff].copy()

# Compute hindcast
alert_entry = models.get("disruption_alerts")
hindcast = pd.Series(dtype=float)
if alert_entry is not None:
    alert_clf_24, _ = alert_entry.get(24, (None, None))
    if alert_clf_24 is not None:
        hindcast = compute_hindcast(port, alert_clf_24, days_back)

# AIS availability
ais_df = load_ais_activity() if port == "houston" else pd.DataFrame()
show_ais_tab = port == "houston" and not ais_df.empty

# ── Tabs ─────────────────────────────────────────────────────────────────────────

tab_labels = ["🌬️ Weather", "⚠️ Disruptions"]
if show_ais_tab:
    tab_labels.append("🚢 Vessel Traffic")

tabs = st.tabs(tab_labels)

# ── Tab 1: Weather timeline ───────────────────────────────────────────────────────

with tabs[0]:
    st.markdown("### Weather Timeline")

    # Primary variable chart
    if weather_var not in hist_slice.columns:
        st.warning(f"Column `{weather_var}` not available in historical data.")
    else:
        var_meta = {
            "wind_speed_10m": ("Wind Speed", "m/s",  "#00d4aa"),
            "wind_gusts_10m": ("Wind Gusts", "m/s",  "#f5a623"),
            "precipitation":  ("Precipitation", "mm/h", "#4a9eff"),
            "pressure_msl":   ("Pressure MSL", "hPa", "#c77dff"),
            "wave_height":    ("Wave Height", "m",   "#7b5eff"),
            "temperature_2m": ("Temperature", "°C",  "#ff8c69"),
        }
        label, unit, color = var_meta.get(weather_var, (weather_var, "", "#aaaaaa"))

        fig_wt = go.Figure()
        # Convert hex colour to rgba for fill
        def _hex_to_rgba(hex_color: str, alpha: float = 0.08) -> str:
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        fig_wt.add_trace(go.Scatter(
            x=hist_slice.index,
            y=hist_slice[weather_var],
            name=f"{label} ({unit})",
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(color) if color.startswith("#") else "rgba(0,212,170,0.08)",
        ))

        # Companion series
        if weather_var == "wind_speed_10m" and "wind_gusts_10m" in hist_slice.columns:
            fig_wt.add_trace(go.Scatter(
                x=hist_slice.index,
                y=hist_slice["wind_gusts_10m"],
                name="Gusts (m/s)",
                line=dict(color="#f5a623", width=1, dash="dot"),
            ))

        # Disruption shading
        if not hindcast.empty:
            hindcast_slice = hindcast[hindcast.index >= cutoff]
            high_risk = hindcast_slice[hindcast_slice >= 0.40]
            if not high_risk.empty:
                blocks = (high_risk.index.to_series().diff() > pd.Timedelta(hours=3)).cumsum()
                for _, grp in high_risk.groupby(blocks):
                    fig_wt.add_vrect(
                        x0=grp.index[0], x1=grp.index[-1],
                        fillcolor="rgba(232,69,69,0.12)",
                        layer="below", line_width=0,
                    )

        fig_wt.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a1118",
            plot_bgcolor="#0f1923",
            height=380,
            margin=dict(l=50, r=20, t=30, b=50),
            xaxis_title="Date (UTC)",
            yaxis_title=f"{label} ({unit})",
            legend=dict(orientation="h", y=-0.2),
            title=dict(text=f"{label} · last {days_back} days", font_color="#aaa", font_size=13),
        )
        st.plotly_chart(fig_wt, use_container_width=True)
        if not hindcast.empty:
            st.caption("Red shading = hours with DisruptionAlert P(24h) ≥ 40%.")

    st.divider()

    # Wind + pressure overview
    st.markdown("#### Wind Speed & Pressure")
    fig_wp = go.Figure()
    if "wind_speed_10m" in hist_slice.columns:
        fig_wp.add_trace(go.Scatter(
            x=hist_slice.index, y=hist_slice["wind_speed_10m"],
            name="Wind Speed (m/s)", line=dict(color="#00d4aa", width=1.2),
            yaxis="y1",
        ))
    if "pressure_msl" in hist_slice.columns:
        fig_wp.add_trace(go.Scatter(
            x=hist_slice.index, y=hist_slice["pressure_msl"],
            name="Pressure (hPa)", line=dict(color="#c77dff", width=1.2),
            yaxis="y2",
        ))
    fig_wp.update_layout(
        template="plotly_dark", paper_bgcolor="#0a1118", plot_bgcolor="#0f1923",
        height=280, margin=dict(l=50, r=60, t=20, b=40),
        yaxis=dict(title="m/s", showgrid=True, gridcolor="#1e3a5f"),
        yaxis2=dict(title="hPa", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig_wp, use_container_width=True)

    # Wave height
    if "wave_height" in hist_slice.columns:
        st.markdown("#### Wave Height")
        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(
            x=hist_slice.index, y=hist_slice["wave_height"],
            name="Wave Height (m)", line=dict(color="#7b5eff", width=1.2),
            fill="tozeroy", fillcolor="rgba(123,94,255,0.08)",
        ))
        fig_wave.update_layout(
            template="plotly_dark", paper_bgcolor="#0a1118", plot_bgcolor="#0f1923",
            height=220, margin=dict(l=50, r=20, t=20, b=40),
            yaxis_title="m", xaxis_title="Date (UTC)",
        )
        st.plotly_chart(fig_wave, use_container_width=True)

# ── Tab 2: Disruption events ──────────────────────────────────────────────────────

with tabs[1]:
    st.markdown("### Disruption Events")

    if hindcast.empty:
        st.info(
            f"No trained DisruptionAlert model for **{PORTS[port]['label']}**. "
            f"Run `conda run -n personal python train.py --port {port}` to enable this tab."
        )
    else:
        hindcast_slice = hindcast[hindcast.index >= cutoff]
        daily_prob = hindcast_slice.resample("D").mean().rename("alert_prob")

        # Daily disruption probability bar chart
        bar_colors = [
            "#e84545" if v >= 0.40 else "#f5a623" if v >= 0.15 else "#00d4aa"
            for v in daily_prob.values
        ]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=daily_prob.index,
            y=daily_prob.values,
            marker_color=bar_colors,
            name="P(disruption in 24h)",
            hovertemplate="%{x|%Y-%m-%d}<br>P: %{y:.1%}<extra></extra>",
        ))
        fig_bar.add_hline(
            y=0.40, line_dash="dot", line_color="#e84545",
            annotation_text="HIGH threshold", annotation_position="top right",
            annotation_font_color="#e84545",
        )
        fig_bar.add_hline(
            y=0.15, line_dash="dot", line_color="#f5a623",
            annotation_text="ELEVATED threshold", annotation_position="top right",
            annotation_font_color="#f5a623",
        )
        fig_bar.update_layout(
            template="plotly_dark", paper_bgcolor="#0a1118", plot_bgcolor="#0f1923",
            height=320, margin=dict(l=50, r=60, t=30, b=50),
            xaxis_title="Date",
            yaxis=dict(title="Daily avg P(disruption in 24h)", tickformat=".0%", range=[0, 1]),
            title=dict(text="Daily Disruption Probability (DisruptionAlert 24h)", font_color="#aaa", font_size=13),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Top 10 highest-risk episodes
        st.markdown("#### Top 10 Highest-Risk Episodes")
        top_h = hindcast_slice.nlargest(200)
        # Cluster into episodes (gap > 6h = new episode)
        if not top_h.empty:
            top_h_sorted = top_h.sort_index()
            episode_id = (top_h_sorted.index.to_series().diff() > pd.Timedelta(hours=6)).cumsum()
            episodes = []
            for ep_num, grp in top_h_sorted.groupby(episode_id):
                peak_ts = grp.idxmax()
                peak_p  = grp.max()
                # Get weather conditions at peak
                row_data = {"Date": peak_ts.strftime("%Y-%m-%d %H:%M"), "P(disruption)": f"{peak_p:.1%}"}
                if peak_ts in hist_slice.index:
                    r = hist_slice.loc[peak_ts]
                    row_data["Wind (m/s)"]  = f"{r.get('wind_speed_10m', float('nan')):.1f}"
                    row_data["Gusts (m/s)"] = f"{r.get('wind_gusts_10m', float('nan')):.1f}"
                    row_data["Wave (m)"]    = f"{r.get('wave_height', float('nan')):.2f}" if "wave_height" in hist_slice.columns else "N/A"
                    row_data["Precip (mm)"] = f"{r.get('precipitation', float('nan')):.2f}"
                else:
                    row_data["Wind (m/s)"] = "N/A"
                    row_data["Gusts (m/s)"] = "N/A"
                    row_data["Wave (m)"]    = "N/A"
                    row_data["Precip (mm)"] = "N/A"
                episodes.append(row_data)

            ep_df = pd.DataFrame(episodes).sort_values("P(disruption)", ascending=False).head(10)
            st.dataframe(ep_df, hide_index=True, use_container_width=True)

        st.divider()

        # Summary statistics
        high_risk_hours = (hindcast_slice >= 0.40).sum()
        elev_risk_hours = ((hindcast_slice >= 0.15) & (hindcast_slice < 0.40)).sum()
        total_hours     = len(hindcast_slice)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("HIGH risk hours", f"{high_risk_hours}", f"{high_risk_hours / max(total_hours, 1):.1%} of period")
        with c2:
            st.metric("ELEVATED risk hours", f"{elev_risk_hours}", f"{elev_risk_hours / max(total_hours, 1):.1%} of period")
        with c3:
            st.metric("Period analysed", f"{total_hours}h", f"{days_back} days")

# ── Tab 3: Vessel Traffic (Houston only) ─────────────────────────────────────────

if show_ais_tab:
    with tabs[2]:
        st.markdown("### AIS Vessel Traffic — Houston Ship Channel")

        # Slice AIS to date range
        if ais_df.index.tz is not None:
            ais_df.index = ais_df.index.tz_localize(None)
        ais_slice = ais_df[ais_df.index >= cutoff].copy()

        if ais_slice.empty:
            st.warning("No AIS data in the selected date range.")
        else:
            # Dual-axis: n_vessels_moving (bars) vs mean_sog (line)
            fig_ais = go.Figure()

            if "n_vessels_moving" in ais_slice.columns:
                fig_ais.add_trace(go.Bar(
                    x=ais_slice.index,
                    y=ais_slice["n_vessels_moving"],
                    name="Vessels Moving",
                    marker_color="#4a9eff",
                    opacity=0.7,
                    yaxis="y1",
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Moving: %{y}<extra></extra>",
                ))

            if "mean_sog" in ais_slice.columns:
                fig_ais.add_trace(go.Scatter(
                    x=ais_slice.index,
                    y=ais_slice["mean_sog"],
                    name="Mean SOG (knots)",
                    line=dict(color="#00d4aa", width=1.5),
                    yaxis="y2",
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Mean SOG: %{y:.1f} kt<extra></extra>",
                ))

            # Overlay DisruptionAlert hindcast
            if not hindcast.empty:
                hindcast_ais = hindcast[hindcast.index >= cutoff]
                if not hindcast_ais.empty:
                    fig_ais.add_trace(go.Scatter(
                        x=hindcast_ais.index,
                        y=hindcast_ais.values,
                        name="DisruptionAlert P(24h)",
                        line=dict(color="#e84545", width=1.5, dash="dot"),
                        yaxis="y3",
                        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>P(disruption): %{y:.1%}<extra></extra>",
                    ))

            fig_ais.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a1118",
                plot_bgcolor="#0f1923",
                height=420,
                margin=dict(l=60, r=80, t=40, b=60),
                xaxis_title="Date (UTC)",
                yaxis=dict(title="Vessels Moving", showgrid=True, gridcolor="#1e3a5f"),
                yaxis2=dict(title="Mean SOG (knots)", overlaying="y", side="right", showgrid=False),
                yaxis3=dict(
                    title="P(disruption in 24h)",
                    overlaying="y", side="right",
                    anchor="free", position=1.0,
                    tickformat=".0%", range=[0, 1],
                    showgrid=False,
                ),
                legend=dict(orientation="h", y=-0.2),
                title=dict(
                    text="AIS Vessel Activity vs DisruptionAlert (24h window)",
                    font_color="#aaa", font_size=13,
                ),
            )
            st.plotly_chart(fig_ais, use_container_width=True)

            st.caption(
                "**Vessel traffic drop method:** disruption is flagged when `n_vessels_moving` falls below "
                "30% of its 28-day rolling median. This AIS-derived signal is used as Houston's ground "
                "truth label for training DisruptionAlert when `houston_ais_activity.parquet` is present. "
                "Data source: NOAA MarineCadastre AIS (daily files, ~5-month lag)."
            )

            st.divider()

            # Summary metrics
            if "n_vessels_total" in ais_slice.columns and "n_vessels_moving" in ais_slice.columns:
                avg_total   = ais_slice["n_vessels_total"].mean()
                avg_moving  = ais_slice["n_vessels_moving"].mean()
                avg_sog     = ais_slice["mean_sog"].mean() if "mean_sog" in ais_slice.columns else float("nan")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Avg vessels (total)", f"{avg_total:.0f}")
                with c2:
                    st.metric("Avg vessels (moving)", f"{avg_moving:.0f}",
                              f"{avg_moving / max(avg_total, 1):.0%} of total")
                with c3:
                    if not np.isnan(avg_sog):
                        st.metric("Avg mean SOG", f"{avg_sog:.1f} kt")

# ── Footer ───────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Data: Open-Meteo archive · NOAA MarineCadastre AIS (Houston) · "
    "Models: DisruptionAlert 24h hindcast"
)
