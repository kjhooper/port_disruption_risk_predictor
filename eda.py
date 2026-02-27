"""
eda.py — Sprint 2 EDA & Feature Analysis dashboard.

Loads {port}_historical_wide.parquet, runs compute_all_features(), and surfaces
statistics, correlations, zone comparisons, and distributions.

Run with: conda run -n personal streamlit run eda.py
"""

import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde

from fetch import (
    PORTS, zone_points, update_or_fetch, fetch_openmeteo_historical,
    OPENMETEO_VARIABLES, OPENMETEO_MARINE_VARIABLES, AIR_QUALITY_VARIABLES,
)
from features import compute_all_features

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Port EDA",
    page_icon="📊",
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
    """
    Partition all DataFrame columns into four disjoint groups:
      'zone'     — starts with any zone prefix from zone_points(port)
      'rolling'  — rolling stats columns (pattern or known extras)
      'raw'      — original Open-Meteo / AQ variable names
      'computed' — everything else
    """
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


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 Port EDA")
    st.markdown("*Sprint 2 — EDA & Feature Engineering*")
    st.divider()

    port = st.selectbox(
        "SELECT PORT",
        options=list(PORTS.keys()),
        format_func=lambda k: PORTS[k]["label"],
    )

    days_back = st.slider("HISTORY (days)", min_value=7, max_value=365, value=90, step=7)

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

col_groups  = classify_columns(feat_df, port)
col_to_group = {col: grp for grp, cols in col_groups.items() for col in cols}
numeric_cols = feat_df.select_dtypes(include="number").columns.tolist()

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(f"# {PORTS[port]['label']} — EDA")
st.markdown(f"*{days_back} days · {len(feat_df):,} rows · {len(feat_df.columns)} columns*")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_stats, tab_ts, tab_corr, tab_zones, tab_dist = st.tabs([
    "📋 Stats", "📈 Time Series", "🔗 Correlation", "🗺 Zones", "📊 Distributions"
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Stats Table
# ══════════════════════════════════════════════════════════════════════════════

with tab_stats:
    st.markdown("### Feature Statistics")
    st.markdown("Surface low-signal and high-missing columns at a glance.")

    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        filter_groups = st.multiselect(
            "Filter by group",
            options=["raw", "computed", "zone", "rolling"],
            default=["raw", "computed", "zone", "rolling"],
        )
    with ctrl2:
        sort_by = st.selectbox(
            "Sort by",
            options=["pct_missing", "pct_zero", "variance_rank", "mean", "std"],
            index=0,
        )
    with ctrl3:
        ascending = st.radio("Order", ["Ascending", "Descending"], index=0) == "Ascending"

    filtered_cols = [c for c in numeric_cols if col_to_group.get(c) in filter_groups]

    if not filtered_cols:
        st.info("No columns match the selected groups.")
    else:
        n_rows = len(feat_df)
        stats_rows = []
        for col in filtered_cols:
            s = feat_df[col].dropna()
            stats_rows.append({
                "column":      col,
                "group":       col_to_group.get(col, "?"),
                "mean":        s.mean()  if len(s) > 0 else float("nan"),
                "std":         s.std()   if len(s) > 0 else float("nan"),
                "min":         s.min()   if len(s) > 0 else float("nan"),
                "max":         s.max()   if len(s) > 0 else float("nan"),
                "pct_missing": feat_df[col].isna().sum() / n_rows * 100,
                "pct_zero":    (feat_df[col] == 0).sum() / n_rows * 100,
            })

        stats_df = pd.DataFrame(stats_rows)
        stats_df["variance_rank"] = stats_df["std"].pow(2).rank(ascending=False).astype("Int64")
        stats_df = stats_df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

        def _color_missing(val):
            if val >= 25:
                return "color: #e84545"
            elif val >= 10:
                return "color: #f5a623"
            return "color: #00d4aa"

        styled = (
            stats_df.style
            .map(_color_missing, subset=["pct_missing"])
            .format({
                "mean": "{:.3f}", "std": "{:.3f}",
                "min": "{:.3f}",  "max": "{:.3f}",
                "pct_missing": "{:.1f}%", "pct_zero": "{:.1f}%",
            })
        )
        st.dataframe(styled, width="stretch", height=400)

        m1, m2, m3 = st.columns(3)
        m1.metric("Low-variance columns (std < 0.01)", int((stats_df["std"] < 0.01).sum()))
        m2.metric("High-missing columns (≥10%)",       int((stats_df["pct_missing"] >= 10).sum()))
        m3.metric("High-zero-rate columns (≥50%)",     int((stats_df["pct_zero"] >= 50).sum()))


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Time Series Explorer
# ══════════════════════════════════════════════════════════════════════════════

with tab_ts:
    st.markdown("### Time Series Explorer")

    defaults = [c for c in ["wind_speed_10m", "onshore_wind", "storm_approach_index"]
                if c in numeric_cols]

    ts_c1, ts_c2 = st.columns([3, 1])
    with ts_c1:
        selected_cols = st.multiselect(
            "Columns to plot",
            options=numeric_cols,
            default=defaults,
        )
    with ts_c2:
        normalise = st.toggle("Normalise (z-score)", value=False)

    # Zone shortcut — auto-adds port + zone versions of a raw variable
    raw_in_df = [c for c in OPENMETEO_VARIABLES + OPENMETEO_MARINE_VARIABLES
                 if c in feat_df.columns]
    zone_shortcut = st.selectbox(
        "Zone shortcut — select a variable to overlay all zone versions",
        options=["(none)"] + raw_in_df,
    )

    plot_cols = list(selected_cols)
    if zone_shortcut != "(none)":
        if zone_shortcut in feat_df.columns and zone_shortcut not in plot_cols:
            plot_cols = [zone_shortcut] + plot_cols
        for z in zone_points(port):
            zc = f"{z['prefix']}_{zone_shortcut}"
            if zc in feat_df.columns and zc not in plot_cols:
                plot_cols.append(zc)

    if not plot_cols:
        st.info("Select at least one column to plot.")
    else:
        plot_df = feat_df[plot_cols].copy()

        if normalise:
            for col in plot_cols:
                mu, sigma = plot_df[col].mean(), plot_df[col].std()
                if sigma and sigma > 0:
                    plot_df[col] = (plot_df[col] - mu) / sigma
            yaxis_label = "z-score"
        else:
            yaxis_label = "value"

        fig_ts = go.Figure()
        for i, col in enumerate(plot_cols):
            color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
            dash  = "dot" if col_to_group.get(col) == "zone" else "solid"
            fig_ts.add_trace(go.Scatter(
                x=feat_df.index,
                y=plot_df[col],
                name=col,
                line=dict(color=color, width=1.5, dash=dash),
                hovertemplate=f"{col}: %{{y:.3f}}<extra></extra>",
            ))

        fig_ts.add_vline(x=datetime.utcnow(), line_dash="dash", line_color="white", opacity=0.3)
        fig_ts.update_layout(**make_layout(
            height=450,
            xaxis_title="Date (UTC)",
            yaxis_title=yaxis_label,
        ))
        st.plotly_chart(fig_ts, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════

with tab_corr:
    st.markdown("### Correlation Heatmap")
    st.markdown("Default: computed features only. Zone columns excluded (see Tab 4).")

    cr_c1, cr_c2 = st.columns(2)
    with cr_c1:
        include_raw     = st.checkbox("Include raw columns",     value=False)
    with cr_c2:
        include_rolling = st.checkbox("Include rolling columns", value=False)

    corr_cols = list(col_groups.get("computed", []))
    if include_raw:
        corr_cols += col_groups.get("raw", [])
    if include_rolling:
        corr_cols += col_groups.get("rolling", [])

    num_set  = set(feat_df.select_dtypes(include="number").columns)
    corr_cols = [c for c in corr_cols if c in num_set]

    if len(corr_cols) > 60:
        variances = feat_df[corr_cols].var().sort_values(ascending=False)
        corr_cols = variances.head(60).index.tolist()
        st.warning("More than 60 columns — showing top 60 by variance.")

    if len(corr_cols) < 2:
        st.info("Need at least 2 numeric columns for a correlation heatmap.")
    else:
        corr_matrix  = feat_df[corr_cols].corr()
        chart_height = max(400, len(corr_cols) * 22)

        fig_corr = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_cols,
            y=corr_cols,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=8),
        ))
        fig_corr.update_layout(**make_layout(
            height=chart_height,
            xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
        ))
        st.plotly_chart(fig_corr, width="stretch")

        st.markdown("#### Top Redundant Pairs (|r| > 0.9)")
        mask        = np.tril(np.ones(corr_matrix.shape, dtype=bool))
        corr_masked = corr_matrix.where(~mask)
        pairs = (
            corr_masked.stack()
            .reset_index()
            .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "correlation"})
        )
        redundant = (
            pairs[pairs["correlation"].abs() > 0.9]
            .sort_values("correlation", key=abs, ascending=False)
            .assign(correlation=lambda d: d["correlation"].round(4))
        )
        if redundant.empty:
            st.info("No feature pairs with |r| > 0.9.")
        else:
            st.dataframe(redundant, width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Zone Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab_zones:
    st.markdown("### Zone Analysis")
    st.markdown("Validate that upstream zone features add signal beyond port-level measurements.")

    zones = zone_points(port)
    if not zones:
        st.info("No zone data configured for this port.")
    else:
        # Variables that exist at port level AND in at least one zone column
        zone_raw_vars: set = set()
        for z in zones:
            prefix = z["prefix"]
            for col in feat_df.columns:
                if col.startswith(f"{prefix}_"):
                    base = col[len(f"{prefix}_"):]
                    if base in feat_df.columns:
                        zone_raw_vars.add(base)

        zoneable_vars = sorted(zone_raw_vars)

        if not zoneable_vars:
            st.info("No zoneable variables found — ensure parquet contains zone-prefixed columns.")
        else:
            zone_var_sel = st.selectbox(
                "Select variable for zone comparison:",
                options=zoneable_vars,
            )

            # ── Section A: Overlay ────────────────────────────────────────────
            st.markdown("#### Section A — Port vs Zone Overlay")
            fig_a = go.Figure()

            if zone_var_sel in feat_df.columns:
                fig_a.add_trace(go.Scatter(
                    x=feat_df.index,
                    y=feat_df[zone_var_sel],
                    name=f"port: {zone_var_sel}",
                    line=dict(color=COLOR_CYCLE[0], width=2),
                ))

            for i, z in enumerate(zones, start=1):
                prefix   = z["prefix"]
                zone_col = f"{prefix}_{zone_var_sel}"
                if zone_col in feat_df.columns:
                    fig_a.add_trace(go.Scatter(
                        x=feat_df.index,
                        y=feat_df[zone_col],
                        name=f"{prefix}: {zone_var_sel} ({z['distance_km']}km)",
                        line=dict(color=COLOR_CYCLE[i % len(COLOR_CYCLE)], width=1.5, dash="dot"),
                    ))

            fig_a.update_layout(**make_layout(
                height=400,
                xaxis_title="Date (UTC)",
                yaxis_title=zone_var_sel,
            ))
            st.plotly_chart(fig_a, width="stretch")

            # ── Section B: Derived Gradients ──────────────────────────────────
            st.markdown("#### Section B — Derived Gradient Features")

            _derived_map = {
                "pressure_msl": (
                    "pressure_gradient",
                    "Positive = port pressure > zone pressure "
                    "(lower pressure offshore → storm approaching).",
                ),
                "cape": (
                    "cape_excess",
                    "Positive = more CAPE energy upstream than at port "
                    "(storm energy building in the approach corridor).",
                ),
                "onshore_wind": (
                    "wind_delta",
                    "Positive = stronger onshore wind upstream "
                    "(wind accelerating toward port).",
                ),
            }

            if zone_var_sel in _derived_map:
                derived_suffix, caption = _derived_map[zone_var_sel]
                fig_b    = go.Figure()
                found_any = False

                for i, z in enumerate(zones):
                    prefix      = z["prefix"]
                    derived_col = f"{prefix}_{derived_suffix}"
                    if derived_col in feat_df.columns:
                        found_any = True
                        fig_b.add_trace(go.Scatter(
                            x=feat_df.index,
                            y=feat_df[derived_col],
                            name=f"{prefix}: {derived_col}",
                            line=dict(color=COLOR_CYCLE[i % len(COLOR_CYCLE)], width=1.5),
                        ))

                if found_any:
                    fig_b.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
                    fig_b.update_layout(**make_layout(
                        height=350,
                        xaxis_title="Date (UTC)",
                        yaxis_title=derived_suffix,
                    ))
                    st.plotly_chart(fig_b, width="stretch")
                    st.caption(caption)
                else:
                    st.info(
                        f"No `{derived_suffix}` columns found. "
                        "Run compute_all_features() to generate derived features."
                    )
            else:
                st.info("No derived gradient columns defined for this variable.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Distributions
# ══════════════════════════════════════════════════════════════════════════════

with tab_dist:
    st.markdown("### Distributions")

    dc1, dc2, dc3 = st.columns([2, 1, 1])
    with dc1:
        dist_col = st.selectbox("Select column:", options=numeric_cols, index=0)
    with dc2:
        n_bins = st.slider("Bins", min_value=20, max_value=200, value=60, step=10)
    with dc3:
        compare_zones = st.checkbox("Compare across zones", value=False)

    series = feat_df[dist_col].dropna()

    if series.empty:
        st.info(f"No non-null data for {dist_col}.")
    else:
        PERCENTILES = [5, 25, 50, 75, 95, 99]
        pct_values  = np.percentile(series, PERCENTILES)

        if not compare_zones:
            fig_d = go.Figure()

            fig_d.add_trace(go.Histogram(
                x=series,
                nbinsx=n_bins,
                histnorm="probability density",
                name=dist_col,
                marker_color="#4a9eff",
                opacity=0.6,
            ))

            try:
                kde     = gaussian_kde(series)
                x_range = np.linspace(float(series.min()), float(series.max()), 300)
                fig_d.add_trace(go.Scatter(
                    x=x_range,
                    y=kde(x_range),
                    name="KDE",
                    line=dict(color="#00d4aa", width=2),
                ))
            except Exception:
                pass  # degenerate distribution (e.g. constant column)

            _pct_colors = ["#666666", "#999999", "#ffffff", "#999999", "#f5a623", "#e84545"]
            for pval, pct, color in zip(pct_values, PERCENTILES, _pct_colors):
                fig_d.add_vline(
                    x=float(pval),
                    line_dash="dash",
                    line_color=color,
                    opacity=0.7,
                    annotation_text=f"p{pct}",
                    annotation_position="top",
                    annotation_font=dict(size=9),
                )

            fig_d.update_layout(**make_layout(
                height=400,
                xaxis_title=dist_col,
                yaxis_title="Density",
            ))
            st.plotly_chart(fig_d, width="stretch")

        else:
            # Zone comparison overlay
            zone_versions = [dist_col] + [
                f"{z['prefix']}_{dist_col}" for z in zone_points(port)
                if f"{z['prefix']}_{dist_col}" in feat_df.columns
            ]
            zone_versions = [c for c in zone_versions if c in feat_df.columns]

            if len(zone_versions) <= 1:
                st.info(f"No zone columns found for `{dist_col}`. Showing port-level only.")

            fig_dz = go.Figure()
            for i, col in enumerate(zone_versions):
                s = feat_df[col].dropna()
                if s.empty:
                    continue
                label = f"port: {dist_col}" if col == dist_col else col
                fig_dz.add_trace(go.Histogram(
                    x=s,
                    nbinsx=n_bins,
                    histnorm="probability density",
                    name=label,
                    marker_color=COLOR_CYCLE[i % len(COLOR_CYCLE)],
                    opacity=0.5,
                ))

            fig_dz.update_layout(
                barmode="overlay",
                **make_layout(height=400, xaxis_title=dist_col, yaxis_title="Density"),
            )
            st.plotly_chart(fig_dz, width="stretch")

        # Percentile table
        st.markdown("#### Percentiles")
        pct_table = (
            pd.DataFrame(
                {"Percentile": [f"p{p}" for p in PERCENTILES], dist_col: pct_values.round(4)}
            )
            .set_index("Percentile")
            .T
        )
        st.dataframe(pct_table, width="stretch")


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption("Sprint 2/5 · EDA & Feature Engineering · Data: Open-Meteo")
