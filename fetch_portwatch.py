"""
fetch_portwatch.py — IMF PortWatch daily port call data for all three ports.

Queries the public ArcGIS FeatureServer hosted by the IMF/Oxford PortWatch
project. No API key required.

Data source:
  https://portwatch.imf.org
  FeatureServer: services9.arcgis.com/weJ1QsnbMYJlCHdG/.../Daily_Ports_Data

Coverage: 2019-01-01 → present (~7 day publication lag), daily resolution.

Port IDs (confirmed from ports database):
  rotterdam → port1114  (NL RTM)
  houston   → port481   (US HOU)
  hong_kong → port474   (HK HKG)
  kaohsiung → port541   (TW KHH)

Usage:
  conda run -n personal python fetch_portwatch.py
  conda run -n personal python fetch_portwatch.py --ports rotterdam hong_kong
  conda run -n personal python fetch_portwatch.py --since 2022-01-01
"""

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

# ── Constants ─────────────────────────────────────────────────────────────────

PORTWATCH_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"
    "/Daily_Ports_Data/FeatureServer/0/query"
)

# Confirmed portids from PortWatch ports database
PORTWATCH_IDS: dict[str, str] = {
    "rotterdam": "port1114",
    "houston":   "port481",
    "hong_kong": "port474",
    "kaohsiung": "port541",
}

MAX_RECORDS_PER_PAGE = 1000   # API hard limit
DEFAULT_FIRST_DATE   = date(2019, 1, 1)

OUTFIELDS = [
    "portid", "portname", "date",
    "portcalls",
    "portcalls_container", "portcalls_dry_bulk",
    "portcalls_general_cargo", "portcalls_roro", "portcalls_tanker",
]


# ── Fetch helpers ─────────────────────────────────────────────────────────────

def _query_page(portid: str, offset: int, where_extra: str = "") -> list[dict]:
    """
    Fetch one page of daily portcall records for a given portid.
    Returns list of feature attribute dicts.
    """
    where = f"portid = '{portid}'"
    if where_extra:
        where += f" AND {where_extra}"

    params = {
        "where":             where,
        "outFields":         ",".join(OUTFIELDS),
        "orderByFields":     "date ASC",
        "resultOffset":      offset,
        "resultRecordCount": MAX_RECORDS_PER_PAGE,
        "f":                 "json",
    }
    resp = requests.get(PORTWATCH_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    features = data.get("features", [])
    return [f["attributes"] for f in features]


def _fetch_port(
    portid: str,
    since: date | None = None,
) -> pd.DataFrame:
    """
    Fetch all daily records for one portid, paginating as needed.
    Returns DataFrame indexed by UTC date (daily).
    """
    where_extra = ""
    if since:
        # ArcGIS date filter: DATE field using epoch milliseconds or ISO string
        since_str = since.strftime("%Y-%m-%d")
        where_extra = f"date >= DATE '{since_str}'"

    all_rows: list[dict] = []
    offset = 0
    while True:
        page = _query_page(portid, offset, where_extra)
        all_rows.extend(page)
        if len(page) < MAX_RECORDS_PER_PAGE:
            break
        offset += MAX_RECORDS_PER_PAGE

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # ArcGIS returns date as ISO string "YYYY-MM-DD"
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d").dt.date
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")

    return df


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_portwatch(
    save_dir: str | Path = "data",
    ports: list[str] | None = None,
    since: date | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch daily portcall data from IMF PortWatch for one or more ports.

    Parameters
    ----------
    save_dir : directory to save parquet files
    ports    : list of port keys (rotterdam/houston/hong_kong/kaohsiung); None = all
    since    : only fetch records from this date onward; None = 2019-01-01

    Saves:
      {save_dir}/{port}_portwatch_activity.parquet

    Returns dict of port → DataFrame indexed by date.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if ports is None:
        ports = list(PORTWATCH_IDS.keys())

    if since is None:
        since = DEFAULT_FIRST_DATE

    results: dict[str, pd.DataFrame] = {}

    for port in ports:
        portid = PORTWATCH_IDS.get(port)
        if portid is None:
            print(f"[fetch_portwatch] Unknown port '{port}', skipping.")
            continue

        if verbose:
            print(f"  {port} ({portid})  fetching from {since} ...", end=" ", flush=True)

        try:
            df = _fetch_port(portid, since=since)
        except Exception as exc:
            print(f"ERROR: {exc}")
            results[port] = pd.DataFrame()
            continue

        if df.empty:
            print("no data returned")
            results[port] = df
            continue

        out_path = save_dir / f"{port}_portwatch_activity.parquet"
        df.to_parquet(out_path)

        if verbose:
            span = f"{df.index.min()} → {df.index.max()}"
            print(f"{len(df)} days  [{span}]  → {out_path}")

        results[port] = df

    return results


def make_portwatch_disruption_label(
    activity: pd.DataFrame,
    weather_index: pd.DatetimeIndex,
    rolling_days: int = 28,
    disruption_threshold: float = 0.30,
    min_baseline_calls: float = 1.0,
    exclude_holidays: bool = False,
    port: str = "default",
) -> pd.Series:
    """
    Binary disruption label derived from daily PortWatch portcalls.

    A day is labelled 1 (disrupted) when portcalls falls below
    disruption_threshold × rolling_baseline, then forward-filled to every
    hour of that day to match the hourly weather index.

    Parameters
    ----------
    activity             : DataFrame with date index and portcalls column
    weather_index        : hourly UTC DatetimeIndex from the weather parquet
    rolling_days         : look-back window for rolling-median baseline
    disruption_threshold : portcalls must drop below this fraction of baseline
    min_baseline_calls   : ignore days where even baseline is near zero

    Returns
    -------
    pd.Series of int8 (0/1), indexed to weather_index
    """
    if activity.empty or "portcalls" not in activity.columns:
        return pd.Series(0, index=weather_index, dtype="int8", name="disruption")

    portcalls = activity["portcalls"].copy().astype(float)

    # Daily baseline: 28-day rolling median
    baseline = portcalls.rolling(window=rolling_days, min_periods=rolling_days // 4).median()

    disrupted_daily = (
        (portcalls < disruption_threshold * baseline) &
        (baseline >= min_baseline_calls)
    ).astype("int8")

    # Zero out holiday-flagged days so M2 only learns weather disruptions.
    if exclude_holidays:
        from features import make_holiday_features, PORT_HOLIDAY_CALENDARS
        if port in PORT_HOLIDAY_CALENDARS:
            dt_idx = pd.DatetimeIndex([pd.Timestamp(d) for d in disrupted_daily.index])
            hol = make_holiday_features(dt_idx, port)
            disrupted_daily[hol["is_holiday"].values == 1] = 0

    # Forward-fill daily label to hourly weather index.
    # Use tz-naive UTC midnight to match the weather parquet index convention.
    daily_dt = pd.DatetimeIndex(
        [pd.Timestamp(d) for d in disrupted_daily.index]
    )
    disrupted_ts = pd.Series(disrupted_daily.values, index=daily_dt, name="disruption")

    # Reindex to hourly weather index: forward-fill within each day
    label = disrupted_ts.reindex(weather_index, method="ffill").fillna(0).astype("int8")
    return label


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch IMF PortWatch daily portcall data for Rotterdam, Houston, Hong Kong, Kaohsiung.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch all three ports (default: from 2019-01-01)
  python fetch_portwatch.py

  # Fetch only Rotterdam and Hong Kong
  python fetch_portwatch.py --ports rotterdam hong_kong

  # Fetch only recent data
  python fetch_portwatch.py --since 2022-01-01

Saves {port}_portwatch_activity.parquet to data/.
Then run: conda run -n personal python train.py --port all
        """,
    )
    parser.add_argument(
        "--ports", nargs="+",
        choices=list(PORTWATCH_IDS.keys()),
        default=None,
        help="Port(s) to fetch (default: all three)",
    )
    parser.add_argument(
        "--since", type=date.fromisoformat, default=None,
        metavar="YYYY-MM-DD",
        help=f"Fetch from this date onward (default: {DEFAULT_FIRST_DATE})",
    )
    parser.add_argument(
        "--save-dir", default="data",
        help="Output directory (default: data/)",
    )
    args = parser.parse_args()

    print("IMF PortWatch — daily port call data\n")
    results = fetch_portwatch(
        save_dir=args.save_dir,
        ports=args.ports,
        since=args.since,
        verbose=True,
    )

    print("\nSummary:")
    for port, df in results.items():
        if df.empty:
            print(f"  {port:<12}  (no data)")
            continue
        calls_mean = df["portcalls"].mean()
        calls_min  = df["portcalls"].min()
        calls_max  = df["portcalls"].max()
        print(
            f"  {port:<12}  {len(df)} days  "
            f"portcalls: mean={calls_mean:.1f}  min={int(calls_min)}  max={int(calls_max)}"
        )

    print("\nNext: conda run -n personal python train.py --port all")
