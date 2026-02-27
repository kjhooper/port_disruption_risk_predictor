"""
fetch_disruptions.py — NCEI Storm Events disruption labels for the Houston Ship Channel.

Downloads NOAA National Centers for Environmental Information (NCEI) Storm Events
CSV files (one per year, ~12 MB each) and filters to marine events in the Houston /
Galveston Bay area to create an hourly binary disruption label.

Why this instead of raw AIS?
  - Total download: ~12 MB × 4 years ≈ 50 MB (vs 400+ GB for full AIS backfill)
  - No data lag: 2025 events are already available
  - Independently validated: USCG/NWS-verified marine wind/fog events
  - Direct causal link: Marine Thunderstorm Wind ≥ 34 kt (gale force) stops ships

Data source (public domain, no API key):
  https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/

Usage:
  conda run -n personal python fetch_disruptions.py
  conda run -n personal python fetch_disruptions.py --first-year 2022
"""

import gzip
import io
import re
from pathlib import Path

import pandas as pd
import requests

# ── Constants ─────────────────────────────────────────────────────────────────

NCEI_BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

# Marine zones within or immediately adjacent to the Houston Ship Channel.
# GALVESTON BAY is the primary zone; HIGH IS TO FREEPORT covers the approach.
HOUSTON_ZONES = [
    "GALVESTON BAY",
    "GALVESTON",          # partial match will also catch "GALVESTON BAY"
    "HIGH IS TO FREEPORT",
    "FREEPORT TO MATAGORDA SHIP",
    "MATAGORDA SHIP CHNL",
    "SABINE LAKE",
    "SABINE PASS",
]

# Marine event types that disrupt port operations
DISRUPTING_EVENT_TYPES = frozenset([
    "Marine Thunderstorm Wind",
    "Marine Strong Wind",
    "Marine Dense Fog",
    "Marine Tropical Storm",
    "Marine Hurricane/Typhoon",
    "Waterspout",
])

# Minimum wind speed (knots) for Marine Thunderstorm Wind to count as disruptive.
# 34 kt = Gale Force (Beaufort 8) — threshold at which the Houston Ship Channel
# typically reduces or suspends large vessel traffic.
WIND_KNOT_THRESHOLD = 34

DEFAULT_FIRST_YEAR = 2022


# ── Helpers ───────────────────────────────────────────────────────────────────

def _list_ncei_files() -> dict[int, str]:
    """Return {year: filename} for all available StormEvents detail files."""
    r = requests.get(NCEI_BASE, timeout=30)
    r.raise_for_status()
    # e.g. StormEvents_details-ftp_v1.0_d2022_c20250721.csv.gz
    pattern = r"(StormEvents_details-ftp_v1\.0_d(\d{4})_c\d+\.csv\.gz)"
    found: dict[int, str] = {}
    for fname, year in re.findall(pattern, r.text):
        y = int(year)
        # Keep the most recent version for each year (last match wins)
        found[y] = fname
    return found


def _download_year(filename: str) -> pd.DataFrame:
    """Download one year's Storm Events detail file and return the raw DataFrame."""
    url = NCEI_BASE + filename
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with gzip.open(io.BytesIO(r.content)) as f:
        return pd.read_csv(f, low_memory=False)


def _filter_houston_marine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only Gulf-of-Mexico marine events near the Houston Ship Channel.

    Applies:
    - STATE == 'GULF OF MEXICO'
    - CZ_NAME in HOUSTON_ZONES (substring match)
    - EVENT_TYPE in DISRUPTING_EVENT_TYPES
    - For Marine Thunderstorm Wind: MAGNITUDE >= WIND_KNOT_THRESHOLD
    """
    gulf = df[df["STATE"] == "GULF OF MEXICO"].copy()

    zone_mask = gulf["CZ_NAME"].str.contains(
        "|".join(HOUSTON_ZONES), case=False, na=False
    )
    gulf = gulf[zone_mask]
    gulf = gulf[gulf["EVENT_TYPE"].isin(DISRUPTING_EVENT_TYPES)]

    # For wind events, enforce minimum magnitude
    is_wind = gulf["EVENT_TYPE"] == "Marine Thunderstorm Wind"
    is_strong = gulf["EVENT_TYPE"] == "Marine Strong Wind"
    wind_ok = (is_wind | is_strong) & (gulf["MAGNITUDE"].fillna(0) >= WIND_KNOT_THRESHOLD)
    non_wind = ~(is_wind | is_strong)
    gulf = gulf[wind_ok | non_wind]

    return gulf


def _to_utc_hour(df: pd.DataFrame) -> pd.Series:
    """
    Parse BEGIN_DATE_TIME (CST-6, i.e. UTC-6) and floor to UTC hour.
    NCEI uses CST year-round regardless of DST; we add 6h for UTC.
    """
    local_dt = pd.to_datetime(df["BEGIN_DATE_TIME"], format="%d-%b-%y %H:%M:%S", errors="coerce")
    return (local_dt + pd.Timedelta(hours=6)).dt.floor("h")


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_houston_disruptions(
    save_dir: str | Path = "data",
    first_year: int = DEFAULT_FIRST_YEAR,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Download NCEI Storm Events for first_year → latest available year,
    filter to Houston Ship Channel marine events, and save an hourly
    disruption label to {save_dir}/houston_storm_events.parquet.

    Returns a DataFrame with columns:
      utc_hour          — DatetimeIndex (UTC, hourly)
      disrupted         — 1 if any qualifying event occurred in that hour
      n_events          — count of qualifying events in that hour
      max_wind_kt       — maximum recorded wind speed in that hour (kt)
      event_types       — comma-joined set of event types in that hour
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_raw      = save_dir / "houston_storm_events.parquet"
    out_activity = save_dir / "houston_storm_activity.parquet"

    if verbose:
        print("Looking up available NCEI Storm Events files ...")
    file_map = _list_ncei_files()
    years_to_fetch = [y for y in sorted(file_map) if y >= first_year]

    if verbose:
        print(f"Years available: {sorted(file_map.keys())}")
        print(f"Fetching: {years_to_fetch}\n")

    all_events: list[pd.DataFrame] = []

    for year in years_to_fetch:
        if verbose:
            print(f"  {year}  downloading ...", end=" ", flush=True)
        try:
            raw_df = _download_year(file_map[year])
        except Exception as exc:
            if verbose:
                print(f"SKIP ({exc})")
            continue

        filtered = _filter_houston_marine(raw_df)
        if verbose:
            print(f"{len(filtered):>4} Houston-area marine events")
        if not filtered.empty:
            all_events.append(filtered)

    if not all_events:
        print("\n[WARN] No qualifying events found.")
        return pd.DataFrame()

    events = pd.concat(all_events, ignore_index=True)

    # Save raw event records
    events.to_parquet(out_raw, index=False)
    if verbose:
        print(f"\nSaved {len(events)} raw events → {out_raw}")

    # Build hourly label
    events["utc_hour"] = _to_utc_hour(events)
    events = events.dropna(subset=["utc_hour"])

    hourly = (
        events.groupby("utc_hour")
        .agg(
            n_events=("EVENT_TYPE", "count"),
            max_wind_kt=("MAGNITUDE", "max"),
            event_types=("EVENT_TYPE", lambda x: ", ".join(sorted(x.unique()))),
        )
        .reset_index()
    )
    hourly["disrupted"] = 1

    hourly.to_parquet(out_activity, index=False)
    if verbose:
        span = f"{hourly['utc_hour'].min().date()} → {hourly['utc_hour'].max().date()}"
        event_rate = len(hourly) / (len(hourly) + (8760 * len(years_to_fetch) - len(hourly))) * 100
        print(f"Saved {len(hourly)} disrupted hours ({span}) → {out_activity}")
        print(f"Event rate: ~{event_rate:.1f}% of all hours in the period")

    return hourly


def make_storm_disruption_label(
    activity: pd.DataFrame,
    weather_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Align storm event disruption hours to a weather DataFrame's UTC index.

    Returns a binary Series (0/1) indexed to weather_index.
    Hours not in the storm events data are labelled 0 (no event recorded).
    """
    label = pd.Series(0, index=weather_index, name="disruption", dtype="int8")
    if activity.empty:
        return label

    event_hours = pd.DatetimeIndex(activity["utc_hour"])
    aligned = label.index.isin(event_hours)
    label[aligned] = 1
    return label


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download NCEI Storm Events disruption label for Houston Ship Channel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Saves two files to data/:
  houston_storm_events.parquet   — raw event records (one row per event)
  houston_storm_activity.parquet — hourly: disrupted, n_events, max_wind_kt

Then update train.py to use make_storm_disruption_label() as the M2 ground truth.
        """,
    )
    parser.add_argument(
        "--first-year", type=int, default=DEFAULT_FIRST_YEAR,
        help=f"First year to include (default: {DEFAULT_FIRST_YEAR})",
    )
    parser.add_argument(
        "--save-dir", default="data",
        help="Output directory (default: data/)",
    )
    args = parser.parse_args()

    print(f"Houston Ship Channel — NCEI Storm Events disruption label\n")
    activity = fetch_houston_disruptions(
        save_dir=args.save_dir,
        first_year=args.first_year,
        verbose=True,
    )

    if not activity.empty:
        print(f"\nEvent type breakdown:")
        # Load raw events to show breakdown
        raw = pd.read_parquet(Path(args.save_dir) / "houston_storm_events.parquet")
        print(raw["EVENT_TYPE"].value_counts().to_string())
        print(f"\nMax wind recorded: {raw['MAGNITUDE'].max():.0f} kt")
        print(f"\nNext: conda run -n personal python train.py --port houston")
