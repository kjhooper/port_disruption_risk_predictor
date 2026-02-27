"""
fetch_ais.py — NOAA MarineCadastre AIS data for Houston Ship Channel.

NOAA serves daily bulk files (all US coastal waters) at:
  2022–2024:  AIS_{year}_{month:02d}_{day:02d}.zip       (~300–400 MB/day)
  2025+:      ais-{year}-{month:02d}-{day:02d}.csv.zst   (~200 MB/day)

Each daily file is streamed into RAM, filtered to the Houston Ship Channel
bounding box and commercial vessel types, then discarded. The filtered result
(~50–200 KB/day) is appended to a growing parquet.

Data source (public domain, no API key):
  https://coast.noaa.gov/htdata/CMSP/AISDataHandler/

Usage:
  # Refresh: download the last 30 days (for regular cron-style runs)
  conda run -n personal python fetch_ais.py

  # Download since a specific date
  conda run -n personal python fetch_ais.py --since 2024-01-01

  # Full 3-year backfill (slow — ~400 MB per day × 1000+ days)
  conda run -n personal python fetch_ais.py --full-history
"""

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import zstandard as zstd

# ── Constants ─────────────────────────────────────────────────────────────────

AIS_BASE_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler"

# Port of Houston / Galveston Bay bounding box
HOUSTON_BBOX = dict(lat_min=29.25, lat_max=29.90, lon_min=-95.40, lon_max=-94.50)

# AIS vessel type codes for commercial traffic
# 52=tug, 55=pilot, 60-69=passenger, 70-79=cargo, 80-89=tanker
COMMERCIAL_VESSEL_TYPES = frozenset(range(60, 90)) | {52, 55}

# Canonical output column names (used in parquet and downstream code)
AIS_USE_COLS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "VesselType"]

AIS_DTYPE = {
    "MMSI":       "Int64",
    "LAT":        "float32",
    "LON":        "float32",
    "SOG":        "float32",
    "VesselType": "Int32",
}

# Column aliases: 2022-2024 used PascalCase, 2025+ uses snake_case
# Maps any variant → canonical name
_COL_ALIASES = {
    "mmsi":           "MMSI",
    "base_date_time": "BaseDateTime",
    "basedatetime":   "BaseDateTime",
    "lat":            "LAT",
    "latitude":       "LAT",
    "lon":            "LON",
    "longitude":      "LON",
    "sog":            "SOG",
    "vesseltype":     "VesselType",
    "vessel_type":    "VesselType",
}

DEFAULT_FIRST_YEAR = 2022
DEFAULT_DAYS_BACK  = 30   # used when neither --since nor --full-history is given
MAX_WORKERS        = 2    # concurrent daily-file downloads (limited by RAM)


# ── URL helpers ───────────────────────────────────────────────────────────────

def _ais_url(d: date) -> str:
    """Return the correct NOAA URL for a given date."""
    if d.year <= 2024:
        return (f"{AIS_BASE_URL}/{d.year}/"
                f"AIS_{d.year}_{d.month:02d}_{d.day:02d}.zip")
    else:
        return (f"{AIS_BASE_URL}/{d.year}/"
                f"ais-{d.year}-{d.month:02d}-{d.day:02d}.csv.zst")


# ── Single-day fetch and filter ───────────────────────────────────────────────

def _read_csv_from_bytes(
    buf: bytes,
    is_zst: bool,
    bbox: dict,
    vessel_types: frozenset,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """
    Decompress (zip or zst), read CSV in chunks, filter to bbox + vessel types.
    Normalises column names so 2022-2024 (PascalCase) and 2025+ (snake_case)
    both produce the same canonical AIS_USE_COLS output.
    Returns filtered DataFrame or empty DataFrame.
    """
    if is_zst:
        dctx = zstd.ZstdDecompressor()
        raw  = dctx.stream_reader(io.BytesIO(buf))
    else:
        zf  = zipfile.ZipFile(io.BytesIO(buf))
        raw = zf.open(zf.namelist()[0])

    # Read header to build usecols list using aliases
    import csv as _csv
    header_sample = raw.read(4096)
    if is_zst:
        # re-open after reading header
        dctx = zstd.ZstdDecompressor()
        raw  = dctx.stream_reader(io.BytesIO(buf))
    else:
        zf  = zipfile.ZipFile(io.BytesIO(buf))
        raw = zf.open(zf.namelist()[0])

    first_line = header_sample.split(b"\n")[0].decode("utf-8", errors="replace")
    file_cols  = [c.strip().strip('"') for c in first_line.split(",")]
    # Map file column → canonical name
    col_map    = {c: _COL_ALIASES.get(c.lower(), c) for c in file_cols}
    use_cols   = [c for c in file_cols if _COL_ALIASES.get(c.lower(), c) in AIS_USE_COLS]

    if not use_cols:
        return pd.DataFrame(columns=AIS_USE_COLS)

    # Build dtype map keyed by original file column names
    rev_alias  = {v: k for k, v in _COL_ALIASES.items()}  # canonical → one alias
    raw_dtype  = {}
    for file_col in use_cols:
        canonical = col_map[file_col]
        if canonical in AIS_DTYPE:
            raw_dtype[file_col] = AIS_DTYPE[canonical]

    parts = []
    for chunk in pd.read_csv(
        raw,
        usecols=use_cols,
        dtype=raw_dtype,
        chunksize=chunksize,
        low_memory=False,
    ):
        # Rename to canonical names
        chunk = chunk.rename(columns=col_map)
        in_bbox = (
            chunk["LAT"].between(bbox["lat_min"], bbox["lat_max"]) &
            chunk["LON"].between(bbox["lon_min"], bbox["lon_max"])
        )
        filtered = chunk[in_bbox]
        if vessel_types and "VesselType" in filtered.columns:
            filtered = filtered[filtered["VesselType"].isin(vessel_types)]
        if not filtered.empty:
            parts.append(filtered[AIS_USE_COLS])

    if not parts:
        return pd.DataFrame(columns=AIS_USE_COLS)

    df = pd.concat(parts, ignore_index=True)
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"], utc=True)
    return df.sort_values("BaseDateTime").reset_index(drop=True)


def _fetch_day(
    d: date,
    bbox: dict = HOUSTON_BBOX,
    vessel_types: frozenset = COMMERCIAL_VESSEL_TYPES,
    timeout: int = 600,
) -> tuple[date, pd.DataFrame]:
    """
    Download and filter one day's AIS file.
    Returns (date, filtered_df). filtered_df is empty on 404 or parse error.
    """
    url    = _ais_url(d)
    is_zst = url.endswith(".zst")

    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        if exc.response.status_code == 404:
            return d, pd.DataFrame(columns=AIS_USE_COLS)
        raise

    # Buffer compressed file in RAM (300–400 MB for zip, ~200 MB for zst)
    buf = b"".join(resp.iter_content(chunk_size=4 * 1024 * 1024))
    resp.close()

    df = _read_csv_from_bytes(buf, is_zst, bbox, vessel_types)
    return d, df


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_houston_ais(
    save_dir:     str | Path = "data",
    since:        date | None = None,
    days_back:    int = DEFAULT_DAYS_BACK,
    full_history: bool = False,
    verbose:      bool = True,
) -> pd.DataFrame:
    """
    Download NOAA AIS data for Houston Ship Channel and append to parquet.

    Priority of date-range arguments:
      full_history=True  → start_date = Jan 1 of DEFAULT_FIRST_YEAR
      since=<date>       → start_date = that date
      (default)          → start_date = today - days_back days

    Already-fetched dates are skipped automatically (resume-safe).
    Results are saved to {save_dir}/houston_ais_raw.parquet.

    Returns the complete filtered DataFrame (all dates combined).
    """
    save_dir  = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path  = save_dir / "houston_ais_raw.parquet"
    today     = date.today()

    # ── Determine date range ──────────────────────────────────────────────────
    if full_history:
        start_date = date(DEFAULT_FIRST_YEAR, 1, 1)
    elif since is not None:
        start_date = since if isinstance(since, date) else date.fromisoformat(str(since))
    else:
        start_date = today - timedelta(days=days_back)

    # end_date: yesterday, but cap at ~5 months ago since NOAA publishes with ~5 month lag
    end_date = min(today - timedelta(days=1), today - timedelta(days=150))

    # ── Which dates already in the parquet? ───────────────────────────────────
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        existing["BaseDateTime"] = pd.to_datetime(existing["BaseDateTime"], utc=True)
        done_dates = set(existing["BaseDateTime"].dt.date.unique())
    else:
        existing   = pd.DataFrame(columns=AIS_USE_COLS)
        done_dates = set()

    # ── Build work queue ──────────────────────────────────────────────────────
    work = [
        start_date + timedelta(days=i)
        for i in range((end_date - start_date).days + 1)
        if (start_date + timedelta(days=i)) not in done_dates
    ]

    if not work:
        if verbose:
            print(f"All dates already downloaded ({start_date} → {end_date}). Nothing to do.")
        return existing

    if verbose:
        print(f"Fetching {len(work)} day(s): {work[0]} → {work[-1]}")
        print(f"(~{len(work) * 350 / 1000:.0f} GB to stream, ~{len(work) * 0.1:.0f} MB retained)\n")

    # ── Parallel download ─────────────────────────────────────────────────────
    new_parts: list[pd.DataFrame] = []
    total_new = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_day, d): d for d in work}
        for fut in as_completed(futures):
            d   = futures[fut]
            try:
                d_actual, df = fut.result()
            except Exception as exc:
                if verbose:
                    print(f"  {d}  ERROR: {exc}")
                continue

            n = len(df)
            if verbose:
                print(f"  {d}  {n:>6,} records")

            if n > 0:
                new_parts.append(df)
                total_new += n

    if total_new == 0:
        if verbose:
            print("\nNo new records found (all dates were 404 or empty after filtering).")
        return existing

    # ── Merge and save ────────────────────────────────────────────────────────
    all_parts = ([existing] if not existing.empty else []) + new_parts
    combined  = pd.concat(all_parts, ignore_index=True)
    combined  = combined.sort_values("BaseDateTime").reset_index(drop=True)
    combined.to_parquet(out_path, index=False)

    if verbose:
        span_start = combined["BaseDateTime"].min().strftime("%Y-%m-%d")
        span_end   = combined["BaseDateTime"].max().strftime("%Y-%m-%d")
        print(f"\nSaved {len(combined):,} records ({span_start} → {span_end}) → {out_path}")

    return combined


def build_port_activity(ais_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw AIS positions into hourly port traffic metrics.

    Returns a UTC-indexed DataFrame with:
      n_vessels_total   — unique MMSIs seen in the bbox per hour
      n_vessels_moving  — unique MMSIs with SOG > 1 kt (actually transiting)
      mean_sog          — mean speed of moving vessels (kt)
    """
    df             = ais_df.copy()
    df["hour"]     = df["BaseDateTime"].dt.floor("h")
    df["moving"]   = df["SOG"] > 1.0

    total   = df.groupby("hour")["MMSI"].nunique().rename("n_vessels_total")
    moving  = df[df["moving"]].groupby("hour")["MMSI"].nunique().rename("n_vessels_moving")
    sog_avg = df[df["moving"]].groupby("hour")["SOG"].mean().rename("mean_sog")

    activity = pd.concat([total, moving, sog_avg], axis=1).fillna(0)
    activity.index = pd.DatetimeIndex(activity.index, tz="UTC")
    return activity.sort_index()


def make_ais_disruption_label(
    activity: pd.DataFrame,
    rolling_days: int = 28,
    disruption_threshold: float = 0.30,
    min_baseline_vessels: float = 2.0,
) -> pd.Series:
    """
    Binary disruption label derived from hourly vessel activity.

    A hour is labelled 1 (disrupted) when n_vessels_moving falls below
    disruption_threshold × rolling_baseline AND the baseline indicates the
    port normally has meaningful traffic (≥ min_baseline_vessels).

    Parameters
    ----------
    rolling_days : int
        Look-back window for computing the rolling-median baseline.
    disruption_threshold : float
        Fraction of baseline below which the port is considered disrupted.
        0.30 = traffic must drop to < 30% of normal to be flagged.
    min_baseline_vessels : float
        Ignore hours where even the baseline is near zero (deep-night
        periods) — avoids false positives from low-but-normal traffic.
    """
    n_moving = activity["n_vessels_moving"].copy()
    baseline = n_moving.rolling(
        window=rolling_days * 24,
        min_periods=rolling_days * 24 // 4,
        center=False,
    ).median()

    disrupted = (
        (n_moving < disruption_threshold * baseline) &
        (baseline >= min_baseline_vessels)
    )
    return disrupted.astype("int8").rename("disruption")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch NOAA AIS data for Houston Ship Channel and rebuild activity parquet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regular refresh (last 30 days)
  python fetch_ais.py

  # Fetch from a specific date forward
  python fetch_ais.py --since 2024-01-01

  # Full backfill from 2022 (slow — ~400 MB per day × 1000+ days)
  python fetch_ais.py --full-history
        """,
    )
    parser.add_argument(
        "--since", type=date.fromisoformat, default=None,
        metavar="YYYY-MM-DD",
        help="Fetch all days from this date to yesterday (inclusive).",
    )
    parser.add_argument(
        "--days-back", type=int, default=DEFAULT_DAYS_BACK,
        help=f"Days back from today when --since is not given (default: {DEFAULT_DAYS_BACK}).",
    )
    parser.add_argument(
        "--full-history", action="store_true",
        help=f"Fetch everything from Jan 1 {DEFAULT_FIRST_YEAR} onward.",
    )
    parser.add_argument(
        "--save-dir", default="data",
        help="Directory to save parquet files (default: data/).",
    )
    args = parser.parse_args()

    print(f"Houston AIS refresh — bounding box: {HOUSTON_BBOX}\n")

    ais_df = fetch_houston_ais(
        save_dir=args.save_dir,
        since=args.since,
        days_back=args.days_back,
        full_history=args.full_history,
        verbose=True,
    )

    if ais_df.empty:
        print("No data. Exiting.")
    else:
        print("\nBuilding hourly port activity metrics ...")
        activity    = build_port_activity(ais_df)
        out_activity = Path(args.save_dir) / "houston_ais_activity.parquet"
        activity.to_parquet(out_activity)
        print(f"Activity: {len(activity):,} hours → {out_activity}")

        label      = make_ais_disruption_label(activity)
        event_rate = float(label.mean()) * 100
        n_events   = int(label.sum())
        print(f"Disruption hours: {n_events:,} ({event_rate:.1f}% of all hours)")
        print("\nNext: conda run -n personal python train.py --port houston")
