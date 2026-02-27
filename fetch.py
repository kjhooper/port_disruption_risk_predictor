"""
fetch.py — Pull weather data from Open-Meteo and NOAA NDBC.
No API keys required for either source.
"""

import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_DAYS_BACK = 1095      # 3 years
ZONE_DISTANCES_KM = [150, 300]
MAX_WORKERS       = 8

# ── Port definitions ──────────────────────────────────────────────────────────

PORTS = {
    "rotterdam": {"lat": 51.95, "lon": 4.14,   "ndbc_buoy": "62081", "label": "Port of Rotterdam", "sea_bearing": 270},
    "houston":   {"lat": 29.75, "lon": -95.35,  "ndbc_buoy": "42035", "label": "Port of Houston",   "sea_bearing": 170},
    "hong_kong": {"lat": 22.28, "lon": 114.16,  "ndbc_buoy": None,    "label": "Port of Hong Kong", "sea_bearing": 90},
    "kaohsiung": {"lat": 22.56, "lon": 120.32,  "ndbc_buoy": None,    "label": "Port of Kaohsiung", "sea_bearing": 120},
}

# ── Variable lists ────────────────────────────────────────────────────────────

OPENMETEO_VARIABLES = [
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "precipitation",
    "pressure_msl",
    "visibility",
    "weather_code",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "cape",
    "lifted_index",
    "cloud_cover",
]

AIR_QUALITY_VARIABLES = [
    "dust",
]

OPENMETEO_MARINE_VARIABLES = [
    "wave_height",
    "wave_period",
    "wind_wave_height",
]

# ── Geometry helpers ──────────────────────────────────────────────────────────

def _offset_latlon(lat: float, lon: float, bearing_deg: float, distance_km: float) -> tuple:
    """
    Compute a new lat/lon offset from (lat, lon) by distance_km along bearing_deg.
    Uses flat-earth approximation; accurate to <0.5% at 300 km.
    """
    bearing_rad = math.radians(bearing_deg)
    dlat = distance_km * math.cos(bearing_rad) / 111.1
    dlon = distance_km * math.sin(bearing_rad) / (111.1 * math.cos(math.radians(lat)))
    return round(lat + dlat, 4), round(lon + dlon, 4)


def zone_points(port: str) -> list:
    """
    Return upstream zone point dicts for the given port.
    Each dict: {prefix, lat, lon, bearing, distance_km}.

    Single-bearing ports (Rotterdam, Houston): prefixes "z150", "z300".
    Multi-bearing ports (Singapore): prefixes "z150b45", "z300b270", etc.
    """
    cfg = PORTS[port]
    sea_bearing = cfg["sea_bearing"]
    bearings = sea_bearing if isinstance(sea_bearing, list) else [sea_bearing]
    multi = len(bearings) > 1

    points = []
    for bearing in bearings:
        for dist in ZONE_DISTANCES_KM:
            prefix = f"z{dist}b{int(bearing)}" if multi else f"z{dist}"
            lat, lon = _offset_latlon(cfg["lat"], cfg["lon"], bearing, dist)
            points.append({
                "prefix":      prefix,
                "lat":         lat,
                "lon":         lon,
                "bearing":     bearing,
                "distance_km": dist,
            })
    return points


# ── Private lat/lon-based fetchers ────────────────────────────────────────────

def _fetch_historical(lat: float, lon: float, days_back: int = DEFAULT_DAYS_BACK) -> pd.DataFrame:
    end_date   = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days_back)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      start_date.isoformat(),
        "end_date":        end_date.isoformat(),
        "hourly":          ",".join(OPENMETEO_VARIABLES),
        "wind_speed_unit": "ms",
        "timezone":        "UTC",
    }
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    raw = resp.json()
    df = pd.DataFrame(raw["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def _fetch_forecast(lat: float, lon: float, days_ahead: int = 7) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "hourly":          ",".join(OPENMETEO_VARIABLES),
        "forecast_days":   days_ahead,
        "wind_speed_unit": "ms",
        "timezone":        "UTC",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    df = pd.DataFrame(raw["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def _fetch_marine_forecast(lat: float, lon: float, days_ahead: int = 7) -> pd.DataFrame:
    """Fetch marine forecast (wave_height, wave_period) from Open-Meteo marine API."""
    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        "latitude":     lat,
        "longitude":    lon,
        "hourly":       ",".join(OPENMETEO_MARINE_VARIABLES),
        "forecast_days": days_ahead,
        "timezone":     "UTC",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
        df = pd.DataFrame(raw["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        return df.set_index("time")
    except Exception as e:
        print(f"[_fetch_marine_forecast] No marine forecast at ({lat}, {lon}): {e}")
        return pd.DataFrame()


def _fetch_marine(lat: float, lon: float, days_back: int = DEFAULT_DAYS_BACK) -> pd.DataFrame:
    end_date   = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days_back)

    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date.isoformat(),
        "end_date":   end_date.isoformat(),
        "hourly":     ",".join(OPENMETEO_MARINE_VARIABLES),
        "timezone":   "UTC",
    }
    try:
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        raw = resp.json()
        df = pd.DataFrame(raw["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        return df.set_index("time")
    except Exception as e:
        print(f"[_fetch_marine] No marine data at ({lat}, {lon}): {e}")
        return pd.DataFrame()


def _fetch_airquality(lat: float, lon: float, days_back: int = DEFAULT_DAYS_BACK) -> pd.DataFrame:
    end_date   = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days_back)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date.isoformat(),
        "end_date":   end_date.isoformat(),
        "hourly":     ",".join(AIR_QUALITY_VARIABLES),
        "timezone":   "UTC",
    }
    try:
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        raw = resp.json()
        df = pd.DataFrame(raw["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        return df.set_index("time")
    except Exception as e:
        print(f"[_fetch_airquality] No air quality data at ({lat}, {lon}): {e}")
        return pd.DataFrame()


def _fetch_buoy(buoy_id: str, days_back: int = 45) -> pd.DataFrame:
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{buoy_id}.txt"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        cols = lines[0].lstrip("#").split()
        data_lines = [l.split() for l in lines[2:]]
        df = pd.DataFrame(data_lines, columns=cols)
        df["time"] = pd.to_datetime(
            df["YY"].astype(str) + "-" +
            df["MM"].astype(str).str.zfill(2) + "-" +
            df["DD"].astype(str).str.zfill(2) + " " +
            df["hh"].astype(str).str.zfill(2) + ":" +
            df["mm"].astype(str).str.zfill(2),
            errors="coerce"
        )
        df = df.set_index("time").sort_index()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        df = df.replace([99, 999, 9999, 99.0, 999.0, 9999.0], float("nan"))
        cutoff = datetime.utcnow() - timedelta(days=days_back)
        return df[df.index >= cutoff]
    except Exception as e:
        print(f"[_fetch_buoy] Failed for buoy {buoy_id}: {e}")
        return pd.DataFrame()


# ── Main parallel fetch ───────────────────────────────────────────────────────

def fetch_port_all(port: str, save_dir: str = "data", days_back: int = DEFAULT_DAYS_BACK) -> dict:
    """
    Fetch all data sources for a port in parallel using ThreadPoolExecutor.

    Saves:
      {port}_forecast.parquet         — 7-day forecast
      {port}_buoy.parquet             — NOAA buoy (if available)
      {port}_historical.parquet       — atmospheric + marine + AQ merged
      {port}_historical_wide.parquet  — historical + all zone data (main pipeline input)

    Returns dict of all result DataFrames.
    """
    cfg = PORTS[port]
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    lat, lon = cfg["lat"], cfg["lon"]
    zones = zone_points(port)

    end_date   = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days_back)

    # Build job dict: key -> (fn, args_tuple)
    jobs = {
        "hist":       (_fetch_historical, (lat, lon, days_back)),
        "fore":       (_fetch_forecast,   (lat, lon, 7)),
        "marine":     (_fetch_marine,     (lat, lon, days_back)),
        "airquality": (_fetch_airquality, (lat, lon, days_back)),
    }
    if cfg.get("ndbc_buoy"):
        jobs["buoy"] = (_fetch_buoy, (cfg["ndbc_buoy"], 45))

    for z in zones:
        jobs[f"zone_{z['prefix']}_hist"] = (_fetch_historical, (z["lat"], z["lon"], days_back))
        jobs[f"zone_{z['prefix']}_fore"] = (_fetch_forecast,   (z["lat"], z["lon"], 7))

    # key -> human-readable location label for progress output
    job_labels = {
        "hist":       f"port ({lat}, {lon})",
        "fore":       f"port ({lat}, {lon})",
        "marine":     f"port ({lat}, {lon})",
        "airquality": f"port ({lat}, {lon})",
    }
    if cfg.get("ndbc_buoy"):
        job_labels["buoy"] = f"buoy {cfg['ndbc_buoy']}"
    for z in zones:
        job_labels[f"zone_{z['prefix']}_hist"] = f"{z['prefix']} ({z['lat']}, {z['lon']})  {z['distance_km']}km bearing={z['bearing']}°"
        job_labels[f"zone_{z['prefix']}_fore"] = f"{z['prefix']} ({z['lat']}, {z['lon']})  {z['distance_km']}km bearing={z['bearing']}°"

    print(f"\n{'='*60}")
    print(f"  {cfg['label']}  —  {len(jobs)} parallel jobs")
    print(f"  Port:  ({lat}, {lon})")
    print(f"  Range: {start_date} → {end_date}  ({days_back} days)")
    if zones:
        print(f"  Zones:")
        for z in zones:
            print(f"    {z['prefix']:12s}  lat={z['lat']:7.4f}  lon={z['lon']:8.4f}  {z['distance_km']}km  bearing={z['bearing']}°")
    print(f"{'='*60}")

    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fn, *args): key for key, (fn, args) in jobs.items()}
        for future in as_completed(futures):
            key = futures[future]
            ts  = datetime.utcnow().strftime("%H:%M:%S")
            try:
                results[key] = future.result()
                df = results[key]
                shape_str = f"{df.shape[0]} rows" if not df.empty else "empty"
                date_str  = ""
                if not df.empty and hasattr(df.index, "min"):
                    date_str = f"  [{df.index.min().date()} → {df.index.max().date()}]"
                print(f"  {ts}  ✓ {key:<30s}  {job_labels.get(key, '')}  {shape_str}{date_str}")
            except Exception as e:
                print(f"  {ts}  ✗ {key:<30s}  {job_labels.get(key, '')}  ERROR: {e}")
                results[key] = pd.DataFrame()

    # ── Save forecast (merge marine wave forecast if available) ───────────────
    fore_df = results.get("fore", pd.DataFrame())
    if not fore_df.empty:
        cfg = PORTS[port]
        marine_fore = _fetch_marine_forecast(cfg["lat"], cfg["lon"], days_ahead=7)
        if not marine_fore.empty:
            for col in ["wave_height", "wave_period"]:
                if col in marine_fore.columns:
                    fore_df[col] = marine_fore[col].reindex(fore_df.index)
        fore_df.attrs.update({"port": port, "source": "open-meteo-forecast"})
        fore_df.to_parquet(f"{save_dir}/{port}_forecast.parquet")

    # ── Save buoy ─────────────────────────────────────────────────────────────
    buoy_df = results.get("buoy", pd.DataFrame())
    if not buoy_df.empty:
        buoy_df.attrs.update({"port": port, "source": "noaa-ndbc"})
        buoy_df.to_parquet(f"{save_dir}/{port}_buoy.parquet")

    # ── Build and save historical (port-level merge) ───────────────────────────
    hist_df   = results.get("hist",       pd.DataFrame())
    marine_df = results.get("marine",     pd.DataFrame())
    aq_df     = results.get("airquality", pd.DataFrame())

    if not hist_df.empty:
        if not marine_df.empty:
            hist_df = hist_df.join(marine_df, how="left", rsuffix="_marine")
        if not aq_df.empty:
            hist_df = hist_df.join(aq_df, how="left", rsuffix="_aq")
        hist_df.attrs.update({"port": port, "source": "open-meteo-archive"})
        hist_df.to_parquet(f"{save_dir}/{port}_historical.parquet")

    # ── Build and save historical_wide (historical + zone data) ───────────────
    wide_df = hist_df.copy() if not hist_df.empty else pd.DataFrame()

    for z in zones:
        prefix    = z["prefix"]
        zone_hist = results.get(f"zone_{prefix}_hist", pd.DataFrame())
        if not zone_hist.empty:
            zone_prefixed = zone_hist.add_prefix(f"{prefix}_")
            if wide_df.empty:
                wide_df = zone_prefixed
            else:
                wide_df = wide_df.join(zone_prefixed, how="left")

    if not wide_df.empty:
        wide_df.attrs["port"] = port
        wide_df.to_parquet(f"{save_dir}/{port}_historical_wide.parquet")

    print(f"Done. Saved to {save_dir}/")

    return {
        "historical":      hist_df,
        "historical_wide": wide_df,
        "forecast":        fore_df,
        "buoy":            buoy_df,
        "marine":          marine_df,
        "airquality":      aq_df,
        **{k: v for k, v in results.items() if k.startswith("zone_")},
    }


# ── Smart incremental update ──────────────────────────────────────────────────

def update_or_fetch(port: str, save_dir: str = "data", full_days_back: int = DEFAULT_DAYS_BACK) -> dict:
    """
    Smart update: if cached parquet exists, only fetch missing recent data.
    - If < 1 hour old: skip entirely.
    - Else: fetch enough days to fill the gap, concat with existing, re-save.
    Falls back to full fetch if no parquet exists.
    """
    wide_path = Path(save_dir) / f"{port}_historical_wide.parquet"
    fore_path = Path(save_dir) / f"{port}_forecast.parquet"

    if wide_path.exists() and fore_path.exists():
        existing  = pd.read_parquet(wide_path)
        last_ts   = existing.index.max()
        age_hours = (datetime.utcnow() - last_ts.to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600

        if age_hours < 1:
            print(f"[update_or_fetch] {port}: data is fresh ({age_hours:.1f}h old), skipping.")
            return {"historical_wide": existing}

        days_to_fetch = max(2, int(age_hours / 24) + 2)
        print(f"[update_or_fetch] {port}: fetching last {days_to_fetch} days to fill {age_hours:.1f}h gap.")

        new_data = fetch_port_all(port, save_dir=save_dir, days_back=days_to_fetch)
        new_wide = new_data.get("historical_wide", pd.DataFrame())

        if not new_wide.empty:
            combined = pd.concat([existing, new_wide])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            combined.to_parquet(str(wide_path))
            print(f"[update_or_fetch] {port}: merged, {len(combined)} rows total.")
            return {"historical_wide": combined}

        return {"historical_wide": existing}

    else:
        print(f"[update_or_fetch] {port}: no cached data found, running full fetch.")
        return fetch_port_all(port, save_dir=save_dir, days_back=full_days_back)


# ── Backward-compatible public wrappers ───────────────────────────────────────

def fetch_openmeteo_historical(port: str, days_back: int = DEFAULT_DAYS_BACK) -> pd.DataFrame:
    """Fetch hourly historical weather for a port. Returns DataFrame indexed by datetime (UTC)."""
    cfg = PORTS[port]
    df  = _fetch_historical(cfg["lat"], cfg["lon"], days_back)
    df.attrs.update({"port": port, "source": "open-meteo-archive"})
    return df


def fetch_openmeteo_forecast(port: str, days_ahead: int = 7) -> pd.DataFrame:
    """Fetch hourly forecast for a port, including wave variables where available."""
    cfg    = PORTS[port]
    df     = _fetch_forecast(cfg["lat"], cfg["lon"], days_ahead)
    marine = _fetch_marine_forecast(cfg["lat"], cfg["lon"], days_ahead)
    if not marine.empty:
        for col in ["wave_height", "wave_period"]:
            if col in marine.columns:
                df[col] = marine[col].reindex(df.index)
    df.attrs.update({"port": port, "source": "open-meteo-forecast"})
    return df


def fetch_openmeteo_marine(port: str, days_back: int = DEFAULT_DAYS_BACK) -> pd.DataFrame:
    """Fetch hourly historical marine conditions. Falls back gracefully if unavailable."""
    cfg = PORTS[port]
    df  = _fetch_marine(cfg["lat"], cfg["lon"], days_back)
    if not df.empty:
        df.attrs.update({"port": port, "source": "open-meteo-marine"})
    return df


def fetch_openmeteo_airquality(port: str, days_back: int = DEFAULT_DAYS_BACK) -> pd.DataFrame:
    """Fetch hourly dust concentration from Open-Meteo Air Quality API."""
    cfg = PORTS[port]
    df  = _fetch_airquality(cfg["lat"], cfg["lon"], days_back)
    if not df.empty:
        df.attrs.update({"port": port, "source": "open-meteo-airquality"})
    return df


def fetch_noaa_buoy(port: str, days_back: int = 45) -> pd.DataFrame:
    """Fetch recent NOAA NDBC buoy observations. Returns empty DataFrame if unavailable."""
    cfg     = PORTS[port]
    buoy_id = cfg.get("ndbc_buoy")
    if not buoy_id:
        print(f"[fetch_noaa_buoy] No NDBC buoy configured for {port}.")
        return pd.DataFrame()
    df = _fetch_buoy(buoy_id, days_back)
    if not df.empty:
        df.attrs.update({"port": port, "source": "noaa-ndbc", "buoy_id": buoy_id})
    return df


fetch_all = fetch_port_all   # backward-compatible alias


def _df_summary(label: str, df: pd.DataFrame, extra: str = "") -> None:
    """Print a one-line summary of shape, lat/lon (if stored in attrs), and date range."""
    if df.empty:
        print(f"  {label:<20s}  (empty){extra}")
        return
    date_str = ""
    if hasattr(df.index, "min") and not df.index.empty:
        try:
            date_str = f"  [{df.index.min().date()} → {df.index.max().date()}]"
        except Exception:
            pass
    print(f"  {label:<20s}  shape={str(df.shape):<14s}  cols={df.shape[1]}{date_str}{extra}")


if __name__ == "__main__":
    port = "rotterdam"
    data = fetch_port_all(port, save_dir="data")
    cfg  = PORTS[port]

    print("\n── Port-level tables ─────────────────────────────────────")
    _df_summary("historical",      data["historical"],
                f"  lat={cfg['lat']}  lon={cfg['lon']}")
    _df_summary("forecast",        data["forecast"],
                f"  lat={cfg['lat']}  lon={cfg['lon']}")
    _df_summary("marine",          data["marine"],
                f"  lat={cfg['lat']}  lon={cfg['lon']}")
    _df_summary("airquality",      data["airquality"],
                f"  lat={cfg['lat']}  lon={cfg['lon']}")
    _df_summary("buoy",            data["buoy"],
                f"  buoy={cfg.get('ndbc_buoy', 'n/a')}")

    print("\n── Zone tables ───────────────────────────────────────────")
    for z in zone_points(port):
        df_zh = data.get(f"zone_{z['prefix']}_hist", pd.DataFrame())
        df_zf = data.get(f"zone_{z['prefix']}_fore", pd.DataFrame())
        loc = f"  lat={z['lat']}  lon={z['lon']}  {z['distance_km']}km  bearing={z['bearing']}°"
        _df_summary(f"{z['prefix']}_hist", df_zh, loc)
        _df_summary(f"{z['prefix']}_fore", df_zf, loc)

    print("\n── historical_wide ───────────────────────────────────────")
    wide = data["historical_wide"]
    _df_summary("historical_wide", wide)
    if not wide.empty:
        zone_cols = [c for c in wide.columns if c[:1] == "z"]
        print(f"  zone cols ({len(zone_cols)}): {zone_cols}")
