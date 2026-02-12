#!/usr/bin/env python3
"""
Spaghetti plot: T2m at a single point (lon -67.75, lat -64.0) cradled by Antarctic Peninsula.
- Target event (black): Feb 1–15 2020 (lead 0–15) from extreme_events.yaml.
- Analogues (from analogues.csv): for each snapshot date, plot T2m from snapshot-7 to snapshot+7 (lead 0–14, 15 days).
- Past analogues: blue (paler = weaker similarity, rank 1=best).
- Present analogues: red (paler = weaker similarity).
Uses ERA5 daily T2m on Lustre: era5_daily_2m_temperature_YYYY.nc
"""
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import yaml
import matplotlib.pyplot as plt

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    "/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/antarctica_peninsula_2020",
)
DATASLICES = os.path.join(PROJECT_ROOT, "dataslices")
EXTREME_EVENTS_YAML = os.environ.get("EXTREME_EVENTS_YAML", os.path.join(DATASLICES, "extreme_events.yaml"))
ANALOGUES_CSV = os.environ.get("ANALOGUES_CSV", os.path.join(DATASLICES, "analogues.csv"))
T2M_DIR = os.environ.get(
    "T2M_DIR",
    "/lustre/soge1/data/analysis/era5/0.28125x0.28125/daily/2m_temperature/nc",
)
T2M_TEMPLATE = "era5_daily_2m_temperature_{year}.nc"
FIGS_DIR = os.environ.get("FIGS_DIR", os.path.join(PROJECT_ROOT, "Figs"))

POINT_LON = -67.75
POINT_LAT = -64.0
LEAD_DAYS = 15  # lead 0..14
K2C = 273.15


def load_target_event(yaml_path, event_name="antarctica_peninsula_2020"):
    """Return (start_date, end_date) for the named event."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    for ev in data.get("events", []):
        if ev.get("name") == event_name:
            start = ev["start_date"]
            end = ev["end_date"]
            return (datetime.strptime(start, "%Y-%m-%d"), datetime.strptime(end, "%Y-%m-%d"))
    raise KeyError(f"Event {event_name!r} not found in {yaml_path}")


def load_analogues(csv_path):
    """Return list of dicts: date (datetime), period ('past'|'present'), rank (1..15)."""
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        date = datetime(int(r["year"]), int(r["month"]), int(r["day"]))
        rows.append({"date": date, "period": r["period"], "rank": int(r["rank"])})
    return rows


def get_t2m_series(start_date, ndays, t2m_dir, point_lon, point_lat):
    """Load ERA5 daily T2m for ndays from start_date at (point_lon, point_lat). Return 1D array in °C."""
    end_date = start_date + timedelta(days=ndays - 1)
    years_needed = set(range(start_date.year, end_date.year + 1))
    series = []
    for year in sorted(years_needed):
        path = os.path.join(t2m_dir, T2M_TEMPLATE.format(year=year))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        ds = xr.open_dataset(path)
        t2m = ds["t2m"]
        # ERA5: lon often 0..360
        lon = t2m.coords["longitude"]
        if float(lon.min()) >= 0 and point_lon < 0:
            lon_val = point_lon + 360
        else:
            lon_val = point_lon
        pt = t2m.sel(latitude=point_lat, longitude=lon_val, method="nearest")
        subset = pt.sel(
            time=slice(
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )
        )
        vals = subset.values - K2C
        series.append(vals)
        ds.close()
    out = np.concatenate(series)
    # ensure exactly ndays (handle leap year / month boundaries)
    return out[:ndays]


def main():
    os.makedirs(FIGS_DIR, exist_ok=True)
    target_start, _ = load_target_event(EXTREME_EVENTS_YAML)
    # Use 15 days (lead 0..14) for all series so target and analogues match
    ndays = LEAD_DAYS

    # Target series (black): 15 days from target start (lead 0 = Feb 1, lead 14 = Feb 15)
    target_series = get_t2m_series(target_start, ndays, T2M_DIR, POINT_LON, POINT_LAT)

    # Analogues: for each, start = snapshot - 7, 16 days
    analogues = load_analogues(ANALOGUES_CSV)
    past_series = []
    present_series = []
    for a in analogues:
        snap = a["date"]
        start = snap - timedelta(days=7)
        try:
            s = get_t2m_series(start, LEAD_DAYS, T2M_DIR, POINT_LON, POINT_LAT)
            if a["period"] == "past":
                past_series.append((a["rank"], s))
            else:
                present_series.append((a["rank"], s))
        except FileNotFoundError as e:
            print("Skip", snap.date(), e)

    fig, ax = plt.subplots(figsize=(8, 5))
    lead = np.arange(LEAD_DAYS)

    # Draw palest first (high rank) so strongest (rank 1) on top
    max_rank = 15
    for rank, s in sorted(past_series, key=lambda x: -x[0]):
        alpha = 0.3 + 0.65 * (1 - (rank - 1) / max_rank)  # rank 1 -> 0.95, rank 15 -> 0.3
        ax.plot(lead, s, color="blue", alpha=alpha, linewidth=1)
    for rank, s in sorted(present_series, key=lambda x: -x[0]):
        alpha = 0.3 + 0.65 * (1 - (rank - 1) / max_rank)
        ax.plot(lead, s, color="red", alpha=alpha, linewidth=1)
    ax.plot(lead, target_series, color="black", linewidth=2.5, label="Target (Feb 2020)")

    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("2 m temperature (°C)")
    ax.set_title(f"T2m at ({POINT_LON}, {POINT_LAT})° — target + analogues")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, LEAD_DAYS - 1)
    plt.tight_layout()
    out = os.path.join(FIGS_DIR, "t2m_spaghetti_peninsula_point.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved", out)


if __name__ == "__main__":
    main()
