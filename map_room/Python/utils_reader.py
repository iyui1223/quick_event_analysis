"""
READER Antarctic surface station I/O and daily-mean computation.

Loads .dat files from BAS READER project. Computes daily means using:
- Standard stations: 6-hourly only (00, 06, 12, 18 UTC)
- Fossil_Bluff: 12-hourly only (00, 12 UTC) for self-consistency

Columns: Year, Month, Day, Hour (UTC), [Minute], SLP (hPa), StnP (hPa), Temp (°C), WindSpd (knots), WindDir (°)
Null = -999.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pandas as pd
import numpy as np

DATA_PAT = re.compile(
    r"^\s*(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(?:(\d{1,2})\s+|\s+)"
    r"(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)"
)
NULL_VAL = -999.0
KNOTS_TO_MS = 0.514444


def _parse_val(x: str) -> float:
    v = float(x)
    return np.nan if v == NULL_VAL or v == -999 else v


def load_reader_surface(path: Path) -> pd.DataFrame | None:
    """
    Load a READER surface .dat file into DataFrame with datetime index.
    Columns: slp (hPa), temp (degC), wspd (knots), wdir (deg). -999 → NaN.
    """
    if not path.exists():
        return None
    lines = path.read_text(errors="replace").splitlines()
    rows = []
    for line in lines:
        m = DATA_PAT.match(line)
        if m:
            g = m.groups()
            y, mo, d, h = int(g[0]), int(g[1]), int(g[2]), int(g[3])
            mi = int(g[4]) if g[4] is not None else 0
            slp = _parse_val(g[5])
            stnp = _parse_val(g[6])
            temp = _parse_val(g[7])
            wspd = _parse_val(g[8])
            wdir = _parse_val(g[9])
            try:
                dt = pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi)
                rows.append({
                    "datetime": dt,
                    "slp": slp, "stnp": stnp, "temp": temp, "wspd": wspd, "wdir": wdir,
                })
            except (ValueError, pd.errors.OutOfBoundsDatetime):
                continue
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    return df


DailyMeanRule = Literal["6h", "12h"]


def compute_daily_mean(
    df: pd.DataFrame,
    rule: DailyMeanRule = "6h",
) -> pd.DataFrame:
    """
    Compute daily mean from synoptic obs. Keep only obs at selected hours.

    Wind components (u_obs, v_obs in m/s) are derived from synoptic wspd+wdir
    *before* daily averaging so the result is a proper vector mean.
    wspd_ms is the scalar daily mean wind speed in m/s.

    rule="6h": 00, 06, 12, 18 UTC (4 obs/day)
    rule="12h": 00, 12 UTC (2 obs/day, for Fossil_Bluff)
    """
    hours = {0, 6, 12, 18} if rule == "6h" else {0, 12}
    mask = df.index.hour.isin(hours)
    sub = df.loc[mask].copy()
    if "wspd" in sub.columns and "wdir" in sub.columns:
        wspd_ms = sub["wspd"] * KNOTS_TO_MS
        wdir_rad = np.deg2rad(sub["wdir"])
        sub["u_obs"] = -wspd_ms * np.sin(wdir_rad)
        sub["v_obs"] = -wspd_ms * np.cos(wdir_rad)
        sub["wspd_ms"] = wspd_ms
    daily = sub.resample("D").mean(numeric_only=True)
    return daily


def load_station_daily(
    reader_dir: Path,
    station: str,
    rule: DailyMeanRule = "6h",
    combine_faraday_vernadsky: bool = True,
) -> pd.DataFrame | None:
    """
    Load READER daily means for a station.

    station: Adelaide, Deception, Esperanza, Fossil_Bluff, FaradayVernadsky, Marambio
    combine_faraday_vernadsky: if True, concatenate Faraday + Vernadsky for FaradayVernadsky
    """
    if station == "Fossil_Bluff":
        rule = "12h"
    elif station == "FaradayVernadsky":
        if combine_faraday_vernadsky:
            df_f = load_reader_surface(reader_dir / "Faraday_surface.dat")
            df_v = load_reader_surface(reader_dir / "Vernadsky_surface.dat")
            if df_f is None and df_v is None:
                return None
            dfs = [d for d in (df_f, df_v) if d is not None and len(d) > 0]
            if not dfs:
                return None
            df = pd.concat(dfs).sort_index()
            return compute_daily_mean(df, rule="6h")
        station = "Faraday"  # fallback
    fname = f"{station}_surface.dat"
    if not (reader_dir / fname).exists():
        return None
    df = load_reader_surface(reader_dir / fname)
    if df is None or len(df) == 0:
        return None
    return compute_daily_mean(df, rule)
