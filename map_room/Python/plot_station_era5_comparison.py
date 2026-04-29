#!/usr/bin/env python3
"""
F08: Compare READER station obs vs reanalysis nearest-neighbour time series.

Stations: Adelaide, Deception, Esperanza, Fossil_Bluff, FaradayVernadsky, Marambio.
Reanalyses: ERA5 daily surface NetCDF/zip archives and JRA-3Q anl_surf125 GRIB2.
Variables: t2m (°C), msl/prmsl (hPa), u10/v10/wspd (m/s).

Reanalysis point extractions are cached under
``<data-dir>/reanalysis_cache/<REANALYSIS>/<Station>.pkl``.

Workflow:
  1. First run (or after source change):  --clear-cache  (rebuilds from raw files)
  2. Subsequent runs:                     --use-cached   (fast figure-only replot)
  3. Default (no flag):                   extract + overwrite cache

Outputs per station:
  - 3-panel daily time series (t2m with trend, SLP, wind speed)
  - Decadal trend (annual + 10-yr running mean + linear fit)
  - Monthly-variance time series for t2m, SLP, wspd (with ACC)
  - Wind-component monthly-variance panel (u, v, wspd) if obs have wdir
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy import stats

from utils_reader import load_station_daily

ROOT = Path(__file__).resolve().parents[1]

ERA5_DIR_DEFAULT = Path(
    "/lustre/soge1/projects/andante/cenv1201/heavy/ERA5/daily/Surf"
)
READER_DIR_DEFAULT = Path(
    "/lustre/soge1/projects/andante/cenv1201/heavy/READER/SURFACE"
)
JRA3Q_DIR_DEFAULT = Path("/lustre/soge1/data/analysis/jra-q3/anl_surf125")

REANALYSIS_STYLES: dict[str, dict] = {
    "ERA5": {"color": "r"},
    "JRA-3Q": {"color": "g"},
}
OBS_STYLE = {"color": "b"}


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _std_coords(ds: xr.Dataset) -> xr.Dataset:
    renames = {}
    for c in list(ds.coords):
        cl = c.lower()
        if "lat" in cl and c != "lat":
            renames[c] = "lat"
        elif "lon" in cl and c != "lon":
            renames[c] = "lon"
        elif ("time" in cl or "valid" in cl) and c != "time":
            renames[c] = "time"
    return ds.rename(renames) if renames else ds


# ---------------------------------------------------------------------------
# ERA5 monthly-zip I/O
# ---------------------------------------------------------------------------

_ERA5_VAR_PATTERNS = {
    "2m_temperature": "t2m",
    "mean_sea_level": "msl",
    "10m_u_component": "u10",
    "10m_v_component": "v10",
}


def _load_month_zip(path: Path) -> xr.Dataset | None:
    """Load t2m, msl, u10, v10 from a monthly ERA5 zip archive."""
    if not path.exists():
        return None
    if not zipfile.is_zipfile(path):
        try:
            ds = _std_coords(xr.open_dataset(path))
            return ds
        except Exception:
            return None

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(path, "r") as zf:
            for member in zf.namelist():
                if any(p in member.lower() for p in _ERA5_VAR_PATTERNS):
                    zf.extract(member, tmp)
        collected: dict[str, xr.DataArray] = {}
        for nc in Path(tmp).rglob("*.nc"):
            name_low = nc.name.lower()
            for pattern, vname in _ERA5_VAR_PATTERNS.items():
                if pattern in name_low:
                    ds_v = _std_coords(xr.open_dataset(str(nc)))
                    collected[vname] = ds_v[list(ds_v.data_vars)[0]]
                    ds_v.close()
                    break
        if "t2m" not in collected or "msl" not in collected:
            return None
        return xr.Dataset(collected)


def extract_era5_point(
    era5_dir: Path,
    lat: float,
    lon: float,
    era5_end_year: int = 2025,
    limit_months: int | None = None,
) -> pd.DataFrame:
    """Extract ERA5 nearest-neighbour daily t2m, msl, u10, v10, wspd."""
    files = sorted(
        f for f in era5_dir.glob("*.nc")
        if len(f.stem) == 6 and f.stem.isdigit()
        and int(f.stem[:4]) <= era5_end_year
    )
    if limit_months:
        files = files[:limit_months]
    if not files:
        return pd.DataFrame()

    rows: list[dict] = []
    for i, fp in enumerate(files):
        if (i + 1) % 60 == 0:
            print(f"    ERA5: {i + 1}/{len(files)} months...", flush=True)
        try:
            ds = _load_month_zip(fp)
            if ds is None:
                continue
            if "lon" in ds.coords and float(ds.lon.min()) < 0:
                ds = ds.assign_coords(lon=(ds.lon % 360)).sortby("lon")
            pt = ds.sel(lat=lat, lon=lon, method="nearest")
            times = pt["time"].values
            t2m = (pt["t2m"].values - 273.15).ravel()
            msl = (pt["msl"].values / 100.0).ravel()
            u10 = pt["u10"].values.ravel() if "u10" in pt else np.full(len(times), np.nan)
            v10 = pt["v10"].values.ravel() if "v10" in pt else np.full(len(times), np.nan)
            n = min(len(times), len(t2m), len(msl))
            for j in range(n):
                wspd = float(np.sqrt(u10[j] ** 2 + v10[j] ** 2))
                rows.append({
                    "time": pd.Timestamp(times[j]),
                    "t2m": float(t2m[j]),
                    "msl": float(msl[j]),
                    "u10": float(u10[j]),
                    "v10": float(v10[j]),
                    "wspd": wspd,
                })
            ds.close()
        except Exception as exc:
            print(f"    Warning: {fp.name}: {exc}", flush=True)
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("time").sort_index()


# ---------------------------------------------------------------------------
# JRA-3Q 6-hourly GRIB2 I/O
# ---------------------------------------------------------------------------

_JRA3Q_FILE_RE = re.compile(r"^anl_surf125\.(\d{10})$")
_JRA3Q_VAR_MAP = {
    "2t": "t2m",
    "prmsl": "msl",
    "10u": "u10",
    "10v": "v10",
}


def _coord_key(lon: float, lat: float) -> tuple[float, float]:
    return round(float(lon) % 360.0, 4), round(float(lat), 4)


def _jra3q_month_groups(
    jra3q_dir: Path,
    end_year: int,
    limit_months: int | None = None,
) -> list[tuple[str, list[Path]]]:
    """Return JRA-3Q 6-hourly GRIB files grouped by YYYYMM."""
    by_month: dict[str, list[Path]] = {}
    for fp in sorted(jra3q_dir.glob("anl_surf125.*")):
        m = _JRA3Q_FILE_RE.match(fp.name)
        if not m:
            continue
        stamp = m.group(1)
        if int(stamp[:4]) > end_year:
            continue
        by_month.setdefault(stamp[:6], []).append(fp)

    groups = sorted(by_month.items())
    if limit_months is not None:
        groups = groups[:limit_months]
    return groups


def _write_cdo_station_grid(
    tmp_dir: Path,
    station_points: dict[str, dict],
) -> Path:
    grid_path = tmp_dir / "jra3q_station_points.grid"
    stations = list(station_points)
    xvals = " ".join(f"{float(station_points[st]['lon']):.6f}" for st in stations)
    yvals = " ".join(f"{float(station_points[st]['lat']):.6f}" for st in stations)
    grid_path.write_text(
        "\n".join([
            "gridtype = unstructured",
            f"gridsize = {len(stations)}",
            f"xvals = {xvals}",
            f"yvals = {yvals}",
            "",
        ])
    )
    return grid_path


def _parse_jra3q_outputtab(
    text: str,
    station_lookup: dict[tuple[float, float], str],
    out: dict[str, dict[pd.Timestamp, dict[str, float]]],
) -> None:
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        date_s, time_s, lon_s, lat_s, name, value_s = parts[:6]
        col = _JRA3Q_VAR_MAP.get(name)
        if col is None:
            continue
        station = station_lookup.get(_coord_key(float(lon_s), float(lat_s)))
        if station is None:
            continue
        try:
            value = float(value_s)
        except ValueError:
            value = np.nan
        ts = pd.Timestamp(f"{date_s} {time_s}")
        out[station].setdefault(ts, {})[col] = value


def _run_cdo_jra3q_extract(
    files: list[Path],
    grid_path: Path,
) -> str:
    cmd = [
        "cdo", "-s",
        "outputtab,date,time,lon,lat,name,value",
        f"-remapnn,{grid_path}",
        "-selname,2t,prmsl,10u,10v",
        "-cat",
        "[",
        *(str(fp) for fp in files),
        "]",
    ]
    res = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if res.returncode != 0:
        msg = res.stderr.strip() or res.stdout.strip()
        raise RuntimeError(msg[-1000:])
    return res.stdout


def _jra3q_to_daily(df_6h: pd.DataFrame) -> pd.DataFrame:
    df = df_6h.copy()
    if "t2m" in df:
        df["t2m"] = df["t2m"] - 273.15
    if "msl" in df:
        df["msl"] = df["msl"] / 100.0
    daily = df.resample("D").mean().dropna(how="all")
    if "u10" in daily and "v10" in daily:
        daily["wspd"] = np.sqrt(daily["u10"] ** 2 + daily["v10"] ** 2)
    return daily


def extract_jra3q_points(
    jra3q_dir: Path,
    station_points: dict[str, dict],
    jra3q_end_year: int = 2025,
    limit_months: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Extract JRA-3Q nearest-neighbour daily t2m, msl, u10, v10, wspd."""
    if shutil.which("cdo") is None:
        raise RuntimeError("JRA-3Q extraction requires the 'cdo' command")

    month_groups = _jra3q_month_groups(
        jra3q_dir,
        end_year=jra3q_end_year,
        limit_months=limit_months,
    )
    if not month_groups:
        return {}

    station_lookup: dict[tuple[float, float], str] = {}
    for station, sc in station_points.items():
        key = _coord_key(sc["lon"], sc["lat"])
        if key in station_lookup:
            raise ValueError(
                f"Duplicate station point for {station} and "
                f"{station_lookup[key]} at lon/lat {key}"
            )
        station_lookup[key] = station

    raw: dict[str, dict[pd.Timestamp, dict[str, float]]] = {
        station: {} for station in station_points
    }
    with tempfile.TemporaryDirectory() as tmp:
        grid_path = _write_cdo_station_grid(Path(tmp), station_points)
        for i, (yyyymm, files) in enumerate(month_groups):
            if (i + 1) % 60 == 0:
                print(
                    f"    JRA-3Q: {i + 1}/{len(month_groups)} months...",
                    flush=True,
                )
            try:
                text = _run_cdo_jra3q_extract(files, grid_path)
                _parse_jra3q_outputtab(text, station_lookup, raw)
            except Exception as exc:
                print(f"    Warning: JRA-3Q {yyyymm}: {exc}", flush=True)

    out: dict[str, pd.DataFrame] = {}
    for station, by_time in raw.items():
        if not by_time:
            continue
        df_6h = pd.DataFrame.from_dict(by_time, orient="index").sort_index()
        df_6h.index.name = "time"
        daily = _jra3q_to_daily(df_6h)
        if len(daily) > 0:
            out[station] = daily
    return out


# ---------------------------------------------------------------------------
# Cache helpers (one sub-dir per reanalysis)
# ---------------------------------------------------------------------------

def _cache_dir(data_dir: Path, reanalysis: str = "ERA5") -> Path:
    return data_dir / "reanalysis_cache" / reanalysis


def _read_cached(cache: Path, station: str) -> pd.DataFrame | None:
    pkl = cache / f"{station}.pkl"
    if pkl.exists():
        return pd.read_pickle(pkl)
    pq = cache / f"{station}.parquet"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except ImportError:
            return None
    return None


def _write_cached(cache: Path, station: str, df: pd.DataFrame) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(cache / f"{station}.parquet", index=True)
    except ImportError:
        df.to_pickle(cache / f"{station}.pkl")


def _load_all_cached(
    cache: Path, stations: list[str],
) -> dict[str, pd.DataFrame] | None:
    out: dict[str, pd.DataFrame] = {}
    for st in stations:
        d = _read_cached(cache, st)
        if d is None:
            return None
        out[st] = d
    return out


def _save_all_cached(
    cache: Path, data: dict[str, pd.DataFrame],
) -> None:
    for st, df in data.items():
        _write_cached(cache, st, df)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_linear_trend(
    x_dates: np.ndarray, y_vals: np.ndarray,
) -> tuple[float, float, float, np.ndarray] | None:
    """Return (slope_per_decade °C/dec, intercept, p_value, trend_line_y)."""
    mask = np.isfinite(y_vals)
    if mask.sum() < 3:
        return None
    x_num = mdates.date2num(x_dates[mask])
    slope, intercept, _r, p, _se = stats.linregress(x_num, y_vals[mask])
    trend_y = slope * mdates.date2num(x_dates) + intercept
    slope_per_decade = slope * 365.25 * 10
    return slope_per_decade, intercept, p, trend_y


def compute_acc(s1: pd.Series, s2: pd.Series) -> float | None:
    """Anomaly correlation coefficient between two monthly series."""
    common = s1.dropna().index.intersection(s2.dropna().index)
    if len(common) < 12:
        return None
    a, b = s1.loc[common], s2.loc[common]
    a_anom, b_anom = a - a.mean(), b - b.mean()
    denom = np.sqrt((a_anom ** 2).sum() * (b_anom ** 2).sum())
    if denom == 0:
        return None
    return float((a_anom * b_anom).sum() / denom)


# ---------------------------------------------------------------------------
# Plotting: helpers
# ---------------------------------------------------------------------------

def _trend_text(ax, label, slope_dec, p, color, y_frac):
    sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
    ax.text(
        0.02, y_frac,
        f"{label}: {slope_dec:+.3f} /decade{sig}",
        transform=ax.transAxes, fontsize=8, color=color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


# ---------------------------------------------------------------------------
# Plot: 3-panel daily time series
# ---------------------------------------------------------------------------

VAR_MAP = [
    # (era5_col, reader_col, ylabel, ylim)
    ("t2m",  "temp",    "2m temperature (°C)",      None),
    ("msl",  "slp",     "Sea level pressure (hPa)",  (900, 1100)),
    ("wspd", "wspd_ms", "Wind speed (m/s)",          None),
]


def plot_timeseries(
    station: str,
    reader_df: pd.DataFrame | None,
    reanalysis_data: dict[str, dict[str, pd.DataFrame]],
    out_dir: Path,
) -> None:
    n = len(VAR_MAP)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    fig.suptitle(f"{station}: Station vs Reanalysis", fontsize=14)

    for i, (ecol, rcol, ylabel, ylim) in enumerate(VAR_MAP):
        ax = axes[i]
        y_pos = 0.05

        # Station obs
        if reader_df is not None and rcol in reader_df.columns:
            v = reader_df[rcol].dropna()
            if len(v) > 0:
                ax.plot(v.index, v.values, color=OBS_STYLE["color"],
                        alpha=0.6, lw=0.5, label="READER")
                if ecol == "t2m":
                    tr = compute_linear_trend(v.index.to_numpy(), v.values)
                    if tr:
                        ax.plot(v.index, tr[3], color=OBS_STYLE["color"],
                                ls="--", lw=1.5, alpha=0.9)
                        _trend_text(ax, "READER", tr[0], tr[2],
                                    OBS_STYLE["color"], y_pos)
                        y_pos += 0.07

        # Reanalyses
        for rean, sty in REANALYSIS_STYLES.items():
            rean_st = reanalysis_data.get(rean, {}).get(station)
            if rean_st is None or ecol not in rean_st.columns:
                continue
            v = rean_st[ecol].dropna()
            if len(v) == 0:
                continue
            ax.plot(v.index, v.values, color=sty["color"],
                    alpha=0.6, lw=0.5, label=rean)
            if ecol == "t2m":
                tr = compute_linear_trend(v.index.to_numpy(), v.values)
                if tr:
                    ax.plot(v.index, tr[3], color=sty["color"],
                            ls="--", lw=1.5, alpha=0.9)
                    _trend_text(ax, rean, tr[0], tr[2], sty["color"], y_pos)
                    y_pos += 0.07

        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(out_dir / f"{station}_timeseries.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {station}_timeseries.png")


# ---------------------------------------------------------------------------
# Plot: decadal trend (annual mean + 10-yr running + linear fit)
# ---------------------------------------------------------------------------

def plot_decadal_trend(
    station: str,
    reader_df: pd.DataFrame | None,
    reanalysis_data: dict[str, dict[str, pd.DataFrame]],
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    y_pos = 0.05

    if reader_df is not None and "temp" in reader_df.columns:
        yearly = reader_df["temp"].resample("YE").mean().dropna()
        decade = yearly.rolling(10, center=True, min_periods=3).mean()
        ax.plot(yearly.index, yearly.values, ".", color=OBS_STYLE["color"],
                alpha=0.5, ms=4, label="READER annual")
        ax.plot(decade.index, decade.values, "-", color=OBS_STYLE["color"],
                lw=2, label="READER 10-yr")
        tr = compute_linear_trend(yearly.index.to_numpy(), yearly.values)
        if tr:
            ax.plot(yearly.index, tr[3], "--", color=OBS_STYLE["color"],
                    lw=1.5, alpha=0.8)
            _trend_text(ax, "READER", tr[0], tr[2], OBS_STYLE["color"], y_pos)
            y_pos += 0.07

    for rean, sty in REANALYSIS_STYLES.items():
        rean_st = reanalysis_data.get(rean, {}).get(station)
        if rean_st is None or "t2m" not in rean_st.columns:
            continue
        yearly = rean_st["t2m"].resample("YE").mean().dropna()
        decade = yearly.rolling(10, center=True, min_periods=3).mean()
        ax.plot(yearly.index, yearly.values, ".", color=sty["color"],
                alpha=0.5, ms=4, label=f"{rean} annual")
        ax.plot(decade.index, decade.values, "-", color=sty["color"],
                lw=2, label=f"{rean} 10-yr")
        tr = compute_linear_trend(yearly.index.to_numpy(), yearly.values)
        if tr:
            ax.plot(yearly.index, tr[3], "--", color=sty["color"],
                    lw=1.5, alpha=0.8)
            _trend_text(ax, rean, tr[0], tr[2], sty["color"], y_pos)
            y_pos += 0.07

    ax.set_ylabel("2m temperature (°C)")
    ax.set_title(f"{station}: Decadal trend (10-yr running mean) with linear fits")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{station}_decadal_trend.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {station}_decadal_trend.png")


# ---------------------------------------------------------------------------
# Plot: monthly-variance time series (one figure per variable) with ACC
# ---------------------------------------------------------------------------

_MONTHLY_VAR_CFG = [
    # tag,  reader_col, era5_col, ylabel
    ("t2m",  "temp",    "t2m",  "Temperature variance (°C²)"),
    ("slp",  "slp",     "msl",  "SLP variance (hPa²)"),
    ("wspd", "wspd_ms", "wspd", "Wind speed variance (m²/s²)"),
]


def plot_monthly_variance(
    station: str,
    reader_df: pd.DataFrame | None,
    reanalysis_data: dict[str, dict[str, pd.DataFrame]],
    out_dir: Path,
) -> None:
    for tag, rcol, ecol, ylabel in _MONTHLY_VAR_CFG:
        fig, ax = plt.subplots(figsize=(14, 4))
        reader_mv: pd.Series | None = None

        if reader_df is not None and rcol in reader_df.columns:
            reader_mv = reader_df[rcol].resample("ME").var().dropna()
            if len(reader_mv) > 0:
                ax.plot(reader_mv.index, reader_mv.values,
                        color=OBS_STYLE["color"], alpha=0.8, lw=0.8,
                        label="READER")

        acc_y = 0.95
        for rean, sty in REANALYSIS_STYLES.items():
            rean_st = reanalysis_data.get(rean, {}).get(station)
            if rean_st is None or ecol not in rean_st.columns:
                continue
            rean_mv = rean_st[ecol].resample("ME").var().dropna()
            if len(rean_mv) == 0:
                continue
            ax.plot(rean_mv.index, rean_mv.values,
                    color=sty["color"], alpha=0.8, lw=0.8, label=rean)
            if reader_mv is not None:
                acc = compute_acc(reader_mv, rean_mv)
                if acc is not None:
                    ax.text(
                        0.98, acc_y, f"ACC({rean}) = {acc:.3f}",
                        transform=ax.transAxes, fontsize=9,
                        ha="right", va="top", color=sty["color"],
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", alpha=0.8),
                    )
                    acc_y -= 0.08

        ax.set_xlabel("Date")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{station}: Monthly variance of daily {tag} (time series)"
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{station}_monthly_variance_{tag}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {station}_monthly_variance_{tag}.png")


# ---------------------------------------------------------------------------
# Plot: SLP correlation scatter (daily mean + monthly mean)
# ---------------------------------------------------------------------------

def _pearson_r(s1: pd.Series, s2: pd.Series) -> tuple[float, int] | None:
    common = s1.dropna().index.intersection(s2.dropna().index)
    if len(common) < 3:
        return None
    a = s1.loc[common].to_numpy()
    b = s2.loc[common].to_numpy()
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    return r, len(common)


def _decade_color_map(decades: list[int]) -> dict[int, tuple[float, float, float]]:
    """
    Build decade colors with a deliberate era split:
      - <= 1970s: blue -> green
      - >= 1980s: red  -> orange
    """
    cmap: dict[int, tuple[float, float, float]] = {}
    pre = sorted(d for d in decades if d <= 1970)
    post = sorted(d for d in decades if d >= 1980)

    def _interp(c0: tuple[float, float, float],
                c1: tuple[float, float, float],
                t: float) -> tuple[float, float, float]:
        return tuple((1.0 - t) * a + t * b for a, b in zip(c0, c1))

    pre_start = (0.10, 0.30, 0.90)   # blue
    pre_end = (0.10, 0.72, 0.35)     # green
    post_start = (0.86, 0.10, 0.10)  # red
    post_end = (0.95, 0.55, 0.10)    # orange

    for i, d in enumerate(pre):
        t = 0.5 if len(pre) == 1 else i / (len(pre) - 1)
        cmap[d] = _interp(pre_start, pre_end, t)
    for i, d in enumerate(post):
        t = 0.5 if len(post) == 1 else i / (len(post) - 1)
        cmap[d] = _interp(post_start, post_end, t)
    return cmap


def plot_slp_correlation(
    station: str,
    reader_df: pd.DataFrame | None,
    reanalysis_data: dict[str, dict[str, pd.DataFrame]],
    out_dir: Path,
) -> None:
    if reader_df is None or "slp" not in reader_df.columns:
        return

    reader_daily = reader_df["slp"].dropna()
    if len(reader_daily) == 0:
        return
    reader_monthly = reader_daily.resample("ME").mean().dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = [
        ("Daily mean SLP", reader_daily, axes[0]),
        ("Monthly mean SLP", reader_monthly, axes[1]),
    ]

    for title, reader_series, ax in panels:
        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        decade_handles: dict[int, object] = {}

        for rean, sty in REANALYSIS_STYLES.items():
            rean_st = reanalysis_data.get(rean, {}).get(station)
            if rean_st is None or "msl" not in rean_st.columns:
                continue

            rean_series = rean_st["msl"].dropna()
            if title.startswith("Monthly"):
                rean_series = rean_series.resample("ME").mean().dropna()

            common = reader_series.index.intersection(rean_series.index)
            if len(common) < 3:
                continue
            x = reader_series.loc[common].to_numpy()
            y = rean_series.loc[common].to_numpy()
            years = common.year
            decades = sorted(np.unique((years // 10) * 10).tolist())
            decade_colors = _decade_color_map(decades)

            for dec in decades:
                m = ((years // 10) * 10) == dec
                if not np.any(m):
                    continue
                sc = ax.scatter(
                    x[m], y[m], s=10, alpha=0.45,
                    color=decade_colors[dec],
                    label=f"{dec}s",
                )
                decade_handles.setdefault(dec, sc)

            all_x.append(x)
            all_y.append(y)

            corr = _pearson_r(reader_series, rean_series)
            if corr is not None:
                r, n = corr
                ax.text(
                    0.03, 0.95 - 0.08 * list(REANALYSIS_STYLES).index(rean),
                    f"{rean}: r={r:.3f}, n={n}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=8, color=sty["color"],
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white", alpha=0.8),
                )

        if all_x and all_y:
            x_all = np.concatenate(all_x)
            y_all = np.concatenate(all_y)
            vmin = float(np.nanmin(np.concatenate([x_all, y_all])))
            vmax = float(np.nanmax(np.concatenate([x_all, y_all])))
            ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1, alpha=0.6)
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
        ax.set_title(title)
        ax.set_xlabel("READER SLP (hPa)")
        ax.set_ylabel("Reanalysis SLP (hPa)")
        ax.grid(True, alpha=0.3)
        if decade_handles:
            handles = [decade_handles[d] for d in sorted(decade_handles)]
            labels = [f"{d}s" for d in sorted(decade_handles)]
            ax.legend(handles, labels, loc="lower right", fontsize=8,
                      title="Decade", title_fontsize=8)

    fig.suptitle(f"{station}: SLP correlation (READER vs Reanalysis)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{station}_slp_correlation_daily_monthly.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {station}_slp_correlation_daily_monthly.png")


# ---------------------------------------------------------------------------
# Plot: wind-component monthly-variance panel (u, v, wspd)
# ---------------------------------------------------------------------------

_WIND_COMP_CFG = [
    # label, reader_col, era5_col, ylabel
    ("U-wind",      "u_obs",   "u10",  "U variance (m²/s²)"),
    ("V-wind",      "v_obs",   "v10",  "V variance (m²/s²)"),
    ("Wind speed",  "wspd_ms", "wspd", "Wspd variance (m²/s²)"),
]


def plot_wind_component_variance(
    station: str,
    reader_df: pd.DataFrame | None,
    reanalysis_data: dict[str, dict[str, pd.DataFrame]],
    out_dir: Path,
) -> None:
    has_obs = (
        reader_df is not None
        and "u_obs" in reader_df.columns
        and "v_obs" in reader_df.columns
    )
    if not has_obs:
        return

    n = len(_WIND_COMP_CFG)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    fig.suptitle(f"{station}: Wind-component monthly variance", fontsize=14)

    for idx, (comp_label, rcol, ecol, ylabel) in enumerate(_WIND_COMP_CFG):
        ax = axes[idx]
        reader_mv: pd.Series | None = None

        if rcol in reader_df.columns:
            reader_mv = reader_df[rcol].resample("ME").var().dropna()
            if len(reader_mv) > 0:
                ax.plot(reader_mv.index, reader_mv.values,
                        color=OBS_STYLE["color"], alpha=0.8, lw=0.8,
                        label="READER")

        acc_y = 0.95
        for rean, sty in REANALYSIS_STYLES.items():
            rean_st = reanalysis_data.get(rean, {}).get(station)
            if rean_st is None or ecol not in rean_st.columns:
                continue
            rean_mv = rean_st[ecol].resample("ME").var().dropna()
            if len(rean_mv) == 0:
                continue
            ax.plot(rean_mv.index, rean_mv.values,
                    color=sty["color"], alpha=0.8, lw=0.8, label=rean)
            if reader_mv is not None:
                acc = compute_acc(reader_mv, rean_mv)
                if acc is not None:
                    ax.text(
                        0.98, acc_y, f"ACC({rean}) = {acc:.3f}",
                        transform=ax.transAxes, fontsize=9,
                        ha="right", va="top", color=sty["color"],
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", alpha=0.8),
                    )
                    acc_y -= 0.08

        ax.set_ylabel(f"{comp_label} — {ylabel}")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{station}_monthly_variance_wind_components.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"  Saved {station}_monthly_variance_wind_components.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="F08: READER station vs reanalysis comparison",
    )
    ap.add_argument("--reader-dir", type=Path, default=READER_DIR_DEFAULT)
    ap.add_argument("--era5-dir", type=Path, default=ERA5_DIR_DEFAULT,
                    help="Directory with ERA5 YYYYMM.nc zip archives")
    ap.add_argument("--jra3q-dir", type=Path, default=JRA3Q_DIR_DEFAULT,
                    help="Directory with JRA-3Q anl_surf125.YYYYMMDDHH GRIB2 files")
    ap.add_argument("--stations-yaml", type=Path,
                    default=ROOT / "Const" / "reader_stations.yaml")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--era5-end-year", type=int, default=2025)
    ap.add_argument("--jra3q-end-year", type=int, default=None,
                    help="Last JRA-3Q year to extract (defaults to --era5-end-year)")
    ap.add_argument("--limit-months", type=int, default=None)
    ap.add_argument("--skip-jra3q", action="store_true",
                    help="Only compare READER with ERA5")
    ap.add_argument("--use-cached", action="store_true",
                    help="Load reanalysis from cache only (no zip I/O)")
    ap.add_argument("--clear-cache", action="store_true",
                    help="Delete existing cache before extraction "
                         "(use once when changing reanalysis sources)")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="F08 data root (cache lives here)")
    args = ap.parse_args()

    if args.use_cached and args.clear_cache:
        print("--use-cached and --clear-cache are mutually exclusive",
              file=sys.stderr)
        sys.exit(2)
    if args.use_cached and args.limit_months is not None:
        print("--use-cached cannot be combined with --limit-months",
              file=sys.stderr)
        sys.exit(2)

    out_dir = args.out_dir or ROOT / "Figs" / "F08_station_era5_comparison"
    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        env_d = os.environ.get("MAP_ROOM_F08_DATA_DIR")
        data_dir = Path(env_d) if env_d else (
            ROOT / "Data" / "F08_station_era5_comparison"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── clear cache ──────────────────────────────────────────────────────
    if args.clear_cache:
        for d in [data_dir / "era5_point_cache",
                  data_dir / "reanalysis_cache"]:
            if d.exists():
                shutil.rmtree(d)
                print(f"Cleared cache: {d}", flush=True)

    # ── station config ───────────────────────────────────────────────────
    with open(args.stations_yaml) as fh:
        cfg = yaml.safe_load(fh)
    stations = list(cfg["stations"].keys())

    # ── load READER daily means ──────────────────────────────────────────
    print("Loading READER daily means...", flush=True)
    reader_data: dict[str, pd.DataFrame] = {}
    for st in stations:
        df = load_station_daily(args.reader_dir, st)
        if df is not None and len(df) > 0:
            reader_data[st] = df
            print(f"  {st}: {df.index.min().date()} – "
                  f"{df.index.max().date()}, n={len(df)}")
        else:
            print(f"  {st}: no data")

    # ── ERA5 ─────────────────────────────────────────────────────────────
    #    Structured as reanalysis_data["ERA5"][station] = DataFrame
    reanalysis_data: dict[str, dict[str, pd.DataFrame]] = {}

    era5_cache = _cache_dir(data_dir, "ERA5")
    if args.use_cached:
        loaded = _load_all_cached(era5_cache, stations)
        if loaded is None:
            missing = [s for s in stations
                       if _read_cached(era5_cache, s) is None]
            print(
                f"Missing ERA5 cache for: {missing}. "
                f"Run without --use-cached to build from {args.era5_dir}.",
                file=sys.stderr, flush=True,
            )
            sys.exit(1)
        reanalysis_data["ERA5"] = loaded
        print("ERA5: loaded from cache (--use-cached)", flush=True)
        for st, df in loaded.items():
            print(f"  {st}: {df.index.min().date()} – "
                  f"{df.index.max().date()}, n={len(df)}")
    else:
        print(
            f"\nExtracting ERA5 nearest-neighbour from {args.era5_dir} ...",
            flush=True,
        )
        era5_stations: dict[str, pd.DataFrame] = {}
        for st in stations:
            sc = cfg["stations"][st]
            df = extract_era5_point(
                args.era5_dir, sc["lat"], sc["lon"],
                era5_end_year=args.era5_end_year,
                limit_months=args.limit_months,
            )
            if len(df) > 0:
                era5_stations[st] = df
                print(f"  {st}: {df.index.min().date()} – "
                      f"{df.index.max().date()}, n={len(df)}")
            else:
                print(f"  {st}: no ERA5 data")

        if era5_stations and args.limit_months is None:
            _save_all_cached(era5_cache, era5_stations)
            print(f"\nCached ERA5 → {era5_cache}", flush=True)
        reanalysis_data["ERA5"] = era5_stations

    # ── JRA-3Q ───────────────────────────────────────────────────────────
    if not args.skip_jra3q:
        jra3q_cache = _cache_dir(data_dir, "JRA-3Q")
        if args.use_cached:
            loaded = _load_all_cached(jra3q_cache, stations)
            if loaded is None:
                missing = [s for s in stations
                           if _read_cached(jra3q_cache, s) is None]
                print(
                    f"Missing JRA-3Q cache for: {missing}. "
                    f"Run without --use-cached to build from {args.jra3q_dir}.",
                    file=sys.stderr, flush=True,
                )
                sys.exit(1)
            reanalysis_data["JRA-3Q"] = loaded
            print("JRA-3Q: loaded from cache (--use-cached)", flush=True)
            for st, df in loaded.items():
                print(f"  {st}: {df.index.min().date()} – "
                      f"{df.index.max().date()}, n={len(df)}")
        else:
            jra3q_end_year = (
                args.jra3q_end_year
                if args.jra3q_end_year is not None
                else args.era5_end_year
            )
            print(
                f"\nExtracting JRA-3Q nearest-neighbour from {args.jra3q_dir} ...",
                flush=True,
            )
            jra3q_stations = extract_jra3q_points(
                args.jra3q_dir,
                {st: cfg["stations"][st] for st in stations},
                jra3q_end_year=jra3q_end_year,
                limit_months=args.limit_months,
            )
            for st in stations:
                df = jra3q_stations.get(st)
                if df is not None and len(df) > 0:
                    print(f"  {st}: {df.index.min().date()} – "
                          f"{df.index.max().date()}, n={len(df)}")
                else:
                    print(f"  {st}: no JRA-3Q data")

            if jra3q_stations and args.limit_months is None:
                _save_all_cached(jra3q_cache, jra3q_stations)
                print(f"\nCached JRA-3Q → {jra3q_cache}", flush=True)
            reanalysis_data["JRA-3Q"] = jra3q_stations

    # ── plotting ─────────────────────────────────────────────────────────
    print("\n--- Plotting ---", flush=True)
    for st in stations:
        rd = reader_data.get(st)
        if rd is None:
            continue
        plot_timeseries(st, rd, reanalysis_data, out_dir)
        plot_decadal_trend(st, rd, reanalysis_data, out_dir)
        plot_monthly_variance(st, rd, reanalysis_data, out_dir)
        plot_slp_correlation(st, rd, reanalysis_data, out_dir)
        plot_wind_component_variance(st, rd, reanalysis_data, out_dir)

    # ── summary CSV ──────────────────────────────────────────────────────
    rows = []
    for st in stations:
        rd = reader_data.get(st)
        row: dict = {"station": st}
        if rd is not None and len(rd) > 0:
            row["reader_start"] = rd.index.min().strftime("%Y-%m-%d")
            row["reader_end"] = rd.index.max().strftime("%Y-%m-%d")
        for rean in REANALYSIS_STYLES:
            ed = reanalysis_data.get(rean, {}).get(st)
            if ed is not None and len(ed) > 0:
                row[f"{rean}_start"] = ed.index.min().strftime("%Y-%m-%d")
                row[f"{rean}_end"] = ed.index.max().strftime("%Y-%m-%d")
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "comparison_summary.csv", index=False)
    print(f"\nSaved {out_dir / 'comparison_summary.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
