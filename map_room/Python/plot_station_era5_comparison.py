#!/usr/bin/env python3
"""
F08: Compare READER station obs vs reanalysis nearest-neighbour time series.

Stations: Adelaide, Deception, Esperanza, Fossil_Bluff, FaradayVernadsky, Marambio.
Variables: t2m (°C), msl (hPa), u10/v10/wspd (m/s).

Reanalysis point extractions are cached under
``<data-dir>/reanalysis_cache/<REANALYSIS>/<Station>.pkl``.

Workflow:
  1. First run (or after source change):  --clear-cache  (rebuilds from raw zips)
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
import shutil
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
JRA3Q_DIR = Path("/lustre/soge1/data/analysis/jra-q3/anl_surf125")

REANALYSIS_STYLES: dict[str, dict] = {
    "ERA5": {"color": "r"},
    # "JRA3Q": {"color": "g"},  # activate when JRA-3Q loader is ready
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
    ap.add_argument("--stations-yaml", type=Path,
                    default=ROOT / "Const" / "reader_stations.yaml")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--era5-end-year", type=int, default=2025)
    ap.add_argument("--limit-months", type=int, default=None)
    ap.add_argument("--use-cached", action="store_true",
                    help="Load reanalysis from cache only (no zip I/O)")
    ap.add_argument("--clear-cache", action="store_true",
                    help="Delete existing cache before extraction "
                         "(use once when changing ERA5 source)")
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

    # ── (future) JRA-3Q slot ─────────────────────────────────────────────
    # When ready, add:
    #   reanalysis_data["JRA3Q"] = _load_or_extract_jra3q(...)
    #   REANALYSIS_STYLES["JRA3Q"] = {"color": "g"}

    # ── plotting ─────────────────────────────────────────────────────────
    print("\n--- Plotting ---", flush=True)
    for st in stations:
        rd = reader_data.get(st)
        if rd is None:
            continue
        plot_timeseries(st, rd, reanalysis_data, out_dir)
        plot_decadal_trend(st, rd, reanalysis_data, out_dir)
        plot_monthly_variance(st, rd, reanalysis_data, out_dir)
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
