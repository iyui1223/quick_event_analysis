#!/usr/bin/env python3
"""
F10: Surface-parcel FPT2000 versus 100 m westerly flow for the AP.

Inputs are hourly ERA5 single-level files downloaded by
download_era5_fpt2000_inputs.py. The main diagnostic is computed grid-cell
first over the western AP mask:

    z_lcl = 125 * (t2m_C - d2m_C)
    FPT2000 = t2m_C + max(0, 2000 - z_lcl) * (2.8 / 1000)

The de-warmed version removes local linear trends from t2m and d2m at each
grid cell, then recomputes FPT2000.
"""

from __future__ import annotations

import argparse
import re
import tempfile
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
MASKS_DEFAULT = ROOT / "Data" / "F04_peninsula_domains" / "all_domain_masks.nc"
IN_DIR_DEFAULT = ROOT / "Data" / "F10_fpt2000_westAP" / "raw"
OUT_DATA_DEFAULT = ROOT / "Data" / "F10_fpt2000_westAP"
OUT_FIG_DEFAULT = ROOT / "Figs" / "F10_fpt2000_westAP"
OUT_TABLE_DEFAULT = ROOT / "Tables"

WEST_DOMAIN = "pen_west_slope"
EAST_DOMAIN = "pen_east_slope"
REQUIRED_VARS = ("t2m", "d2m", "u100")
OPTIONAL_VARS = ("sp",)

FPT_TARGET_M = 2000.0
DRY_LAPSE_C_PER_KM = 9.8
MOIST_LAPSE_C_PER_KM = 7.0
LAPSE_DELTA_C_PER_M = (DRY_LAPSE_C_PER_KM - MOIST_LAPSE_C_PER_KM) / 1000.0


def _std_coords(ds: xr.Dataset) -> xr.Dataset:
    renames = {}
    for name in set(ds.coords) | set(ds.dims):
        low = str(name).lower()
        if "lat" in low and name != "lat" and "lat" not in ds.coords:
            renames[name] = "lat"
        elif "lon" in low and name != "lon" and "lon" not in ds.coords:
            renames[name] = "lon"
        elif ("time" in low or "valid" in low) and name != "time" and "time" not in ds.coords:
            renames[name] = "time"
    if renames:
        ds = ds.rename(renames)
    if "lat" in ds.coords and ds.sizes.get("lat", 0) > 1 and bool(ds.lat[0] > ds.lat[-1]):
        ds = ds.sortby("lat")
    if "lon" in ds.coords and float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon % 360)).sortby("lon")
    return ds


def _canonical_var(name: str) -> str | None:
    low = str(name).lower()
    if low == "t2m" or ("2m_temperature" in low and "dewpoint" not in low):
        return "t2m"
    if low == "d2m" or "2m_dewpoint_temperature" in low:
        return "d2m"
    if low == "sp" or "surface_pressure" in low:
        return "sp"
    if low == "u100" or "100m_u_component" in low:
        return "u100"
    return None


def _drop_extra_dims(da: xr.DataArray) -> xr.DataArray:
    for dim in list(da.dims):
        if dim in {"time", "lat", "lon"}:
            continue
        if dim == "expver":
            da = da.max(dim=dim, skipna=True)
        elif da.sizes[dim] == 1:
            da = da.isel({dim: 0}, drop=True)
        else:
            raise ValueError(f"Unexpected non-singleton dimension {dim!r} in {da.name!r}")
    return da


def _extract_vars(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    ds = _std_coords(ds)
    out: dict[str, xr.DataArray] = {}
    for name in ds.data_vars:
        canon = _canonical_var(str(name))
        if canon is None or canon in out:
            continue
        da = _drop_extra_dims(ds[name])
        if not {"time", "lat", "lon"}.issubset(set(da.dims)):
            continue
        out[canon] = da.rename(canon)
    return out


def _open_input_file(path: Path) -> xr.Dataset:
    path = Path(path)
    if zipfile.is_zipfile(path):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            out: dict[str, xr.DataArray] = {}
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(tmpdir)
            for nc in sorted(tmpdir.rglob("*.nc")):
                ds = xr.open_dataset(nc)
                try:
                    found = _extract_vars(ds)
                    for key, da in found.items():
                        if key not in out:
                            out[key] = da.load()
                finally:
                    ds.close()
            if not out:
                raise ValueError(f"No usable NetCDF members found in zip file: {path}")
            return xr.Dataset(out)

    ds = xr.open_dataset(path)
    try:
        found = _extract_vars(ds)
        if not found:
            raise ValueError(f"No required variables found in {path}")
        loaded = {key: da.load() for key, da in found.items()}
        return xr.Dataset(loaded)
    finally:
        ds.close()


def _year_from_path(path: Path) -> int | None:
    matches = re.findall(r"(?:19|20)\d{2}", path.stem)
    if not matches:
        return None
    return int(matches[-1])


def _discover_input_files(
    input_dir: Path,
    input_glob: str,
    start_year: int,
    end_year: int,
) -> list[Path]:
    files = sorted(input_dir.glob(input_glob))
    if not files:
        files = sorted(input_dir.glob("*.nc")) + sorted(input_dir.glob("*.zip"))
    filtered = []
    for path in files:
        year = _year_from_path(path)
        if year is not None and not (start_year <= year <= end_year):
            continue
        filtered.append(path)
    return filtered


def _select_utc_hour(ds: xr.Dataset, utc_hour: int | None) -> xr.Dataset:
    if utc_hour is None:
        return ds
    idx = pd.DatetimeIndex(ds.time.values)
    keep = idx.hour == utc_hour
    if not keep.any():
        raise ValueError(f"No timestamps found for UTC hour {utc_hour:02d}")
    return ds.isel(time=np.where(keep)[0])


def _mask_bbox(masks: xr.Dataset, domains: tuple[str, ...], buffer_deg: float) -> tuple[slice, slice]:
    union = None
    for dom in domains:
        if dom not in masks:
            raise ValueError(f"Missing mask domain: {dom}")
        m = masks[dom] > 0
        union = m if union is None else (union | m)
    if union is None or int(union.sum()) == 0:
        raise ValueError(f"Selected domains contain no cells: {domains}")
    valid_lat = masks.lat.where(union.any("lon"), drop=True)
    valid_lon = masks.lon.where(union.any("lat"), drop=True)
    lat0 = float(valid_lat.min()) - buffer_deg
    lat1 = float(valid_lat.max()) + buffer_deg
    lon0 = float(valid_lon.min()) - buffer_deg
    lon1 = float(valid_lon.max()) + buffer_deg
    return slice(lat0, lat1), slice(lon0, lon1)


def _load_inputs(
    input_dir: Path,
    input_glob: str,
    masks_sub: xr.Dataset,
    lat_slice: slice,
    lon_slice: slice,
    utc_hour: int | None,
    start_year: int,
    end_year: int,
) -> xr.Dataset:
    files = _discover_input_files(input_dir, input_glob, start_year=start_year, end_year=end_year)
    if not files:
        raise FileNotFoundError(f"No input files found in {input_dir} matching {input_glob!r}")

    datasets: list[xr.Dataset] = []
    print(f"Loading {len(files)} input files from {input_dir}")
    for i, path in enumerate(files, start=1):
        print(f"  [{i:3d}/{len(files)}] {path.name}", end=" ", flush=True)
        ds = _open_input_file(path)
        ds = _std_coords(ds)
        ds = _select_utc_hour(ds, utc_hour)
        ds = ds.sel(lat=lat_slice, lon=lon_slice)
        missing = [v for v in REQUIRED_VARS if v not in ds]
        if missing:
            raise ValueError(f"{path} missing required variables: {missing}")
        ds = ds.interp(lat=masks_sub.lat, lon=masks_sub.lon, method="nearest")
        datasets.append(ds.load())
        ds.close()
        print("ok")

    combined = xr.concat(datasets, dim="time").sortby("time")
    idx = pd.DatetimeIndex(combined.time.values)
    keep = ~idx.duplicated(keep="first")
    combined = combined.isel(time=np.where(keep)[0])
    return combined


def _weights_like(mask: xr.DataArray) -> xr.DataArray:
    lat_rad = np.deg2rad(mask.lat)
    w_lat = xr.DataArray(np.cos(lat_rad), coords={"lat": mask.lat}, dims=["lat"])
    return w_lat.broadcast_like(mask)


def _weighted_mean(da: xr.DataArray, mask: xr.DataArray, weights2d: xr.DataArray) -> xr.DataArray:
    valid_mask = mask > 0
    masked = da.where(valid_mask)
    wmask = weights2d.where(valid_mask)
    return (masked * wmask).sum(dim=["lat", "lon"]) / wmask.sum(dim=["lat", "lon"])


def _decimal_year(times: xr.DataArray) -> xr.DataArray:
    idx = pd.DatetimeIndex(times.values)
    year = idx.year.astype(float)
    doy0 = (idx.dayofyear - 1).astype(float)
    frac_day = (idx.hour + idx.minute / 60.0 + idx.second / 3600.0) / 24.0
    days_in_year = np.where(idx.is_leap_year, 366.0, 365.0)
    vals = year + (doy0 + frac_day) / days_in_year
    return xr.DataArray(vals, coords={"time": times}, dims=["time"], name="decimal_year")


def _detrend_to_reference(
    da: xr.DataArray,
    ref_year: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    x = _decimal_year(da.time)
    finite = da.notnull()
    xmean = x.where(finite).mean("time")
    ymean = da.mean("time", skipna=True)
    xc = x - xmean
    yc = da - ymean
    denom = (xc ** 2).where(finite).sum("time", skipna=True)
    slope = (xc * yc).where(finite).sum("time", skipna=True) / denom
    detrended = da - slope * (x - ref_year)
    slope.attrs["units"] = f"{da.attrs.get('units', 'units')} per year"
    return detrended, slope


def _fpt_fields(t2m_k: xr.DataArray, d2m_k: xr.DataArray) -> xr.Dataset:
    t2m_c = t2m_k - 273.15
    d2m_c = d2m_k - 273.15
    z_lcl = (125.0 * (t2m_c - d2m_c)).clip(min=0.0)
    lift_depth = (FPT_TARGET_M - z_lcl).clip(min=0.0)
    boost = lift_depth * LAPSE_DELTA_C_PER_M
    fpt = t2m_c + boost
    return xr.Dataset(
        {
            "t2m_c": t2m_c,
            "d2m_c": d2m_c,
            "z_lcl_m": z_lcl,
            "fpt2000_c": fpt,
            "fpt2000_boost_c": boost,
        }
    )


def _compute_timeseries(
    ds: xr.Dataset,
    masks_sub: xr.Dataset,
    detrend_ref_year: float,
) -> tuple[xr.Dataset, xr.Dataset]:
    west_mask = masks_sub[WEST_DOMAIN]
    ap_mask = ((masks_sub[WEST_DOMAIN] > 0) | (masks_sub[EAST_DOMAIN] > 0)).astype(np.int8)
    weights2d = _weights_like(west_mask)

    raw = _fpt_fields(ds["t2m"], ds["d2m"])
    t2m_dw, t2m_slope = _detrend_to_reference(ds["t2m"], detrend_ref_year)
    d2m_dw, d2m_slope = _detrend_to_reference(ds["d2m"], detrend_ref_year)
    dew = _fpt_fields(t2m_dw, d2m_dw)

    out_vars = {
        "u100_ap_ms": _weighted_mean(ds["u100"], ap_mask, weights2d),
        "t2m_west_c": _weighted_mean(raw["t2m_c"], west_mask, weights2d),
        "d2m_west_c": _weighted_mean(raw["d2m_c"], west_mask, weights2d),
        "z_lcl_west_m": _weighted_mean(raw["z_lcl_m"], west_mask, weights2d),
        "fpt2000_west_c": _weighted_mean(raw["fpt2000_c"], west_mask, weights2d),
        "fpt2000_boost_west_c": _weighted_mean(raw["fpt2000_boost_c"], west_mask, weights2d),
        "t2m_dewarmed_west_c": _weighted_mean(dew["t2m_c"], west_mask, weights2d),
        "d2m_dewarmed_west_c": _weighted_mean(dew["d2m_c"], west_mask, weights2d),
        "z_lcl_dewarmed_west_m": _weighted_mean(dew["z_lcl_m"], west_mask, weights2d),
        "fpt2000_dewarmed_west_c": _weighted_mean(dew["fpt2000_c"], west_mask, weights2d),
        "fpt2000_boost_dewarmed_west_c": _weighted_mean(dew["fpt2000_boost_c"], west_mask, weights2d),
    }
    if "sp" in ds:
        out_vars["sp_west_pa"] = _weighted_mean(ds["sp"], west_mask, weights2d)

    out = xr.Dataset(out_vars, coords={"time": ds.time})
    for name in out:
        if name.endswith("_c"):
            out[name].attrs["units"] = "degC"
        elif name.endswith("_m"):
            out[name].attrs["units"] = "m"
        elif name.endswith("_ms"):
            out[name].attrs["units"] = "m s-1"
        elif name.endswith("_pa"):
            out[name].attrs["units"] = "Pa"

    trend_ds = xr.Dataset(
        {
            "t2m_trend_k_per_year": t2m_slope,
            "d2m_trend_k_per_year": d2m_slope,
        }
    )
    return out, trend_ds


def _decade_ranges(start_year: int, end_year: int) -> list[tuple[int, int]]:
    out = []
    y0 = start_year
    while y0 <= end_year:
        y1 = min(y0 + 10, end_year)
        out.append((y0, y1))
        y0 += 10
    return out


def _decade_start_for_year(year: int, start_year: int) -> int:
    return start_year + ((year - start_year) // 10) * 10


def _included_end_for_bin(decade_start: int, decade_end_label: int) -> int:
    if decade_end_label == decade_start + 10:
        return decade_end_label - 1
    return decade_end_label


def _interp_color(c0: tuple[float, float, float], c1: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    return tuple((1.0 - t) * a + t * b for a, b in zip(c0, c1))


def _mix_with(color: tuple[float, float, float], target: tuple[float, float, float], amt: float) -> tuple[float, float, float]:
    return tuple((1.0 - amt) * c + amt * t for c, t in zip(color, target))


def _build_decade_colors(decades: list[tuple[int, int]]) -> dict[tuple[int, int], tuple[float, float, float]]:
    pre = [d for d in decades if _included_end_for_bin(d[0], d[1]) < 1979]
    post = [d for d in decades if d[0] >= 1979]
    crossover = [d for d in decades if d not in pre and d not in post]
    colors: dict[tuple[int, int], tuple[float, float, float]] = {}
    pre_start = mcolors.to_rgb("#1f5aa6")
    pre_end = mcolors.to_rgb("#2fa36b")
    post_start = mcolors.to_rgb("#f1c232")
    post_end = mcolors.to_rgb("#e67e22")
    for i, dec in enumerate(pre):
        t = 0.5 if len(pre) == 1 else i / (len(pre) - 1)
        colors[dec] = _interp_color(pre_start, pre_end, t)
    for i, dec in enumerate(post):
        t = 0.5 if len(post) == 1 else i / (len(post) - 1)
        colors[dec] = _interp_color(post_start, post_end, t)
    for dec in crossover:
        colors[dec] = mcolors.to_rgb("#b9c24f")
    return colors


def _samples_dataframe(ds_ts: xr.Dataset, start_year: int, end_year: int, present_start: int) -> pd.DataFrame:
    df = ds_ts.to_dataframe().reset_index()
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["hour_utc"] = df["time"].dt.hour
    df["decade_start"] = df["year"].apply(lambda y: _decade_start_for_year(int(y), start_year))
    df["decade_end"] = df["decade_start"].apply(lambda y: min(int(y) + 10, end_year))
    df["decade_year_end_included"] = df.apply(
        lambda row: _included_end_for_bin(int(row["decade_start"]), int(row["decade_end"])),
        axis=1,
    )
    df["decade"] = df["decade_start"].astype(str) + "-" + df["decade_end"].astype(str)
    df["period"] = np.where(df["year"] >= present_start, "present", "past")
    df["is_westerly_u100"] = df["u100_ap_ms"] > 0.0
    return df


def _save_summary_tables(df: pd.DataFrame, out_table_dir: Path) -> None:
    out_table_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_table_dir / "FPT2000_westAP_samples.csv"
    df.to_csv(sample_path, index=False)
    print(f"Saved table -> {sample_path} ({len(df)} rows)")

    value_cols = [
        "u100_ap_ms",
        "t2m_west_c",
        "d2m_west_c",
        "z_lcl_west_m",
        "fpt2000_west_c",
        "fpt2000_boost_west_c",
        "fpt2000_dewarmed_west_c",
        "fpt2000_boost_dewarmed_west_c",
    ]
    rows = []
    for (d0, d1), sub in df.groupby(["decade_start", "decade_end"], sort=True):
        raw_warm_westerly = (sub["fpt2000_west_c"] > 0.0) & (sub["u100_ap_ms"] > 0.0)
        dewarmed_warm_westerly = (sub["fpt2000_dewarmed_west_c"] > 0.0) & (sub["u100_ap_ms"] > 0.0)
        row = {
            "decade_start": int(d0),
            "decade_end": int(d1),
            "year_start_included": int(sub["year"].min()),
            "year_end_included": int(sub["year"].max()),
            "n_samples": int(len(sub)),
            "n_westerly_u100": int(sub["is_westerly_u100"].sum()),
            "pct_westerly_u100": 100.0 * float(sub["is_westerly_u100"].mean()),
            "n_fpt2000_gt0_and_westerly": int(raw_warm_westerly.sum()),
            "pct_fpt2000_gt0_and_westerly": 100.0 * float(raw_warm_westerly.mean()),
            "n_fpt2000_dewarmed_gt0_and_westerly": int(dewarmed_warm_westerly.sum()),
            "pct_fpt2000_dewarmed_gt0_and_westerly": 100.0 * float(dewarmed_warm_westerly.mean()),
        }
        for col in value_cols:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_median"] = float(sub[col].median())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary_path = out_table_dir / "FPT2000_westAP_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved table -> {summary_path} ({len(summary)} rows)")

    bins = [-np.inf, 0.0, 2.0, 4.0, 6.0, np.inf]
    labels = ["u100_lt_0", "u100_0_2", "u100_2_4", "u100_4_6", "u100_ge_6"]
    binned = df.copy()
    binned["u100_bin"] = pd.cut(binned["u100_ap_ms"], bins=bins, labels=labels, right=False)
    bin_rows = []
    for (period, ubin), sub in binned.groupby(["period", "u100_bin"], observed=True):
        if sub.empty:
            continue
        bin_rows.append(
            {
                "period": period,
                "u100_bin": str(ubin),
                "n_samples": int(len(sub)),
                "u100_ap_ms_mean": float(sub["u100_ap_ms"].mean()),
                "fpt2000_west_c_mean": float(sub["fpt2000_west_c"].mean()),
                "fpt2000_dewarmed_west_c_mean": float(sub["fpt2000_dewarmed_west_c"].mean()),
                "fpt2000_boost_west_c_mean": float(sub["fpt2000_boost_west_c"].mean()),
            }
        )
    bins_path = out_table_dir / "FPT2000_westAP_u100_bins.csv"
    pd.DataFrame(bin_rows).to_csv(bins_path, index=False)
    print(f"Saved table -> {bins_path} ({len(bin_rows)} rows)")


def _plot_faceted_scatter(
    df: pd.DataFrame,
    decades: list[tuple[int, int]],
    out_png: Path,
    y_col: str,
    t2m_col: str,
    title: str,
    present_start: int,
    figure_dpi: int,
    draw_segments: bool,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    colors = _build_decade_colors(decades)
    n_dec = len(decades)
    ncols = 3 if n_dec >= 7 else 2
    nrows = int(np.ceil(n_dec / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.1 * ncols, 3.6 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    finite = df[np.isfinite(df["u100_ap_ms"]) & np.isfinite(df[y_col])]
    if finite.empty:
        raise ValueError(f"No finite data to plot for {y_col}")

    xpad = max(0.5, 0.05 * (finite["u100_ap_ms"].max() - finite["u100_ap_ms"].min()))
    ypad = max(0.5, 0.05 * (finite[y_col].max() - finite[y_col].min()))
    xlim = (finite["u100_ap_ms"].min() - xpad, finite["u100_ap_ms"].max() + xpad)
    ymin = min(float(finite[y_col].min()), float(df[t2m_col].min()))
    ymax = max(float(finite[y_col].max()), float(df[t2m_col].max()))
    ylim = (ymin - ypad, ymax + ypad)

    period_specs = {
        "past": {"marker": "o", "label": f"Past (<{present_start})"},
        "present": {"marker": "^", "label": f"Present (>={present_start})"},
    }

    for i, dec in enumerate(decades):
        ax = axes[i]
        d0, d1 = dec
        sub = df[(df["decade_start"] == d0) & (df["decade_end"] == d1)].copy()
        color = colors[dec]
        if sub.empty:
            ax.set_visible(False)
            continue

        warm_westerly = (sub[y_col] > 0.0) & (sub["u100_ap_ms"] > 0.0)
        n_warm_westerly = int(warm_westerly.sum())
        pct_warm_westerly = 100.0 * n_warm_westerly / len(sub)

        if draw_segments:
            seg_color = _mix_with(color, (0.0, 0.0, 0.0), 0.25)
            ax.vlines(
                sub["u100_ap_ms"],
                sub[t2m_col],
                sub[y_col],
                colors=[seg_color],
                linewidth=0.45,
                alpha=0.16,
                zorder=1,
            )

        for period, spec in period_specs.items():
            part = sub[sub["period"] == period]
            if part.empty:
                continue
            ax.scatter(
                part["u100_ap_ms"],
                part[y_col],
                s=14,
                c=[color],
                marker=spec["marker"],
                alpha=0.68,
                edgecolors="none",
                zorder=2,
            )

        ax.axvline(0.0, color="#333333", lw=0.8, ls="--", alpha=0.75)
        ax.axhline(0.0, color="#333333", lw=0.8, ls=":", alpha=0.55)
        ax.text(
            0.98,
            0.96,
            f"FPT>0 & U>0\n{n_warm_westerly}/{len(sub)} ({pct_warm_westerly:.1f}%)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#222222",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.0},
            zorder=4,
        )
        ax.set_title(f"{d0}-{d1} (n={len(sub)})", fontsize=10)
        ax.grid(True, color="#bbbbbb", alpha=0.25, ls=":")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    for j in range(n_dec, len(axes)):
        axes[j].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker=spec["marker"], color="none", markerfacecolor="#777777",
               markeredgecolor="none", markersize=6, label=spec["label"])
        for spec in period_specs.values()
    ]
    if draw_segments:
        legend_handles.append(Line2D([0], [0], color="#777777", lw=1.0, alpha=0.6, label="T2m to FPT2000"))
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles), frameon=False)

    for ax in axes:
        if ax.get_visible():
            ax.set_xlabel("U100_AP (m s-1)")
            ax.set_ylabel("FPT2000 west AP (degC)")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.055, 1, 0.96])
    fig.savefig(out_png, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute and plot FPT2000 west-AP versus U100_AP for February 15-local-time samples."
    )
    ap.add_argument("--input-dir", type=Path, default=IN_DIR_DEFAULT)
    ap.add_argument("--input-glob", default="era5_fpt2000_inputs_feb*utc_*.nc")
    ap.add_argument("--masks", type=Path, default=MASKS_DEFAULT)
    ap.add_argument("--out-data-dir", type=Path, default=OUT_DATA_DEFAULT)
    ap.add_argument("--out-fig-dir", type=Path, default=OUT_FIG_DEFAULT)
    ap.add_argument("--out-table-dir", type=Path, default=OUT_TABLE_DEFAULT)
    ap.add_argument("--start-year", type=int, default=1959)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--present-start", type=int, default=1988)
    ap.add_argument("--utc-hour", type=int, default=19,
                    help="Only use timestamps at this UTC hour; pass -1 to disable filtering.")
    ap.add_argument("--mask-buffer-deg", type=float, default=0.5)
    ap.add_argument("--detrend-reference-year", type=float, default=None,
                    help="Reference year for de-warming. Default: --start-year.")
    ap.add_argument("--figure-dpi", type=int, default=170)
    ap.add_argument("--no-segments", action="store_true",
                    help="Do not draw T2m-to-FPT2000 warming segments.")
    args = ap.parse_args()

    if args.start_year > args.end_year:
        raise ValueError("--start-year cannot exceed --end-year")
    utc_hour = None if args.utc_hour < 0 else args.utc_hour
    detrend_ref_year = float(args.start_year if args.detrend_reference_year is None else args.detrend_reference_year)

    args.out_data_dir.mkdir(parents=True, exist_ok=True)
    args.out_fig_dir.mkdir(parents=True, exist_ok=True)
    args.out_table_dir.mkdir(parents=True, exist_ok=True)

    masks = _std_coords(xr.open_dataset(args.masks))
    try:
        lat_slice, lon_slice = _mask_bbox(
            masks=masks,
            domains=(WEST_DOMAIN, EAST_DOMAIN),
            buffer_deg=args.mask_buffer_deg,
        )
        masks_sub = masks.sel(lat=lat_slice, lon=lon_slice)
        ds = _load_inputs(
            input_dir=args.input_dir,
            input_glob=args.input_glob,
            masks_sub=masks_sub,
            lat_slice=lat_slice,
            lon_slice=lon_slice,
            utc_hour=utc_hour,
            start_year=args.start_year,
            end_year=args.end_year,
        )
        ds = ds.sel(time=slice(f"{args.start_year}-01-01", f"{args.end_year}-12-31"))
        if ds.sizes.get("time", 0) == 0:
            raise RuntimeError("No data remain after applying year/time filters.")

        ds_ts, trend_ds = _compute_timeseries(
            ds=ds,
            masks_sub=masks_sub,
            detrend_ref_year=detrend_ref_year,
        )
        ds_ts.attrs.update(
            {
                "description": "F10 west-AP FPT2000 and AP U100 time series",
                "fpt_formula": "t2m_C + max(0, 2000 - max(0, 125*(t2m_C-d2m_C))) * 2.8/1000",
                "west_domain": WEST_DOMAIN,
                "u100_domain": f"{WEST_DOMAIN}|{EAST_DOMAIN}",
                "detrend_reference_year": detrend_ref_year,
                "utc_hour_filter": "none" if utc_hour is None else int(utc_hour),
                "source_input_dir": str(args.input_dir),
                "masks": str(args.masks),
            }
        )
        out_nc = args.out_data_dir / "fpt2000_westap_timeseries.nc"
        ds_ts.to_netcdf(out_nc)
        print(f"Saved data -> {out_nc}")

        trend_nc = args.out_data_dir / "fpt2000_input_trends_gridcell.nc"
        trend_ds.to_netcdf(trend_nc)
        print(f"Saved data -> {trend_nc}")

        df = _samples_dataframe(
            ds_ts=ds_ts,
            start_year=args.start_year,
            end_year=args.end_year,
            present_start=args.present_start,
        )
        _save_summary_tables(df, args.out_table_dir)

        decades = _decade_ranges(args.start_year, args.end_year)
        _plot_faceted_scatter(
            df=df,
            decades=decades,
            out_png=args.out_fig_dir / "U100_vs_FPT2000_decade_raw.png",
            y_col="fpt2000_west_c",
            t2m_col="t2m_west_c",
            title="Western-side AP FPT2000 during February 15-local-time samples",
            present_start=args.present_start,
            figure_dpi=args.figure_dpi,
            draw_segments=not args.no_segments,
        )
        _plot_faceted_scatter(
            df=df,
            decades=decades,
            out_png=args.out_fig_dir / "U100_vs_FPT2000_decade_dewarmed.png",
            y_col="fpt2000_dewarmed_west_c",
            t2m_col="t2m_dewarmed_west_c",
            title="De-warmed western-side AP FPT2000 during February 15-local-time samples",
            present_start=args.present_start,
            figure_dpi=args.figure_dpi,
            draw_segments=not args.no_segments,
        )
    finally:
        masks.close()


if __name__ == "__main__":
    main()
