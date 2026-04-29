#!/usr/bin/env python3
"""
F09: Peninsula warm-extreme histograms by decade (ONDJFM), split by westerly flow.

Workflow:
  1) Build (or reuse) cached daily area-mean west/east peninsula series
     for t2m, u10, v10 from ERA5 monthly surface archives (YYYYMM.nc zip files).
  2) Apply adjustable N-day rolling mean.
  3) For each decade (1949-1958, 1959-1968, ...), plot stacked histograms
     of daily-mean temperature where:
       - bottom bar = westerly days (wind-from direction 270 +/- 30 deg)
       - top bar    = non-westerly days
  4) Export CSVs for warmest top 5%, 1%, 0.5% events per decade/domain.

Defaults are chosen to match current project layout and user request.
"""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]

SURF_DIR_DEFAULT = Path(
    "/lustre/soge1/projects/andante/cenv1201/heavy/ERA5/daily/Surf/slices"
)
MASKS_DEFAULT = ROOT / "Data" / "F04_peninsula_domains" / "all_domain_masks.nc"
OUT_DATA_DEFAULT = ROOT / "Data" / "F09_peninsula_extreme_histograms"
OUT_FIG_DEFAULT = ROOT / "Figs" / "F09_peninsula_extreme_histograms"

DOMAINS = ["pen_west_slope", "pen_east_slope"]
VARS = ["t2m", "u10", "v10"]

# ERA5 member name patterns inside monthly zip files
VAR_PATTERNS = {
    "t2m": ["2m_temperature", "t2m"],
    "u10": ["10m_u_component", "u10"],
    "v10": ["10m_v_component", "v10"],
}


def _parse_months(text: str) -> list[int]:
    vals = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = int(tok)
        if m < 1 or m > 12:
            raise ValueError(f"Invalid month: {m}")
        vals.append(m)
    if not vals:
        raise ValueError("No months parsed from --months")
    return sorted(set(vals))


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
    if renames:
        ds = ds.rename(renames)

    if ds.lat[0] > ds.lat[-1]:
        ds = ds.sortby("lat")
    if float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon % 360)).sortby("lon")
    return ds


def _find_member(zf: zipfile.ZipFile, patterns: list[str]) -> str | None:
    names = zf.namelist()
    for name in names:
        low = name.lower()
        if any(p in low for p in patterns):
            return name
    return None


def _open_member_subset(member_path: Path, lat_slice: slice, lon_slice: slice) -> xr.DataArray:
    ds = xr.open_dataset(member_path)
    ds = _std_coords(ds)
    vname = list(ds.data_vars)[0]
    da = ds[vname].sel(lat=lat_slice, lon=lon_slice).load()
    ds.close()
    return da


def load_month_surface_subset(path: Path, lat_slice: slice, lon_slice: slice) -> xr.Dataset | None:
    """
    Load monthly t2m/u10/v10 subset from:
      - zip archive with .nc members (typical in this project), or
      - regular netcdf containing multiple vars.
    """
    path = Path(path)
    if not path.exists():
        return None

    # Case 1: zip archive
    if zipfile.is_zipfile(path):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            with zipfile.ZipFile(path, "r") as zf:
                extracted: dict[str, xr.DataArray] = {}
                for var in VARS:
                    member = _find_member(zf, VAR_PATTERNS[var])
                    if member is None:
                        continue
                    out_fp = Path(zf.extract(member, path=tmpdir))
                    extracted[var] = _open_member_subset(out_fp, lat_slice, lon_slice)
                if not all(v in extracted for v in VARS):
                    return None
                return xr.Dataset(extracted)

    # Case 2: direct netcdf
    try:
        ds = xr.open_dataset(path)
        ds = _std_coords(ds)
        out = {}
        for v in ds.data_vars:
            vlow = str(v).lower()
            if "2m_temperature" in vlow or v == "t2m":
                out["t2m"] = ds[v].sel(lat=lat_slice, lon=lon_slice).load()
            elif "10m_u_component" in vlow or v == "u10":
                out["u10"] = ds[v].sel(lat=lat_slice, lon=lon_slice).load()
            elif "10m_v_component" in vlow or v == "v10":
                out["v10"] = ds[v].sel(lat=lat_slice, lon=lon_slice).load()
        ds.close()
        if not all(v in out for v in VARS):
            return None
        return xr.Dataset(out)
    except Exception:
        return None


def weighted_domain_mean(da: xr.DataArray, mask: xr.DataArray, weights2d: xr.DataArray) -> xr.DataArray:
    masked = da.where(mask > 0)
    wmask = weights2d.where(mask > 0)
    numer = (masked * wmask).sum(dim=["lat", "lon"])
    denom = wmask.sum(dim=["lat", "lon"])
    return numer / denom


def _discover_month_files(surf_dir: Path, start_year: int, end_year: int, months: set[int]) -> list[Path]:
    out = []
    for fp in sorted(surf_dir.glob("*.nc")):
        stem = fp.stem
        if len(stem) != 6 or not stem.isdigit():
            continue
        year = int(stem[:4])
        month = int(stem[4:6])
        if year < start_year or year > end_year:
            continue
        if month not in months:
            continue
        out.append(fp)
    return out


def build_or_load_daily_cache(
    surf_dir: Path,
    masks_path: Path,
    cache_path: Path,
    start_year: int,
    end_year: int,
    months: list[int],
    force_rebuild: bool,
) -> xr.Dataset:
    if cache_path.exists() and not force_rebuild:
        ds_cache = xr.open_dataset(cache_path)
        cache_months_txt = str(ds_cache.attrs.get("months", "")).strip()
        cache_months = set()
        if cache_months_txt:
            for tok in cache_months_txt.split(","):
                tok = tok.strip()
                if tok:
                    cache_months.add(int(tok))
        cache_start = int(ds_cache.attrs.get("start_year", -9999))
        cache_end = int(ds_cache.attrs.get("end_year", -9999))
        req_months = set(months)
        period_ok = (cache_start <= start_year) and (cache_end >= end_year)
        months_ok = (cache_months == req_months)
        if period_ok and months_ok:
            print(f"Using cache: {cache_path}")
            return ds_cache
        ds_cache.close()
        print(
            "Cache exists but does not match requested coverage; rebuilding "
            f"(cache years {cache_start}-{cache_end}, req {start_year}-{end_year}; "
            f"cache months {sorted(cache_months)}, req {sorted(req_months)})."
        )

    if not masks_path.exists():
        raise FileNotFoundError(f"Masks not found: {masks_path}")

    masks = xr.open_dataset(masks_path)
    masks = _std_coords(masks)
    if not all(d in masks for d in DOMAINS):
        missing = [d for d in DOMAINS if d not in masks]
        raise ValueError(f"Missing mask variables: {missing}")

    # Smallest bounding box that covers west/east peninsula masks
    union = (masks[DOMAINS[0]] > 0) | (masks[DOMAINS[1]] > 0)
    valid_lat = masks.lat.where(union.any("lon"), drop=True)
    valid_lon = masks.lon.where(union.any("lat"), drop=True)
    lat_min = float(valid_lat.min())
    lat_max = float(valid_lat.max())
    lon_min = float(valid_lon.min())
    lon_max = float(valid_lon.max())
    lat_slice = slice(lat_min, lat_max)
    lon_slice = slice(lon_min, lon_max)

    masks_sub = masks.sel(lat=lat_slice, lon=lon_slice)
    lat_rad = np.deg2rad(masks_sub.lat)
    w_lat = xr.DataArray(np.cos(lat_rad), coords={"lat": masks_sub.lat}, dims=["lat"])
    weights2d = w_lat.broadcast_like(masks_sub[DOMAINS[0]])

    files = _discover_month_files(
        surf_dir=surf_dir,
        start_year=start_year,
        end_year=end_year,
        months=set(months),
    )
    if not files:
        raise FileNotFoundError(
            f"No monthly files found in {surf_dir} for years {start_year}-{end_year} and months {months}"
        )

    print(f"Building cache from {len(files)} monthly files...")
    month_datasets: list[xr.Dataset] = []

    for i, fp in enumerate(files, start=1):
        print(f"  [{i:4d}/{len(files)}] {fp.name}", end=" ", flush=True)
        ds_month = load_month_surface_subset(fp, lat_slice=lat_slice, lon_slice=lon_slice)
        if ds_month is None:
            print("skip (missing vars or unreadable)")
            continue

        ds_month = _std_coords(ds_month)
        out_vars = {}
        for var in VARS:
            if var not in ds_month:
                continue
            for dom in DOMAINS:
                out_vars[f"{var}_{dom}"] = weighted_domain_mean(
                    ds_month[var], masks_sub[dom], weights2d
                )
        month_out = xr.Dataset(out_vars)
        month_datasets.append(month_out)
        ds_month.close()
        print("ok")

    if not month_datasets:
        raise RuntimeError("No monthly data processed; cache cannot be built.")

    out = xr.concat(month_datasets, dim="time").sortby("time")

    # Remove duplicated timestamps, keeping first occurrence
    t_idx = pd.DatetimeIndex(out.time.values)
    keep = ~t_idx.duplicated(keep="first")
    out = out.isel(time=np.where(keep)[0])

    out.attrs["description"] = "Cached daily area means for peninsula west/east domains"
    out.attrs["domains"] = ",".join(DOMAINS)
    out.attrs["months"] = ",".join(str(m) for m in months)
    out.attrs["start_year"] = int(start_year)
    out.attrs["end_year"] = int(end_year)
    out.attrs["source_dir"] = str(surf_dir)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(cache_path)
    print(f"Saved cache -> {cache_path}")

    masks.close()
    return out


def _rolling_mean(da: xr.DataArray, n_days: int, center: bool) -> xr.DataArray:
    if n_days <= 1:
        return da
    return da.rolling(time=n_days, min_periods=n_days, center=center).mean()


def _wind_from_direction_deg(u: pd.Series, v: pd.Series) -> pd.Series:
    # Meteorological "from" direction:
    # 0=N, 90=E, 180=S, 270=W
    direction = (180.0 + np.degrees(np.arctan2(u.values, v.values))) % 360.0
    return pd.Series(direction, index=u.index)


def _decade_ranges(start_year: int, end_year: int) -> list[tuple[int, int]]:
    out = []
    y0 = start_year
    while y0 <= end_year:
        y1 = min(y0 + 9, end_year)
        out.append((y0, y1))
        y0 += 10
    return out


def _interp_color(c0: tuple[float, float, float], c1: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    return tuple((1.0 - t) * a + t * b for a, b in zip(c0, c1))


def _mix_with(color: tuple[float, float, float], target: tuple[float, float, float], amt: float) -> tuple[float, float, float]:
    return tuple((1.0 - amt) * c + amt * t for c, t in zip(color, target))


def _build_decade_colors(decades: list[tuple[int, int]]) -> dict[tuple[int, int], tuple[float, float, float]]:
    """
    Requested palette:
      - pre-satellite (<1979): green-blue shades
      - satellite era (>=1979): yellow-orange shades
    """
    pre = [d for d in decades if d[1] < 1979]
    post = [d for d in decades if d[0] >= 1979]
    crossover = [d for d in decades if d not in pre and d not in post]

    cmap: dict[tuple[int, int], tuple[float, float, float]] = {}
    pre_start = mcolors.to_rgb("#1f5aa6")   # blue
    pre_end = mcolors.to_rgb("#2fa36b")     # green
    post_start = mcolors.to_rgb("#f1c232")  # yellow
    post_end = mcolors.to_rgb("#e67e22")    # orange

    for i, dec in enumerate(pre):
        t = 0.5 if len(pre) == 1 else i / (len(pre) - 1)
        cmap[dec] = _interp_color(pre_start, pre_end, t)
    for i, dec in enumerate(post):
        t = 0.5 if len(post) == 1 else i / (len(post) - 1)
        cmap[dec] = _interp_color(post_start, post_end, t)
    for dec in crossover:
        # If a decade straddles 1979, use a neutral bridge color.
        cmap[dec] = mcolors.to_rgb("#b9c24f")
    return cmap


def _make_domain_dataframe(ds_daily: xr.Dataset, domain: str, mean_days: int, center: bool) -> pd.DataFrame:
    t2m = _rolling_mean(ds_daily[f"t2m_{domain}"], mean_days, center=center) - 273.15
    u10 = _rolling_mean(ds_daily[f"u10_{domain}"], mean_days, center=center)
    v10 = _rolling_mean(ds_daily[f"v10_{domain}"], mean_days, center=center)

    time_index = pd.DatetimeIndex(ds_daily.time.values)
    df = pd.DataFrame(
        {
            "time": time_index,
            "t2m_c": t2m.values,
            "u10_ms": u10.values,
            "v10_ms": v10.values,
        }
    ).set_index("time")
    df = df.dropna(subset=["t2m_c", "u10_ms", "v10_ms"])
    df["wdir_from_deg"] = _wind_from_direction_deg(df["u10_ms"], df["v10_ms"])
    # pure westerly +/- 30deg => [240, 300]
    df["is_westerly"] = (df["wdir_from_deg"] >= 240.0) & (df["wdir_from_deg"] <= 300.0)
    df["year"] = df.index.year
    df["month"] = df.index.month
    return df


def _save_percentile_csvs(
    df_by_domain: dict[str, pd.DataFrame],
    decades: list[tuple[int, int]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    percentile_specs = [
        (95.0, "warmest_top_5pct.csv"),
        (99.0, "warmest_top_1pct.csv"),
        (99.5, "warmest_top_0p5pct.csv"),
    ]
    summary_rows: list[dict] = []

    for pct, fname in percentile_specs:
        rows = []
        for domain, df in df_by_domain.items():
            for d0, d1 in decades:
                sub = df[(df["year"] >= d0) & (df["year"] <= d1)].copy()
                if sub.empty:
                    continue
                thr = float(np.nanpercentile(sub["t2m_c"].values, pct))
                hot = sub[sub["t2m_c"] >= thr].copy()
                if hot.empty:
                    continue
                hot = hot.reset_index()
                hot.insert(0, "domain", domain)
                hot.insert(1, "decade_start", d0)
                hot.insert(2, "decade_end", d1)
                hot.insert(3, "percentile", pct)
                hot.insert(4, "threshold_c", thr)
                rows.append(hot)
                summary_rows.append(
                    {
                        "domain": domain,
                        "decade_start": d0,
                        "decade_end": d1,
                        "percentile": pct,
                        "threshold_c": thr,
                        "n_events": int(len(hot)),
                    }
                )
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        out.to_csv(out_dir / fname, index=False)
        print(f"Saved CSV -> {out_dir / fname} ({len(out)} rows)")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "warm_extreme_thresholds_summary.csv", index=False)
    print(f"Saved CSV -> {out_dir / 'warm_extreme_thresholds_summary.csv'} ({len(summary)} rows)")


def _plot_histograms_for_domain(
    domain: str,
    df: pd.DataFrame,
    decades: list[tuple[int, int]],
    out_dir: Path,
    mean_days: int,
    months: list[int],
    bin_width_c: float,
    figure_dpi: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    finite = df[np.isfinite(df["t2m_c"].values)]
    if finite.empty:
        print(f"No data for {domain}; skip plot.")
        return

    tmin = float(np.floor(finite["t2m_c"].min() / bin_width_c) * bin_width_c)
    tmax = float(np.ceil(finite["t2m_c"].max() / bin_width_c) * bin_width_c)
    bins = np.arange(tmin, tmax + bin_width_c, bin_width_c)
    if len(bins) < 3:
        bins = np.array([tmin - 0.5, tmin, tmin + 0.5])

    n_dec = len(decades)
    ncols = 2
    nrows = int(np.ceil(n_dec / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), sharex=True, sharey=False)
    axes = np.atleast_1d(axes).ravel()

    dec_colors = _build_decade_colors(decades)
    bar_x = bins[:-1]
    width = np.diff(bins)
    added_legend = False

    for i, dec in enumerate(decades):
        ax = axes[i]
        d0, d1 = dec
        sub = df[(df["year"] >= d0) & (df["year"] <= d1)]
        if sub.empty:
            ax.set_visible(False)
            continue

        base = dec_colors[dec]
        w_color = _mix_with(base, (0.0, 0.0, 0.0), 0.20)   # slightly darker
        nw_color = _mix_with(base, (1.0, 1.0, 1.0), 0.35)  # lighter

        vals_w = sub.loc[sub["is_westerly"], "t2m_c"].values
        vals_nw = sub.loc[~sub["is_westerly"], "t2m_c"].values
        c_w, _ = np.histogram(vals_w, bins=bins)
        c_nw, _ = np.histogram(vals_nw, bins=bins)

        ax.bar(
            bar_x, c_w, width=width, align="edge", color=w_color,
            edgecolor="none", label="Westerly (bottom)"
        )
        ax.bar(
            bar_x, c_nw, width=width, align="edge", bottom=c_w, color=nw_color,
            edgecolor="none", label="Non-westerly (top)"
        )

        p95 = float(np.nanpercentile(sub["t2m_c"].values, 95.0))
        p99 = float(np.nanpercentile(sub["t2m_c"].values, 99.0))
        p995 = float(np.nanpercentile(sub["t2m_c"].values, 99.5))
        ax.axvline(p95, color="#555555", ls="--", lw=0.8, alpha=0.7)
        ax.axvline(p99, color="#333333", ls=":", lw=0.9, alpha=0.9)
        ax.axvline(p995, color="#111111", ls="-.", lw=0.9, alpha=0.9)

        n_total = len(sub)
        frac_w = 100.0 * float(sub["is_westerly"].mean())
        ax.set_title(f"{d0}-{d1} (n={n_total}, westerly={frac_w:.1f}%)", fontsize=10)
        ax.grid(True, alpha=0.2, ls=":")
        if not added_legend:
            ax.legend(loc="upper left", fontsize=8)
            added_legend = True

    for j in range(n_dec, len(axes)):
        axes[j].set_visible(False)

    month_txt = ",".join(f"{m:02d}" for m in months)
    domain_label = "West Peninsula" if domain == "pen_west_slope" else "East Peninsula"
    fig.suptitle(
        f"{domain_label}: ONDJFM Temperature Histograms by Decade\n"
        f"Stacked by Westerly (270°±30°), {mean_days}-day mean, months={month_txt}",
        fontsize=12,
    )
    for ax in axes:
        if ax.get_visible():
            ax.set_xlabel("Temperature (°C)")
            ax.set_ylabel("Count of days")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_png = out_dir / f"{domain}_ondjfm_decadal_hist_{mean_days}d.png"
    fig.savefig(out_png, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="F09: Peninsula decadal warm-extreme histograms (ONDJFM, west/east, westerly split)."
    )
    ap.add_argument("--surf-dir", type=Path, default=SURF_DIR_DEFAULT)
    ap.add_argument("--masks", type=Path, default=MASKS_DEFAULT)
    ap.add_argument("--out-data-dir", type=Path, default=OUT_DATA_DEFAULT)
    ap.add_argument("--out-fig-dir", type=Path, default=OUT_FIG_DEFAULT)
    ap.add_argument("--cache", type=Path, default=None,
                    help="Path to cached daily domain means netcdf.")
    ap.add_argument("--start-year", type=int, default=1949)
    ap.add_argument("--end-year", type=int, default=None,
                    help="Default: max available year in --surf-dir for selected months.")
    ap.add_argument("--months", type=str, default="10,11,12,1,2,3",
                    help="Comma-separated months. Default ONDJFM (10,11,12,1,2,3).")
    ap.add_argument("--mean-days", type=int, default=1,
                    help="Rolling mean window in days (1, 2, 5, ...).")
    ap.add_argument("--rolling-center", action="store_true",
                    help="Use centered rolling mean instead of trailing.")
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="Force rebuilding cached daily domain means.")
    ap.add_argument("--bin-width-c", type=float, default=1.0,
                    help="Histogram bin width in degree C.")
    ap.add_argument("--figure-dpi", type=int, default=170)
    args = ap.parse_args()

    months = _parse_months(args.months)
    args.out_data_dir.mkdir(parents=True, exist_ok=True)
    args.out_fig_dir.mkdir(parents=True, exist_ok=True)

    cache_path = args.cache or (args.out_data_dir / "peninsula_domain_daily_means_ondjfm.nc")

    # If end-year omitted, discover from available monthly files.
    if args.end_year is None:
        years = []
        for fp in sorted(args.surf_dir.glob("*.nc")):
            stem = fp.stem
            if len(stem) != 6 or not stem.isdigit():
                continue
            yy = int(stem[:4])
            mm = int(stem[4:6])
            if mm in months:
                years.append(yy)
        if not years:
            raise RuntimeError(f"No matching monthly files in {args.surf_dir} for months={months}")
        end_year = max(years)
    else:
        end_year = args.end_year

    ds_daily = build_or_load_daily_cache(
        surf_dir=args.surf_dir,
        masks_path=args.masks,
        cache_path=cache_path,
        start_year=args.start_year,
        end_year=end_year,
        months=months,
        force_rebuild=args.rebuild_cache,
    )

    # Build per-domain dataframes from cached daily means.
    df_by_domain: dict[str, pd.DataFrame] = {}
    for domain in DOMAINS:
        df = _make_domain_dataframe(
            ds_daily=ds_daily,
            domain=domain,
            mean_days=args.mean_days,
            center=args.rolling_center,
        )
        df = df[df["month"].isin(months)]
        df = df[(df["year"] >= args.start_year) & (df["year"] <= end_year)]
        df_by_domain[domain] = df

    decades = _decade_ranges(args.start_year, end_year)
    if not decades:
        raise RuntimeError("No decades found for the selected period.")

    # Save percentile CSVs
    _save_percentile_csvs(
        df_by_domain=df_by_domain,
        decades=decades,
        out_dir=args.out_data_dir,
    )

    # Save daily table too (convenient for re-plot tuning)
    daily_rows = []
    for dom, df in df_by_domain.items():
        tmp = df.copy().reset_index()
        tmp.insert(0, "domain", dom)
        daily_rows.append(tmp)
    daily_all = pd.concat(daily_rows, ignore_index=True)
    daily_csv = args.out_data_dir / f"peninsula_daily_means_{args.mean_days}d.csv"
    daily_all.to_csv(daily_csv, index=False)
    print(f"Saved CSV -> {daily_csv} ({len(daily_all)} rows)")

    # Plot per domain
    for domain, df in df_by_domain.items():
        _plot_histograms_for_domain(
            domain=domain,
            df=df,
            decades=decades,
            out_dir=args.out_fig_dir,
            mean_days=args.mean_days,
            months=months,
            bin_width_c=args.bin_width_c,
            figure_dpi=args.figure_dpi,
        )

    ds_daily.close()
    print("Done.")


if __name__ == "__main__":
    main()
