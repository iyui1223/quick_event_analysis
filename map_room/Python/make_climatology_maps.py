"""
F01: Compute DJFM daily-mean climatology for two eras and plot polar maps.
Uses year-by-year processing with lazy loading (one year, one month at a time)
to avoid OOM; pattern follows kotesaki_tools/climatology/Python/climatology_calc.py.
Era A: 1948-1987, Era B: 1988-2025. Months: Dec(12), Jan(1), Feb(2), Mar(3).
Output: Data/F01_climatology/{era}/{month}/clim.nc, Figs/.../t2m_msl.png and diff.
"""

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr

from utils_era5_io import (
    get_project_root,
    get_era5_t2m_paths,
    get_era5_msl_paths,
    open_era5_one_year,
)
from utils_plot_polar import plot_polar_map

ROOT = get_project_root()


def load_params():
    import yaml
    with open(ROOT / "Const" / "params.yaml") as f:
        return yaml.safe_load(f)


def _get_var(ds: xr.Dataset, names: list) -> xr.DataArray:
    for v in names:
        if v in ds:
            return ds[v]
    return ds[list(ds.data_vars)[0]]


def compute_monthly_climatology_year_by_year(
    era_name: str,
    start_year: int,
    end_year: int,
    month: int,
    t2m_dir: Path,
    msl_dir: Path,
    lat_cutoff: float,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Compute climatology for one (era, month) by accumulating over years.
    Only one year's worth of one month is in memory at a time.
    Returns (clim_t2m, clim_msl) with same coords (lat, lon).
    """
    t2m_paths = get_era5_t2m_paths(t2m_dir, start_year, end_year)
    msl_paths = get_era5_msl_paths(msl_dir, start_year, end_year)
    # Years that have both variables
    t2m_years = {int(p.stem.split("_")[-1]) for p in t2m_paths}
    msl_years = {int(p.stem.split("_")[-1]) for p in msl_paths}
    years = sorted(t2m_years & msl_years)
    if not years:
        raise FileNotFoundError(
            f"No common years for t2m/msl in {start_year}-{end_year} "
            f"(t2m: {len(t2m_paths)}, msl: {len(msl_paths)})"
        )
    t2m_by_year = {y: next(p for p in t2m_paths if p.stem.endswith(f"_{y}")) for y in years}
    msl_by_year = {y: next(p for p in msl_paths if p.stem.endswith(f"_{y}")) for y in years}

    month_sum_t2m = None
    month_sum_msl = None
    n_days = 0
    template_t2m = None

    for year in years:
        ds_t2m = open_era5_one_year(t2m_by_year[year], lat_south_of=lat_cutoff, chunks={"time": 31})
        ds_msl = open_era5_one_year(msl_by_year[year], lat_south_of=lat_cutoff, chunks={"time": 31})
        da_t2m = _get_var(ds_t2m, ["t2m", "T2m", "2t"])
        da_msl = _get_var(ds_msl, ["msl", "MSL", "mslp"])
        # Select this calendar month only (small: ~28-31 time steps)
        t2m_month = da_t2m.sel(time=da_t2m.time.dt.month == month)
        msl_month = da_msl.sel(time=da_msl.time.dt.month == month)
        # Load into memory (only this month)
        t2m_month = t2m_month.load()
        msl_month = msl_month.load()
        n = t2m_month.sizes.get("time", 0)
        if n == 0:
            ds_t2m.close()
            ds_msl.close()
            del ds_t2m, ds_msl, da_t2m, da_msl, t2m_month, msl_month
            gc.collect()
            continue
        # Accumulate sum over days
        t2m_sum = t2m_month.sum(dim="time")
        msl_sum = msl_month.sum(dim="time")
        if month_sum_t2m is None:
            month_sum_t2m = t2m_sum.values.astype(np.float64)
            month_sum_msl = msl_sum.values.astype(np.float64)
            template_t2m = t2m_sum  # no time dim after sum(dim="time")
        else:
            month_sum_t2m += t2m_sum.values
            month_sum_msl += msl_sum.values
        n_days += n
        ds_t2m.close()
        ds_msl.close()
        del ds_t2m, ds_msl, da_t2m, da_msl, t2m_month, msl_month, t2m_sum, msl_sum
        gc.collect()

    if month_sum_t2m is None or n_days == 0:
        raise RuntimeError(f"No data for era {era_name} month {month}")

    clim_t2m = month_sum_t2m / n_days
    clim_msl = month_sum_msl / n_days
    # Rebuild DataArrays with coords from template (drop time)
    dims = list(template_t2m.dims)
    coords = {c: template_t2m.coords[c] for c in dims}
    clim_t2m_da = xr.DataArray(clim_t2m, dims=dims, coords=coords, name="t2m")
    clim_msl_da = xr.DataArray(clim_msl, dims=dims, coords=coords, name="msl")
    return clim_t2m_da, clim_msl_da


def run_era(
    era_name: str,
    start_year: int,
    end_year: int,
    t2m_dir: Path,
    msl_dir: Path,
    lat_cutoff: float,
    months: list,
    data_dir: Path,
    figs_dir: Path,
) -> None:
    """Compute and save climatology NetCDF and figures for one era (year-by-year, low memory)."""
    month_names = {"12": "12", "1": "01", "2": "02", "3": "03"}
    for month in months:
        mstr = month_names.get(str(month), str(month).zfill(2))
        print(f"  {era_name} month {mstr}...", flush=True)
        clim_t2m, clim_msl = compute_monthly_climatology_year_by_year(
            era_name, start_year, end_year, month, t2m_dir, msl_dir, lat_cutoff
        )
        out_nc = data_dir / era_name / mstr / "clim.nc"
        out_nc.parent.mkdir(parents=True, exist_ok=True)
        # Add time=1 for GrADS ctl (t,y,x)
        t0 = xr.DataArray([np.datetime64("2000-01-01")], dims=["time"])
        ds_out = xr.Dataset(
            {"t2m": clim_t2m.expand_dims(time=t0), "msl": clim_msl.expand_dims(time=t0)}
        )
        ds_out.to_netcdf(out_nc)
        fig_dir = figs_dir / era_name / mstr
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_polar_map(
            clim_t2m,
            fig_dir / "t2m_msl.png",
            title=f"Climatology {era_name} month {mstr}",
            cmap="viridis",
            contours=clim_msl,
            contour_levels=14,
            colorbar_label="t2m (K)",
        )
        del clim_t2m, clim_msl
        gc.collect()


def run_diff(
    past_name: str,
    present_name: str,
    months: list,
    data_dir: Path,
    figs_dir: Path,
) -> None:
    """Era B − Era A difference maps (reads small clim NetCDFs)."""
    month_names = {"12": "12", "1": "01", "2": "02", "3": "03"}
    for month in months:
        mstr = month_names.get(str(month), str(month).zfill(2))
        past_ds = xr.open_dataset(data_dir / past_name / mstr / "clim.nc")
        pres_ds = xr.open_dataset(data_dir / present_name / mstr / "clim.nc")
        diff_t2m = pres_ds["t2m"] - past_ds["t2m"]
        diff_msl = pres_ds["msl"] - past_ds["msl"]
        out_fig = figs_dir / "diff" / mstr / "t2m_msl_diff.png"
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        plot_polar_map(
            diff_t2m,
            out_fig,
            title=f"Difference (present − past) month {mstr}",
            cmap="RdBu_r",
            contours=diff_msl,
            contour_levels=12,
            colorbar_label="t2m diff (K)",
        )
        past_ds.close()
        pres_ds.close()


def main():
    p = argparse.ArgumentParser(description="F01 Climatology maps DJFM (year-by-year, low memory)")
    p.add_argument("--data-dir", default=None, help="Data output dir (default: PROJECT_ROOT/Data/F01_climatology)")
    p.add_argument("--figs-dir", default=None, help="Figs output dir (default: PROJECT_ROOT/Figs/F01_climatology)")
    args = p.parse_args()
    params = load_params()
    domain = params.get("domain", {})
    lat_cutoff = domain.get("lat_cutoff", -55)
    eras = params.get("eras", {})
    past = eras["past"]
    present = eras["present"]
    months = params.get("months", [12, 1, 2, 3])

    t2m_dir = Path(os.environ.get("ERA5_T2M_DIR", str(ROOT / "Data")))
    msl_dir = Path(os.environ.get("ERA5_MSL_DIR", str(ROOT / "Data")))
    # Use passed paths as-is (Slurm passes DATA_DIR/F01_climatology); default adds F01_climatology
    data_dir = (
        Path(args.data_dir)
        if args.data_dir
        else Path(os.environ.get("DATA_DIR", str(ROOT / "Data"))) / "F01_climatology"
    )
    figs_dir = (
        Path(args.figs_dir)
        if args.figs_dir
        else Path(os.environ.get("FIGS_DIR", str(ROOT / "Figs"))) / "F01_climatology"
    )

    print("F01 climatology (year-by-year, one month at a time)", flush=True)
    run_era(
        past["name"],
        past["start_year"],
        past["end_year"],
        t2m_dir,
        msl_dir,
        lat_cutoff,
        months,
        data_dir,
        figs_dir,
    )
    run_era(
        present["name"],
        present["start_year"],
        present["end_year"],
        t2m_dir,
        msl_dir,
        lat_cutoff,
        months,
        data_dir,
        figs_dir,
    )
    print("Difference maps...", flush=True)
    run_diff(past["name"], present["name"], months, data_dir, figs_dir)
    print("F01 done. Outputs under", data_dir, "and", figs_dir)


if __name__ == "__main__":
    main()
