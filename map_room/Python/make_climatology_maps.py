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
from typing import Optional, Tuple

import numpy as np
import xarray as xr

from utils_era5_io import (
    get_project_root,
    get_era5_t2m_paths,
    get_era5_msl_paths,
    get_era5_u_paths,
    open_era5_one_year,
)
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


def _sel_level(da: xr.DataArray, level_hpa: int) -> xr.DataArray:
    """Select pressure level from DataArray; level coord may be level, lev, plev, etc."""
    for dim in ("level", "lev", "plev", "pressure"):
        if dim in da.coords:
            out = da.sel({dim: level_hpa}, method="nearest").squeeze()
            return out
    raise ValueError(f"No level dimension in {list(da.coords)}; cannot select {level_hpa} hPa")


def compute_monthly_climatology_u_year_by_year(
    era_name: str,
    start_year: int,
    end_year: int,
    month: int,
    u_dir: Path,
    lat_cutoff: float,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Compute U850 and U500 climatology for one (era, month) by accumulating over years.
    Reads era5_daily_u_component_of_wind_YYYY.nc (all levels in one file), selects 850 and 500 hPa.
    Returns (clim_u850, clim_u500) with same coords (lat, lon).
    """
    paths = get_era5_u_paths(u_dir, start_year, end_year)
    if not paths:
        raise FileNotFoundError(f"No U-wind files in {u_dir} for {start_year}-{end_year}")

    month_sum_u850 = None
    month_sum_u500 = None
    n_days = 0
    template = None

    for path in paths:
        year = int(path.stem.split("_")[-1])
        ds = open_era5_one_year(path, lat_south_of=lat_cutoff, chunks={"time": 31})
        da_u = _get_var(ds, ["u", "U"])
        da_u850 = _sel_level(da_u, 850)
        da_u500 = _sel_level(da_u, 500)
        u850_month = da_u850.sel(time=da_u850.time.dt.month == month).load()
        u500_month = da_u500.sel(time=da_u500.time.dt.month == month).load()
        n = u850_month.sizes.get("time", 0)
        if n == 0:
            ds.close()
            del ds, da_u, da_u850, da_u500, u850_month, u500_month
            gc.collect()
            continue
        u850_sum = u850_month.sum(dim="time")
        u500_sum = u500_month.sum(dim="time")
        if month_sum_u850 is None:
            month_sum_u850 = u850_sum.values.astype(np.float64)
            month_sum_u500 = u500_sum.values.astype(np.float64)
            template = u850_sum
        else:
            month_sum_u850 += u850_sum.values
            month_sum_u500 += u500_sum.values
        n_days += n
        ds.close()
        del ds, da_u, da_u850, da_u500, u850_month, u500_month, u850_sum, u500_sum
        gc.collect()

    if month_sum_u850 is None or n_days == 0:
        raise RuntimeError(f"No U data for era {era_name} month {month}")

    clim_u850 = month_sum_u850 / n_days
    clim_u500 = month_sum_u500 / n_days
    dims = list(template.dims)
    coords = {c: template.coords[c] for c in dims}
    clim_u850_da = xr.DataArray(clim_u850, dims=dims, coords=coords, name="u850")
    clim_u500_da = xr.DataArray(clim_u500, dims=dims, coords=coords, name="u500")
    return clim_u850_da, clim_u500_da


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
    u_dir: Optional[Path] = None,
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
        del clim_t2m, clim_msl
        gc.collect()

        # U850/U500 climatology if U dir set and has data (grads_ctl: all levels in one file)
        if u_dir is not None:
            try:
                clim_u850, clim_u500 = compute_monthly_climatology_u_year_by_year(
                    era_name, start_year, end_year, month, u_dir, lat_cutoff
                )
                out_u = data_dir / era_name / mstr / "clim_u.nc"
                t0 = xr.DataArray([np.datetime64("2000-01-01")], dims=["time"])
                ds_u = xr.Dataset(
                    {
                        "u850": clim_u850.expand_dims(time=t0),
                        "u500": clim_u500.expand_dims(time=t0),
                    }
                )
                ds_u.to_netcdf(out_u)
                del clim_u850, clim_u500, ds_u
                gc.collect()
            except FileNotFoundError as e:
                print(f"    Skip U (no data): {e}", flush=True)


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
    u_dir = Path(os.environ["ERA5_U_DIR"]) if os.environ.get("ERA5_U_DIR") else None
    if u_dir is not None and not u_dir.exists():
        print("ERA5_U_DIR missing; skipping U climatology.", flush=True)
        u_dir = None

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
        u_dir=u_dir,
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
        u_dir=u_dir,
    )
    print("F01 done. NetCDFs under", data_dir, flush=True)
    print("Run Sh/run_plot_climatology_grads.sh for GrADS figures.", flush=True)


if __name__ == "__main__":
    main()
