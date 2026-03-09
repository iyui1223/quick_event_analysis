"""
ERA5 I/O and path helpers for map_room pipeline.
Uses xarray with dask-friendly chunking. Variable names: t2m, msl.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import xarray as xr


def get_project_root() -> Path:
    """Project root (map_room). Assumes this file is in PROJECT_ROOT/Python/."""
    return Path(__file__).resolve().parent.parent


def get_era5_t2m_paths(era5_t2m_dir: Path, start_year: int, end_year: int) -> List[Path]:
    """Return list of existing ERA5 daily 2m temperature files for year range."""
    paths = []
    for year in range(start_year, end_year + 1):
        p = era5_t2m_dir / f"era5_daily_2m_temperature_{year}.nc"
        if p.exists():
            paths.append(p)
    return sorted(paths)


def get_era5_msl_paths(era5_msl_dir: Path, start_year: int, end_year: int) -> List[Path]:
    """Return list of existing ERA5 daily MSL pressure files for year range."""
    paths = []
    for year in range(start_year, end_year + 1):
        p = era5_msl_dir / f"era5_daily_mean_sea_level_pressure_{year}.nc"
        if p.exists():
            paths.append(p)
    return sorted(paths)


def get_era5_u_paths(era5_u_dir: Path, start_year: int, end_year: int) -> List[Path]:
    """
    Return list of existing ERA5 daily U-wind files for year range.
    grads_ctl layout: era5_daily_u_component_of_wind_YYYY.nc (all 37 levels in one file).
    """
    if not era5_u_dir.exists():
        return []
    paths = []
    for year in range(start_year, end_year + 1):
        p = era5_u_dir / f"era5_daily_u_component_of_wind_{year}.nc"
        if p.exists():
            paths.append(p)
    return sorted(paths)


def open_era5_daily(
    paths: List[Path],
    var_name: str = "t2m",
    chunks: Optional[dict] = None,
    lat_south_of: Optional[float] = None,
) -> xr.Dataset:
    """
    Open multi-year ERA5 daily NetCDF with xarray (lazy).
    Standardizes coords to lat, lon, time. Optionally slice to lat <= lat_south_of.
    Chunking: default time=365 for dask-friendly annual chunks.
    """
    if chunks is None:
        chunks = {"time": 365}
    ds = xr.open_mfdataset(
        paths,
        combine="by_coords",
        chunks=chunks,
        parallel=False,
        engine="netcdf4",
    )
    # Standardize coord names
    renames = {}
    for c in list(ds.coords):
        cl = c.lower()
        if "lat" in cl and c != "latitude":
            renames[c] = "latitude"
        elif "lon" in cl and c != "longitude":
            renames[c] = "longitude"
        elif ("time" in cl or "valid" in cl) and c != "time":
            renames[c] = "time"
    if renames:
        ds = ds.rename(renames)
    # Common ERA5 names
    if "latitude" in ds.coords and "lat" not in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords and "lon" not in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if lat_south_of is not None:
        # lat typically -90 to 90; south of 55S means lat <= -55
        ds = ds.sel(lat=ds.lat <= lat_south_of)
    return ds


def load_era5_t2m_south(
    era5_t2m_dir: Path,
    start_year: int,
    end_year: int,
    lat_cutoff: float = -55.0,
    chunks: Optional[dict] = None,
) -> xr.DataArray:
    """Load t2m daily for given year range, south of lat_cutoff (lazy)."""
    paths = get_era5_t2m_paths(Path(era5_t2m_dir), start_year, end_year)
    if not paths:
        raise FileNotFoundError(f"No t2m files in {era5_t2m_dir} for {start_year}-{end_year}")
    ds = open_era5_daily(paths, var_name="t2m", chunks=chunks, lat_south_of=lat_cutoff)
    # Find t2m variable (may be t2m or similar)
    for v in ["t2m", "T2m", "2t"]:
        if v in ds:
            return ds[v]
    return ds[list(ds.data_vars)[0]]


def load_era5_msl_south(
    era5_msl_dir: Path,
    start_year: int,
    end_year: int,
    lat_cutoff: float = -55.0,
    chunks: Optional[dict] = None,
) -> xr.DataArray:
    """Load msl daily for given year range, south of lat_cutoff (lazy)."""
    paths = get_era5_msl_paths(Path(era5_msl_dir), start_year, end_year)
    if not paths:
        raise FileNotFoundError(f"No msl files in {era5_msl_dir} for {start_year}-{end_year}")
    ds = open_era5_daily(paths, var_name="msl", chunks=chunks, lat_south_of=lat_cutoff)
    for v in ["msl", "MSL", "mslp"]:
        if v in ds:
            return ds[v]
    return ds[list(ds.data_vars)[0]]


def open_era5_one_year(
    path: Path,
    lat_south_of: Optional[float] = None,
    lat_min: Optional[float] = None,
    chunks: Optional[dict] = None,
) -> xr.Dataset:
    """
    Open a single year ERA5 daily file (lazy, chunked).
    Use .sel(time=...).load() to pull only needed months and avoid OOM.
    lat_south_of: keep lat <= this (e.g. -55). lat_min: keep lat >= this (e.g. -80).
    """
    if chunks is None:
        chunks = {"time": 31}
    ds = xr.open_dataset(str(path), chunks=chunks, engine="netcdf4")
    renames = {}
    for c in list(ds.coords):
        cl = c.lower()
        if "lat" in cl and c not in ("lat", "latitude"):
            renames[c] = "lat" if "latitude" in str(c).lower() else "lat"
        elif "lon" in cl and c not in ("lon", "longitude"):
            renames[c] = "lon" if "longitude" in str(c).lower() else "lon"
        elif ("time" in cl or "valid" in cl) and c != "time":
            renames[c] = "time"
    if "latitude" in ds.coords and "lat" not in ds.coords:
        renames["latitude"] = "lat"
    if "longitude" in ds.coords and "lon" not in ds.coords:
        renames["longitude"] = "lon"
    if renames:
        ds = ds.rename(renames)
    if lat_south_of is not None:
        ds = ds.sel(lat=ds.lat <= lat_south_of)
    if lat_min is not None:
        ds = ds.sel(lat=ds.lat >= lat_min)
    return ds
