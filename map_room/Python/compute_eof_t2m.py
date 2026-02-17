"""
F02 (and F03): EOF analysis of t2m anomalies south of 55S.
Uses precomputed climatology from NetCDF (e.g. grads_ctl / kotesaki_tools).
Anomaly = daily value - climatology(day-of-year); no groupby.
Covariance computed incrementally (year-by-year) to avoid OOM.
Area weight: sqrt(cos(lat)). F03: optional time weight (Gaussian on doy).
"""

import argparse
import gc
import os
from pathlib import Path

import numpy as np
import xarray as xr

from utils_era5_io import (
    get_project_root,
    get_era5_t2m_paths,
    open_era5_one_year,
)
from utils_weights import sqrt_cos_lat_weight

try:
    from sklearn.decomposition import IncrementalPCA
except ImportError:
    IncrementalPCA = None

ROOT = get_project_root()


def load_params():
    import yaml
    with open(ROOT / "Const" / "params.yaml") as f:
        return yaml.safe_load(f)


def _coarsen_factor(grid_deg: float, input_resolution_deg: float = 0.25) -> int:
    """Integer factor to coarsen from input_resolution_deg to grid_deg (e.g. 1 deg from 0.25 -> 4)."""
    return max(1, int(round(grid_deg / input_resolution_deg)))


def load_climatology_south(
    clim_path: Path,
    lat_cutoff: float = -55.0,
    lat_min: float = -80.0,
    coarsen_factor: int = 1,
) -> tuple:
    """
    Load daily climatology (366 days) from NetCDF; slice to lat_min <= lat <= lat_cutoff.
    If coarsen_factor > 1, average to coarser grid (e.g. 4 -> 1 deg from 0.25 deg).
    Returns (clim, lat, lon): clim (366, n_lat, n_lon), 1d lat/lon.
    """
    ds = xr.open_dataset(str(clim_path))
    for c in list(ds.coords):
        if "lat" in c.lower() and c != "lat":
            ds = ds.rename({c: "lat"})
        if "lon" in c.lower() and c != "lon":
            ds = ds.rename({c: "lon"})
    if "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    da = ds["t2m"] if "t2m" in ds else ds[list(ds.data_vars)[0]]
    da = da.sel(lat=(ds.lat.values >= lat_min) & (ds.lat.values <= lat_cutoff))
    if "time" not in da.dims and "t" in da.dims:
        da = da.rename({"t": "time"})
    if coarsen_factor > 1:
        da = da.coarsen(
            lat=coarsen_factor, lon=coarsen_factor, boundary="trim"
        ).mean()
    clim = da.values
    lat = da.lat.values
    lon = da.lon.values
    ds.close()
    return clim, lat, lon


def anomaly_from_clim_doy(
    t2m_vals: np.ndarray,
    time_doy: np.ndarray,
    clim: np.ndarray,
) -> np.ndarray:
    """
    Subtract day-of-year climatology. t2m_vals (n_time, n_lat, n_lon), time_doy (n_time,) 1..366.
    clim (366, n_lat, n_lon). Returns anomaly same shape as t2m_vals.
    """
    n_time = t2m_vals.shape[0]
    anom = np.empty_like(t2m_vals, dtype=np.float64)
    for t in range(n_time):
        doy = int(time_doy[t])
        if doy < 1:
            doy = 1
        if doy > 366:
            doy = 366
        anom[t] = t2m_vals[t] - clim[doy - 1]
    return anom


def eof_from_incremental_pca(
    ipca,
    n_modes: int,
    lat: np.ndarray,
    lon: np.ndarray,
    w2d: np.ndarray,
) -> tuple:
    """Unweight components for plotting; build xr.DataArrays. components_ are (n_modes, n_space)."""
    n_lat, n_lon = len(lat), len(lon)
    evecs = ipca.components_[:n_modes].T  # (n_space, n_modes)
    variance_fraction = ipca.explained_variance_ratio_[:n_modes]
    eof_patterns = []
    for k in range(n_modes):
        eof_flat = evecs[:, k].reshape(n_lat, n_lon)
        eof_unweighted = eof_flat / np.where(w2d != 0, w2d, 1.0)
        eof_patterns.append(eof_unweighted)
    eof_da = xr.DataArray(
        np.stack(eof_patterns),
        dims=("mode", "lat", "lon"),
        coords={"mode": np.arange(1, n_modes + 1), "lat": lat, "lon": lon},
    )
    var_frac = xr.DataArray(
        variance_fraction[:n_modes],
        dims=("mode",),
        coords={"mode": np.arange(1, n_modes + 1)},
    )
    return eof_da, var_frac, ipca


def main():
    ap = argparse.ArgumentParser(description="F02/F03 EOF of t2m anomalies (climatology from file)")
    ap.add_argument("--out-dir", default=None, help="Output dir for EOFs.nc")
    ap.add_argument("--taper-djf", action="store_true", help="F03: Gaussian calendar taper (15 Jan)")
    ap.add_argument("--n-modes", type=int, default=4)
    ap.add_argument("--clim-nc", default=None, help="Climatology NetCDF (overrides params)")
    args = ap.parse_args()
    params = load_params()
    domain = params.get("domain", {})
    lat_cutoff = domain.get("lat_cutoff", -55)
    lat_min = domain.get("lat_min", -80)
    eof_cfg = params.get("eof", {})
    grid_deg = float(eof_cfg.get("grid_deg", 1.0))
    input_res = float(eof_cfg.get("input_resolution_deg", 0.25))
    coarsen_factor = _coarsen_factor(grid_deg, input_res)
    n_modes = args.n_modes or eof_cfg.get("n_modes", 4)
    present = params["eras"]["present"]
    start_year = present["start_year"]
    end_year = present["end_year"]
    t2m_dir = Path(os.environ.get("ERA5_T2M_DIR", str(ROOT / "Data")))
    clim_path = Path(
        args.clim_nc
        or os.environ.get("CLIMATOLOGY_T2M_NC")
        or eof_cfg.get("climatology_netcdf", "")
    )
    if not clim_path or not clim_path.exists():
        raise FileNotFoundError(
            f"Climatology NetCDF not found: {clim_path}. "
            "Set --clim-nc or CLIMATOLOGY_T2M_NC or eof.climatology_netcdf in params.yaml. "
            "See plan.md / grads_ctl for paths (e.g. kotesaki_tools/climatology/clim_T2m_1991-2020.nc)."
        )
    if args.taper_djf:
        step_id = "F03_eof_tapered_djf"
        center_doy = eof_cfg.get("taper_center_doy", 15)
        sigma_days = eof_cfg.get("taper_sigma_days", 45)
    else:
        step_id = "F02_eof"
    data_dir = Path(os.environ.get("DATA_DIR", str(ROOT / "Data")))
    out_dir = Path(args.out_dir or str(data_dir / step_id))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load climatology once (small); domain -80 <= lat <= -55, optional coarsening
    print("Loading climatology from", clim_path, flush=True)
    clim, lat, lon = load_climatology_south(
        clim_path, lat_cutoff=lat_cutoff, lat_min=lat_min, coarsen_factor=coarsen_factor
    )
    n_lat, n_lon = len(lat), len(lon)
    n_space = n_lat * n_lon
    w = sqrt_cos_lat_weight(xr.DataArray(lat, dims=["lat"], coords={"lat": lat})).values
    w2d = np.broadcast_to(w[:, np.newaxis], (n_lat, n_lon))

    t2m_paths = get_era5_t2m_paths(t2m_dir, start_year, end_year)
    years = sorted({int(p.stem.split("_")[-1]) for p in t2m_paths})
    t2m_by_year = {y: next(p for p in t2m_paths if p.stem.endswith(f"_{y}")) for y in years}

    # Optional time weight (F03): apply sqrt(weight) to each row so covariance is weighted
    def get_time_weight(doy_array):
        if not args.taper_djf:
            return np.ones(len(doy_array))
        sigma = float(sigma_days)
        diff = np.abs(doy_array - center_doy)
        diff = np.minimum(diff, 365 - diff)
        return np.sqrt(np.exp(-0.5 * (diff / sigma) ** 2))

    if IncrementalPCA is None:
        raise RuntimeError("sklearn.decomposition.IncrementalPCA required; pip install scikit-learn")

    # Pass 1: fit IncrementalPCA year-by-year (one year of (n_days, n_space) at a time)
    ipca = IncrementalPCA(n_components=n_modes)
    time_coords = []

    for year in years:
        ds = open_era5_one_year(
            t2m_by_year[year], lat_south_of=lat_cutoff, lat_min=lat_min, chunks={"time": 31}
        )
        da = ds["t2m"] if "t2m" in ds else ds[list(ds.data_vars)[0]]
        if coarsen_factor > 1:
            da = da.coarsen(
                lat=coarsen_factor, lon=coarsen_factor, boundary="trim"
            ).mean()
        t2m_vals = da.load().values
        time = da.time.values
        ds.close()
        doy = xr.DataArray(time, dims=["time"]).dt.dayofyear.values
        anom = anomaly_from_clim_doy(t2m_vals, doy, clim)
        anom = anom * w2d[np.newaxis, :, :]
        tw = get_time_weight(doy)
        anom = anom * tw[:, np.newaxis, np.newaxis]
        X = anom.reshape(anom.shape[0], -1)
        ipca.partial_fit(X)
        time_coords.append(time)
        del ds, da, t2m_vals, anom, doy, X
        gc.collect()

    eof_da, var_frac, ipca = eof_from_incremental_pca(ipca, n_modes, lat, lon, w2d)

    # Pass 2: compute PCs (year-by-year): PC = (X - mean_) @ components_.T
    pc_list = []
    for year in years:
        ds = open_era5_one_year(
            t2m_by_year[year], lat_south_of=lat_cutoff, lat_min=lat_min, chunks={"time": 31}
        )
        da = ds["t2m"] if "t2m" in ds else ds[list(ds.data_vars)[0]]
        if coarsen_factor > 1:
            da = da.coarsen(
                lat=coarsen_factor, lon=coarsen_factor, boundary="trim"
            ).mean()
        t2m_vals = da.load().values
        doy = xr.DataArray(da.time.values, dims=["time"]).dt.dayofyear.values
        ds.close()
        anom = anomaly_from_clim_doy(t2m_vals, doy, clim)
        anom = anom * w2d[np.newaxis, :, :]
        X = anom.reshape(anom.shape[0], -1)
        pc_list.append(ipca.transform(X))
        del ds, da, t2m_vals, anom, doy, X
        gc.collect()

    time_axis = np.concatenate(time_coords)
    pcs = xr.DataArray(
        np.concatenate(pc_list, axis=0),
        dims=("time", "mode"),
        coords={"time": time_axis, "mode": np.arange(1, n_modes + 1)},
    )

    ds_out = xr.Dataset(
        {"eof_pattern": eof_da, "pc": pcs, "variance_fraction": var_frac},
        attrs={
            "anomaly_baseline": "daily_climatology_from_file",
            "climatology_netcdf": str(clim_path),
            "area_weight": "sqrt_cos_lat",
        },
    )
    out_nc = out_dir / "EOFs.nc"
    ds_out.to_netcdf(out_nc)
    print("Wrote", out_nc)


if __name__ == "__main__":
    main()
