"""
Compute lat-weighted area-mean time series of t2m and msl per domain.

Data source: /lustre/soge1/projects/andante/cenv1201/heavy/ERA5/daily/Surf/YYYYMM.nc
Files are zip archives from CDS containing one NetCDF per variable.
Domain masks: Data/F04_peninsula_domains/all_domain_masks.nc

Weight = cos(lat) so each grid cell contributes by physical area.
Output: Data/F05_domain_timeseries/domain_timeseries.nc
"""

import argparse
import zipfile
from pathlib import Path
import tempfile

import numpy as np
import xarray as xr

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

SLICES_DIR = Path("/lustre/soge1/projects/andante/cenv1201/heavy/ERA5/daily/Surf")
MASKS_PATH = Path(__file__).resolve().parent.parent / "Data" / "F04_peninsula_domains" / "all_domain_masks.nc"

# Slice region: 60S-90S, lon 255-345 (105W-15W)
LAT_MIN = -90.0
LAT_MAX = -60.0
LON_MIN = 255.0
LON_MAX = 345.0

DOMAIN_NAMES = ["west_ocean", "east_ocean", "inland", "pen_west_slope", "pen_east_slope"]
VARIABLES = ["t2m", "msl"]


def _std_coords(ds: xr.Dataset) -> xr.Dataset:
    """Standardize coord names to lat, lon, time."""
    renames = {}
    for c in list(ds.coords):
        if "lat" in c.lower() and c != "lat":
            renames[c] = "lat"
        elif "lon" in c.lower() and c != "lon":
            renames[c] = "lon"
        elif ("time" in c.lower() or "valid" in c.lower()) and c != "time":
            renames[c] = "time"
    if renames:
        ds = ds.rename(renames)
    return ds


def _ensure_vars(ds: xr.Dataset) -> xr.Dataset | None:
    """Extract t2m and msl under standard names."""
    vmap = {"2m_temperature": "t2m", "t2m": "t2m", "mean_sea_level_pressure": "msl", "msl": "msl"}
    out = {}
    for v in ds.data_vars:
        vlow = str(v).lower()
        if "2m_temperature" in vlow or v == "t2m":
            out["t2m"] = ds[v]
        elif "mean_sea_level" in vlow or v == "msl":
            out["msl"] = ds[v]
    if "t2m" not in out or "msl" not in out:
        return None
    return xr.Dataset(out)


def _load_month_slice(path: Path) -> xr.Dataset | None:
    """Load t2m and msl from a monthly slice (zip or single netcdf)."""
    path = Path(path)
    if not path.exists():
        return None

    if zipfile.is_zipfile(path):
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(tmp)
            t2m_ds = msl_ds = None
            for nc in Path(tmp).rglob("*.nc"):
                name = nc.name.lower()
                if "2m_temperature" in name:
                    t2m_ds = xr.open_dataset(str(nc))
                elif "mean_sea_level" in name:
                    msl_ds = xr.open_dataset(str(nc))
            if t2m_ds is None or msl_ds is None:
                return None
            t2m_ds = _std_coords(t2m_ds)
            msl_ds = _std_coords(msl_ds)
            t2m_var = list(t2m_ds.data_vars)[0]
            msl_var = list(msl_ds.data_vars)[0]
            ds = xr.Dataset({
                "t2m": t2m_ds[t2m_var].rename("t2m"),
                "msl": msl_ds[msl_var].rename("msl"),
            })
            t2m_ds.close()
            msl_ds.close()
            return ds
    else:
        ds = xr.open_dataset(path)
        ds = _std_coords(ds)
        ds = _ensure_vars(ds)
        return ds


def _standardize(ds: xr.Dataset) -> xr.Dataset:
    """Ensure lat ascending, lon 0-360."""
    if "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    if ds.lat[0] > ds.lat[-1]:
        ds = ds.sortby("lat")
    if float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon % 360)).sortby("lon")
    return ds


def weighted_domain_mean(da: xr.DataArray, mask: xr.DataArray, weight: xr.DataArray) -> xr.DataArray:
    """Area-weighted mean over masked grid points. weight = cos(lat)."""
    masked = da.where(mask > 0)
    wmasked = weight.where(mask > 0)
    numer = (masked * wmasked).sum(dim=["lat", "lon"])
    denom = wmasked.sum(dim=["lat", "lon"])
    return numer / denom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slices-dir", default=str(SLICES_DIR))
    parser.add_argument("--masks", default=str(MASKS_PATH))
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    slices_dir = Path(args.slices_dir)
    masks_path = Path(args.masks)
    out_dir = Path(args.out_dir) if args.out_dir else root / "Data" / "F05_domain_timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover available months
    files = sorted(slices_dir.glob("*.nc"))
    if not files:
        print(f"No slice files in {slices_dir}")
        return

    print(f"Slices dir: {slices_dir}  ({len(files)} files)")
    print(f"Masks: {masks_path}")

    # Load masks and crop to slice region (60S-90S, lon 255-345)
    # Masks lat is ascending (-55..-90); use slice(low, high) = slice(-90, -60)
    masks_ds = xr.open_dataset(masks_path)
    masks_crop = masks_ds.sel(
        lat=slice(LAT_MIN, LAT_MAX),  # -90 to -60
        lon=slice(LON_MIN, LON_MAX),
    )
    lat = masks_crop.lat
    lon = masks_crop.lon

    # Area weight: cos(lat) in radians
    lat_rad = np.deg2rad(lat)
    weight = np.cos(lat_rad)
    weight = xr.DataArray(weight, coords={"lat": lat}, dims=["lat"])
    weight = weight.broadcast_like(masks_crop["west_ocean"])

    # Accumulators
    series = {v: {d: [] for d in DOMAIN_NAMES} for v in VARIABLES}
    all_times = []

    for i, fp in enumerate(files):
        yymm = fp.stem
        if len(yymm) != 6 or not yymm.isdigit():
            continue
        print(f"  [{i+1}/{len(files)}] {yymm}", end=" ", flush=True)
        ds = None
        try:
            ds = _load_month_slice(fp)
            if ds is None:
                print("(no t2m/msl)")
                continue
            ds = _standardize(ds)
            # Crop: 60S-90S, lon 255-345 (after _standardize, lat is ascending -90..90)
            ds = ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
            if ds.sizes["lat"] == 0 or ds.sizes["lon"] == 0:
                print("(empty crop)")
                continue
            # Align to mask grid
            ds = ds.interp(lat=lat, lon=lon, method="nearest")
            times = ds.time.values
            all_times.extend(list(times))
            for var in VARIABLES:
                if var not in ds:
                    continue
                da = ds[var]
                for domain in DOMAIN_NAMES:
                    mask = masks_crop[domain]
                    m = weighted_domain_mean(da, mask, weight)
                    series[var][domain].append(m)
            print("ok")
        except Exception as e:
            print(f"err: {e}")
        finally:
            if ds is not None:
                ds.close()

    if not all_times:
        print("No data processed.")
        return

    # Build output dataset (all series have same length = total days processed)
    ntimes = len(all_times)
    times = np.array(all_times, dtype="datetime64[ns]")
    out = {}
    for var in VARIABLES:
        for domain in DOMAIN_NAMES:
            if series[var][domain]:
                arr = np.concatenate([np.atleast_1d(a.values) for a in series[var][domain]])
                arr = arr[:ntimes]
                out[f"{var}_{domain}"] = (["time"], arr)
    out_ds = xr.Dataset(
        out,
        coords={"time": times},
        attrs={
            "description": "Lat-weighted area-mean t2m and msl per domain",
            "source": str(slices_dir),
            "domain_definitions": "Data/F04_peninsula_domains/all_domain_masks.nc",
        },
    )
    out_path = out_dir / "domain_timeseries.nc"
    out_ds.to_netcdf(out_path)
    print(f"\nSaved → {out_path}")
    print(f"Time range: {out_ds.time.values[0]} to {out_ds.time.values[-1]}")
    print(f"Variables: {list(out_ds.data_vars)}")


if __name__ == "__main__":
    main()
