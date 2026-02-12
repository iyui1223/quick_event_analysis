#!/usr/bin/env python3
"""
Map 2: Histogram of mean windspeed over Antarctic Peninsula (land only), February.
- Reads CDS download: YYYY02.nc (zip with u10, v10, t2m .nc inside).
- Masks sea using ERA5 land-sea mask; for each day takes mean windspeed over land.
- Past: 1948–1987, Present: 1988–current. Plots two histograms.
- Optional: decade gradation in bin colour (1988–97 pink → 2018–present strong red).
"""
import os
import zipfile
import tempfile
import glob
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Paths (absolute Lustre for Slurm; override via env if needed)
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    "/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/antarctica_peninsula_2020",
)
DATA_DIR = os.environ.get(
    "WIND_DATA_DIR",
    os.path.join(PROJECT_ROOT, "dataslices", "era5_daily_peninsula"),
)
LSM_PATH = "/lustre/soge1/data/analysis/era5/0.28125x0.28125/invariant/land-sea_mask/nc/era5_invariant_land-sea_mask_20000101.nc"
FIGS_DIR = os.environ.get("FIGS_DIR", os.path.join(PROJECT_ROOT, "Figs"))
# Peninsula: 90°W–40°W, 75°S–65°S
LON_SLICE = slice(-90, -40)   # or 270, 320 in 0..360
LAT_SLICE = slice(-75, -65)
LAND_THRESHOLD = 0.5

PAST_RANGE = (1948, 1987)
PRESENT_RANGE = (1988, 2026)


def _unzip_month(zip_path, out_dir):
    """Unzip one YYYY02.nc into out_dir; return out_dir."""
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    return out_dir


def _find_u_v_nc(dir_path):
    """Return (path_to_u_nc, path_to_v_nc) from extracted files."""
    files = [f for f in os.listdir(dir_path) if f.endswith(".nc")]
    u_path = v_path = None
    for f in files:
        if "10m_u" in f and "wind" in f:
            u_path = os.path.join(dir_path, f)
        elif "10m_v" in f and "wind" in f:
            v_path = os.path.join(dir_path, f)
    return u_path, v_path


def load_land_mask():
    """Load ERA5 land-sea mask and subset to peninsula. Return 2D (lat, lon) land mask (True = land)."""
    ds = xr.open_dataset(LSM_PATH)
    # lsm may have scale_factor/add_offset
    lsm = ds["lsm"].isel(time=0)
    if hasattr(lsm, "values") and lsm.dtype == np.int16:
        # typical ERA5 packed: scale_factor, add_offset
        sf = getattr(ds["lsm"], "scale_factor", 1.0)
        ao = getattr(ds["lsm"], "add_offset", 0.0)
        lsm = lsm.astype(float) * sf + ao
    # Subset to peninsula. Global mask: lat 90 -> -90 (descending), lon 0..360
    if "longitude" in lsm.dims:
        lon = lsm.coords["longitude"]
        if float(lon.min()) >= 0:
            lsm = lsm.sel(latitude=slice(-65, -75), longitude=slice(270, 320))
        else:
            lsm = lsm.sel(latitude=slice(-65, -75), longitude=LON_SLICE)
    else:
        lsm = lsm.sel(lat=slice(-65, -75), lon=LON_SLICE)
    land = (lsm.values >= LAND_THRESHOLD).squeeze()
    ds.close()
    return land


def load_wind_speed_for_month(zip_path, land_mask):
    """
    Unzip zip_path, load u10 and v10, compute daily mean windspeed over land.
    Return 1D array of length n_days (mean windspeed per day).
    """
    with tempfile.TemporaryDirectory() as tmp:
        _unzip_month(zip_path, tmp)
        u_path, v_path = _find_u_v_nc(tmp)
        if u_path is None or v_path is None:
            return None
        du = xr.open_dataset(u_path)
        dv = xr.open_dataset(v_path)
        # Variable names: u10, v10; dims valid_time, latitude, longitude
        u = du["u10"] if "u10" in du else du[list(du.data_vars)[0]]
        v = dv["v10"] if "v10" in dv else dv[list(dv.data_vars)[0]]
        speed = np.sqrt(u**2 + v**2)
        # land_mask: (41, 201) lat -65..-75, same order as CDS
        land = land_mask
        # speed: (valid_time, latitude, longitude) -> for each time, mask by land and take mean
        land_broadcast = np.broadcast_to(land, speed.values.shape)
        masked = np.where(land_broadcast, speed.values, np.nan)
        daily_means = np.nanmean(masked, axis=(1, 2))
        du.close()
        dv.close()
        return daily_means


def collect_all_daily_means(data_dir, land_mask):
    """Scan data_dir for YYYY02.nc (zip), load each, return dict year -> array of daily means."""
    pattern = os.path.join(data_dir, "*02.nc")
    results = {}
    for path in sorted(glob.glob(pattern)):
        m = re.match(r".*(\d{4})02\.nc$", path)
        if not m:
            continue
        year = int(m.group(1))
        # Skip if it's a zip; else could be already-extracted dir later
        if not zipfile.is_zipfile(path):
            continue
        arr = load_wind_speed_for_month(path, land_mask)
        if arr is not None:
            results[year] = arr
    return results


def main():
    os.makedirs(FIGS_DIR, exist_ok=True)
    land_mask = load_land_mask()
    all_means = collect_all_daily_means(DATA_DIR, land_mask)
    if not all_means:
        print("No *02.nc zip files found in", DATA_DIR)
        return

    # Group by decade, tracking year for each value
    past_by_decade = {
        "1948-1957": [],
        "1958-1967": [],
        "1968-1977": [],
        "1978-1987": [],
    }
    present_by_decade = {
        "1988-1997": [],
        "1998-2007": [],
        "2008-2017": [],
        "2018-2026": [],
    }
    
    for year, arr in all_means.items():
        if PAST_RANGE[0] <= year <= PAST_RANGE[1]:
            if 1948 <= year <= 1957:
                past_by_decade["1948-1957"].extend(arr.tolist())
            elif 1958 <= year <= 1967:
                past_by_decade["1958-1967"].extend(arr.tolist())
            elif 1968 <= year <= 1977:
                past_by_decade["1968-1977"].extend(arr.tolist())
            elif 1978 <= year <= 1987:
                past_by_decade["1978-1987"].extend(arr.tolist())
        elif PRESENT_RANGE[0] <= year <= PRESENT_RANGE[1]:
            if 1988 <= year <= 1997:
                present_by_decade["1988-1997"].extend(arr.tolist())
            elif 1998 <= year <= 2007:
                present_by_decade["1998-2007"].extend(arr.tolist())
            elif 2008 <= year <= 2017:
                present_by_decade["2008-2017"].extend(arr.tolist())
            elif 2018 <= year <= 2026:
                present_by_decade["2018-2026"].extend(arr.tolist())
        elif year < PAST_RANGE[0]:
            # e.g. 1947
            if year <= 1957:
                past_by_decade["1948-1957"].extend(arr.tolist())

    # Collect all values for bin range
    all_vals = []
    for d in past_by_decade.values():
        all_vals.extend(d)
    for d in present_by_decade.values():
        all_vals.extend(d)
    all_vals = np.array(all_vals)
    if all_vals.size == 0:
        print("No data in past or present range.")
        return
    bins = np.linspace(0, max(30, np.nanmax(all_vals) * 1.05), 25)

    # Colors: past (blue gradation), present (pink→red gradation)
    past_colors = ["#87CEEB", "#4682B4", "#1E90FF", "#000080"]  # light to dark blue
    present_colors = ["#FFB6C1", "#FF69B4", "#FF6B6B", "#DC143C"]  # pink → strong red
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # Past: stacked by decade
    past_data = [past_by_decade[k] for k in sorted(past_by_decade.keys())]
    past_labels = sorted(past_by_decade.keys())
    ax1.hist(
        past_data,
        bins=bins,
        stacked=True,
        color=past_colors,
        alpha=0.8,
        edgecolor="k",
        linewidth=0.3,
        label=past_labels,
    )
    ax1.set_title(f"Past ({PAST_RANGE[0]}-{PAST_RANGE[1]})")
    ax1.set_xlabel("Mean windspeed over peninsula land [m/s]")
    ax1.set_ylabel("Count")
    ax1.legend(loc="upper right", fontsize=8)
    
    # Present: stacked by decade
    present_data = [present_by_decade[k] for k in sorted(present_by_decade.keys())]
    present_labels = sorted(present_by_decade.keys())
    ax2.hist(
        present_data,
        bins=bins,
        stacked=True,
        color=present_colors,
        alpha=0.8,
        edgecolor="k",
        linewidth=0.3,
        label=present_labels,
    )
    ax2.set_title(f"Present ({PRESENT_RANGE[0]}-{PRESENT_RANGE[1]})")
    ax2.set_xlabel("Mean windspeed over peninsula land [m/s]")
    ax2.legend(loc="upper right", fontsize=8)
    
    fig.suptitle("February: ERA5 10m wind (land-masked peninsula) — by decade")
    plt.tight_layout()
    out = os.path.join(FIGS_DIR, "peninsula_wind_histogram_past_present.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved", out)


if __name__ == "__main__":
    main()
