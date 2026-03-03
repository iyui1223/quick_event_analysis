"""
Define Antarctic Peninsula sub-domains (west / east slope) from ERA5 invariants.

Loads ERA5 land-sea mask and surface geopotential, identifies land grid points
within the peninsula bounding box, finds the ridge line (highest elevation per
latitude band), and classifies west vs east slope.

Produces:
  - Polar stereographic figure of topography + peninsula box + ridge line
  - NetCDF masks (peninsula, west_slope, east_slope) on the ERA5 grid
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

G = 9.80665  # m s-2


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_params(yaml_path: Path) -> dict:
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def load_invariant(nc_path: str, var: str) -> xr.DataArray:
    """Load an ERA5 invariant field and standardise coords to lat/lon."""
    ds = xr.open_dataset(nc_path)
    renames = {}
    if "latitude" in ds.coords:
        renames["latitude"] = "lat"
    if "longitude" in ds.coords:
        renames["longitude"] = "lon"
    if renames:
        ds = ds.rename(renames)
    da = ds[var].squeeze(drop=True)
    if da.lat[0] > da.lat[-1]:
        da = da.sortby("lat")
    return da


# ---------------------------------------------------------------------------
# Domain logic
# ---------------------------------------------------------------------------

def crop_to_box(da: xr.DataArray, box: dict) -> xr.DataArray:
    return da.sel(
        lat=slice(box["lat_min"], box["lat_max"]),
        lon=slice(box["lon_min"], box["lon_max"]),
    )


def find_ridge_lon(elev_land: xr.DataArray) -> xr.DataArray:
    """
    For each latitude row, find the longitude of the highest land point.
    Returns a DataArray indexed by lat.  NaN where no land exists.
    """
    ridge_lon = elev_land.idxmax(dim="lon")
    return ridge_lon


def classify_slopes(
    lsm_pen: xr.DataArray,
    elev_pen: xr.DataArray,
    ridge_lon: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Classify peninsula land points as west or east of the ridge.
    Returns (west_mask, east_mask) — boolean DataArrays on the peninsula grid.
    """
    land = lsm_pen > 0.5
    lon_2d = xr.broadcast(elev_pen.lon, elev_pen.lat)[0]
    ridge_2d = ridge_lon.broadcast_like(elev_pen)

    west = land & (lon_2d <= ridge_2d)
    east = land & (lon_2d > ridge_2d)
    return west, east


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _lon360_to_180(lon):
    return np.where(lon > 180, lon - 360, lon)


def _geodetic_box(box: dict, n_per_edge: int = 80):
    """
    Return (lons, lats) arrays tracing a bounding box along geodetic arcs.
    Each edge is sampled with *n_per_edge* points so that lines of constant
    latitude / longitude curve correctly on non-cylindrical projections.
    Longitudes are returned in [-180, 180].
    """
    lo0, lo1 = box["lon_min"], box["lon_max"]
    la0, la1 = box["lat_min"], box["lat_max"]
    bottom = (np.linspace(lo0, lo1, n_per_edge), np.full(n_per_edge, la0))
    right  = (np.full(n_per_edge, lo1), np.linspace(la0, la1, n_per_edge))
    top    = (np.linspace(lo1, lo0, n_per_edge), np.full(n_per_edge, la1))
    left   = (np.full(n_per_edge, lo0), np.linspace(la1, la0, n_per_edge))
    lons = np.concatenate([bottom[0], right[0], top[0], left[0]])
    lats = np.concatenate([bottom[1], right[1], top[1], left[1]])
    return _lon360_to_180(lons), lats


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_domains(
    elev_wrapper: xr.DataArray,
    lsm_wrapper: xr.DataArray,
    elev_pen: xr.DataArray,
    lsm_pen: xr.DataArray,
    ridge_lon: xr.DataArray,
    west_mask: xr.DataArray,
    east_mask: xr.DataArray,
    pen_box: dict,
    out_path: Path,
) -> None:
    """
    Two-panel figure:
      Left  — wrapper region topography with peninsula box overlay
      Right — peninsula zoom with west/east colouring and ridge line
    """
    fig = plt.figure(figsize=(16, 7))

    if HAS_CARTOPY:
        proj = ccrs.SouthPolarStereo(central_longitude=-90)
        data_crs = ccrs.PlateCarree()

        # --- panel 1: wrapper topography (land-masked) ---
        ax1 = fig.add_subplot(1, 2, 1, projection=proj)
        ax1.set_extent([-135, -35, -85, -55], crs=data_crs)

        lon_w = _lon360_to_180(elev_wrapper.lon.values)
        lat_w = elev_wrapper.lat.values
        elev_m = elev_wrapper.values / G
        lsm_v = lsm_wrapper.values

        elev_masked = np.where(lsm_v > 0.5, elev_m, np.nan)
        p1 = ax1.pcolormesh(
            lon_w, lat_w, elev_masked,
            transform=data_crs, cmap="terrain",
            vmin=0, vmax=3500,
        )
        ax1.contour(
            lon_w, lat_w, lsm_v,
            levels=[0.5], colors="k", linewidths=0.8,
            transform=data_crs,
        )

        box_lons, box_lats = _geodetic_box(pen_box)
        ax1.plot(box_lons, box_lats, "r-", linewidth=2, transform=data_crs)

        ax1.gridlines(draw_labels=False, color="grey", alpha=0.4)
        plt.colorbar(p1, ax=ax1, shrink=0.6, label="Elevation (m)")
        ax1.set_title("ERA5 topography — wrapper region")

        # --- panel 2: peninsula zoom ---
        ax2 = fig.add_subplot(1, 2, 2, projection=proj)
        pen_extent = [
            float(_lon360_to_180(np.array(pen_box["lon_min"]))),
            float(_lon360_to_180(np.array(pen_box["lon_max"]))),
            pen_box["lat_min"], pen_box["lat_max"],
        ]
        ax2.set_extent(pen_extent, crs=data_crs)

        lon_p = _lon360_to_180(elev_pen.lon.values)
        lat_p = elev_pen.lat.values
        elev_pen_m = elev_pen.values / G
        lsm_pen_v = lsm_pen.values

        elev_pen_land = np.where(lsm_pen_v > 0.5, elev_pen_m, np.nan)

        west_v = west_mask.values.astype(float)
        east_v = east_mask.values.astype(float)
        slope_field = np.full_like(elev_pen_m, np.nan)
        slope_field[west_v > 0.5] = 0
        slope_field[east_v > 0.5] = 1

        from matplotlib.colors import ListedColormap
        cmap_slope = ListedColormap(["#3182bd", "#e6550d"])
        ax2.pcolormesh(
            lon_p, lat_p, slope_field,
            transform=data_crs, cmap=cmap_slope, vmin=-0.5, vmax=1.5, alpha=0.6,
        )

        ax2.contourf(
            lon_p, lat_p, elev_pen_land,
            levels=np.arange(0, 3200, 200),
            transform=data_crs, cmap="terrain", alpha=0.5,
        )
        ax2.contour(
            lon_p, lat_p, elev_pen_land,
            levels=np.arange(0, 3200, 400),
            transform=data_crs, colors="k", linewidths=0.4,
        )

        ax2.contour(
            lon_p, lat_p, lsm_pen_v,
            levels=[0.5], colors="k", linewidths=1.0,
            transform=data_crs,
        )

        valid = ridge_lon.dropna("lat")
        ridge_lons_180 = _lon360_to_180(valid.values)
        ax2.plot(
            ridge_lons_180, valid.lat.values,
            "k-o", markersize=3, linewidth=1.5,
            transform=data_crs, label="Ridge line",
        )

        ax2.gridlines(draw_labels=False, color="grey", alpha=0.4)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.set_title("Peninsula domains (blue=west, orange=east)")

    else:
        ax1 = fig.add_subplot(1, 2, 1)
        (elev_wrapper / G).plot(ax=ax1, cmap="terrain", vmin=0, vmax=3500)
        ax1.set_title("ERA5 topography — wrapper region")

        ax2 = fig.add_subplot(1, 2, 2)
        slope_field = np.full(elev_pen.shape, np.nan)
        slope_field[west_mask.values > 0.5] = 0
        slope_field[east_mask.values > 0.5] = 1
        ax2.pcolormesh(elev_pen.lon, elev_pen.lat, slope_field, cmap="coolwarm")
        valid = ridge_lon.dropna("lat")
        ax2.plot(valid.values, valid.lat.values, "k-o", markersize=3)
        ax2.set_title("Peninsula west/east")

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved domain figure → {out_path}")


def plot_elevation_only(
    elev_wrapper: xr.DataArray,
    lsm_wrapper: xr.DataArray,
    pen_box: dict,
    out_path: Path,
) -> None:
    """Single-panel wide-area topography map with coastline and peninsula box."""
    fig = plt.figure(figsize=(10, 9))

    if HAS_CARTOPY:
        proj = ccrs.SouthPolarStereo(central_longitude=-90)
        data_crs = ccrs.PlateCarree()
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent([-135, -35, -85, -55], crs=data_crs)

        lon_w = _lon360_to_180(elev_wrapper.lon.values)
        lat_w = elev_wrapper.lat.values
        elev_m = elev_wrapper.values / G
        lsm_v = lsm_wrapper.values

        elev_masked = np.where(lsm_v > 0.5, elev_m, np.nan)
        p = ax.pcolormesh(
            lon_w, lat_w, elev_masked,
            transform=data_crs, cmap="terrain", vmin=0, vmax=3500,
        )
        ax.contour(
            lon_w, lat_w, lsm_v,
            levels=[0.5], colors="k", linewidths=0.8,
            transform=data_crs,
        )

        box_lons, box_lats = _geodetic_box(pen_box)
        ax.plot(box_lons, box_lats, "r-", linewidth=2.5, transform=data_crs)

        ax.gridlines(draw_labels=False, color="grey", alpha=0.4)
        plt.colorbar(p, ax=ax, shrink=0.6, label="Elevation (m, geopotential / g)")
        ax.set_title("ERA5 surface elevation — Antarctic region")
    else:
        ax = fig.add_subplot(1, 1, 1)
        (elev_wrapper / G).plot(ax=ax, cmap="terrain")
        ax.set_title("ERA5 elevation")

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved elevation figure → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Define Antarctic Peninsula domains")
    parser.add_argument("--params", default=None, help="Path to params.yaml")
    parser.add_argument("--out-dir", default=None, help="Output directory for masks and figures")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    params_path = Path(args.params) if args.params else root / "Const" / "params.yaml"
    params = load_params(params_path)

    out_dir = Path(args.out_dir) if args.out_dir else root / "Data" / "F04_peninsula_domains"
    fig_dir = root / "Figs" / "F04_peninsula_domains"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    wrapper = params["wrapper"]
    pen_box = params["peninsula"]
    inv = params["era5_invariant"]

    print("Loading ERA5 invariant fields …")
    lsm = load_invariant(inv["lsm"], "lsm")
    elev = load_invariant(inv["geopotential"], "z")

    # --- crop to wrapper region ---
    lsm_wrap = crop_to_box(lsm, wrapper)
    elev_wrap = crop_to_box(elev, wrapper)

    # --- elevation-only map ---
    plot_elevation_only(elev_wrap, lsm_wrap, pen_box, fig_dir / "elevation_wrapper.png")

    # --- crop to peninsula ---
    lsm_pen = crop_to_box(lsm, pen_box)
    elev_pen = crop_to_box(elev, pen_box)

    land_pen = lsm_pen > 0.5
    elev_land = elev_pen.where(land_pen)

    print(f"Peninsula grid: {lsm_pen.shape}  land points: {int(land_pen.sum())}")

    # --- ridge line ---
    ridge_lon = find_ridge_lon(elev_land)
    print("Ridge line (lon at each lat):")
    for lat_val in ridge_lon.lat.values:
        rlon = float(ridge_lon.sel(lat=lat_val))
        if not np.isnan(rlon):
            print(f"  lat={lat_val:7.2f}  →  ridge lon={rlon:.2f}  ({rlon - 360:.1f}°E)")

    # --- classify west / east ---
    west_mask, east_mask = classify_slopes(lsm_pen, elev_pen, ridge_lon)
    n_west = int(west_mask.sum())
    n_east = int(east_mask.sum())
    print(f"West slope: {n_west} points   East slope: {n_east} points")

    # --- domain figure ---
    plot_domains(
        elev_wrap, lsm_wrap, elev_pen, lsm_pen,
        ridge_lon, west_mask, east_mask, pen_box,
        fig_dir / "peninsula_domains.png",
    )

    # --- save masks as NetCDF ---
    mask_ds = xr.Dataset(
        {
            "peninsula_land": land_pen.astype(np.int8),
            "west_slope": west_mask.astype(np.int8),
            "east_slope": east_mask.astype(np.int8),
            "elevation_m": (elev_pen / G),
            "ridge_lon": ridge_lon,
        },
        attrs={
            "description": "Antarctic Peninsula domain masks derived from ERA5 invariants",
            "peninsula_box": str(pen_box),
        },
    )
    mask_path = out_dir / "peninsula_masks.nc"
    mask_ds.to_netcdf(mask_path)
    print(f"Saved masks → {mask_path}")


if __name__ == "__main__":
    main()
