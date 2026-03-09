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


def build_full_domain_masks(
    lsm: xr.DataArray,
    pen_box: dict,
    ridge_lon: xr.DataArray,
) -> xr.Dataset:
    """
    Build all analysis domains on the full Antarctic grid (lat ≤ -55, all lon).

    Domains (all boolean → int8):
      peninsula_land   — land inside peninsula bounding box
      pen_west_slope   — peninsula land west of ridge
      pen_east_slope   — peninsula land east of ridge
      west_ocean       — ocean, lon 255-300, lat ≤ -60, trimmed east of AP
      east_ocean       — ocean, lon 300-345, lat ≤ -60, minus peninsula box
      inland           — land, lon 255-345, lat ≤ -75
    """
    lat = lsm.lat
    lon = lsm.lon
    land = lsm > 0.5
    ocean = ~land

    # ---- Peninsula (reindex onto full grid, fill False) ----
    pen_land_local = land.sel(
        lat=slice(pen_box["lat_min"], pen_box["lat_max"]),
        lon=slice(pen_box["lon_min"], pen_box["lon_max"]),
    )
    peninsula = pen_land_local.reindex_like(lsm, fill_value=False)

    elev_pen = lsm.sel(
        lat=slice(pen_box["lat_min"], pen_box["lat_max"]),
        lon=slice(pen_box["lon_min"], pen_box["lon_max"]),
    )
    lon_pen = elev_pen.lon
    lat_pen = elev_pen.lat

    ridge_2d = ridge_lon.reindex(lat=lat_pen, method="nearest").broadcast_like(elev_pen)
    lon_2d_pen = xr.broadcast(lon_pen, lat_pen)[0]

    pen_west_local = pen_land_local & (lon_2d_pen <= ridge_2d)
    pen_east_local = pen_land_local & (lon_2d_pen > ridge_2d)
    pen_west = pen_west_local.reindex_like(lsm, fill_value=False)
    pen_east = pen_east_local.reindex_like(lsm, fill_value=False)

    # ---- West Ocean: lon 255-300, lat ≤ -60, ocean ----
    in_west_box = (lon >= 255) & (lon <= 300) & (lat <= -60)
    west_ocean = ocean & in_west_box

    # Trim fringe east of AP: for each lat in peninsula range,
    # find the easternmost land point in the west-ocean lon range
    # and remove ocean east of it.
    for lat_val in lat.values:
        if lat_val > pen_box["lat_min"] and lat_val < pen_box["lat_max"]:
            row_land = land.sel(lat=lat_val, lon=slice(255, 300))
            if row_land.any():
                land_lons = row_land.lon.values[row_land.values]
                if len(land_lons) > 0:
                    east_coast = float(land_lons.max())
                    west_ocean.loc[dict(lat=lat_val)] = (
                        west_ocean.sel(lat=lat_val) & (lon <= east_coast)
                    )

    # ---- East Ocean: lon 300-345, lat ≤ -60, ocean ----
    # Includes ocean surrounding the peninsula east of 60°W (lon 300).
    in_east_box = (lon >= 300) & (lon <= 345) & (lat <= -60)
    east_ocean = ocean & in_east_box

    # ---- Inland: lon 255-345, lat ≤ -75, land ----
    in_inland_box = (lon >= 255) & (lon <= 345) & (lat <= -75)
    inland = land & in_inland_box

    masks = xr.Dataset(
        {
            "peninsula_land": peninsula.astype(np.int8),
            "pen_west_slope": pen_west.astype(np.int8),
            "pen_east_slope": pen_east.astype(np.int8),
            "west_ocean": west_ocean.astype(np.int8),
            "east_ocean": east_ocean.astype(np.int8),
            "inland": inland.astype(np.int8),
        },
        attrs={
            "description": "Antarctic Peninsula analysis domains on ERA5 grid",
            "peninsula_box": str(pen_box),
            "west_ocean_box": "lon 255-300, lat <= -60, ocean, trimmed E of AP",
            "east_ocean_box": "lon 300-345, lat <= -60, ocean (incl. around peninsula)",
            "inland_box": "lon 255-345, lat <= -75, land",
        },
    )

    for name in masks.data_vars:
        n = int(masks[name].sum())
        print(f"  {name:20s}  {n:6d} grid points")

    return masks


def plot_all_domains(
    masks: xr.Dataset,
    lsm: xr.DataArray,
    pen_box: dict,
    t2m_clim: xr.DataArray | None,
    out_path: Path,
) -> None:
    """
    Single-panel polar stereographic map covering all Antarctica.
    Colour-shaded domains, coastline, and optional t2m March climatology contours.
    """
    if not HAS_CARTOPY:
        print("Cartopy required — skipping all-domain plot.")
        return

    from matplotlib.colors import ListedColormap

    proj = ccrs.SouthPolarStereo()
    data_crs = ccrs.PlateCarree()
    # Reduce line-segment threshold so cartopy subdivides contour paths
    # into short arcs that project as smooth curves at high latitudes.
    data_crs._threshold = data_crs._threshold / 100

    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([-180, 180, -90, -55], crs=data_crs)

    lon_v = _lon360_to_180(masks.lon.values)
    lat_v = masks.lat.values

    # Plot each domain as a separate contourf to avoid pcolormesh
    # reprojection artifacts on polar stereo.
    domain_styles = [
        ("west_ocean",     "#6baed6"),
        ("east_ocean",     "#fd8d3c"),
        ("inland",         "#a1d99b"),
        ("pen_west_slope", "#3182bd"),
        ("pen_east_slope", "#e6550d"),
    ]
    for var, color in domain_styles:
        field = masks[var].values.astype(float)
        ax.contourf(
            lon_v, lat_v, field,
            levels=[0.5, 1.5], colors=[color], alpha=0.55,
            transform=data_crs,
        )

    # t2m climatology contours
    if t2m_clim is not None:
        t2m = t2m_clim.squeeze(drop=True)
        lon_t = _lon360_to_180(t2m.lon.values)
        lat_t = t2m.lat.values
        levels_t = np.arange(210, 275, 5)
        cs = ax.contour(
            lon_t, lat_t, t2m.values,
            levels=levels_t, colors="k", linewidths=0.5,
            transform=data_crs,
        )
        ax.clabel(cs, levels_t[::2], fontsize=6, fmt="%.0f")

    # ERA5 coastline
    lsm_v = lsm.sel(lat=masks.lat, lon=masks.lon).values
    ax.contour(
        lon_v, lat_v, lsm_v,
        levels=[0.5], colors="k", linewidths=0.8,
        transform=data_crs,
    )

    ax.gridlines(draw_labels=False, color="grey", alpha=0.3)

    # legend
    import matplotlib.patches as mpatches
    legend_items = [
        mpatches.Patch(color="#6baed6", alpha=0.6, label="West Ocean"),
        mpatches.Patch(color="#fd8d3c", alpha=0.6, label="East Ocean"),
        mpatches.Patch(color="#a1d99b", alpha=0.6, label="Inland"),
        mpatches.Patch(color="#3182bd", alpha=0.6, label="Peninsula West"),
        mpatches.Patch(color="#e6550d", alpha=0.6, label="Peninsula East"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=8, framealpha=0.8)

    title = "Analysis domains — all Antarctica"
    if t2m_clim is not None:
        title += "\nContours: March t2m climatology (K, 1988-2025)"
    ax.set_title(title, fontsize=11)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved all-domain figure → {out_path}")


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


def _load_etopo_antarctic(cache_path: Path, lat_north: float = -55.0) -> xr.DataArray:
    """
    Fetch ETOPO2022 surface elevation for Antarctica via NOAA THREDDS.
    Downloads at stride-5 (~5-arcmin) resolution to keep transfer fast,
    then caches locally as NetCDF.  Returns DataArray in metres.
    """
    if cache_path.exists():
        print(f"Loading cached ETOPO → {cache_path}")
        return xr.open_dataarray(cache_path)

    print("Downloading ETOPO2022 Antarctic subset via THREDDS (stride=5) …")
    url = (
        "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/"
        "60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"
    )
    ds = xr.open_dataset(url)
    sub = ds["z"].sel(lat=slice(-90, lat_north)).isel(
        lat=slice(None, None, 5), lon=slice(None, None, 5),
    )
    da = sub.load()
    da.name = "etopo_elev"
    da.attrs["units"] = "m"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(cache_path)
    print(f"Cached ETOPO → {cache_path}  shape={da.shape}")
    return da


def _regrid_to_target(src: xr.DataArray, tgt_lat, tgt_lon) -> xr.DataArray:
    """Bilinear interpolation of *src* onto target lat/lon grid."""
    return src.interp(lat=tgt_lat, lon=tgt_lon, method="linear")


def plot_era5_vs_reference(
    elev_antarctic: xr.DataArray,
    lsm_antarctic: xr.DataArray,
    cache_dir: Path,
    out_path: Path,
) -> None:
    """
    Single-panel all-Antarctica comparison:
      - bwr shading: ERA5 minus ETOPO2022 elevation difference
      - Thin contours: ETOPO2022 at 500 m intervals
      - Thick contours: ERA5 at 500 m intervals
      - Coastline from Natural Earth + ERA5 LSM
    """
    if not HAS_CARTOPY:
        print("Cartopy required for reference comparison — skipping.")
        return

    etopo = _load_etopo_antarctic(cache_dir / "etopo_antarctic.nc")

    era5_m = elev_antarctic / G

    # Align ETOPO lon (-180..180) → ERA5 lon (0..360) for difference
    etopo_360 = etopo.assign_coords(lon=(etopo.lon % 360)).sortby("lon")
    etopo_on_era5 = _regrid_to_target(etopo_360, era5_m.lat, era5_m.lon)
    diff = era5_m - etopo_on_era5

    proj = ccrs.SouthPolarStereo()
    data_crs = ccrs.PlateCarree()
    data_crs._threshold = data_crs._threshold / 100

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([-180, 180, -90, -55], crs=data_crs)

    lon_e = _lon360_to_180(era5_m.lon.values)
    lat_e = era5_m.lat.values
    lsm_v = lsm_antarctic.values
    diff_land = np.where(lsm_v > 0.5, diff.values, np.nan)

    vmax = float(np.nanpercentile(np.abs(diff_land), 98))
    vmax = max(vmax, 50)
    p = ax.pcolormesh(
        lon_e, lat_e, diff_land,
        transform=data_crs, cmap="bwr",
        vmin=-vmax, vmax=vmax, alpha=0.7,
    )

    # --- ETOPO contours (thin) ---
    levels = np.arange(500, 4500, 500)
    ax.contour(
        etopo.lon.values, etopo.lat.values, etopo.values,
        levels=levels, colors="0.35", linewidths=0.6,
        transform=data_crs,
    )

    # --- ERA5 contours (thick, land only) ---
    era5_land = np.where(lsm_v > 0.5, era5_m.values, np.nan)
    ax.contour(
        lon_e, lat_e, era5_land,
        levels=levels, colors="k", linewidths=1.4,
        transform=data_crs,
    )

    # --- coastline ---
    ax.contour(
        lon_e, lat_e, lsm_v,
        levels=[0.5], colors="k", linewidths=0.8, linestyles="--",
        transform=data_crs,
    )
    try:
        ax.coastlines(resolution="50m", linewidth=0.5, color="k")
    except Exception:
        pass

    ax.gridlines(draw_labels=False, color="grey", alpha=0.3)
    cb = plt.colorbar(p, ax=ax, shrink=0.6, pad=0.05)
    cb.set_label("ERA5 − ETOPO2022 elevation difference (m)")

    ax.set_title(
        "Antarctic elevation: ERA5 (thick contour) vs ETOPO2022 (thin contour)\n"
        "Contours every 500 m  ·  Shading = difference (bwr)",
        fontsize=11,
    )

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison figure → {out_path}")


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

    # --- full Antarctic cap for ETOPO comparison ---
    antarctic_box = {"lat_min": -90.0, "lat_max": -55.0, "lon_min": 0.0, "lon_max": 359.75}
    lsm_ant = crop_to_box(lsm, antarctic_box)
    elev_ant = crop_to_box(elev, antarctic_box)
    plot_era5_vs_reference(elev_ant, lsm_ant, out_dir, fig_dir / "era5_vs_reference.png")

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

    # --- peninsula zoom figure ---
    plot_domains(
        elev_wrap, lsm_wrap, elev_pen, lsm_pen,
        ridge_lon, west_mask, east_mask, pen_box,
        fig_dir / "peninsula_domains.png",
    )

    # --- full-grid domain masks (ocean, inland, peninsula) ---
    antarctic_box = {"lat_min": -90.0, "lat_max": -55.0, "lon_min": 0.0, "lon_max": 359.75}
    lsm_full = crop_to_box(lsm, antarctic_box)

    print("\nBuilding full-grid domain masks …")
    all_masks = build_full_domain_masks(lsm_full, pen_box, ridge_lon)

    # --- load March climatology for overlay ---
    clim_mar_path = root / "Data" / "F01_climatology" / "1988_2025" / "03" / "clim.nc"
    t2m_clim = None
    if clim_mar_path.exists():
        ds_clim = xr.open_dataset(clim_mar_path)
        for v in ["t2m", "T2m", "2t"]:
            if v in ds_clim:
                t2m_clim = ds_clim[v]
                break
        if t2m_clim is None and ds_clim.data_vars:
            t2m_clim = ds_clim[list(ds_clim.data_vars)[0]]
        if t2m_clim is not None:
            if "latitude" in t2m_clim.coords:
                t2m_clim = t2m_clim.rename({"latitude": "lat"})
            if "longitude" in t2m_clim.coords:
                t2m_clim = t2m_clim.rename({"longitude": "lon"})
            print(f"Loaded March t2m climatology from {clim_mar_path}")
    else:
        print(f"March climatology not found at {clim_mar_path} — skipping overlay.")

    plot_all_domains(all_masks, lsm, pen_box, t2m_clim, fig_dir / "all_domains.png")

    # --- save masks ---
    all_masks["elevation_m"] = crop_to_box(elev, antarctic_box) / G
    all_masks["ridge_lon"] = ridge_lon
    mask_path = out_dir / "all_domain_masks.nc"
    all_masks.to_netcdf(mask_path)
    print(f"Saved all masks → {mask_path}")


if __name__ == "__main__":
    main()
