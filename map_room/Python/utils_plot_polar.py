"""
Polar stereographic plotting for Antarctic domain (south of 55S).
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Optional cartopy for polar projection
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


def _polar_projection():
    if HAS_CARTOPY:
        return ccrs.SouthPolarStereo(central_longitude=0)
    return None


def plot_polar_map(
    data: xr.DataArray,
    out_path: Path,
    title: str = "",
    cmap: str = "RdBu_r",
    contours: Optional[xr.DataArray] = None,
    contour_levels: Optional[list] = None,
    colorbar_label: Optional[str] = None,
    figsize: tuple = (8, 8),
) -> None:
    """
    Plot 2D field (lat, lon) on polar stereographic projection (Antarctic).
    Optionally overlay contours from another array (e.g. msl).
    """
    fig = plt.figure(figsize=figsize)
    if HAS_CARTOPY:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
        ax.set_extent([-180, 180, -90, -55], crs=ccrs.PlateCarree())
        # Skip LAND/OCEAN features: they require Cartopy to create shapefile dirs under
        # ~/.local/share/cartopy, which often fails on HPC (read-only or permission denied).
        # ax.add_feature(cfeature.LAND, ...)  # disabled for batch
        ax.gridlines(draw_labels=False)
        # Plot data in PlateCarree and let projection transform
        lon = data.lon.values
        lat = data.lat.values
        # Handle lon 0-360 vs -180-180
        if lon.min() >= 0 and lon.max() > 180:
            data_plot = data.assign_coords(lon=(data.lon + 180) % 360 - 180)
        else:
            data_plot = data
        p = ax.pcolormesh(
            data_plot.lon,
            data_plot.lat,
            data_plot.values,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
        )
        if contours is not None:
            if contour_levels is None:
                contour_levels = 12
            ax.contour(
                contours.lon,
                contours.lat,
                contours.values,
                levels=contour_levels,
                transform=ccrs.PlateCarree(),
                colors="k",
                linewidths=0.5,
            )
        plt.colorbar(p, ax=ax, shrink=0.7, label=colorbar_label)
    else:
        ax = fig.add_subplot(1, 1, 1)
        data.plot(ax=ax, cmap=cmap, cbar_kwargs={"label": colorbar_label})
        if contours is not None:
            if contour_levels is None:
                contour_levels = 12
            contours.plot.contour(ax=ax, levels=contour_levels, colors="k", linewidths=0.5)
    ax.set_title(title or data.name or "Map")
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_eof_mode_map(
    eof_pattern: xr.DataArray,
    mode: int,
    out_path: Path,
    title: Optional[str] = None,
    units: str = "K",
) -> None:
    """Plot single EOF spatial pattern (unweighted, physical units) on polar map."""
    tit = title or f"EOF mode {mode}"
    plot_polar_map(
        eof_pattern,
        out_path,
        title=tit,
        cmap="RdBu_r",
        colorbar_label=units,
    )
