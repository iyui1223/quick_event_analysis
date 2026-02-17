"""
Spatial and temporal weights for map_room EOF analysis.
- Area weight: sqrt(cos(lat)) for equal-area contribution on regular lat/lon.
- Optional: Gaussian calendar-day taper for DJF emphasis (F03).
"""

from typing import Optional

import numpy as np
import xarray as xr


def sqrt_cos_lat_weight(lat: xr.DataArray) -> xr.DataArray:
    """
    Area weight w(lat) = sqrt(cos(lat)) in radians.
    Use for EOF so that each grid point contributes by physical area.
    """
    lat_rad = np.deg2rad(lat)
    w = np.sqrt(np.cos(lat_rad))
    return xr.DataArray(w, coords=lat.coords, dims=lat.dims)


def apply_spatial_weights(da: xr.DataArray, lat_dim: str = "lat") -> xr.DataArray:
    """Multiply data by sqrt(cos(lat)). lat_dim can be 'lat' or 'latitude'."""
    w = sqrt_cos_lat_weight(da[lat_dim])
    return da * w


def gaussian_calendar_taper(
    time: xr.DataArray,
    center_doy: int = 15,
    sigma_days: float = 45.0,
) -> xr.DataArray:
    """
    Gaussian weight in day-of-year, centered on center_doy (e.g. 15 = 15 Jan).
    Wraps around year: distance in circular calendar (365 days).
    """
    doy = time.dt.dayofyear
    # Distance on circular calendar
    diff = np.abs(doy - center_doy)
    diff = np.minimum(diff, 365 - diff)
    w = np.exp(-0.5 * (diff.astype(float) / sigma_days) ** 2)
    return xr.DataArray(w, coords=time.coords, dims=time.dims)


def apply_time_taper(da: xr.DataArray, center_doy: int = 15, sigma_days: float = 45.0) -> xr.DataArray:
    """Multiply data by Gaussian calendar taper (e.g. for F03 DJF emphasis)."""
    time = da.time
    w = gaussian_calendar_taper(time, center_doy=center_doy, sigma_days=sigma_days)
    return da * w
