#!/usr/bin/env python3
"""
One-off: add time=1 to existing F01 clim.nc files that have (lat, lon) only,
so GrADS ctl (t,y,x) reads them correctly. Overwrites in place.
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "Data" / "F01_climatology"
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])

    for nc in sorted(data_dir.glob("*/*/clim.nc")):
        ds = xr.open_dataset(nc)
        if "time" in ds.dims:
            print("Skip (has time):", nc)
            ds.close()
            continue
        # (lat, lon) -> (time=1, lat, lon)
        t0 = xr.DataArray([np.datetime64("2000-01-01")], dims=["time"])
        out = xr.Dataset(
            {
                "t2m": ds["t2m"].expand_dims(time=t0),
                "msl": ds["msl"].expand_dims(time=t0),
            },
            attrs=ds.attrs,
        )
        ds.close()
        tmp = nc.with_suffix(".nc.tmp")
        out.to_netcdf(tmp)
        tmp.replace(nc)
        print("Fixed:", nc)

if __name__ == "__main__":
    main()
