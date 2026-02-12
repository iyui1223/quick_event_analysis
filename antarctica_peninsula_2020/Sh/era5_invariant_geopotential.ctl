* ERA5 invariant surface geopotential (single file)
* Grid: 0.25 x 0.25 deg global, 1440x721. Orography [m] = z/9.80665
* Real path for Slurm / no symlinks
*
DSET /lustre/soge1/data/analysis/era5/0.28125x0.28125/invariant/geopotential/nc/era5_invariant_geopotential_20000101.nc
DTYPE netcdf
OPTIONS yrev
TITLE ERA5 invariant surface geopotential
UNDEF -32767
XDEF 1440 LINEAR 0 0.25
YDEF 721 LINEAR -90 0.25
ZDEF 1 LINEAR 1 1
TDEF 1 LINEAR 01jan2000 1dy
VARS 1
z 0 t,y,x Geopotential [m**2 s**-2]
ENDVARS
