* ERA5 invariant land-sea mask (for coastline overlay)
* Same path as antarctica_peninsula_2020; absolute path for batch.
*
DSET /lustre/soge1/data/analysis/era5/0.28125x0.28125/invariant/land-sea_mask/nc/era5_invariant_land-sea_mask_20000101.nc
DTYPE netcdf
OPTIONS yrev
TITLE ERA5 land-sea mask
UNDEF -32767
XDEF 1440 LINEAR 0 0.25
YDEF 721 LINEAR -90 0.25
ZDEF 1 LINEAR 1 1
TDEF 1 LINEAR 01jan2000 1dy
VARS 1
lsm 0 t,y,x Land-sea mask (0-1)
ENDVARS
