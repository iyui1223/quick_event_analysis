* F01 climatology (t2m + msl), one era/month. 0.25 deg, south of 55S.
* DSET replaced by run script with path to clim.nc
* NetCDF: t2m(lat,lon) msl(lat,lon); lat 141 from -55 to -90 (file order north->south).
* YDEF positive increment; yrev so file row 1 (-55) = north.
*
DSET ^clim.nc
DTYPE netcdf
OPTIONS yrev
TITLE Climatology t2m and msl
UNDEF -9e9
XDEF 1440 LINEAR 0 0.25
YDEF 141 LINEAR -90 0.25
ZDEF 1 LINEAR 1 1
TDEF 1 LINEAR 01jan2000 1dy
VARS 2
t2m 0 t,y,x 2m temperature [K]
msl 0 t,y,x Mean sea level pressure [Pa]
ENDVARS
