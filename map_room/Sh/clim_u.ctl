* F01 U850/U500 climatology. Same grid as clim.ctl. DSET replaced by run script.
*
DSET ^clim_u.nc
DTYPE netcdf
OPTIONS yrev
TITLE Climatology U850 U500
UNDEF -9e9
XDEF 1440 LINEAR 0 0.25
YDEF 141 LINEAR -90 0.25
ZDEF 1 LINEAR 1 1
TDEF 1 LINEAR 01jan2000 1dy
VARS 2
u850 0 t,y,x U 850 hPa [m/s]
u500 0 t,y,x U 500 hPa [m/s]
ENDVARS
