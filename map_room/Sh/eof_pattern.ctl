* EOF spatial patterns (mode, lat, lon) — map_room F02/F03
* Grid: 1 deg, -80 to -55 lat, 0-360 lon. Use as file 1; set t 1..4 for modes.
*
DSET ^EOFs.nc
DTYPE netcdf
OPTIONS yrev
TITLE EOF patterns t2m
UNDEF -9e9
XDEF 360 LINEAR 0.375 1
YDEF 25 LINEAR -79.375 1
ZDEF 1 LINEAR 1 1
TDEF 4 LINEAR 01jan2000 1dy
VARS 1
eof_pattern 0 t,y,x EOF pattern [K]
ENDVARS
