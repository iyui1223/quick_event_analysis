* plot_climatology_u_grads.gs
* Plot F01 U climatology: U850 shaded + U500 contours + LSM. TITLE_ERA, TITLE_MONTH set by run script.
*
'reinit'
'run color.gs'
'open clim_u.ctl'
'open era5_lsm.ctl'

'set display color white'
'set grads off'
'set mproj sps'
'set lat -90 -55'
'set lon 0 360'
'set map 1 1 5'

* U850 shaded (e.g. -20 to 20 m/s)
'color -20 20 2 -kind blue->white->red -gxout shaded'
'set gxout shaded'
'd u850.1'
'run cbarn'

* U500 contours
'set gxout contour'
'set ccolor 1'
'set cint 5'
'd u500.1'

* Land-sea boundary
'set dfile 2'
'set t 1'
'set gxout contour'
'set clevs 0.5'
'set clab off'
'set ccolor 1'
'set cthick 5'
'd lsm.2'

'set dfile 1'
'draw title TITLE_ERA month TITLE_MONTH U850 (shaded) U500 (contours)'
'printim u850_u500.png white'
'quit'
