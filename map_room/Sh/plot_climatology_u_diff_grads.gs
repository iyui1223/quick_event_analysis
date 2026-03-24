* plot_climatology_u_diff_grads.gs
* Plot F01 U diff: (present − past) U850 shaded + U500 diff contours + LSM.
* Expects clim_u_past.ctl (dfile 1), clim_u_present.ctl (dfile 2). TITLE_MONTH set by run script.
*
'reinit'
'run color.gs'
'open clim_u_past.ctl'
'open clim_u_present.ctl'
'open era5_lsm.ctl'

'set display color white'
'set grads off'
'set mproj sps'
'set lat -90 -55'
'set lon 0 360'
'set map 1 1 5'

* Present − past (file 1=past, file 2=present)
'define u850diff = u850.2 - u850.1'
'define u500diff = u500.2 - u500.1'

* Diverging palette for U850 diff (e.g. -5 to 5 m/s)
'color -5 5 0.5 -kind blue->white->red -gxout shaded'
'set gxout shaded'
'd u850diff'
'run cbarn'

* U500 diff contours
'set gxout contour'
'set ccolor 1'
'set cint 1'
'd u500diff'

* Land-sea boundary
'set dfile 3'
'set t 1'
'set gxout contour'
'set clevs 0.5'
'set clab off'
'set ccolor 1'
'set cthick 5'
'd lsm.3'

'set dfile 1'
'draw title Diff (1988_2025 minus 1948_1987) month TITLE_MONTH U850/U500'
'printim u850_u500_diff.png white'
'quit'
