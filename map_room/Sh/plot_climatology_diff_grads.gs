* plot_climatology_diff_grads.gs
* Plot F01 climatology diff: (present − past) t2m (C) shaded + msl diff contours + LSM.
* Expects clim_past.ctl (dfile 1) and clim_present.ctl (dfile 2). TITLE_MONTH replaced by run script.
*
'reinit'
'run color.gs'
'open clim_past.ctl'
'open clim_present.ctl'
'open era5_lsm.ctl'

'set display color white'
'set grads off'
'set mproj sps'
'set lat -90 -55'
'set lon 0 360'
'set map 1 1 5'

* Present − past (file 1=past, file 2=present; diff in K = diff in C)
'define t2mcdiff = (t2m.2 - 273.15) - (t2m.1 - 273.15)'
'define msldiff = msl.2 - msl.1'

* Diverging palette for t2m diff (e.g. -2 to 2 C by 0.2)
'color -2 2 0.1 -kind blue->white->red -gxout shaded'
* 'set gxout shaded'
'd t2mcdiff'
'run cbarn'

* MSL diff contours
'set gxout contour'
'set ccolor 1'
'set cint 100'
'd msldiff'

* Land-sea boundary
'set dfile 3'
'set t 1'
'set gxout contour'
'set clevs 0.5'
'set clab off'
'set ccolor 5'
'set cthick 5'
'd lsm.3'

'set dfile 1'
'draw title Diff (1988_2025 minus 1948_1987) month TITLE_MONTH (C) and MSL'
'printim t2m_msl_diff.png white'
'quit'
