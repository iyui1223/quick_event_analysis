* plot_climatology_grads.gs
* Plot F01 climatology: t2m (Celsius) shaded + msl contours + land-sea boundary.
* Uses color.gs for temperature palette. No 180 rotation.
*
'reinit'
'run color.gs'
'open clim.ctl'
'open era5_lsm.ctl'

'set display color white'
'set grads off'
'set mproj sps'
'set lat -90 -55'
'set lon 0 360'
'set map 1 1 5'

* t2m in Celsius (clim.ctl has t2m in K)
'define t2mc = t2m.1 - 273.15'

* Temperature palette: cold to warm (e.g. -25 to 5 C by 2)
* Use blue->cyan->white->yellow->red style
'color -24 6 0.5 -kind blue->cyan->white->yellow->red -gxout shaded'
'set gxout shaded'
'd t2mc'
'run cbarn'

* MSL contours (optional, thin lines)
'set gxout contour'
'set ccolor 1'
'set cint 1000'
* 'set clab off'
'd msl.1'

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
'draw title TITLE_ERA month TITLE_MONTH (C) and MSL'
'printim t2m_msl.png white'
'quit'
