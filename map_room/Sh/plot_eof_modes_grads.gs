* plot_eof_modes_grads.gs
* Plot EOF spatial patterns (modes 1-4) with land-sea boundary.
* Uses Kodama color.gs for BWR (blue->white->red), clevs -0.2 to 0.2 by 0.02.
* Map rotated 180 deg. cbarn for colorbar.
*
'reinit'
'run color.gs'
'open eof_pattern.ctl'
'open era5_lsm.ctl'

'set display color white'
'set grads off'
'set mproj sps'
'set lat -80 -55'
'set lon 0 360'
'set map 1 1 5'
* 'set grid off'

* BWR gradation via color.gs: -0.2 to 0.2 by 0.02, blue->white->red

m = 1
while (m <= 4)
  'set dfile 1'
  'set t 'm
  'color -0.1 0.1 0.002 -kind blue->white->red -gxout shaded'
  'd eof_pattern.1'
  'run cbarn'
  'set gxout contour'  
  'set clevs -0.1 -0.09 -0.08 -0.07 -0.06 -0.05 -0.04 -0.03 -0.02 -0.01 0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1'
  'set ccolor 15'
  'd eof_pattern.1'
  
  'set dfile 2'
  'set t 1'
  'set gxout contour'
  'set clevs 0.5'
  'set clab off'
  'set ccolor 5'
  'set cthick 5'
  'd lsm.2'
  'set clab on'  
  'set dfile 1'
  'draw title EOF mode 'm' (Antarctic t2m, -80 to -55S)'
  'printim mode_0'm'.png white'
  'c'
  m = m + 1
endwhile

'quit'
