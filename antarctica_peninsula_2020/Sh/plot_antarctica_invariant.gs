* plot_antarctica_invariant.gs
* Combined shaded orography + land-sea boundary (single image)
'reinit'
'open era5_invariant_land_sea_mask.ctl'
'open era5_invariant_geopotential.ctl'

'set display color white'
'set grads off'

* Projection & domain
'set mproj sps'
'set lat -90 -60'
'set lon 0 360'
'set t 1'
'set map 1 1 5'


* ========== Plot shaded orography (base layer) ==========
* Use the mask to mask out the sea -- or sea get all level 0 colour that's not my intention.
'set gxout shaded'
* convert geopotential to metres
* shading does not have to be every 200 meters as there are not enough rols anyway.
* 'd z.2/9.80665'

* ========== Overplot orography contour lines (thin) ==========
'set gxout contour'
'set clevs 0 200 400 600 800 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000'
* 'set clab off'
* choose a subtle color index for contour lines (tweak if needed)
* 'set ccols 16' 
'd z.2/9.80665'

* ========== Overplot land-sea boundary as single contour (top layer) ==========
'set gxout contour'
* threshold at 0.5 for 0/1 mask; this draws the ice/land boundary cleanly
'set clevs 0.5'
'set clab off'
* color index for boundary (1 is usually black)
'set ccols 5'
'd lsm.1'

* ========== Title and save ==========
'draw title ERA5 land-sea boundary and elevation'
'printim ../Figs/era5_antarctica_shaded_orog_lsm.png white'
'quit'
