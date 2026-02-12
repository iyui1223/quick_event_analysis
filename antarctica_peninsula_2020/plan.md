Make quick analysis maps for an Antarctic heatwave event took place early Febrary in 2020.

Todos (updated):
1. Download surface wind speed for the area 90w − 40w and 75s to 65s (covers Antarctic peninsula) from ERA-5 site. 
2. Find the surface temperature (T2m) data for the area 90w − 40w and 75s to 65s from .ctl file loc. use cdo to slice the area. 
3. make Maps (details provided below). Probably for this project, python may be a better choice.

Maps to make: 
0. Orography check for ERA-5. 
    Look at Land-sea mask and orography to understand how good Antarctic is represented by the dataset.
    Files can be found here:
    /lustre/soge1/data/analysis/era5/0.28125x0.28125/invariant/land-sea_mask/nc/era5_invariant_land-sea_mask_20000101.nc 
    /lustre/soge1/data/analysis/era5/0.28125x0.28125/invariant/slope_of_sub-gridscale_orography/nc/era5_invariant_slope_of_sub-gridscale_orography_20000101.nc 
    and Example .ctl files are stored 
    /lustre/soge1/projects/andante/cenv1201/scripts/data_handling/grads_ctl

    0.1 create the .ctl files for each .nc files there.
    Make simple grads scripts for entire Antarctic continent by polar stereo coordinate, and plot both land-sea mask and orography both upon it.
    save image.
    if you could provide a starter .gs file, I can make nesessaly visual adjustment.

1. Histogram of the temperature distribution over peninsula's cradled ice shelves. (skip for now until Shapefile issues solved)
    From the sliced T2m data, pick areas covering ice shelves. For the Februaries between 1948 to current. Separate the period between past: 1948-1987 and present: 1988-2026. 
    Make histograms each for past and present. 
    Use celcius rather than the native Kelvin degrees.
    -- picking up the area covered by ice shelves -- is there a shape file for ice shelves + antarctic? Or should we use the ERA-5 ice files?
    If it is possible, the bin colouring has gradation within, showing each decade fall in each bins more often. say 1988-1997 has pink, 1998-2007 has stronger pink, 2008-2017 pale red, 2018-present strong red. 


2. Histogram of the windspeed distribution over peninsula, as an indicater of Fehn frequency. 
    Mask out the sea from the downloaded windspeed data slice using era-5's land sea mask -- to obtain the grid data covering the peninsula. Then calculate the average windspeed for each calendar dates in Februaries between 1948 to current. Separate the period between past: 1948-1987 and present: 1988-2026.
    Make histgrams for each past and present periods.
    If it is possible, the bin colouring has gradation within, showing each decade fall in each bins more often. say 1988-1997 has pink, 1998-2007 has stronger pink, 2008-2017 pale red, 2018-present strong red. 
    Land-sea mask can be found here
    /lustre/soge1/data/analysis/era5/0.28125x0.28125/invariant/land-sea_mask/nc/era5_invariant_land-sea_mask_20000101.nc

For this project. You can sometimes neglect the "good" coding practice provided as Skill.md. This is because it is for a small testing project, and hardcoding paths and reducing the number of files are often more optimal.

Useful resources
.ctl files for existing netcdf files are stocked at the following location. You can find the file locations, and the format from its content.
/lustre/soge1/projects/andante/cenv1201/scripts/data_handling/grads_ctl

Notes from previous errors:
Slurm symlink issues
Slurm environment at Oxford HPC does not like symlinks. Try to use absolute paths instead.
e.g. current root directry is 
O /lustre/soge1/projects/andante/cenv1201/proj/analogue/Sh
X /soge-home/users/cenv1201/andante/cenv1201/proj/analogue/Sh (symlink)

Disk access issues (occational -- will let you know if there is a potential)
It seems Lustre disk sometimes malfunctions -- like simple file header read hungs forever. If another electrical incident occures, I will let you know so try to add that type of errors in your thinking. (it may be files are broken not your code)

grads issues:
I am accessing HPC via cursor (that means establishing a server access) and also often visualization job may want to be passed on to the slurm scheduling system. So expect no x windows system.
grads execution may then be 
grads -blcx (for batch mode no X window available -- lc optional)

