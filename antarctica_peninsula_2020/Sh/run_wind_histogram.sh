#!/bin/bash
# Run Map 2: wind-speed histogram (past vs present). Use absolute path for Slurm.
set -e
PROJECT_ROOT="${PROJECT_ROOT:-/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/antarctica_peninsula_2020}"
source /lustre/soge1/projects/andante/cenv1201/venvs/template/bin/activate
export WIND_DATA_DIR="${PROJECT_ROOT}/dataslices/era5_daily_peninsula"
export FIGS_DIR="${PROJECT_ROOT}/Figs"
python3 "${PROJECT_ROOT}/Python/plot_wind_histogram.py"
