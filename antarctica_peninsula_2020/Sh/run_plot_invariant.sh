#!/bin/bash
# Run GrADS plot for land-sea mask and orography (no X11).
# Use absolute path for Slurm: set PROJECT_ROOT if needed.
set -e
PROJECT_ROOT="${PROJECT_ROOT:-/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/antarctica_peninsula_2020}"
mkdir -p "${PROJECT_ROOT}/Figs"
cd "${PROJECT_ROOT}/Sh"
grads -blcx "run plot_antarctica_invariant.gs"
