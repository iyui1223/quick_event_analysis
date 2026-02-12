#!/bin/bash
# Spaghetti plot: T2m at peninsula point, target + analogues. Use absolute path for Slurm.
set -e
PROJECT_ROOT="${PROJECT_ROOT:-/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/antarctica_peninsula_2020}"
source /lustre/soge1/projects/andante/cenv1201/venvs/template/bin/activate
export FIGS_DIR="${PROJECT_ROOT}/Figs"
python3 "${PROJECT_ROOT}/Python/plot_t2m_spaghetti.py"
