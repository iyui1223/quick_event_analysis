#!/bin/bash
#SBATCH -J era5-peninsula
#SBATCH -o era5-peninsula_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=Short

# === Antarctic Peninsula: 10m wind + 2m temp, February only, 1948–present ===
# Use absolute Lustre path (no symlinks) for Slurm
BASE="/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/antarctica_peninsula_2020/dataslices"
OUTPUT_DIR="${BASE}/era5_daily_peninsula"

source /lustre/soge1/projects/andante/cenv1201/venvs/template/bin/activate

mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"

export OUTPUT_DIR="${OUTPUT_DIR}"
export START_YEAR=1948
export MONTHS="2"
export DAILY_STAT="daily_mean"
export TIME_ZONE="utc+00:00"
export FREQUENCY="6_hourly"
export VARIABLES="10m_u_component_of_wind,10m_v_component_of_wind,2m_temperature"
# export FORCE=1

python "${BASE}/download.py"