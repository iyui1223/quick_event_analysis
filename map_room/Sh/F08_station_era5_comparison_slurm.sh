#!/bin/bash
#SBATCH --job-name=F08_station
#SBATCH --partition=Long
#SBATCH --time=18:00:00
#SBATCH --mem=16G
#SBATCH --output=../Log/F08_station_era5_comparison/slurm.out
#SBATCH --error=../Log/F08_station_era5_comparison/slurm.err
set -euo pipefail

SCRIPT_DIR="/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/map_room/Sh"
ROOT="$(readlink -f "${SCRIPT_DIR}/..")"
source "$(readlink -f "${ROOT}/Const/env_settings.sh")"

STEP_ID="F08_station_era5_comparison"
F08_DATA="${DATA_DIR}/${STEP_ID}"
mkdir -p "${LOG_DIR}/${STEP_ID}" "${FIGS_DIR}/${STEP_ID}" "${F08_DATA}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/Python:${PYTHONPATH:-}"
export MAP_ROOM_F08_DATA_DIR="${F08_DATA}"

# ────────────────────────────────────────────────────────────────────────────
# Modes:
#   1. First run after changing sources: add --clear-cache
#      (deletes old era5_point_cache + reanalysis_cache, re-extracts raw files)
#   2. Default: extract + build/overwrite cache (omit both flags)
#   3. Fast replot from cache: add --use-cached after ERA5 and JRA-3Q caches exist
# ────────────────────────────────────────────────────────────────────────────

${PYTHON} Python/plot_station_era5_comparison.py \
    --reader-dir "${READER_DIR}" \
    --era5-dir  "${ERA5_DAILY_SURF}" \
    --jra3q-dir "${JRA3Q_SURF}" \
    --era5-end-year "${ERA5_END_YEAR}" \
    --data-dir "${F08_DATA}" \
    --out-dir  "${FIGS_DIR}/${STEP_ID}" \
    2>&1 | tee "${LOG_DIR}/${STEP_ID}/plot.log"

echo "=== ${STEP_ID} complete ==="
echo "Outputs: ${FIGS_DIR}/${STEP_ID}/"
echo "Cache:   ${F08_DATA}/reanalysis_cache/{ERA5,JRA-3Q}/"
echo ""
echo "For fast replots after caches exist, add --use-cached to the Python command."
