#!/bin/bash
#SBATCH --job-name=F01_climatology
#SBATCH --output=/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/map_room/Log/F01_climatology_maps/slurm_%j.out
#SBATCH --error=/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/map_room/Log/F01_climatology_maps/slurm_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=GPU
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail

PROJECT_ROOT="/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/map_room"
source "${PROJECT_ROOT}/Const/env_settings.sh"

STEP_ID="F01_climatology_maps"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/Work}/${STEP_ID}"
mkdir -p "${WORK_DIR}" "${LOG_DIR}/${STEP_ID}" "${DATA_DIR}/F01_climatology" "${FIGS_DIR}/F01_climatology"

cd "${WORK_DIR}"
export PYTHONPATH="${PROJECT_ROOT}/Python:${PYTHONPATH:-}"
# Use writable dirs on compute nodes (avoid permission errors on home)
export MPLCONFIGDIR="${WORK_DIR}/.matplotlib"
export CARTOPY_USER_BASE_DIR="${WORK_DIR}/.cartopy"
mkdir -p "${MPLCONFIGDIR}" "${CARTOPY_USER_BASE_DIR}"

"${PYTHON}" "${PROJECT_ROOT}/Python/make_climatology_maps.py" \
  --data-dir "${DATA_DIR}/F01_climatology" \
  --figs-dir "${FIGS_DIR}/F01_climatology" \
  2>&1 | tee "${LOG_DIR}/${STEP_ID}/make_climatology_run.log"

echo "Done. NetCDF: ${DATA_DIR}/F01_climatology. Figs: ${FIGS_DIR}/F01_climatology"
