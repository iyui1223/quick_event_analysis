#!/bin/bash
#SBATCH --job-name=F02_eof
#SBATCH --output=/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/map_room/Log/F02_eof_t2m_anoms/slurm_%j.out
#SBATCH --error=/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/map_room/Log/F02_eof_t2m_anoms/slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=hort
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

PROJECT_ROOT="/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/map_room"
source "${PROJECT_ROOT}/Const/env_settings.sh"

STEP_ID="F02_eof_t2m_anoms"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/Work}/${STEP_ID}"
mkdir -p "${WORK_DIR}" "${LOG_DIR}/${STEP_ID}" "${DATA_DIR}/F02_eof" "${FIGS_DIR}/F02_eof"

cd "${WORK_DIR}"
export PYTHONPATH="${PROJECT_ROOT}/Python:${PYTHONPATH:-}"

"${PYTHON}" "${PROJECT_ROOT}/Python/compute_eof_t2m.py" \
  --out-dir "${DATA_DIR}/F02_eof" \
  2>&1 | tee "${LOG_DIR}/${STEP_ID}/compute_eof_run.log"

"${PYTHON}" "${PROJECT_ROOT}/Python/plot_eof_modes.py" \
  --eof-nc "${DATA_DIR}/F02_eof/EOFs.nc" \
  --fig-dir "${FIGS_DIR}/F02_eof" \
  --step F02_eof \
  2>&1 | tee "${LOG_DIR}/${STEP_ID}/plot_eof_run.log"

echo "Done. EOFs: ${DATA_DIR}/F02_eof/EOFs.nc. Figs: ${FIGS_DIR}/F02_eof"
