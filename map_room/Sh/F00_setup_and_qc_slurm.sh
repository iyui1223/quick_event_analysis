#!/bin/bash
#SBATCH --job-name=F00_setup_qc
#SBATCH --output=/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/map_room/Log/F00_setup_and_qc/slurm_%j.out
#SBATCH --error=/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/map_room/Log/F00_setup_and_qc/slurm_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=Short
#SBATCH --mem=512M

set -euo pipefail

PROJECT_ROOT="/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/map_room"
source "${PROJECT_ROOT}/Const/env_settings.sh"

STEP_ID="F00_setup_and_qc"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/Work}/${STEP_ID}"
mkdir -p "${WORK_DIR}" "${LOG_DIR}/${STEP_ID}" "${DATA_DIR}/${STEP_ID}"

cd "${WORK_DIR}"
export PYTHONPATH="${PROJECT_ROOT}/Python:${PYTHONPATH:-}"

"${PYTHON}" "${PROJECT_ROOT}/Python/setup_and_qc.py" \
  2>&1 | tee "${LOG_DIR}/${STEP_ID}/setup_and_qc_run.log"

# Manifest and qc_summary are written to Data/F00_setup_and_qc by the script
echo "Done. Check ${DATA_DIR}/${STEP_ID}/inputs_manifest.json and qc_summary.txt"
