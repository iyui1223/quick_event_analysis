#!/bin/bash
#SBATCH --job-name=F06_corr
#SBATCH --partition=Long
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --output=../Log/F06_domain_correlations/slurm.out
#SBATCH --error=../Log/F06_domain_correlations/slurm.err
set -euo pipefail

SCRIPT_DIR="/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/map_room/Sh"
ROOT="$(readlink -f "${SCRIPT_DIR}/..")"
source "$(readlink -f "${ROOT}/Const/env_settings.sh")"

STEP_ID="F06_domain_correlations"

mkdir -p "${LOG_DIR}/${STEP_ID}" "${DATA_DIR}/${STEP_ID}"

cd "${ROOT}"

${PYTHON} Python/compute_domain_correlations.py \
    --params "${PARAMS_YAML}" \
    --input "${DATA_DIR}/F05_domain_timeseries/domain_timeseries.nc" \
    --out-dir "${DATA_DIR}/${STEP_ID}" \
    2>&1 | tee "${LOG_DIR}/${STEP_ID}/compute.log"

echo "=== ${STEP_ID} complete ==="
