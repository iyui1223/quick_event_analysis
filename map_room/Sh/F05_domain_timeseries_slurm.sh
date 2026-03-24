#!/bin/bash
#SBATCH --job-name=F05_domain_ts
#SBATCH --partition=Short
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --output=../Log/F05_domain_timeseries/slurm_%j.out
#SBATCH --error=../Log/F05_domain_timeseries/slurm_%j.err
set -euo pipefail

SCRIPT_DIR="/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/map_room/Sh"
ROOT="$(readlink -f "${SCRIPT_DIR}/..")"
source "$(readlink -f "${ROOT}/Const/env_settings.sh")"

STEP_ID="F05_domain_timeseries"
SLICES_DIR="${SLICES_DIR:-/lustre/soge1/projects/andante/cenv1201/heavy/ERA5/daily/Surf/slices}"

mkdir -p "${LOG_DIR}/${STEP_ID}" "${DATA_DIR}/${STEP_ID}"

cd "${ROOT}"

${PYTHON} Python/compute_domain_timeseries.py \
    --slices-dir "${SLICES_DIR}" \
    --masks "${ROOT}/Data/F04_peninsula_domains/all_domain_masks.nc" \
    --out-dir "${DATA_DIR}/${STEP_ID}" \
    2>&1 | tee "${LOG_DIR}/${STEP_ID}/compute.log"

echo "=== ${STEP_ID} complete ==="
