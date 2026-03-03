#!/bin/bash
#SBATCH --job-name=F04_peninsula_domains
#SBATCH --partition=Short
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --output=Log/F04_peninsula_domains/slurm_%j.out
#SBATCH --error=Log/F04_peninsula_domains/slurm_%j.err
set -euo pipefail

# ---- load shared environment ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(readlink -f "${SCRIPT_DIR}/../Const/env_settings.sh")"

STEP_ID="F04_peninsula_domains"
WORK="${WORK_DIR}/${STEP_ID}"
OUT="${WORK}/out"

mkdir -p "${WORK}" "${OUT}" "${LOG_DIR}/${STEP_ID}" \
         "${DATA_DIR}/${STEP_ID}" "${FIGS_DIR}/${STEP_ID}"

# ---- symlink code ----
ln -sf "$(readlink -f "${PROJECT_ROOT}/Python/define_peninsula_domains.py")" "${WORK}/"
ln -sf "$(readlink -f "${PROJECT_ROOT}/Const/params.yaml")"                  "${WORK}/"

cd "${WORK}"

# ---- run ----
${PYTHON} define_peninsula_domains.py \
    --params params.yaml \
    --out-dir "${OUT}" \
    2>&1 | tee "${LOG_DIR}/${STEP_ID}/define_domains.log"

# ---- collect outputs ----
cp -u "${OUT}"/*.nc  "${DATA_DIR}/${STEP_ID}/" 2>/dev/null || true
cp -u "${FIGS_DIR}/${STEP_ID}"/*.png "${FIGS_DIR}/${STEP_ID}/" 2>/dev/null || true

echo "=== ${STEP_ID} complete ==="
