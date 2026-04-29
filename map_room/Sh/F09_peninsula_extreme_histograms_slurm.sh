#!/bin/bash
#SBATCH --job-name=F09_pen_hist
#SBATCH --partition=Short
#SBATCH --time=03:00:00
#SBATCH --mem=10G
#SBATCH --output=../Log/F09_peninsula_extreme_histograms/slurm_%j.out
#SBATCH --error=../Log/F09_peninsula_extreme_histograms/slurm_%j.err
set -euo pipefail

SCRIPT_DIR="/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/map_room/Sh"
ROOT="$(readlink -f "${SCRIPT_DIR}/..")"
source "$(readlink -f "${ROOT}/Const/env_settings.sh")"

STEP_ID="F09_peninsula_extreme_histograms"
SURF_DIR="${SLICES_DIR:-/lustre/soge1/projects/andante/cenv1201/heavy/ERA5/daily/Surf/slices}"
MASKS="${ROOT}/Data/F04_peninsula_domains/all_domain_masks.nc"

# Tunable knobs (override with: sbatch --export=MEAN_DAYS=5,BIN_WIDTH_C=0.5,...)
MEAN_DAYS="${MEAN_DAYS:-1}"
BIN_WIDTH_C="${BIN_WIDTH_C:-1.0}"
START_YEAR="${START_YEAR:-1949}"
END_YEAR="${END_YEAR:-}"  # empty => auto-detect latest available month
MONTHS="${MONTHS:-10,11,12,1,2,3}"  # ONDJFM
REBUILD_CACHE="${REBUILD_CACHE:-0}" # 1 to force recomputation

mkdir -p "${LOG_DIR}/${STEP_ID}" "${DATA_DIR}/${STEP_ID}" "${FIGS_DIR}/${STEP_ID}"

cd "${ROOT}"

CMD=(
    "${PYTHON}" "Python/plot_peninsula_extreme_histograms.py"
    "--surf-dir" "${SURF_DIR}"
    "--masks" "${MASKS}"
    "--out-data-dir" "${DATA_DIR}/${STEP_ID}"
    "--out-fig-dir" "${FIGS_DIR}/${STEP_ID}"
    "--start-year" "${START_YEAR}"
    "--months" "${MONTHS}"
    "--mean-days" "${MEAN_DAYS}"
    "--bin-width-c" "${BIN_WIDTH_C}"
)

if [[ -n "${END_YEAR}" ]]; then
    CMD+=( "--end-year" "${END_YEAR}" )
fi
if [[ "${REBUILD_CACHE}" == "1" ]]; then
    CMD+=( "--rebuild-cache" )
fi

"${CMD[@]}" 2>&1 | tee "${LOG_DIR}/${STEP_ID}/compute.log"

echo "=== ${STEP_ID} complete ==="
