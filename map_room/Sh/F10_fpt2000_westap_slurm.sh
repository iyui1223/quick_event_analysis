#!/bin/bash
#SBATCH --job-name=F10_fpt2000
#SBATCH --partition=Long
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --output=../Log/F10_fpt2000_westAP/slurm_%j.out
#SBATCH --error=../Log/F10_fpt2000_westAP/slurm_%j.err
set -euo pipefail

SCRIPT_DIR="/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/map_room/Sh"
ROOT="$(readlink -f "${SCRIPT_DIR}/..")"
source "$(readlink -f "${ROOT}/Const/env_settings.sh")"

STEP_ID="F10_fpt2000_westAP"
RAW_DIR="${DATA_DIR}/${STEP_ID}/raw"
DATA_OUT="${DATA_DIR}/${STEP_ID}"
FIG_OUT="${FIGS_DIR}/FPT2000_westAP"
TABLE_OUT="${ROOT}/Tables"
MASKS="${ROOT}/Data/F04_peninsula_domains/all_domain_masks.nc"

# Override with sbatch --export=NAME=value,...
START_YEAR="${START_YEAR:-1959}"
END_YEAR="${END_YEAR:-2025}"
UTC_HOUR="${UTC_HOUR:-19}"
BUFFER_DEG="${BUFFER_DEG:-0.5}"
ROUND_DEG="${ROUND_DEG:-0.25}"
PRESENT_START="${PRESENT_START:-1988}"
RUN_DOWNLOAD="${RUN_DOWNLOAD:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
CDS_DRY_RUN="${CDS_DRY_RUN:-0}"
OVERWRITE_DOWNLOAD="${OVERWRITE_DOWNLOAD:-0}"
LEGACY_CDS_FORMAT="${LEGACY_CDS_FORMAT:-0}"
NO_SEGMENTS="${NO_SEGMENTS:-0}"

mkdir -p "${LOG_DIR}/${STEP_ID}" "${RAW_DIR}" "${DATA_OUT}" "${FIG_OUT}" "${TABLE_OUT}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/Python:${PYTHONPATH:-}"

if [[ "${RUN_DOWNLOAD}" == "1" ]]; then
    DOWNLOAD_CMD=(
        "${PYTHON}" "Python/download_era5_fpt2000_inputs.py"
        "--masks" "${MASKS}"
        "--out-dir" "${RAW_DIR}"
        "--start-year" "${START_YEAR}"
        "--end-year" "${END_YEAR}"
        "--utc-hour" "${UTC_HOUR}"
        "--buffer-deg" "${BUFFER_DEG}"
        "--round-deg" "${ROUND_DEG}"
    )
    if [[ "${CDS_DRY_RUN}" == "1" ]]; then
        DOWNLOAD_CMD+=( "--dry-run" )
    fi
    if [[ "${OVERWRITE_DOWNLOAD}" == "1" ]]; then
        DOWNLOAD_CMD+=( "--overwrite" )
    fi
    if [[ "${LEGACY_CDS_FORMAT}" == "1" ]]; then
        DOWNLOAD_CMD+=( "--legacy-format-key" )
    fi

    "${DOWNLOAD_CMD[@]}" 2>&1 | tee "${LOG_DIR}/${STEP_ID}/download.log"
fi

if [[ "${RUN_ANALYSIS}" == "1" ]]; then
    ANALYSIS_CMD=(
        "${PYTHON}" "Python/plot_fpt2000_westap.py"
        "--input-dir" "${RAW_DIR}"
        "--masks" "${MASKS}"
        "--out-data-dir" "${DATA_OUT}"
        "--out-fig-dir" "${FIG_OUT}"
        "--out-table-dir" "${TABLE_OUT}"
        "--start-year" "${START_YEAR}"
        "--end-year" "${END_YEAR}"
        "--utc-hour" "${UTC_HOUR}"
        "--present-start" "${PRESENT_START}"
        "--mask-buffer-deg" "${BUFFER_DEG}"
    )
    if [[ "${NO_SEGMENTS}" == "1" ]]; then
        ANALYSIS_CMD+=( "--no-segments" )
    fi

    "${ANALYSIS_CMD[@]}" 2>&1 | tee "${LOG_DIR}/${STEP_ID}/analysis.log"
fi

echo "=== ${STEP_ID} complete ==="
echo "Raw ERA5 inputs: ${RAW_DIR}"
echo "Data outputs:    ${DATA_OUT}"
echo "Figures:         ${FIG_OUT}"
echo "Tables:          ${TABLE_OUT}"
