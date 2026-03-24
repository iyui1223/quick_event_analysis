#!/bin/bash
#SBATCH --job-name=F07_reader
#SBATCH --partition=Short
#SBATCH --time=00:15:00
#SBATCH --mem=1G
#SBATCH --output=../Log/F07_reader_download/slurm.out
#SBATCH --error=../Log/F07_reader_download/slurm.err
set -euo pipefail

SCRIPT_DIR="/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/map_room/Sh"
ROOT="$(readlink -f "${SCRIPT_DIR}/..")"
source "$(readlink -f "${ROOT}/Const/env_settings.sh")"

STEP_ID="F07_reader_download"
mkdir -p "${LOG_DIR}/${STEP_ID}" "${FIGS_DIR}/${STEP_ID}"

cd "${ROOT}"

# 1. Download READER surface data
echo "=== Downloading READER surface data ==="
bash Sh/download_reader_surface.sh 2>&1 | tee "${LOG_DIR}/${STEP_ID}/download.log"

# 2. Analyze temporal frequencies
echo "=== Analyzing observation frequencies ==="
${PYTHON} Python/analyze_reader_frequencies.py \
    --reader-dir "${READER_DIR}" \
    --out-dir "${FIGS_DIR}/${STEP_ID}" \
    2>&1 | tee "${LOG_DIR}/${STEP_ID}/analyze.log"

echo "=== ${STEP_ID} complete ==="
echo "Outputs: ${FIGS_DIR}/${STEP_ID}/"
echo "  - reader_frequency_summary.csv"
echo "  - reader_frequency_by_decade.csv"
echo "  - reader_obs_per_day_by_station.png"
echo "  - reader_frequency_by_decade.png"
