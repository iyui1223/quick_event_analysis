#!/bin/bash
# Re-plot EOF mode maps with GrADS (land-sea boundary). Run after EOFs.nc exists.
# Usage: ./run_plot_eof_modes_grads.sh [F02_eof|F03_eof_tapered_djf]
set -euo pipefail

PROJECT_ROOT="/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/map_room"
source "${PROJECT_ROOT}/Const/env_settings.sh"

STEP_ID="${1:-F02_eof}"
EOFNCDIR="${DATA_DIR}/${STEP_ID}"
FIGDIR="${FIGS_DIR}/${STEP_ID}"
WORK_GRADS="${WORK_DIR}/${STEP_ID}/grads_figs"

if [[ ! -f "${EOFNCDIR}/EOFs.nc" ]]; then
  echo "ERROR: ${EOFNCDIR}/EOFs.nc not found. Run F02 or F03 first." >&2
  exit 1
fi

mkdir -p "${WORK_GRADS}" "${FIGDIR}"

# eof_pattern.ctl with DSET = absolute path to EOFs.nc
sed "s|DSET .*|DSET ${EOFNCDIR}/EOFs.nc|" "${PROJECT_ROOT}/Sh/eof_pattern.ctl" > "${WORK_GRADS}/eof_pattern.ctl"
cp "${PROJECT_ROOT}/Sh/era5_lsm.ctl" "${WORK_GRADS}/"
cp "${PROJECT_ROOT}/Sh/plot_eof_modes_grads.gs" "${WORK_GRADS}/"
# color.gs (Kodama) for BWR gradation: prefer Sh/, else project root
if [[ -f "${PROJECT_ROOT}/Sh/color.gs" ]]; then
  cp "${PROJECT_ROOT}/Sh/color.gs" "${WORK_GRADS}/"
elif [[ -f "${PROJECT_ROOT}/color.gs" ]]; then
  cp "${PROJECT_ROOT}/color.gs" "${WORK_GRADS}/"
fi

cd "${WORK_GRADS}"
grads -blcx "run plot_eof_modes_grads.gs"
mv -f mode_0*.png "${FIGDIR}/"
cd - >/dev/null

echo "GrADS EOF maps (with coastline) written to ${FIGDIR}/mode_01.png ... mode_04.png"
