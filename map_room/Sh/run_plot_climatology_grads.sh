#!/bin/bash
# Re-plot F01 climatology maps with GrADS (t2m in Celsius, land-sea boundary).
# Run after F01 has produced Data/F01_climatology/{era}/{month}/clim.nc
set -euo pipefail

PROJECT_ROOT="/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/map_room"
source "${PROJECT_ROOT}/Const/env_settings.sh"

DATA_CLIM="${DATA_DIR}/F01_climatology"
FIG_CLIM="${FIGS_DIR}/F01_climatology"
WORK_GRADS="${WORK_DIR}/F01_climatology_grads"

mkdir -p "${WORK_GRADS}" "${FIG_CLIM}"

# Copy scripts and ctls once (LSM, color.gs, cbarn.gs)
cp "${PROJECT_ROOT}/Sh/era5_lsm.ctl" "${WORK_GRADS}/"
cp "${PROJECT_ROOT}/Sh/cbarn.gs" "${WORK_GRADS}/"
if [[ -f "${PROJECT_ROOT}/Sh/color.gs" ]]; then
  cp "${PROJECT_ROOT}/Sh/color.gs" "${WORK_GRADS}/"
elif [[ -f "${PROJECT_ROOT}/color.gs" ]]; then
  cp "${PROJECT_ROOT}/color.gs" "${WORK_GRADS}/"
fi

month_name() { case "$1" in 12) echo Dec ;; 01) echo Jan ;; 02) echo Feb ;; 03) echo Mar ;; *) echo "$1" ;; esac; }

for era in 1948_1987 1988_2025; do
  for month in 12 01 02 03; do
    CLIMNC="${DATA_CLIM}/${era}/${month}/clim.nc"
    if [[ ! -f "${CLIMNC}" ]]; then
      echo "Skip (missing): ${CLIMNC}"
      continue
    fi
    mkdir -p "${FIG_CLIM}/${era}/${month}"
    MONTH_LABEL=$(month_name "$month")
    sed "s|DSET .*|DSET ${CLIMNC}|" "${PROJECT_ROOT}/Sh/clim.ctl" > "${WORK_GRADS}/clim.ctl"
    sed "s|TITLE_ERA|${era}|g; s|TITLE_MONTH|${MONTH_LABEL}|g" "${PROJECT_ROOT}/Sh/plot_climatology_grads.gs" > "${WORK_GRADS}/plot_climatology_grads.gs"
    cd "${WORK_GRADS}"
    grads -blcx "run plot_climatology_grads.gs"
    mv -f t2m_msl.png "${FIG_CLIM}/${era}/${month}/t2m_msl.png"
    cd - >/dev/null
    echo "  ${era} month ${month}"
  done
done

# U850/U500 climatology (if clim_u.nc exists)
for era in 1948_1987 1988_2025; do
  for month in 12 01 02 03; do
    CLIMUNC="${DATA_CLIM}/${era}/${month}/clim_u.nc"
    if [[ ! -f "${CLIMUNC}" ]]; then
      continue
    fi
    mkdir -p "${FIG_CLIM}/${era}/${month}"
    MONTH_LABEL=$(month_name "$month")
    sed "s|DSET .*|DSET ${CLIMUNC}|" "${PROJECT_ROOT}/Sh/clim_u.ctl" > "${WORK_GRADS}/clim_u.ctl"
    sed "s|TITLE_ERA|${era}|g; s|TITLE_MONTH|${MONTH_LABEL}|g" "${PROJECT_ROOT}/Sh/plot_climatology_u_grads.gs" > "${WORK_GRADS}/plot_climatology_u_grads.gs"
    cd "${WORK_GRADS}"
    grads -blcx "run plot_climatology_u_grads.gs"
    mv -f u850_u500.png "${FIG_CLIM}/${era}/${month}/u850_u500.png"
    cd - >/dev/null
    echo "  U ${era} month ${month}"
  done
done

# Diff maps (present − past) per month
for month in 12 01 02 03; do
  PASTNC="${DATA_CLIM}/1948_1987/${month}/clim.nc"
  PRESNC="${DATA_CLIM}/1988_2025/${month}/clim.nc"
  if [[ ! -f "${PASTNC}" || ! -f "${PRESNC}" ]]; then
    echo "Skip diff (missing clim): month ${month}"
    continue
  fi
  mkdir -p "${FIG_CLIM}/diff/${month}"
  MONTH_LABEL=$(month_name "$month")
  sed "s|DSET .*|DSET ${PASTNC}|" "${PROJECT_ROOT}/Sh/clim.ctl" > "${WORK_GRADS}/clim_past.ctl"
  sed "s|DSET .*|DSET ${PRESNC}|" "${PROJECT_ROOT}/Sh/clim.ctl" > "${WORK_GRADS}/clim_present.ctl"
  sed "s|TITLE_MONTH|${MONTH_LABEL}|g" "${PROJECT_ROOT}/Sh/plot_climatology_diff_grads.gs" > "${WORK_GRADS}/plot_climatology_diff_grads.gs"
  cd "${WORK_GRADS}"
  grads -blcx "run plot_climatology_diff_grads.gs"
  mv -f t2m_msl_diff.png "${FIG_CLIM}/diff/${month}/t2m_msl_diff.png"
  cd - >/dev/null
  echo "  diff month ${month}"
done

# U diff (present − past) per month
for month in 12 01 02 03; do
  PASTUNC="${DATA_CLIM}/1948_1987/${month}/clim_u.nc"
  PRESUNC="${DATA_CLIM}/1988_2025/${month}/clim_u.nc"
  if [[ ! -f "${PASTUNC}" || ! -f "${PRESUNC}" ]]; then
    continue
  fi
  mkdir -p "${FIG_CLIM}/diff/${month}"
  MONTH_LABEL=$(month_name "$month")
  sed "s|DSET .*|DSET ${PASTUNC}|" "${PROJECT_ROOT}/Sh/clim_u.ctl" > "${WORK_GRADS}/clim_u_past.ctl"
  sed "s|DSET .*|DSET ${PRESUNC}|" "${PROJECT_ROOT}/Sh/clim_u.ctl" > "${WORK_GRADS}/clim_u_present.ctl"
  sed "s|TITLE_MONTH|${MONTH_LABEL}|g" "${PROJECT_ROOT}/Sh/plot_climatology_u_diff_grads.gs" > "${WORK_GRADS}/plot_climatology_u_diff_grads.gs"
  cd "${WORK_GRADS}"
  grads -blcx "run plot_climatology_u_diff_grads.gs"
  mv -f u850_u500_diff.png "${FIG_CLIM}/diff/${month}/u850_u500_diff.png"
  cd - >/dev/null
  echo "  U diff month ${month}"
done

echo "GrADS climatology and diff maps written to ${FIG_CLIM}/"
