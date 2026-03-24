#!/bin/bash
# Download READER Antarctic surface station data for Peninsula stations.
# Saves to READER_DIR (default: heavy/READER/SURFACE).
# Run locally: ./Sh/download_reader_surface.sh
# Or: bash Sh/download_reader_surface.sh

set -euo pipefail

BASE_URL="https://legacy.bas.ac.uk/met/READER/ANTARCTIC_METEOROLOGICAL_DATA/SURFACE"
SCRIPT_DIR="/lustre/soge1/projects/andante/cenv1201/proj/quick_event_analysis/map_room/Sh"
ROOT="$(readlink -f "${SCRIPT_DIR}/..")"
# Use READER_DIR from env (set by Const/env_settings.sh) or default
READER_DIR="${READER_DIR:-/lustre/soge1/projects/andante/cenv1201/heavy/READER/SURFACE}"

# Peninsula stations from plan.md (files available on BAS)
STATIONS=(
    Adelaide
    Deception
    Esperanza
    Faraday
    Fossil_Bluff
    Marambio
    O_Higgins
    Palmer
    Rothera
    San_Martin
    Vernadsky
    # Jubany (King George Island) - not in plan but peninsula
    Jubany
)

mkdir -p "${READER_DIR}"
cd "${READER_DIR}"

echo "Downloading READER surface data to ${READER_DIR}"
for st in "${STATIONS[@]}"; do
    f="${st}_surface.dat"
    url="${BASE_URL}/${f}"
    if [[ -f "$f" ]]; then
        echo "  [skip] $f (exists)"
    else
        echo "  [get]  $f"
        wget -q -O "$f" "$url" || echo "  [WARN] Failed: $url"
    fi
done

echo "Done. Files in ${READER_DIR}"
