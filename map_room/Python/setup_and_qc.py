"""
F00: Discover ERA5 input paths and write QC manifest.
Reads paths from environment (ERA5_T2M_DIR, ERA5_MSL_DIR) or params;
writes Data/F00_setup_and_qc/inputs_manifest.json and qc_summary.txt.
"""

import json
import os
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent


def load_params():
    try:
        import yaml
        with open(ROOT / "Const" / "params.yaml") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def main():
    params = load_params()
    eras = params.get("eras", {})
    past = eras.get("past", {})
    present = eras.get("present", {})
    start_year = past.get("start_year", 1948)
    end_year = present.get("end_year", 2025)

    t2m_dir = Path(os.environ.get("ERA5_T2M_DIR", str(ROOT / "Data")))
    msl_dir = Path(os.environ.get("ERA5_MSL_DIR", str(ROOT / "Data")))

    from utils_era5_io import get_era5_t2m_paths, get_era5_msl_paths

    t2m_paths = get_era5_t2m_paths(t2m_dir, start_year, end_year)
    msl_paths = get_era5_msl_paths(msl_dir, start_year, end_year)

    manifest = {
        "era5_t2m_dir": str(t2m_dir),
        "era5_msl_dir": str(msl_dir),
        "year_range": [start_year, end_year],
        "t2m_files": [str(p) for p in t2m_paths],
        "msl_files": [str(p) for p in msl_paths],
        "n_t2m": len(t2m_paths),
        "n_msl": len(msl_paths),
    }

    out_dir = ROOT / "Data" / "F00_setup_and_qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = out_dir / "inputs_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_file}")

    # QC summary (missing recent years e.g. 2022+ is acceptable)
    expected_years = end_year - start_year + 1
    n_ok = min(len(t2m_paths), len(msl_paths))
    qc_lines = [
        "Map room F00 QC summary",
        "=========================",
        f"Year range: {start_year}-{end_year} (expected up to {expected_years} years)",
        f"t2m files found: {len(t2m_paths)}",
        f"msl files found: {len(msl_paths)}",
        "",
        "Note: Missing recent years (2022 and after) is acceptable; pipeline uses available years.",
        "",
        "QC checklist:",
        f"  [{'x' if n_ok >= 70 else ' '}] Time coverage sufficient (missing 2022+ OK)",
        "  [ ] Calendar: leap days present in daily data (document in code)",
        "  [ ] Chunking: use chunks={'time': 365} for dask",
        "  [ ] Variables: t2m (K), msl (Pa) confirmed in NetCDF",
    ]
    qc_file = out_dir / "qc_summary.txt"
    with open(qc_file, "w") as f:
        f.write("\n".join(qc_lines))
    print(f"Wrote {qc_file}")

    if n_ok < 70:
        print("Warning: fewer than 70 years of data; check ERA5_*_DIR and year range.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
