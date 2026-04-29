#!/usr/bin/env python3
"""
F10: Download minimal ERA5 hourly single-level inputs for FPT2000.

The download area is derived from the F04 peninsula masks by default. It covers
only the west/east peninsula land masks plus a small buffer, then requests all
February days at the UTC hour nearest 15 local time over the western AP.
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
from pathlib import Path
from typing import Iterable

import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
MASKS_DEFAULT = ROOT / "Data" / "F04_peninsula_domains" / "all_domain_masks.nc"
OUT_DIR_DEFAULT = ROOT / "Data" / "F10_fpt2000_westAP" / "raw"

DEFAULT_DOMAINS = ("pen_west_slope", "pen_east_slope")
DEFAULT_VARIABLES = (
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "100m_u_component_of_wind",
)


def _std_coords(ds: xr.Dataset) -> xr.Dataset:
    renames = {}
    for name in set(ds.coords) | set(ds.dims):
        low = str(name).lower()
        if "lat" in low and name != "lat" and "lat" not in ds.coords:
            renames[name] = "lat"
        elif "lon" in low and name != "lon" and "lon" not in ds.coords:
            renames[name] = "lon"
        elif ("time" in low or "valid" in low) and name != "time" and "time" not in ds.coords:
            renames[name] = "time"
    if renames:
        ds = ds.rename(renames)
    if "lat" in ds.coords and ds.sizes.get("lat", 0) > 1 and bool(ds.lat[0] > ds.lat[-1]):
        ds = ds.sortby("lat")
    if "lon" in ds.coords and float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon % 360)).sortby("lon")
    return ds


def _round_down(value: float, step: float) -> float:
    return math.floor(value / step) * step


def _round_up(value: float, step: float) -> float:
    return math.ceil(value / step) * step


def _lon360_to_180(value: float) -> float:
    return ((value + 180.0) % 360.0) - 180.0


def _parse_domains(text: str) -> tuple[str, ...]:
    domains = tuple(tok.strip() for tok in text.split(",") if tok.strip())
    if not domains:
        raise ValueError("No domains parsed.")
    return domains


def _parse_variables(text: str) -> list[str]:
    vals = [tok.strip() for tok in text.split(",") if tok.strip()]
    if not vals:
        raise ValueError("No variables parsed.")
    return vals


def _area_from_masks(
    masks_path: Path,
    domains: Iterable[str],
    buffer_deg: float,
    round_deg: float,
) -> tuple[list[float], dict[str, float]]:
    if not masks_path.exists():
        raise FileNotFoundError(f"Masks not found: {masks_path}")

    masks = _std_coords(xr.open_dataset(masks_path))
    try:
        missing = [d for d in domains if d not in masks]
        if missing:
            raise ValueError(f"Missing mask variables in {masks_path}: {missing}")

        union = None
        for dom in domains:
            m = masks[dom] > 0
            union = m if union is None else (union | m)
        if union is None or int(union.sum()) == 0:
            raise ValueError(f"Selected mask domains contain no grid cells: {domains}")

        valid_lat = masks.lat.where(union.any("lon"), drop=True)
        valid_lon = masks.lon.where(union.any("lat"), drop=True)
        lat_south = float(valid_lat.min()) - buffer_deg
        lat_north = float(valid_lat.max()) + buffer_deg
        lon_west_360 = float(valid_lon.min()) - buffer_deg
        lon_east_360 = float(valid_lon.max()) + buffer_deg

        lat_south = max(-90.0, _round_down(lat_south, round_deg))
        lat_north = min(90.0, _round_up(lat_north, round_deg))
        lon_west_360 = _round_down(lon_west_360, round_deg) % 360.0
        lon_east_360 = _round_up(lon_east_360, round_deg) % 360.0

        lon_west = _lon360_to_180(lon_west_360)
        lon_east = _lon360_to_180(lon_east_360)
        if lon_west > lon_east:
            raise ValueError(
                "Mask-derived longitude range crosses the dateline; pass --area manually."
            )

        area = [lat_north, lon_west, lat_south, lon_east]
        meta = {
            "mask_lat_south": float(valid_lat.min()),
            "mask_lat_north": float(valid_lat.max()),
            "mask_lon_west_360": float(valid_lon.min()),
            "mask_lon_east_360": float(valid_lon.max()),
            "download_north": area[0],
            "download_west": area[1],
            "download_south": area[2],
            "download_east": area[3],
            "buffer_deg": buffer_deg,
            "round_deg": round_deg,
        }
        return area, meta
    finally:
        masks.close()


def _days_in_february(year: int) -> list[str]:
    ndays = calendar.monthrange(year, 2)[1]
    return [f"{day:02d}" for day in range(1, ndays + 1)]


def _build_request(
    year: int,
    variables: list[str],
    area: list[float],
    utc_hour: int,
    legacy_format_key: bool,
) -> dict:
    request = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [str(year)],
        "month": ["02"],
        "day": _days_in_february(year),
        "time": [f"{utc_hour:02d}:00"],
        "area": area,
    }
    if legacy_format_key:
        request["format"] = "netcdf"
    else:
        request["data_format"] = "netcdf"
        request["download_format"] = "unarchived"
    return request


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download minimal ERA5 F10 inputs over the F04 peninsula-mask bounding box."
    )
    ap.add_argument("--masks", type=Path, default=MASKS_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--dataset", default="reanalysis-era5-single-levels")
    ap.add_argument("--start-year", type=int, default=1959)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--utc-hour", type=int, default=19,
                    help="UTC hour used as the first-pass 15 local time proxy.")
    ap.add_argument("--buffer-deg", type=float, default=0.5,
                    help="Buffer added around the F04 peninsula mask bbox.")
    ap.add_argument("--round-deg", type=float, default=0.25,
                    help="Round the download bbox outward to this degree step.")
    ap.add_argument("--domains", default=",".join(DEFAULT_DOMAINS),
                    help="Comma-separated F04 mask variables used to derive the bbox.")
    ap.add_argument("--variables", default=",".join(DEFAULT_VARIABLES),
                    help="Comma-separated CDS variable names.")
    ap.add_argument("--area", nargs=4, type=float, metavar=("NORTH", "WEST", "SOUTH", "EAST"),
                    help="Manual CDS area override in degrees north/west/south/east.")
    ap.add_argument("--legacy-format-key", action="store_true",
                    help="Use legacy CDS request key format=netcdf instead of data_format/download_format.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print/write request metadata without contacting CDS.")
    args = ap.parse_args()

    if not (0 <= args.utc_hour <= 23):
        raise ValueError("--utc-hour must be between 0 and 23")
    if args.start_year > args.end_year:
        raise ValueError("--start-year cannot exceed --end-year")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    domains = _parse_domains(args.domains)
    variables = _parse_variables(args.variables)

    if args.area:
        area = list(args.area)
        bbox_meta = {
            "download_north": area[0],
            "download_west": area[1],
            "download_south": area[2],
            "download_east": area[3],
            "manual_area": True,
        }
    else:
        area, bbox_meta = _area_from_masks(
            masks_path=args.masks,
            domains=domains,
            buffer_deg=args.buffer_deg,
            round_deg=args.round_deg,
        )
        bbox_meta["manual_area"] = False

    metadata = {
        "dataset": args.dataset,
        "years": [args.start_year, args.end_year],
        "month": "02",
        "utc_hour": args.utc_hour,
        "local_time_note": "19 UTC is used as a first-pass proxy for 15 local time near 60W.",
        "variables": variables,
        "domains": list(domains),
        "masks": str(args.masks),
        "area": area,
        "bbox": bbox_meta,
    }
    _write_json(args.out_dir / "download_request_metadata.json", metadata)

    print("ERA5 F10 download area [N, W, S, E]:", area)
    print("Variables:", ", ".join(variables))
    print(f"Years: {args.start_year}-{args.end_year}; February {args.utc_hour:02d}:00 UTC")

    first_request = _build_request(
        year=args.start_year,
        variables=variables,
        area=area,
        utc_hour=args.utc_hour,
        legacy_format_key=args.legacy_format_key,
    )
    _write_json(args.out_dir / f"request_example_{args.start_year}.json", first_request)

    if args.dry_run:
        print("Dry run only; no CDS request submitted.")
        return

    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError(
            "cdsapi is not installed in the active Python environment. "
            "Install/configure cdsapi or rerun with --dry-run to inspect the request."
        ) from exc

    client = cdsapi.Client()
    for year in range(args.start_year, args.end_year + 1):
        out_path = args.out_dir / f"era5_fpt2000_inputs_feb{args.utc_hour:02d}utc_{year}.nc"
        if out_path.exists() and not args.overwrite:
            print(f"[{year}] exists, skip: {out_path}")
            continue
        request = _build_request(
            year=year,
            variables=variables,
            area=area,
            utc_hour=args.utc_hour,
            legacy_format_key=args.legacy_format_key,
        )
        print(f"[{year}] retrieving -> {out_path}")
        client.retrieve(args.dataset, request, str(out_path))

    print("Done.")


if __name__ == "__main__":
    main()
