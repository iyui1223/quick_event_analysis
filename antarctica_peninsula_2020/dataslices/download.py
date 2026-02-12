#!/usr/bin/env python3
"""
ERA5 daily stats downloader (monthly files as YYYYMM.nc)

- Dataset: derived-era5-single-levels-daily-statistics
- Variables: 10m u/v wind + 2m temperature (daily_mean) for Antarctic Peninsula slice
- Area: 90°W–40°W, 75°S–65°S (CDS order: N,W,S,E = -65,-90,-75,-40)
- Loops from START_YEAR (env) or 1948; MONTHS (env) e.g. "2" = February only
- Saves to OUTPUT_DIR (env); skips existing files unless FORCE=1
"""

import os
import sys
import time
import calendar
from datetime import date
import cdsapi

DATASET = "derived-era5-single-levels-daily-statistics"

# Antarctic Peninsula: 90°W–40°W, 75°S–65°S → CDS area [North, West, South, East]
AREA_PENINSULA = [-65, -90, -75, -40]

# ---- Config via environment variables (with sensible defaults)
START_YEAR = int(os.environ.get("START_YEAR", "1948"))
END_YEAR = int(os.environ.get("END_YEAR", str(date.today().year)))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.getcwd())
FORCE = os.environ.get("FORCE", "0") == "1"
VARIABLES = os.environ.get("VARIABLES", "10m_u_component_of_wind,10m_v_component_of_wind,2m_temperature").split(",")
DAILY_STAT = os.environ.get("DAILY_STAT", "daily_mean")
TIME_ZONE = os.environ.get("TIME_ZONE", "utc+00:00")
FREQUENCY = os.environ.get("FREQUENCY", "6_hourly")
# MONTHS: comma-separated 1-12, e.g. "2" = February only (default all months)
_months_env = os.environ.get("MONTHS", "")
MONTHS = [int(x.strip()) for x in _months_env.split(",") if x.strip()] if _months_env else list(range(1, 13))
AREA = [int(x) for x in os.environ.get("AREA", "").split(",")] if os.environ.get("AREA") else AREA_PENINSULA
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))

# If END_YEAR is this year, only go up to *this month*
today = date.today()
END_MONTH_LIMIT = today.month if END_YEAR == today.year else 12

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Build a reusable client (reads ~/.cdsapirc)
client = cdsapi.Client()  # respects proxies/cert settings if present

def days_in_month(y: int, m: int) -> list:
    """Return a zero-padded list of days '01'..'28-31' for given y/m."""
    _, ndays = calendar.monthrange(y, m)  # handles leap years
    return [f"{d:02d}" for d in range(1, ndays + 1)]

def month_request_payload(y: int, m: int) -> dict:
    """Construct request for one month (with area slice)."""
    return {
        "product_type": "reanalysis",
        "variable": VARIABLES,
        "year": f"{y}",
        "month": f"{m:02d}",
        "day": days_in_month(y, m),
        "daily_statistic": DAILY_STAT,
        "time_zone": TIME_ZONE,
        "frequency": FREQUENCY,
        "area": AREA,  # [North, West, South, East] degrees
    }

def download_month(y: int, m: int, target_path: str) -> None:
    """Retrieve a single month with retries and backoff."""
    req = month_request_payload(y, m)

    backoff = 10  # seconds
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # You can pass 'target=' directly to retrieve to write the file
            client.retrieve(DATASET, req, target_path)
            return
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            # Print and backoff
            print(f"[{y}-{m:02d}] attempt {attempt}/{MAX_RETRIES} failed: {e}", file=sys.stderr)
            time.sleep(backoff)
            backoff = min(backoff * 2, 600)  # cap backoff at 10 min

def main():
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Years: {START_YEAR}..{END_YEAR} (current year month limit: {END_MONTH_LIMIT if END_YEAR==today.year else 'none'})")
    print(f"Months: {MONTHS}")
    print(f"Variable(s): {VARIABLES}")
    print(f"Area: {AREA}")
    print(f"Daily statistic: {DAILY_STAT}, Time zone: {TIME_ZONE}, Frequency: {FREQUENCY}")
    print(f"Force re-download: {FORCE}, Max retries: {MAX_RETRIES}")

    for y in range(START_YEAR, END_YEAR + 1):
        last_month = END_MONTH_LIMIT if y == END_YEAR else 12
        for m in sorted(set(MONTHS) & set(range(1, last_month + 1))):
            fname = f"{y}{m:02d}.nc"
            target = os.path.join(OUTPUT_DIR, fname)

            if os.path.exists(target) and os.path.getsize(target) > 0 and not FORCE:
                print(f"[SKIP] {fname} exists.")
                continue

            print(f"[GET ] {fname}")
            try:
                download_month(y, m, target)
                print(f"[DONE] {fname}")
            except Exception as e:
                # If a partial file exists (e.g., from interrupted download), remove it to avoid confusion
                if os.path.exists(target) and os.path.getsize(target) == 0:
                    try:
                        os.remove(target)
                    except Exception:
                        pass
                print(f"[FAIL] {fname}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()