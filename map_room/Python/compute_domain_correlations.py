"""
F06: Preprocess domain time series and compute sliding-window bivariate correlations.

Steps:
  0. Remove seasonal cycle (daily climatology)
  1. Optional detrending (linear, ENSO, SAM, PDO) - switchable
  2. Apply DJF seasonal filter (2-tanh style)
  3. Sliding 10-year window correlation, 5-year step, with optional day lags
  4. Save to CSV

Indices: use centralized analogue Const (PSL .data format: year + 12 monthly values).
  Run analogue/Sh/index_scatter.sh to download if missing.
"""

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_params(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def djf_weight(doy: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """
    Smooth 0-1-0 seasonal filter emphasizing austral summer (DJF).
    Uses 2 tanh ramps: one at Dec 1, one at Mar 1.
    doy: day-of-year 1-366.
    """
    # Ramp up around Dec 1 (doy 335), ramp down around Mar 1 (doy 60)
    ramp_up = 0.5 * (1 + np.tanh((doy - 350.0) / sigma))
    ramp_down = 0.5 * (1 - np.tanh((doy - 60.0) / sigma))
    return ramp_up + ramp_down


def plot_djf_filter(sigma: float, out_path: Path) -> None:
    """Visualize the DJF filter for user tuning."""
    doy = np.arange(1, 367)
    w = djf_weight(doy, sigma)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(doy, w)
    ax.axvspan(1, 90, alpha=0.2, color="blue", label="Jan-Feb")
    ax.axvspan(335, 366, alpha=0.2, color="blue", label="Dec")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Weight")
    ax.set_title(f"DJF seasonal filter (2-tanh, sigma={sigma} days)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 366)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved filter viz → {out_path}")


def remove_seasonal(da: xr.DataArray) -> xr.DataArray:
    """Remove daily climatology (seasonal cycle)."""
    clim = da.groupby(da.time.dt.dayofyear).mean(dim="time")
    return da.groupby(da.time.dt.dayofyear) - clim


def _parse_psl_year_month(path: Path) -> dict[tuple[int, int], float]:
    """
    Parse PSL .data format: lines with 4-digit year + 12 monthly values.
    Returns dict[(year, month)] -> value. -99.99 / -999 treated as NaN.
    """
    vals: dict[tuple[int, int], float] = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.match(r"^\s*(\d{4})\s+(.+)$", line.strip())
            if not m:
                continue
            year = int(m.group(1))
            tokens = re.split(r"\s+", m.group(2).strip())
            nums = []
            for t in tokens:
                t = t.replace("*", "").replace("NA", "nan")
                try:
                    v = float(t)
                except ValueError:
                    continue
                if v <= -99:
                    v = np.nan
                nums.append(v)
            if len(nums) >= 12:
                for month, v in enumerate(nums[:12], start=1):
                    vals[(year, month)] = v
    return vals


def load_index(path: Path | str) -> pd.Series | None:
    """
    Load climate index. Supports:
    - PSL .data format (year + 12 monthly cols) → monthly dates (15th)
    - CSV with columns date, value
    Returns Series indexed by date.
    """
    path = Path(path) if path else None
    if not path or not path.exists():
        return None
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "date" not in df.columns or "value" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["value"]
    # PSL .data format
    vals = _parse_psl_year_month(path)
    if not vals:
        return None
    dates = [pd.Timestamp(y, m, 15) for (y, m) in sorted(vals)]
    series = pd.Series([vals[(d.year, d.month)] for d in dates], index=dates)
    return series


def detrend_ts(
    da: xr.DataArray,
    linear: bool,
    enso: pd.Series | None,
    sam: pd.Series | None,
    pdo: pd.Series | None,
) -> xr.DataArray:
    """
    Regress out selected modes. Uses OLS: y = a + b*x1 + ... + residuals.
    Indices are interpolated to daily if monthly.
    """
    out = da.values.copy()
    times = pd.DatetimeIndex(da.time.values)
    n = len(times)

    predictors = []
    if linear:
        predictors.append(np.arange(n, dtype=float))
    for name, idx in [("enso", enso), ("sam", sam), ("pdo", pdo)]:
        if idx is None:
            continue
        idx_daily = idx.reindex(times, method="nearest").values
        if np.any(np.isfinite(idx_daily)):
            predictors.append(idx_daily)

    if not predictors:
        return da

    X = np.column_stack([np.ones(n)] + [np.asarray(p, dtype=float) for p in predictors])
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(out)
    if valid.sum() < 10:
        return da

    Xv, yv = X[valid], out[valid]
    beta = np.linalg.lstsq(Xv, yv, rcond=None)[0]
    fit = X @ beta
    out = out - fit
    return xr.DataArray(out, coords=da.coords, dims=da.dims)


def apply_djf_filter(da: xr.DataArray, sigma: float) -> xr.DataArray:
    """Weight by DJF filter and return (masked non-DJF as NaN or just weighted for corr)."""
    doy = da.time.dt.dayofyear.values
    w = djf_weight(doy, sigma)
    weighted = da * w
    # For correlation we want to focus on DJF; zero weight → exclude (mask)
    return weighted.where(w > 0.01)


def sliding_correlation(
    ds: xr.Dataset,
    var: str,
    domains: list[str],
    window_years: int,
    step_years: int,
    lag_days: list[int],
) -> pd.DataFrame:
    """Compute correlation matrix for each sliding window. Returns long-form CSV-friendly df."""
    years = np.unique(ds.time.dt.year.values)
    y0, y1 = int(years.min()), int(years.max())
    rows = []

    for start in range(y0, y1 - window_years + 2, step_years):
        end = start + window_years
        mask = (ds.time.dt.year >= start) & (ds.time.dt.year < end)
        sub = ds.sel(time=mask)
        if sub.sizes["time"] < 365:  # need at least 1 year of data
            continue

        for (d1, d2) in itertools.combinations(domains, 2):
            v1 = f"{var}_{d1}"
            v2 = f"{var}_{d2}"
            if v1 not in sub or v2 not in sub:
                continue
            a = sub[v1]
            b = sub[v2]

            for lag in lag_days:
                # Positive lag compares domain_a(t) with domain_b(t + lag).
                b_lagged = b.shift(time=-lag) if lag != 0 else b
                av = a.values
                bv = b_lagged.values
                valid = np.isfinite(av) & np.isfinite(bv)
                if valid.sum() < 100:
                    continue
                r = np.corrcoef(av[valid], bv[valid])[0, 1]
                rows.append({
                    "var": var,
                    "domain_a": d1,
                    "domain_b": d2,
                    "lag_days": int(lag),
                    "window_start": start,
                    "window_end": end - 1,
                    "correlation": r,
                    "n_valid": int(valid.sum()),
                })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--plot-filter-only", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    params_path = Path(args.params) if args.params else root / "Const" / "params.yaml"
    params = load_params(params_path)
    corr_cfg = params.get("correlation", {})

    out_dir = Path(args.out_dir) if args.out_dir else root / "Data" / "F06_domain_correlations"
    fig_dir = root / "Figs" / "F06_domain_correlations"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    input_nc = Path(args.input) if args.input else root / corr_cfg.get("input_nc", "Data/F05_domain_timeseries/domain_timeseries.nc")
    sigma = corr_cfg.get("djf_filter_sigma", 15.0)
    window_years = corr_cfg.get("window_years", 10)
    step_years = corr_cfg.get("step_years", 5)
    lag_days = corr_cfg.get("lag_days", [-3, -2, -1, 0, 1, 2, 3])
    detrend_cfg = corr_cfg.get("detrend", {})
    indices_cfg = corr_cfg.get("indices", {})

    # Plot filter
    plot_djf_filter(sigma, fig_dir / "djf_filter.png")
    if args.plot_filter_only:
        return

    if not input_nc.exists():
        print(f"Input not found: {input_nc}")
        return

    ds = xr.open_dataset(input_nc)
    domain_suffixes = ["west_ocean", "east_ocean", "inland", "pen_west_slope", "pen_east_slope"]
    vars_in = [v for v in ds.data_vars if any(d in v for d in domain_suffixes)]
    base_vars = list({v.split("_")[0] for v in vars_in})
    base_vars = [v for v in ["t2m", "msl"] if v in base_vars]

    # Load indices from centralized analogue Const (indices_dir)
    idx_dir = Path(corr_cfg.get("indices_dir", "")) or root / "Data" / "indices"
    enso = load_index(idx_dir / p) if (p := indices_cfg.get("enso")) else None
    sam = load_index(idx_dir / p) if (p := indices_cfg.get("sam")) else None
    pdo = load_index(idx_dir / p) if (p := indices_cfg.get("pdo")) else None

    processed = {}
    for bv in base_vars:
        doms = [d for d in domain_suffixes if f"{bv}_{d}" in ds]
        if not doms:
            continue
        da_dict = {d: ds[f"{bv}_{d}"] for d in doms}
        out_ds = xr.Dataset()
        for d in doms:
            da = da_dict[d]
            da = remove_seasonal(da)
            da = detrend_ts(
                da,
                linear=detrend_cfg.get("linear", False),
                enso=enso if detrend_cfg.get("enso") else None,
                sam=sam if detrend_cfg.get("sam") else None,
                pdo=pdo if detrend_cfg.get("pdo") else None,
            )
            da = apply_djf_filter(da, sigma)
            out_ds[f"{bv}_{d}"] = da
        processed[bv] = out_ds

    # Correlations
    all_dfs = []
    for bv, sub in processed.items():
        doms = [k.replace(f"{bv}_", "", 1) for k in sub.data_vars if k.startswith(f"{bv}_")]
        df = sliding_correlation(sub, bv, doms, window_years, step_years, lag_days)
        all_dfs.append(df)

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        csv_path = out_dir / "domain_correlations.csv"
        result.to_csv(csv_path, index=False)
        print(f"Saved → {csv_path}  ({len(result)} rows)")

    ds.close()


if __name__ == "__main__":
    main()
