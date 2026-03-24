#!/usr/bin/env python3
"""
Analyze temporal observation frequency of READER Antarctic surface station data.

For each station .dat file, computes:
- Obs per day (histogram over time)
- Dominant frequency (1h, 3h, 6h, etc.) and how it varies by period
- Data availability (date range, gaps)

Output: summary table + figure showing frequency by station and time period.

Usage:
  python analyze_reader_frequencies.py --reader-dir /path/to/READER/SURFACE [--out-dir Figs/F07_reader_frequencies]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Columns: Year, Month, Day, Hour (UTC), [Minute], SLP, StnP, Temp, WindSpd, WindDir
# Null = -999. Some stations (e.g. Palmer) omit Minute.
DATA_PAT = re.compile(r"^\s*(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(?:(\d{1,2})\s+|-?\d)")


def load_reader_surface(path: Path) -> pd.DataFrame | None:
    """Load a READER surface .dat file into a DataFrame with datetime index."""
    if not path.exists():
        return None
    lines = path.read_text(errors="replace").splitlines()
    rows = []
    for line in lines:
        m = DATA_PAT.match(line)
        if m:
            g = m.groups()
            y, mo, d, h = int(g[0]), int(g[1]), int(g[2]), int(g[3])
            mi = int(g[4]) if g[4] is not None else 0
            try:
                dt = pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi)
                rows.append(dt)
            except (ValueError, pd.errors.OutOfBoundsDatetime):
                continue
    if not rows:
        return None
    return pd.DataFrame({"datetime": rows}).set_index("datetime").sort_index()


def obs_per_day(series: pd.DatetimeIndex) -> pd.Series:
    """Count observations per calendar day."""
    return series.to_series().resample("D").count()


def dominant_frequency(obs_per_day_series: pd.Series) -> tuple[str, float]:
    """
    Infer dominant observation frequency from obs-per-day distribution.
    Returns (label, fraction) e.g. ("3-hourly", 0.85).
    """
    valid = obs_per_day_series[obs_per_day_series > 0]
    if valid.empty:
        return "unknown", 0.0
    vc = valid.value_counts().sort_values(ascending=False)
    total = vc.sum()
    # Map obs/day to nominal frequency
    freq_map = {
        24: "1-hourly",
        8: "3-hourly",
        6: "4-hourly",
        4: "6-hourly",
        3: "8-hourly",
        2: "12-hourly",
        1: "daily",
    }
    best_n = int(vc.index[0])
    best_frac = vc.iloc[0] / total
    label = freq_map.get(best_n, f"{best_n}/day")
    return label, float(best_frac)


def analyze_station(path: Path) -> dict:
    """Analyze one station file. Returns dict with stats."""
    df = load_reader_surface(path)
    if df is None or len(df) == 0:
        return {"station": path.stem.replace("_surface", ""), "error": "no data"}

    idx = df.index
    obs_day = obs_per_day(idx)
    obs_day_valid = obs_day[obs_day > 0]

    dom_freq, dom_frac = dominant_frequency(obs_day)

    # Frequency distribution (obs per day)
    vc = obs_day_valid.value_counts().sort_index()
    freq_dist = {int(k): v for k, v in vc.items()}

    # Decadal breakdown: dominant frequency per decade
    decades = []
    for decade_start in range(1940, 2030, 10):
        mask = (idx >= f"{decade_start}-01-01") & (idx < f"{decade_start + 10}-01-01")
        sub = idx[mask]
        if len(sub) == 0:
            continue
        sub_obs = obs_per_day(sub)
        f, frac = dominant_frequency(sub_obs)
        decades.append({"decade": f"{decade_start}s", "freq": f, "frac": frac, "n_days": (sub_obs > 0).sum()})

    return {
        "station": path.stem.replace("_surface", ""),
        "start": idx.min().strftime("%Y-%m-%d"),
        "end": idx.max().strftime("%Y-%m-%d"),
        "n_obs": len(df),
        "n_days_with_data": int((obs_day > 0).sum()),
        "dominant_freq": dom_freq,
        "dominant_frac": dom_frac,
        "freq_distribution": freq_dist,
        "decades": decades,
    }


def main():
    ap = argparse.ArgumentParser(description="Analyze READER station temporal frequencies")
    ap.add_argument("--reader-dir", type=Path, default=None, help="Path to READER SURFACE dir")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory for figures/tables")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    reader_dir = args.reader_dir or Path("/lustre/soge1/projects/andante/cenv1201/heavy/READER/SURFACE")
    out_dir = args.out_dir or root / "Figs" / "F07_reader_frequencies"

    if not reader_dir.exists():
        print(f"ERROR: {reader_dir} does not exist. Run download_reader_surface.sh first.")
        return 1

    dat_files = sorted(reader_dir.glob("*_surface.dat"))
    if not dat_files:
        print(f"No *_surface.dat files in {reader_dir}")
        return 1

    results = []
    for p in dat_files:
        r = analyze_station(p)
        results.append(r)

    # Summary table
    rows = []
    for r in results:
        if "error" in r:
            rows.append({"Station": r["station"], "Status": r["error"], "Start": "", "End": "", "N_obs": "", "Dominant_freq": "", "Frac": ""})
        else:
            rows.append({
                "Station": r["station"],
                "Status": "ok",
                "Start": r["start"],
                "End": r["end"],
                "N_obs": r["n_obs"],
                "Dominant_freq": r["dominant_freq"],
                "Frac": f"{r['dominant_frac']:.0%}",
            })
    df_summary = pd.DataFrame(rows)
    print("\n=== READER station temporal frequency summary ===\n")
    print(df_summary.to_string(index=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "reader_frequency_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Decadal frequency table (for stations with data)
    dec_rows = []
    for r in results:
        if "error" in r or "decades" not in r:
            continue
        for d in r["decades"]:
            dec_rows.append({
                "Station": r["station"],
                "Decade": d["decade"],
                "Freq": d["freq"],
                "Frac": f"{d['frac']:.0%}",
                "N_days": d["n_days"],
            })
    if dec_rows:
        df_dec = pd.DataFrame(dec_rows)
        dec_path = out_dir / "reader_frequency_by_decade.csv"
        df_dec.to_csv(dec_path, index=False)
        print(f"Saved: {dec_path}")

    # Figure: obs-per-day distribution per station
    fig, axes = plt.subplots(4, 3, figsize=(12, 12), sharex=False)
    axes = axes.flatten()
    for i, r in enumerate(results):
        ax = axes[i]
        if "error" in r:
            ax.text(0.5, 0.5, f"{r['station']}\n{r['error']}", ha="center", va="center")
            ax.set_xlim(0, 25)
            continue
        path = reader_dir / f"{r['station']}_surface.dat"
        df = load_reader_surface(path)
        if df is None or df.empty:
            ax.text(0.5, 0.5, f"{r['station']}\nno data", ha="center", va="center")
            continue
        obs_day = obs_per_day(df.index)
        vc = obs_day[obs_day > 0].value_counts().sort_index()
        ax.bar(vc.index, vc.values, color="steelblue", edgecolor="navy", alpha=0.8)
        ax.axvline(8, color="red", ls="--", alpha=0.7, label="3-hourly (8/day)")
        ax.axvline(4, color="orange", ls="--", alpha=0.7, label="6-hourly (4/day)")
        ax.set_title(f"{r['station']} ({r['start'][:4]}-{r['end'][:4]})\n{r['dominant_freq']} ({r['dominant_frac']:.0%})")
        ax.set_xlabel("Observations per day")
        ax.set_ylabel("Number of days")
        ax.legend(fontsize=7)
    for j in range(len(results), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("READER Peninsula stations: distribution of observations per day", fontsize=12)
    fig.tight_layout()
    fig_path = out_dir / "reader_obs_per_day_by_station.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # Figure: frequency over time (heatmap or timeline)
    stations_ok = [r for r in results if "error" not in r and "decades" in r]
    if stations_ok:
        dec_vals = sorted(set(d["decade"] for r in stations_ok for d in r["decades"]))
        freq_order = ["1-hourly", "3-hourly", "4-hourly", "6-hourly", "8-hourly", "12-hourly", "daily", "unknown"]
        freq_map = {f: i for i, f in enumerate(freq_order)}

        arr = np.full((len(stations_ok), len(dec_vals)), np.nan)
        for i, r in enumerate(stations_ok):
            dec_dict = {d["decade"]: d["freq"] for d in r["decades"]}
            for j, dec in enumerate(dec_vals):
                if dec in dec_dict:
                    arr[i, j] = freq_map.get(dec_dict[dec], len(freq_order) - 1)

        fig2, ax2 = plt.subplots(figsize=(14, 6))
        im = ax2.imshow(arr, aspect="auto", cmap="viridis", vmin=0, vmax=len(freq_order) - 1)
        ax2.set_yticks(range(len(stations_ok)))
        ax2.set_yticklabels([r["station"] for r in stations_ok], fontsize=9)
        ax2.set_xticks(range(len(dec_vals)))
        ax2.set_xticklabels(dec_vals, rotation=45)
        ax2.set_xlabel("Decade")
        cbar = plt.colorbar(im, ax=ax2, ticks=range(len(freq_order)))
        cbar.ax.set_yticklabels(freq_order)
        cbar.set_label("Dominant obs frequency")
        ax2.set_title("READER stations: dominant observation frequency by decade")
        fig2.tight_layout()
        fig2_path = out_dir / "reader_frequency_by_decade.png"
        fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {fig2_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
