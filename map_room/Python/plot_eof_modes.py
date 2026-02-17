"""
Plot EOF modes: spatial pattern (polar map) and PC time series.
Reads Data/F02_eof/EOFs.nc or Data/F03_eof_tapered_djf/EOFs.nc.
Writes Figs/F02_eof/mode_01.png ... mode_04.png (or F03 path).
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from utils_plot_polar import plot_eof_mode_map

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="Plot EOF modes")
    ap.add_argument("--eof-nc", default=None, help="Path to EOFs.nc")
    ap.add_argument("--fig-dir", default=None, help="Output figure dir (default: Figs/F02_eof or F03)")
    ap.add_argument("--step", default="F02_eof", choices=["F02_eof", "F03_eof_tapered_djf"], help="Step id for default paths")
    args = ap.parse_args()
    data_dir = Path(os.environ.get("DATA_DIR", str(ROOT / "Data")))
    figs_dir = Path(os.environ.get("FIGS_DIR", str(ROOT / "Figs")))
    eof_nc = Path(args.eof_nc or str(data_dir / args.step / "EOFs.nc"))
    fig_dir = Path(args.fig_dir or str(figs_dir / args.step))
    if not eof_nc.exists():
        raise FileNotFoundError(f"EOFs file not found: {eof_nc}")
    ds = xr.open_dataset(eof_nc)
    eof = ds["eof_pattern"]
    pcs = ds["pc"]
    var_frac = ds["variance_fraction"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_modes = eof.sizes.get("mode", 4)
    for i in range(1, int(n_modes) + 1):
        mode_da = eof.sel(mode=i)
        vf = float(var_frac.sel(mode=i).values) * 100
        plot_eof_mode_map(
            mode_da,
            i,
            fig_dir / f"mode_{i:02d}.png",
            title=f"EOF mode {i} ({vf:.1f}% var)",
            units="K",
        )
    # Combined PC time series (optional): one figure with 4 subplots
    fig, axes = plt.subplots(n_modes, 1, figsize=(10, 2 * n_modes), sharex=True)
    if n_modes == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        m = i + 1
        pcs.sel(mode=m).plot(ax=ax, color="k", linewidth=0.5)
        ax.set_ylabel(f"PC{m}")
        ax.set_title(f"Mode {m} ({float(var_frac.sel(mode=m).values)*100:.1f}%)")
    plt.suptitle("PC time series")
    plt.tight_layout()
    plt.savefig(fig_dir / "pc_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()
    ds.close()
    print("Plots written to", fig_dir)


if __name__ == "__main__":
    main()
