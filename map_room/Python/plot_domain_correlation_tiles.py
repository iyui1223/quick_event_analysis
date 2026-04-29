#!/usr/bin/env python3
"""
Plot domain correlations over time as tile (heatmap) plots.

Reads domain_correlations.csv and produces heatmaps where:
  - Rows: domain pairs (6 choose 2 = 15, or however many in the data)
  - Columns: time windows (e.g. 1948–1957, 1953–1962, …)
  - Colour/text: correlation value (-1 to 1)
  - Separate output files for each lag_days value when present

Creates one figure per variable (t2m, msl, etc.) with a diverging colormap
centered at 0.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def make_pair_label(a: str, b: str) -> str:
    """Short label for domain pair (a–b)."""
    return f"{a}\n{b}"


def plot_correlation_tiles(
    df: pd.DataFrame,
    var: str,
    out_path: Path,
    lag_days: int = 0,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "RdBu_r",
    figsize: tuple | None = None,
    fontsize: int = 9,
    annotate: bool = True,
) -> None:
    """
    Plot correlation values as tiles: domain pairs (rows) vs time windows (cols).
    """
    sub = df[df["var"] == var].copy()
    if "lag_days" in sub.columns:
        sub = sub[sub["lag_days"] == lag_days].copy()
    if sub.empty:
        return

    sub["pair"] = sub.apply(lambda r: make_pair_label(r["domain_a"], r["domain_b"]), axis=1)
    sub["window"] = sub["window_start"].astype(str) + "–" + sub["window_end"].astype(str)

    pivot = sub.pivot_table(
        index="pair",
        columns="window",
        values="correlation",
        aggfunc="first",
    )

    # Stable ordering: lexicographic by domain_a, domain_b
    pair_order = sub.drop_duplicates("pair").sort_values(["domain_a", "domain_b"])["pair"].tolist()
    pivot = pivot.reindex([p for p in pair_order if p in pivot.index])

    if figsize is None:
        n_cols = max(1, len(pivot.columns))
        n_rows = max(1, len(pivot.index))
        figsize = (max(16, 0.95 * n_cols + 4), max(9, 0.55 * n_rows + 3))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=fontsize)
    ax.set_xlabel("Time window (10-yr, 5-yr step)")
    ax.set_ylabel("Domain pair")
    lag_title = f", lag {lag_days:+d} day" if lag_days else ", lag 0 days"
    ax.set_title(f"Domain correlations ({var}{lag_title})")

    # Colour bar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlation")

    # Optional: annotate tiles with correlation values
    if annotate and pivot.size <= 400:  # skip only if the grid becomes too dense
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    txt = f"{val:.2f}"
                    tc = "white" if abs(val) > 0.5 else "black"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=tc)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot domain correlation tiles")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Data/F06_domain_correlations/domain_correlations.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as input, or Figs/F06_domain_correlations)",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Do not annotate tiles with correlation values",
    )
    args = parser.parse_args()

    inp = args.input.resolve()
    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    if args.out_dir is not None:
        out_dir = Path(args.out_dir).resolve()
    else:
        # Default: Figs/F06_domain_correlations (project_root/Figs/...)
        proj = inp.parent.parent.parent if "F06_domain_correlations" in str(inp) else Path.cwd()
        out_dir = Path(proj) / "Figs" / "F06_domain_correlations"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    if "lag_days" not in df.columns:
        df["lag_days"] = 0
    vars_ = df["var"].unique()

    def lag_label(lag: int) -> str:
        if lag < 0:
            return f"lag_m{abs(lag)}d"
        if lag > 0:
            return f"lag_p{lag}d"
        return "lag_0d"

    for var in sorted(vars_):
        for lag in sorted(df["lag_days"].dropna().astype(int).unique()):
            if lag == 0:
                out_path = out_dir / f"correlation_tiles_{var}.png"
            else:
                out_path = out_dir / f"correlation_tiles_{var}_{lag_label(lag)}.png"
            plot_correlation_tiles(df, var, out_path, lag_days=lag, annotate=not args.no_annotate)


if __name__ == "__main__":
    main()
