#!/usr/bin/env python3
"""Quick comparison of several sweep_results.csv files in one amplitude plot
(not thesis style). Writes to results/coupled_comparison/ by default.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from viv_analysis.plotting.plot_style import apply_thesis_style
apply_thesis_style()

from viv_analysis.utils import PROJECT_ROOT


def parse_args():
    """Command-line options."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep_csv", nargs="+", required=True,
                   help="Paths to two or more sweep_results.csv files to compare")
    p.add_argument("--label", nargs="+", default=None,
                   help="Legend label per --sweep_csv, same order (default: parent dir name)")
    p.add_argument("--title", default="Closed-loop coupled inference comparison")
    p.add_argument("--out_dir", default=None,
                   help="Output dir for merged CSV/plot (default: results/coupled_comparison)")
    p.add_argument("--out_name", default="comparison",
                   help="Base filename for the merged .csv/.png (default: comparison)")
    return p.parse_args()


def main():
    """Plot the given sweeps together."""
    args = parse_args()
    if len(args.sweep_csv) < 2:
        raise SystemExit("Need at least 2 --sweep_csv paths to compare.")

    paths = [Path(p) for p in args.sweep_csv]
    labels = args.label or [p.parent.name for p in paths]
    if len(labels) != len(paths):
        raise SystemExit("--label count must match --sweep_csv count.")

    dfs = []
    cfd_ref = None
    for path, label in zip(paths, labels):
        if not path.exists():
            raise SystemExit(f"Not found: {path}")
        df = pd.read_csv(path)
        model_cols = [c for c in df.columns if c not in ("Ur", "CFD")]
        if len(model_cols) != 1:
            raise SystemExit(f"{path}: expected exactly one model column, got {model_cols}")
        sub = df[["Ur", "CFD", model_cols[0]]].rename(columns={model_cols[0]: label})
        dfs.append(sub.set_index("Ur"))
        if cfd_ref is None:
            cfd_ref = df.set_index("Ur")["CFD"]
        elif not cfd_ref.equals(df.set_index("Ur")["CFD"]):
            print(f"WARNING: {path} has different CFD reference values/Ur coverage "
                  f"than the first sweep -- comparison may not be apples-to-apples.")

    merged = pd.concat([d.drop(columns="CFD") for d in dfs], axis=1)
    merged.insert(0, "CFD", cfd_ref)
    merged = merged.sort_index().reset_index()

    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "results" / "coupled_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.out_name}.csv"
    merged.to_csv(out_csv, index=False)
    print(f"Saved merged comparison CSV to {out_csv}")

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.plot(merged["Ur"], merged["CFD"], "o-", color="black", lw=2,
            label="CFD (ground truth)", zorder=3)
    markers = ["s", "^", "D", "v", "P", "X"]
    for i, label in enumerate(labels):
        ax.plot(merged["Ur"], merged[label], marker=markers[i % len(markers)],
                linestyle="--", label=label, zorder=2)
    ax.set_xlabel(r"Reduced velocity $U_r$")
    ax.set_ylabel("Steady-state $A/D$")
    ax.set_title(args.title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_png = out_dir / f"{args.out_name}.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved comparison plot to {out_png}")

    print("\nPer-Ur comparison:")
    print(merged.to_string(index=False))

    print("\nMean absolute error vs CFD:")
    for label in labels:
        mae = (merged[label] - merged["CFD"]).abs().mean()
        print(f"  {label}: {mae:.4f}")


if __name__ == "__main__":
    main()
