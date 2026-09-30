#!/usr/bin/env python3
r"""Two closed-loop sweeps in one amplitude-response figure, e.g. dimensional vs
nondimensional inputs (thesis Fig. 5.10).

Example:
    PYTHONPATH=src python -m viv_analysis.plotting.regenerate_ablation_comparison_plot \
        --primary_coupled_eval_dir <sweep A> --primary_model_column <col> --primary_label <label> \
        --secondary_coupled_eval_dir <sweep B> --secondary_model_column <col> --secondary_label <label> \
        --output_dir <folder>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from viv_analysis.plotting.plot_style import apply_thesis_style
apply_thesis_style()

from viv_analysis.plotting.thesis_plots import plot_amplitude_response_thesis
from viv_analysis.utils import PROJECT_ROOT


def _load_sweep(coupled_eval_dir: str, model_column: str) -> pd.DataFrame:
    """Read sweep_results.csv from a sweep folder."""
    csv_path = PROJECT_ROOT / "results" / coupled_eval_dir / "sweep_results.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found.")
    df = pd.read_csv(csv_path)
    if model_column not in df.columns:
        raise SystemExit(
            f"--model_column '{model_column}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}")
    return df[["Ur", "CFD", model_column]]


def main():
    """Merge the two sweeps and plot them together."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--primary_coupled_eval_dir", required=True)
    p.add_argument("--primary_model_column", required=True)
    p.add_argument("--primary_label", required=True)
    p.add_argument("--secondary_coupled_eval_dir", required=True)
    p.add_argument("--secondary_model_column", required=True)
    p.add_argument("--secondary_label", required=True)
    p.add_argument("--secondary_color", default="orange", choices=["orange", "green"])
    p.add_argument("--output_dir", required=True,
                   help="results/-relative directory to write thesis_figures/ under "
                        "(this is a merge of two existing runs, so it doesn't belong "
                        "to either one's own directory).")
    p.add_argument("--out_name", default="amplitude_response_comparison")
    p.add_argument("--dataset_note", default=None)
    p.add_argument("--font_scale", type=float, default=1.0,
                   help="Font-size multiplier for the figure. Pass "
                        "1/display_fraction, e.g. 1/0.8 for 0.8\\linewidth.")
    args = p.parse_args()

    primary = _load_sweep(args.primary_coupled_eval_dir, args.primary_model_column)
    secondary = _load_sweep(args.secondary_coupled_eval_dir, args.secondary_model_column)

    if set(primary["Ur"]) != set(secondary["Ur"]):
        raise SystemExit(
            f"Ur grids differ between the two sweeps: "
            f"primary-only={sorted(set(primary['Ur']) - set(secondary['Ur']))}, "
            f"secondary-only={sorted(set(secondary['Ur']) - set(primary['Ur']))}")

    merged = primary.merge(secondary, on="Ur", suffixes=("", "_secondary"))
    mismatch = merged[(merged["CFD"] - merged["CFD_secondary"]).abs() > 1e-9]
    if not mismatch.empty:
        raise SystemExit(
            f"CFD reference disagrees between the two sweeps at Ur="
            f"{mismatch['Ur'].tolist()} -- they were not run against the same "
            f"reference; refusing to plot them as if they were.")

    merged = merged.drop(columns=["CFD_secondary"])

    thesis_dir = PROJECT_ROOT / "results" / args.output_dir / "thesis_figures"
    r = plot_amplitude_response_thesis(
        merged,
        model_column=args.primary_model_column,
        model_label=args.primary_label,
        output_dir=thesis_dir,
        literature_columns={
            args.secondary_model_column: {"label": args.secondary_label, "color": args.secondary_color},
        },
        dataset_note=args.dataset_note,
        out_name=args.out_name,
        font_scale=args.font_scale,
    )
    print(f"wrote {r['pdf_path']}")


if __name__ == "__main__":
    main()
