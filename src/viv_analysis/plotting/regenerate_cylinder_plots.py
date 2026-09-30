#!/usr/bin/env python3
r"""Closed-loop cylinder figures from an evaluate_all.py sweep
(thesis Fig. 5.3, 5.8, 5.9 and Appendix E): time series per case and the
amplitude response over Ur. Reads the sweep folder
results/<model_subdir>_coupled_eval/ and writes to its thesis_figures/.

Example:
    PYTHONPATH=src python -m viv_analysis.plotting.regenerate_cylinder_plots \
        --model_subdir <model folder>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viv_analysis.plotting.plot_style import apply_thesis_style
apply_thesis_style()

from viv_analysis.config import cylinder200_U
from viv_analysis.evaluate_all import load_full_cfd_df
from viv_analysis.plotting.thesis_plots import (
    plot_amplitude_response_thesis, plot_coupled_thesis, ur_tag,
)
from viv_analysis.utils import PROJECT_ROOT, format_ur_label, present_model_label

_UR_RE = re.compile(r"coupled_cylinder200_Ur([0-9.]+)_(.+)\.npz$")


def _load_case(npz_path: Path, cfd_df: pd.DataFrame):
    """Load one coupled npz."""
    m = _UR_RE.match(npz_path.name)
    if not m:
        return None
    Ur = float(m.group(1))
    d = np.load(npz_path, allow_pickle=True)
    t, h, CL, D = d["t"], d["h"], d["cl"], float(d["D"])
    t_handoff = None
    receipt_path = npz_path.with_suffix(".receipt.json")
    if receipt_path.exists():
        t_handoff = json.load(open(receipt_path)).get("handoff_time_s")

    case_label = format_ur_label(Ur)
    case_df = cfd_df[cfd_df["case"].astype(str) == case_label]
    if case_df.empty:
        print(f"  [skip] no CFD case found for Ur={Ur} in cfd_df (npz={npz_path.name})")
        return None
    return dict(Ur=Ur, t=t, h=h, CL=CL, D=D, t_handoff=t_handoff,
               CFD_t=case_df["time"].to_numpy(), CFD_h=case_df["disp"].to_numpy(),
               CFD_cl=case_df["cl"].to_numpy())


def replot_timeseries(coupled_eval_dir: Path, cfd_df: pd.DataFrame, dataset: str = "cylinder200") -> list[Path]:
    """Quick-look time series (non-thesis style)."""
    written = []
    for npz_path in sorted(coupled_eval_dir.glob("coupled_cylinder200_Ur*.npz")):
        case = _load_case(npz_path, cfd_df)
        if case is None:
            continue
        tag = _UR_RE.match(npz_path.name).group(2)
        t, h, CL, D = case["t"], case["h"], case["CL"], case["D"]

        fig, axes = plt.subplots(2, 1, figsize=(13, 6), constrained_layout=True)
        axes[0].plot(case["CFD_t"], case["CFD_h"] / D, lw=0.8, color="black", alpha=0.7, label="CFD")
        axes[0].plot(t, h / D, lw=1, color="tab:blue", alpha=0.9,
                     label=present_model_label(dataset, "GRU coupled"))
        if case["t_handoff"] is not None:
            axes[0].axvline(case["t_handoff"], color="green", ls="--", lw=1, alpha=0.7,
                            label=f"handoff t={case['t_handoff']:.1f}s")
        axes[0].axhline(0, color="0.8", lw=1, ls=":")
        axes[0].set_ylabel(r"$h/D$")
        axes[0].set_xlim(0, 200)
        axes[0].legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            frameon=False,
            borderaxespad=0.0,
            columnspacing=1.2,
            handlelength=1.8,
            handletextpad=0.5,
        )
        axes[0].grid(True, alpha=0.3)


        axes[1].plot(case["CFD_t"], case["CFD_cl"], lw=0.8, color="black", alpha=0.7, label="CFD")
        axes[1].plot(t, CL, lw=1, color="tab:orange", alpha=0.9,
                     label=present_model_label(dataset, "GRU coupled"))
        if case["t_handoff"] is not None:
            axes[1].axvline(case["t_handoff"], color="green", ls="--", lw=1, alpha=0.7)
        axes[1].axhline(0, color="0.8", lw=1, ls=":")
        axes[1].set_ylabel("$C_L$")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 200)
        out_png = coupled_eval_dir / f"coupled_viv_Ur{case['Ur']}_{tag}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        written.append(out_png)
        print(f"  wrote {out_png.name}")
    return written


def replot_sweep_summary(coupled_eval_dir: Path, dataset: str = "cylinder200") -> Path | None:
    """Quick-look amplitude response (non-thesis style)."""
    csv_path = coupled_eval_dir / "sweep_results.csv"
    if not csv_path.exists():
        print(f"  [skip] no sweep_results.csv in {coupled_eval_dir}")
        return None
    df = pd.read_csv(csv_path)
    model_cols = [c for c in df.columns
                 if c not in ("Ur", "CFD", "closure_mode")
                 and not c.endswith(("_stability_label", "_A_star_rel_error", "_pass"))]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df["Ur"], df["CFD"], "o-", color="black", label="CFD")
    for lbl in model_cols:
        ax.plot(df["Ur"], df[lbl], "s--", label=lbl)
    ax.set_xlabel("$U_r$")
    ax.set_ylabel("Steady-state $A/D$")
    forcing = df["closure_mode"].iloc[0] if "closure_mode" in df.columns and len(df) else "v1_additive"
    ax.set_title(f"Coupled GRU-Structural VIV: closed-loop A/D vs $U_r$ ({dataset}, {forcing})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_png = coupled_eval_dir / "sweep_results.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_png.name}")
    return out_png


def replot_timeseries_thesis(coupled_eval_dir: Path, cfd_df: pd.DataFrame, dataset: str,
                             dataset_note: str | None,
                             appendix_font_scale: float = 1.0,
                             linewidth: float = 0.6) -> dict[float, dict]:
    """Thesis-style closed-loop time series for every case."""
    thesis_dir = coupled_eval_dir / "thesis_figures"
    if appendix_font_scale != 1.0:
        thesis_dir = thesis_dir / "appendix_narrow"
    results = {}
    for npz_path in sorted(coupled_eval_dir.glob("coupled_cylinder200_Ur*.npz")):
        case = _load_case(npz_path, cfd_df)
        if case is None:
            continue
        case_label = ur_tag(case["Ur"])
        model_mask = case["t"] <= 200

        cfd_mask = case["CFD_t"] <= 200
        r = plot_coupled_thesis(
            case["t"][model_mask], case["h"][model_mask], case["CL"][model_mask], case["D"],
            case["CFD_t"][cfd_mask], case["CFD_h"][cfd_mask], case["CFD_cl"][cfd_mask],
            case_label=case_label, output_dir=thesis_dir,
            condition_label=f"$U_r={case['Ur']:g}$", t_handoff=case["t_handoff"],
            h_mode="raw", dataset_note=dataset_note,
            U=cylinder200_U(case["Ur"]), font_scale=appendix_font_scale, linewidth=linewidth,
        )
        results[case["Ur"]] = r
        print(f"  wrote {r['pdf_path'].name}")
    return results


def replot_sweep_summary_thesis(coupled_eval_dir: Path, model_column: str,
                                dataset_note: str | None) -> dict:
    """Thesis-style amplitude response over Ur."""
    csv_path = coupled_eval_dir / "sweep_results.csv"
    if not csv_path.exists():
        print(f"  [skip] no sweep_results.csv in {coupled_eval_dir}")
        return {}
    df = pd.read_csv(csv_path)
    if model_column not in df.columns:
        raise KeyError(
            f"--model_column '{model_column}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )
    thesis_dir = coupled_eval_dir / "thesis_figures"
    r = plot_amplitude_response_thesis(
        df, model_column=model_column, output_dir=thesis_dir,
        dataset_note=dataset_note,
    )
    print(f"  wrote {r['pdf_path'].name}")
    return r


def select_representative_cases(coupled_eval_dir: Path, model_column: str) -> dict:
    """Cases shown in the main text; the others go to the appendix."""
    csv_path = coupled_eval_dir / "sweep_results.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path).sort_values("Ur")
    if df.empty:
        return {}
    low_ur = float(df["Ur"].iloc[0])
    high_ur = float(df["Ur"].iloc[-1])
    peak_ur = float(df.loc[df["CFD"].idxmax(), "Ur"])
    return {"low": low_ur, "near_peak": peak_ur, "high": high_ur}


def main():
    """Regenerate the figures for each model."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_subdir", action="append", required=True,
                  help="e.g. gru_cylinder200_dim_context_noacc (repeatable). "
                       "Looks for results/<model_subdir>_coupled_eval/.")
    p.add_argument("--dataset", default="cylinder200")
    p.add_argument("--model_column", default=None,
                  help="Explicit sweep_results.csv column to plot as the model "
                       "series in the amplitude-response figure (required for "
                       "--thesis; auto-column-discovery was removed on purpose "
                       "-- a sweep CSV can carry non-amplitude columns).")
    p.add_argument("--diagnostic", action="store_true",
                  help="Also (re)write the fast ad hoc-styled diagnostic PNGs "
                       "in place (old behavior). Off by default now that the "
                       "thesis path is the primary deliverable.")
    p.add_argument("--thesis", action="store_true", default=True,
                  help="Write thesis-quality PDF+PNG+caption figures to "
                       "results/<model>_coupled_eval/thesis_figures/ (default on).")
    p.add_argument("--no_thesis", dest="thesis", action="store_false")
    p.add_argument("--appendix_font_scale", type=float, default=1.0,
                   help="Font-size multiplier for closed-loop per-case "
                        "figures, written to thesis_figures/appendix_narrow/ "
                        "instead of thesis_figures/ so the full-width "
                        "originals (for whichever case is used as the "
                        "main-text representative figure) are untouched. "
                        "Pass 1/display_fraction, e.g. 1/0.45.")
    args = p.parse_args()

    if args.thesis and args.model_column is None:
        raise SystemExit("--model_column is required for --thesis (see --help)")

    print("Loading CFD dataset once (shared across all model subdirs)...")
    cfd_df = load_full_cfd_df(args.dataset)

    for subdir in args.model_subdir:
        coupled_eval_dir = PROJECT_ROOT / "results" / f"{subdir}_coupled_eval"
        if not coupled_eval_dir.is_dir():
            print(f"[skip] {coupled_eval_dir} does not exist")
            continue
        print(f"\n=== {subdir} ===")

        if args.diagnostic:
            replot_timeseries(coupled_eval_dir, cfd_df, args.dataset)
            replot_sweep_summary(coupled_eval_dir, args.dataset)

        if args.thesis:
            dataset_note = f"{args.dataset}, {subdir}"
            replot_timeseries_thesis(coupled_eval_dir, cfd_df, args.dataset, dataset_note,
                                     appendix_font_scale=args.appendix_font_scale)
            replot_sweep_summary_thesis(coupled_eval_dir, args.model_column, dataset_note)

            rep = select_representative_cases(coupled_eval_dir, args.model_column)
            if rep:
                print(f"  Recommended main-chapter cases (data-driven pick; override freely):")
                print(f"    low-Ur:    Ur={rep['low']:g}  -> {ur_tag(rep['low'])}")
                print(f"    near-peak: Ur={rep['near_peak']:g}  -> {ur_tag(rep['near_peak'])}")
                print(f"    high-Ur:   Ur={rep['high']:g}  -> {ur_tag(rep['high'])}")
                print(f"    + amplitude_response.pdf")
                print(f"    (all other Ur cases in thesis_figures/ are appendix material)")


if __name__ == "__main__":
    main()
