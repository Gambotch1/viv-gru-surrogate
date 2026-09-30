#!/usr/bin/env python3
r"""Closed-loop bridge figures from an evaluate_all.py sweep folder
(thesis Fig. 6.6, 6.7 and Appendix H). The amplitude response uses the
reference status of each CFD case (reference_quality.py), so only settled
limit cycles get an amplitude point. Writes to <sweep folder>/thesis_figures/.

Example:
    PYTHONPATH=src python -m viv_analysis.plotting.regenerate_bridge_plots \
        --coupled_eval_dir <sweep folder> --model_column <column in sweep_results.csv>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from viv_analysis.plotting.plot_style import apply_thesis_style
apply_thesis_style()

from viv_analysis.config import config
from viv_analysis.evaluate_all import load_full_cfd_df
from viv_analysis.reference_quality import build_reference_status_table
from viv_analysis.plotting.thesis_plots import (
    plot_amplitude_response_status_aware_thesis, plot_coupled_thesis, ur_tag,
)
from viv_analysis.utils import PROJECT_ROOT, format_ur_label

_UR_RE = re.compile(r"coupled_bridge_Ur([0-9.]+)_(.+)\.npz$")


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


def replot_timeseries_thesis(coupled_eval_dir: Path, cfd_df: pd.DataFrame,
                             dataset_note: str | None,
                             appendix_font_scale: float = 1.0) -> dict[float, dict]:
    """Thesis-style closed-loop time series for every case."""
    thesis_dir = coupled_eval_dir / "thesis_figures"
    if appendix_font_scale != 1.0:
        thesis_dir = thesis_dir / "appendix_narrow"
    results = {}
    for npz_path in sorted(coupled_eval_dir.glob("coupled_bridge_Ur*.npz")):
        case = _load_case(npz_path, cfd_df)
        if case is None:
            continue
        case_label = ur_tag(case["Ur"])

        model_mask = case["t"] <= 200

        cfd_mask = case["CFD_t"] <= 200
        fn = config["bridge_fn_hz"]
        U = case["Ur"] * fn * case["D"]
        r = plot_coupled_thesis(
            case["t"][model_mask], case["h"][model_mask], case["CL"][model_mask], case["D"],
            case["CFD_t"][cfd_mask], case["CFD_h"][cfd_mask], case["CFD_cl"][cfd_mask],
            case_label=case_label, output_dir=thesis_dir,
            condition_label=f"$U_r={case['Ur']:g}$", t_handoff=case["t_handoff"],
            h_mode="mean_removed", dataset_note=dataset_note,
            U=U, font_scale=appendix_font_scale,
        )
        results[case["Ur"]] = r
        print(f"  wrote {r['pdf_path'].name}")
    return results


def _load_partition_by_ur(model_column: str) -> dict[float, str]:
    """Train/val/test partition of each Ur, from the model's run_config.json."""
    from viv_analysis.utils import parse_ur_label

    metrics_path = PROJECT_ROOT / "results" / model_column / "metrics_gru.json"
    if not metrics_path.exists():
        print(f"  [warn] {metrics_path} not found -- the response-curve figure "
              f"will have no train/val/test partition to show.")
        return {}
    case_split = json.loads(metrics_path.read_text())["case_split"]
    return {
        round(parse_ur_label(case), 6): kind
        for kind, cases in case_split.items()
        for case in cases
    }


def replot_sweep_summary_thesis(coupled_eval_dir: Path, model_column: str,
                                dataset_note: str | None,
                                font_scale: float = 1.0) -> dict:
    """Status-aware amplitude response over Ur."""
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

    print("  Classifying CFD reference quality for every swept Ur "
          "(reference_quality.build_reference_status_table)...")
    status_table = build_reference_status_table()
    status_by_ur = dict(zip(status_table["Ur"].round(6), status_table["reference_status"]))
    df["reference_status"] = df["Ur"].round(6).map(status_by_ur)
    missing = df[df["reference_status"].isna()]
    if not missing.empty:
        raise KeyError(
            f"No reference_status found for Ur value(s) {sorted(missing['Ur'].tolist())} "
            f"-- build_reference_status_table() covers the current retained bridge "
            f"dataset; a swept Ur outside that set would silently get no status.")

    partition_by_ur = _load_partition_by_ur(model_column)
    df["partition"] = df["Ur"].round(6).map(partition_by_ur)
    missing_partition = df[df["partition"].isna()]
    if partition_by_ur and not missing_partition.empty:
        raise KeyError(
            f"No train/val/test partition found for Ur value(s) "
            f"{sorted(missing_partition['Ur'].tolist())} -- metrics_gru.json's "
            f"case_split should cover every case in the retained dataset.")

    thesis_dir = coupled_eval_dir / "thesis_figures"
    r = plot_amplitude_response_status_aware_thesis(
        df, model_column=model_column, status_column="reference_status",
        partition_column="partition", output_dir=thesis_dir, dataset_note=dataset_note,
        font_scale=font_scale,
    )
    print(f"  wrote {r['pdf_path'].name}  ({r['n_settled_lco']}/{r['n_total']} settled_lco)")
    return r


def select_representative_cases(coupled_eval_dir: Path) -> dict:
    """Cases shown in the main text; the others go to the appendix."""
    csv_path = coupled_eval_dir / "sweep_results.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path).sort_values("Ur")
    if df.empty:
        return {}
    status_table = build_reference_status_table()
    status_by_ur = dict(zip(status_table["Ur"].round(6), status_table["reference_status"]))

    def _status(ur):
        return status_by_ur.get(round(float(ur), 6), "unknown")

    low_ur = float(df["Ur"].iloc[0])
    high_ur = float(df["Ur"].iloc[-1])
    peak_ur = float(df.loc[df["CFD"].idxmax(), "Ur"])
    return {
        "low": (low_ur, _status(low_ur)),
        "near_peak": (peak_ur, _status(peak_ur)),
        "high": (high_ur, _status(high_ur)),
    }


def main():
    """Regenerate the bridge closed-loop figures."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coupled_eval_dir", required=True,
                  help="results/-relative directory of a prior "
                       "`evaluate_all.py --dataset bridge` sweep, e.g. "
                       "gru_bridge_p0_nd_context_noacc_final22_coupled_eval")
    p.add_argument("--model_column", required=True,
                  help="Explicit sweep_results.csv column to plot as the model "
                       "series in the amplitude-response figure (auto-column-"
                       "discovery was removed on purpose -- a sweep CSV can "
                       "carry non-amplitude columns).")
    p.add_argument("--dataset_note", default=None)
    p.add_argument("--appendix_font_scale", type=float, default=1.0,
                   help="Font-size multiplier for closed-loop per-case "
                        "figures, written to thesis_figures/appendix_narrow/ "
                        "instead of thesis_figures/. Pass 1/display_fraction, "
                        "e.g. 1/0.45.")
    args = p.parse_args()

    coupled_eval_dir = PROJECT_ROOT / "results" / args.coupled_eval_dir
    if not coupled_eval_dir.is_dir():
        raise SystemExit(f"{coupled_eval_dir} does not exist")

    print("Loading bridge CFD dataset...")
    cfd_df = load_full_cfd_df("bridge")

    dataset_note = args.dataset_note or f"bridge, {args.coupled_eval_dir}"
    print(f"\n=== {args.coupled_eval_dir} ===")
    replot_timeseries_thesis(coupled_eval_dir, cfd_df, dataset_note,
                             appendix_font_scale=args.appendix_font_scale)
    replot_sweep_summary_thesis(coupled_eval_dir, args.model_column, dataset_note)

    rep = select_representative_cases(coupled_eval_dir)
    if rep:
        print(f"  Recommended main-chapter cases (data-driven pick; override freely):")
        for tag, (ur, status) in rep.items():
            print(f"    {tag:10s}: Ur={ur:g}  -> {ur_tag(ur)}  [{status}]")
        print(f"    + amplitude_response_status_aware.pdf")
        print(f"    (all other Ur cases in thesis_figures/ are appendix material)")


if __name__ == "__main__":
    main()
