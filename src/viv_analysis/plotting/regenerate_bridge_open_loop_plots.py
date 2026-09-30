#!/usr/bin/env python3
r"""Open-loop (teacher-forced) figures and tables for the bridge models
(thesis Fig. 6.5 and Appendix G). Writes to results/<model_subdir>/thesis_figures/.

Example:
    PYTHONPATH=src python -m viv_analysis.plotting.regenerate_bridge_open_loop_plots \
        --model_subdir gru_bridge_nd_context_noacc
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from viv_analysis.config import bridge_structural_params, config
from viv_analysis.preprocess import load_bridge_df_cached
from viv_analysis.plotting.regenerate_teacher_forcing_plots import (
    _load_model_artifacts, _verify_matches_slow_reference, fast_teacher_forcing,
)
from viv_analysis.plotting.thesis_plots import (
    build_metrics_table, plot_open_loop_representative, plot_tf_result_thesis,
)
from viv_analysis.train_gru import apply_nd_transform
from viv_analysis.utils import parse_ur_label


def main():
    """Make the open-loop figures and table for each bridge model."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_subdir", action="append", required=True)
    p.add_argument("--skip_verify", action="store_true")
    p.add_argument("--appendix_font_scale", type=float, default=1.0,
                   help="Font-size multiplier for the per-case appendix "
                        "figures (plot_tf_result_thesis) ONLY -- the "
                        "representative main-text figure is unaffected. "
                        "Pass 1/display_fraction for the width these "
                        "figures will actually be included at, e.g. "
                        "1/0.48 for two-up at 0.48\\textwidth.")
    p.add_argument("--font_scale", type=float, default=1.0,
                   help="Font-size multiplier for the representative "
                        "main-text figure (plot_open_loop_representative) "
                        "ONLY. Pass 1/display_fraction, e.g. 1/0.8 for "
                        "0.8\\textwidth.")
    p.add_argument("--linewidth", type=float, default=None,
                   help="Overrides both CFD/GRU trace linewidths (both "
                        "figures) to this exact absolute value.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = config["bridge_D_ref"]; fn = config["bridge_fn_hz"]
    t_star_release = config["bridge_t_star_release"]
    sp = bridge_structural_params()

    physical_params_note = (
        rf"$D={D:g}\,\mathrm{{m}}$, $f_n={fn:g}\,\mathrm{{Hz}}$, "
        rf"$\zeta={config['bridge_zeta']:g}$"
    )

    print("Loading + preprocessing bridge CFD dataset once (shared across models)...")
    raw_df = load_bridge_df_cached(fn_hz=fn, d_ref=D, bridge_structural_params=sp)

    for model_subdir in args.model_subdir:
        print(f"\n=== {model_subdir} ===")
        art = _load_model_artifacts(model_subdir, device)
        df = raw_df
        if art["nd_inputs"]:
            df = apply_nd_transform(df, nd_inputs=True, D=D, fn=fn, input_cols=art["input_cols"])

        thesis_dir = art["output_dir"] / "thesis_figures"
        case_results = []
        for case_name in art["test_cases"]:
            case_df = df[df["case"] == case_name].copy()
            if case_df.empty:
                print(f"  [skip] {case_name} not found")
                continue
            case_df_scaled = case_df.copy()
            case_df_scaled[art["input_cols"]] = art["x_scaler"].transform(
                case_df_scaled[art["input_cols"]].to_numpy(dtype=np.float32))
            case_df_scaled["cl"] = art["y_scaler"].transform(
                case_df_scaled["cl"].to_numpy(dtype=np.float32).reshape(-1, 1)).ravel()

            ur = parse_ur_label(case_name)
            U = ur * fn * D
            release_t = t_star_release * D / U

            if not args.skip_verify:
                _verify_matches_slow_reference(art, case_df_scaled, release_t, device)

            cl_pred, cl_true, times = fast_teacher_forcing(
                art["model"], case_df_scaled, art["input_cols"], art["seq_len"], release_t,
                art["y_scaler"], case_name, device, art["use_ur_context"], art["ur_stats"],
            )

            condition_label = rf"the bridge test case at $U={U:.2f}\,\mathrm{{m/s}}$ ($U_r={ur:g}$)"
            result = plot_tf_result_thesis(
                cl_pred, cl_true, times, case_label=case_name, output_dir=thesis_dir,
                condition_label=condition_label, physical_params_note=physical_params_note,
                U=U, D=D, font_scale=args.appendix_font_scale, linewidth=args.linewidth,
            )
            rmse = float(np.sqrt(np.mean((cl_pred - cl_true) ** 2)))
            mae = float(np.mean(np.abs(cl_pred - cl_true)))
            print(f"  {case_name} (U={U:.2f} m/s): R2={result['r2']:.4f}  RMSE={rmse:.4f}  "
                  f"MAE={mae:.4f}  n={len(times)}  -> {result['pdf_path']}")
            case_results.append(dict(
                case=rf"$U={U:.2f}\,\mathrm{{m/s}}$ ($U_r={ur:g}$)", ur=ur, U=U,
                r2=result["r2"], rmse=rmse, mae=mae,
                amplitude=float(cl_true.max() - cl_true.min()),
                cl_pred=cl_pred, cl_true=cl_true, times=times, condition_label=condition_label,
            ))

        if not case_results:
            continue

        rep = max(case_results, key=lambda r: r["amplitude"])
        rep_result = plot_open_loop_representative(
            rep["cl_pred"], rep["cl_true"], rep["times"], case_label=f"Ur{rep['ur']:g}",
            output_dir=thesis_dir, condition_label=rep["condition_label"],
            physical_params_note=physical_params_note,
            U=rep["U"], D=D, font_scale=args.font_scale, linewidth=args.linewidth,
        )
        print(f"  Representative case: {rep['condition_label']} -> {rep_result['pdf_path']}")

        table_caption = (
            f"Open-loop performance on the held-out bridge test cases for {model_subdir} "
            f"({physical_params_note})."
        )
        table = build_metrics_table(
            [{"case": r["case"], "r2": r["r2"], "rmse": r["rmse"], "mae": r["mae"]}
             for r in sorted(case_results, key=lambda r: r["ur"])],
            output_dir=thesis_dir, caption=table_caption,
        )
        print(f"  Metrics table -> {table['tex_path']}")


if __name__ == "__main__":
    main()
