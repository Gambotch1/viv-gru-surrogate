#!/usr/bin/env python3
r"""Open-loop (teacher-forced) figures and metric tables for the cylinder models
(thesis Fig. 5.4, 5.7 and Appendix D).

Runs the trained model over each test case with the CFD kinematics as input.
A fast batched version is checked against the slow step-by-step one before use
(skip with --skip_verify). Writes to results/<model_subdir>/thesis_figures/.

Example:
    PYTHONPATH=src python -m viv_analysis.plotting.regenerate_teacher_forcing_plots \
        --model_subdir gru_cylinder200_nd_context_noacc
"""


from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from viv_analysis.config import config, cylinder200_release_time, prepare_gru_config
from viv_analysis.models.gru import VIV_GRU
from viv_analysis.preprocess import compute_kinematics, merge_dataframes
from viv_analysis.plotting.thesis_plots import (
    build_metrics_table, plot_open_loop_representative, plot_tf_result_thesis,
)
from viv_analysis.train_gru import (
    apply_nd_transform, resolve_nd_reference_scales, teacher_forcing_rollout,
)
from viv_analysis.utils import PROJECT_ROOT, parse_ur_label, segment_by_time_gaps


def fast_teacher_forcing(model, case_df_scaled, input_cols, seq_len, release_t,
                         y_scaler, case_name, device, use_ur_context, ur_stats,
                         batch_size=1024):
    """Open-loop prediction over a whole case in batches (same result as train_gru.teacher_forcing_rollout, faster)."""
    ordered = case_df_scaled.sort_values("time").reset_index(drop=True)
    signal = ordered[input_cols].to_numpy(dtype=np.float32)

    if use_ur_context:
        ur_mean, ur_std = ur_stats
        ur_std = float(ur_std) if abs(float(ur_std)) > 0 else 1.0
        ur_val = parse_ur_label(str(case_name))
        ur_scaled = (ur_val - float(ur_mean)) / ur_std
        ur_col = np.full((signal.shape[0], 1), ur_scaled, dtype=np.float32)
        signal = np.hstack([signal, ur_col])

    cl_true_s = ordered["cl"].to_numpy(dtype=np.float32)
    times = ordered["time"].to_numpy(dtype=np.float32)
    release_idx = int(np.searchsorted(times, release_t))

    idx_chunks = []
    for seg_start, seg_end in segment_by_time_gaps(times):
        seg_pred_start = seg_start + seq_len
        if seg_start <= release_idx < seg_end:
            seg_pred_start = max(seg_pred_start, release_idx + seq_len)
        if seg_pred_start < seg_end:
            idx_chunks.append(np.arange(seg_pred_start, seg_end))
    idxs = np.concatenate(idx_chunks) if idx_chunks else np.array([], dtype=int)

    preds = np.empty(len(idxs), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for b0 in range(0, len(idxs), batch_size):
            batch_idx = idxs[b0:b0 + batch_size]
            windows = np.stack([signal[i - seq_len:i] for i in batch_idx], axis=0)
            x = torch.from_numpy(windows).to(device)
            p, _ = model(x)
            preds[b0:b0 + len(batch_idx)] = p.cpu().numpy()

    cl_pred = y_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
    cl_true = y_scaler.inverse_transform(cl_true_s[idxs].reshape(-1, 1)).ravel()
    return cl_pred, cl_true, times[idxs]


def _load_model_artifacts(model_subdir: str, device: str):
    """Model, scalers, split and settings from a model folder."""
    d = PROJECT_ROOT / "results" / model_subdir
    import json
    rc = json.load(open(d / "run_config.json"))
    with open(d / "x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open(d / "y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)
    with open(d / "ur_stats.pkl", "rb") as f:
        ur_stats_pkl = pickle.load(f)
    input_cols = rc["input_cols"]
    use_ur_context = bool(rc["use_ur_context"])
    nd_inputs = bool(rc["nd_inputs"])
    input_size = len(input_cols) + (1 if use_ur_context else 0)
    model = VIV_GRU(input_size=input_size, hidden_size=config["hidden_size"],
                    num_layers=config["num_layers"], dropout=config["dropout"]).to(device)
    model.load_state_dict(torch.load(d / "gru_best.pt", map_location=device))
    model.eval()
    return dict(model=model, x_scaler=x_scaler, y_scaler=y_scaler,
               ur_stats=(ur_stats_pkl["mean"], ur_stats_pkl["std"]),
               input_cols=input_cols, use_ur_context=use_ur_context, nd_inputs=nd_inputs,
               test_cases=rc["test_cases"], train_cases=rc.get("train_cases", []),
               val_cases=rc.get("val_cases", []), seq_len=rc["seq_len"], output_dir=d)


def _verify_matches_slow_reference(art, case_df_scaled, release_t, device, n_check_steps=3000):
    """Check that the fast open-loop prediction matches the step-by-step reference."""
    seq_len = art["seq_len"]
    ordered_full = case_df_scaled.sort_values("time").reset_index(drop=True)
    times_full = ordered_full["time"].to_numpy(dtype=np.float32)
    release_idx_full = int(np.searchsorted(times_full, release_t))
    slice_start = max(0, release_idx_full - 10)
    short_df = ordered_full.iloc[slice_start: slice_start + seq_len + n_check_steps].copy()
    case_name = short_df["case"].iloc[0]

    cl_pred_fast, cl_true_fast, t_fast = fast_teacher_forcing(
        art["model"], short_df, art["input_cols"], seq_len, release_t,
        art["y_scaler"], case_name, device, art["use_ur_context"], art["ur_stats"],
        batch_size=256,
    )
    cl_pred_slow, cl_true_slow, t_slow = teacher_forcing_rollout(
        art["model"], short_df, art["input_cols"], seq_len, release_t,
        art["y_scaler"], case_name, device, use_ur_context=art["use_ur_context"],
        ur_stats=art["ur_stats"],
    )
    np.testing.assert_allclose(t_fast, t_slow, atol=1e-4)
    np.testing.assert_allclose(cl_true_fast, cl_true_slow, atol=1e-4)
    np.testing.assert_allclose(cl_pred_fast, cl_pred_slow, atol=1e-3, rtol=1e-3)
    print(f"  [verified] fast path matches teacher_forcing_rollout on {len(t_fast)} steps "
          f"(max|pred diff|={np.max(np.abs(cl_pred_fast-cl_pred_slow)):.2e})")


def main():
    """Make the open-loop figures and tables for each model."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_subdir", action="append", required=True)
    p.add_argument("--dataset", default="cylinder200")
    p.add_argument("--skip_verify", action="store_true")
    p.add_argument("--width_in", type=float, default=None,
                   help="Generate the representative-case figure (plot_open_"
                        "loop_representative) at this width in inches "
                        "instead of the default full TEXT_WIDTH_IN -- changes "
                        "the figure's own proportions (line/marker sizes "
                        "scale with the smaller canvas too), not just text. "
                        "Prefer --font_scale for a subfigure grid where the "
                        "plot itself should keep its normal full-width "
                        "proportions and only the text needs to read larger.")
    p.add_argument("--font_scale", type=float, default=1.0,
                   help="Multiplies the representative-case figure's axis/"
                        "tick/legend font sizes by this factor without "
                        "changing figsize -- for a figure displayed smaller "
                        "than full \\textwidth (e.g. inside a 0.49\\textwidth "
                        "subfigure), pass 1/display_fraction (e.g. 1/0.49) "
                        "so the deliberately oversized text lands back at a "
                        "normal readable size once LaTeX shrinks the whole "
                        "page. Only affects plot_open_loop_representative.")
    p.add_argument("--appendix_font_scale", type=float, default=1.0,
                   help="Same idea as --font_scale, but for the per-case "
                        "appendix figures (plot_tf_result_thesis) instead of "
                        "the representative one. These use a distinct "
                        "filename per case (open_loop_Ur*.pdf vs open_loop_"
                        "representative_*.pdf), so regenerating in place "
                        "never clobbers the representative figure.")
    p.add_argument("--linewidth_scale", type=float, default=1.0,
                   help="Multiplies both the CFD and GRU trace linewidths "
                        "by this factor (both figures: representative and "
                        "per-case appendix). E.g. 0.7 for visibly thinner "
                        "lines where the two traces overlap closely. "
                        "Ignored if --linewidth is given.")
    p.add_argument("--linewidth", type=float, default=None,
                   help="Overrides both the CFD and GRU trace linewidths "
                        "to this exact absolute value (both figures), "
                        "instead of scaling each trace's own base width.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = prepare_gru_config(args.dataset, config)
    D_nd, fn_nd = resolve_nd_reference_scales(args.dataset, cfg)

    print("Loading + preprocessing CFD dataset once (shared across models)...")
    raw_df = merge_dataframes(dataset=args.dataset)
    raw_df = compute_kinematics(raw_df, dataset=args.dataset)

    physical_params_note = r"$Re=200$, $m^*=10$, $\zeta=0.01$" if args.dataset == "cylinder200" else None

    for model_subdir in args.model_subdir:
        print(f"\n=== {model_subdir} ===")
        art = _load_model_artifacts(model_subdir, device)
        df = raw_df
        if art["nd_inputs"]:
            df = apply_nd_transform(df, nd_inputs=True, D=D_nd, fn=fn_nd, input_cols=art["input_cols"])

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
            U_nd = ur * fn_nd * D_nd
            release_t = cylinder200_release_time(ur)

            if not args.skip_verify:
                _verify_matches_slow_reference(art, case_df_scaled, release_t, device)

            cl_pred, cl_true, times = fast_teacher_forcing(
                art["model"], case_df_scaled, art["input_cols"], art["seq_len"], release_t,
                art["y_scaler"], case_name, device, art["use_ur_context"], art["ur_stats"],
            )

            condition_label = f"the cylinder200 test case at $U_r={ur:g}$"
            thesis_dir = art["output_dir"] / "thesis_figures"
            result = plot_tf_result_thesis(
                cl_pred, cl_true, times, case_label=case_name, output_dir=thesis_dir,
                condition_label=condition_label, physical_params_note=physical_params_note,
                U=U_nd, D=D_nd,
                font_scale=args.appendix_font_scale, linewidth_scale=args.linewidth_scale,
                linewidth=args.linewidth,
            )
            rmse = float(np.sqrt(np.mean((cl_pred - cl_true) ** 2)))
            mae = float(np.mean(np.abs(cl_pred - cl_true)))
            print(f"  {case_name}: R2={result['r2']:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  "
                  f"n={len(times)}  -> {result['pdf_path']}")
            case_results.append(dict(
                case=f"$U_r={ur:g}$", ur=ur, U=U_nd, r2=result["r2"], rmse=rmse, mae=mae,
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
            U=rep["U"], D=D_nd,
            width_in=args.width_in, font_scale=args.font_scale,
            linewidth_scale=args.linewidth_scale, linewidth=args.linewidth,
        )
        print(f"  Representative case: {rep['condition_label']} -> {rep_result['pdf_path']}")

        table_caption = (
            f"Open-loop performance on the four held-out test cases for {model_subdir} "
            f"({physical_params_note})." if physical_params_note else
            f"Open-loop performance on the four held-out test cases for {model_subdir}."
        )
        table = build_metrics_table(
            [{"case": r["case"], "r2": r["r2"], "rmse": r["rmse"], "mae": r["mae"]}
             for r in sorted(case_results, key=lambda r: r["ur"])],
            output_dir=thesis_dir, caption=table_caption,
        )
        print(f"  Metrics table -> {table['tex_path']}")


if __name__ == "__main__":
    main()
