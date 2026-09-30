#!/usr/bin/env python3
r"""Pooled teacher-forcing residual for the residual forcing test (thesis Sec. 6.6.2).

For every training case of the model, the open-loop residual cl_true - cl_tf
is computed (first --skip_s seconds dropped) and all residuals are joined into
one record. The target case is excluded as long as it is not a training case
of the model.

Writes results/<model_subdir>/pooled_tf_residual_train_cases.npz, which
coupled_inference.py reads with --residual_npz.

Example:
    PYTHONPATH=src python -m viv_analysis.build_pooled_tf_residual \
        --model_subdir gru_bridge_nd_context_noacc --target_ur 6.7385
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
from viv_analysis.train_gru import apply_nd_transform
from viv_analysis.utils import PROJECT_ROOT


def main():
    """Compute and save the pooled residual."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_subdir", required=True)
    p.add_argument("--target_ur", type=float, required=True,
                   help="Ur of the coupled run this pooled residual will be used for "
                        "(stamped into the output file's 'Ur' field; not a source case).")
    p.add_argument("--skip_s", type=float, default=5.0,
                   help="Seconds trimmed from the start of each case's residual "
                        "(matches coupled_inference.py's existing single-case convention).")
    p.add_argument("--skip_verify", action="store_true")
    p.add_argument("--out", default=None,
                   help="Output npz path (default: results/<model_subdir>/"
                        "pooled_tf_residual_train_cases.npz)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = config["bridge_D_ref"]; fn = config["bridge_fn_hz"]
    t_star_release = config["bridge_t_star_release"]
    sp = bridge_structural_params()

    print("Loading + preprocessing bridge CFD dataset...")
    raw_df = load_bridge_df_cached(fn_hz=fn, d_ref=D, bridge_structural_params=sp)

    art = _load_model_artifacts(args.model_subdir, device)
    if not art["train_cases"]:
        raise SystemExit(f"{args.model_subdir}'s run_config.json has no train_cases recorded")
    df = raw_df
    if art["nd_inputs"]:
        df = apply_nd_transform(df, nd_inputs=True, D=D, fn=fn, input_cols=art["input_cols"])

    dt_ref = None
    pieces = []
    used, skipped = [], []
    for case_name in art["train_cases"]:
        case_df = df[df["case"] == case_name].copy()
        if case_df.empty:
            print(f"  [skip] {case_name}: not present in current bridge cache")
            skipped.append(case_name)
            continue
        case_df_scaled = case_df.copy()
        case_df_scaled[art["input_cols"]] = art["x_scaler"].transform(
            case_df_scaled[art["input_cols"]].to_numpy(dtype=np.float32))
        case_df_scaled["cl"] = art["y_scaler"].transform(
            case_df_scaled["cl"].to_numpy(dtype=np.float32).reshape(-1, 1)).ravel()

        from viv_analysis.utils import parse_ur_label
        ur = parse_ur_label(case_name)
        U = ur * fn * D
        release_t = t_star_release * D / U

        if not args.skip_verify:
            _verify_matches_slow_reference(art, case_df_scaled, release_t, device)

        cl_pred, cl_true, times = fast_teacher_forcing(
            art["model"], case_df_scaled, art["input_cols"], art["seq_len"], release_t,
            art["y_scaler"], case_name, device, art["use_ur_context"], art["ur_stats"],
        )
        dt = float(np.median(np.diff(times)))
        if dt_ref is None:
            dt_ref = dt
        elif abs(dt - dt_ref) / dt_ref > 1e-3:
            raise ValueError(f"{case_name}: dt={dt:.6g} inconsistent with dt_ref={dt_ref:.6g} "
                              f"-- pooling assumes a shared sample rate across cases")

        skip_n = int(round(args.skip_s / dt))
        resid = (cl_true - cl_pred)[skip_n:]
        pieces.append(resid)
        used.append(case_name)
        print(f"  {case_name}: n={len(cl_true)}  resid_std={resid.std():.5f}  "
              f"(kept {len(resid)} samples after {args.skip_s:g}s skip)")

    if not pieces:
        raise SystemExit("no training cases were usable -- nothing to pool")

    pooled = np.concatenate(pieces).astype(np.float64)
    print(f"\nPooled {len(used)}/{len(art['train_cases'])} training cases "
          f"({len(skipped)} skipped: {skipped}) -> {len(pooled)} samples "
          f"({len(pooled) * dt_ref:.1f}s), pooled_std={pooled.std():.5f}")

    out_dir = PROJECT_ROOT / "results" / args.model_subdir
    out_path = (out_dir / "pooled_tf_residual_train_cases.npz") if args.out is None else args.out
    np.savez(
        out_path,
        cl_true=pooled, cl_tf=np.zeros_like(pooled),
        dt=dt_ref, fn=fn, Ur=args.target_ur,
        source_cases=np.array(used), skipped_cases=np.array(skipped),
        skip_s=args.skip_s,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
