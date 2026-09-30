"""Rank the configurations of a stage on validation results (median over seeds); --freeze writes the selection manifest."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import REQUIRED_SEEDS, STUDY_ROOT, git_commit, write_json

RETENTION_THRESHOLD = -0.005
BASELINE_H, BASELINE_L = 64, 2


def _closed_loop_rank_cols(dataset: str, agg: pd.DataFrame) -> list[tuple[str, bool]]:
    if dataset == "cylinder200":
        pairs = [
            ("closed_loop_validation_stable_count_median", False),
            ("closed_loop_validation_median_abs_rel_amp_error_median", True),
            ("closed_loop_validation_worst_abs_rel_amp_error_median", True),
            ("closed_loop_validation_median_abs_f_osc_rel_error_median", True),
        ]
    else:
        pairs = [
            ("closed_loop_validation_stable_count_among_settled_lco_median", False),
            ("closed_loop_validation_median_abs_rel_amp_error_among_settled_lco_median", True),
            ("closed_loop_validation_worst_abs_rel_amp_error_among_settled_lco_median", True),
            ("_bridge_non_lco_rms_ratio_deviation_median", True),
            ("closed_loop_validation_median_abs_f_osc_rel_error_median", True),
        ]
    return [(c, asc) for c, asc in pairs if c in agg.columns]


def build_pareto_table(dataset: str, agg: pd.DataFrame) -> pd.DataFrame:
    baseline = agg[(agg["hidden_size"] == BASELINE_H) & (agg["num_layers"] == BASELINE_L)]
    if baseline.empty:
        raise SystemExit(f"No fresh H={BASELINE_H},L={BASELINE_L} baseline found in "
                          f"the aggregated results -- cannot compute retention deltas.")
    baseline_r2 = float(baseline["open_loop_val_r2_median"].iloc[0])
    baseline_r2_iqr = float(baseline["open_loop_val_r2_iqr"].iloc[0] or 0.0)

    df = agg.copy()
    df = df[df["all_valid"]]
    df["delta_r2_val_vs_baseline"] = df["open_loop_val_r2_median"] - baseline_r2
    df["passes_retention_threshold"] = df["delta_r2_val_vs_baseline"] >= RETENTION_THRESHOLD

    rms_col = "closed_loop_validation_mean_rms_ratio_among_non_lco_median"
    if dataset == "bridge" and rms_col in df.columns:
        df["_bridge_non_lco_rms_ratio_deviation_median"] = (df[rms_col] - 1.0).abs()

    review_flags = []
    if df["passes_retention_threshold"].sum() == 0:
        review_flags.append("NO candidate satisfies delta_R2_val >= -0.005 -- "
                             "the threshold may be too strict for this grid; flagging "
                             "for review rather than silently rejecting everything.")
    if df["passes_retention_threshold"].all() and len(df) > 1:
        review_flags.append("EVERY candidate satisfies delta_R2_val >= -0.005 -- "
                             "the threshold is not discriminating between architectures "
                             "here; open-loop retention alone will not drive the ranking.")
    if baseline_r2_iqr > abs(RETENTION_THRESHOLD):
        review_flags.append(
            f"Baseline (H{BASELINE_H},L{BASELINE_L}) open-loop val R2 IQR across seeds "
            f"({baseline_r2_iqr:.4f}) exceeds the retention threshold magnitude "
            f"({abs(RETENTION_THRESHOLD)}) -- seed noise in the baseline itself is "
            f"comparable to the gate. Treat rejections/acceptances near the boundary "
            f"as unreliable.")

    rank_pairs = _closed_loop_rank_cols(dataset, df)
    if not rank_pairs:
        review_flags.append("No closed-loop validation columns found -- ranking falls "
                             "back to open-loop retention only. Run "
                             "evaluate_closed_loop.py + collect_results.py first.")

    df["short_intermediate_rollout_evidence"] = (
        "not computed -- this study's closed-loop protocol is full-duration only "
        "(500s cylinder200 / 300s bridge, matching bridge's actual max CFD "
        "reference duration), per the task's explicit requirement that "
        "short-horizon improvement alone is not sufficient evidence."
    )

    tie_break_pairs = [
        (c, True) for c in
        ["trainable_parameter_count_median", "closed_loop_mean_inference_time_s_median"]
        if c in df.columns
    ]

    sort_pairs = [("passes_retention_threshold", False)] + rank_pairs + tie_break_pairs
    sort_pairs = [(c, asc) for c, asc in sort_pairs if c in df.columns]
    sort_cols = [c for c, _ in sort_pairs]
    ascending = [asc for _, asc in sort_pairs]
    df = df.sort_values(sort_cols, ascending=ascending, na_position="last")
    df["hierarchical_rank"] = range(1, len(df) + 1)
    df.attrs["review_flags"] = review_flags
    df.attrs["baseline_r2"] = baseline_r2
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["cylinder200", "bridge"])
    p.add_argument("--stage", type=int, required=True, choices=[1, 2])
    p.add_argument("--freeze", action="store_true",
                   help="Write manifests/selection_manifest_{dataset}_stage{stage}"
                        ".json with frozen=true, naming the selected "
                        "CONFIGURATION's 3 seed-specific run_dirs. Filename is "
                        "PER-DATASET (not shared between cylinder200 and "
                        "bridge) -- both datasets need their own frozen "
                        "selection simultaneously for Stage 2 job generation, "
                        "and a shared filename would let freezing one dataset "
                        "silently overwrite the other's. This is the ONLY way "
                        "test evaluation can later be unlocked (via "
                        "unlock_test_evaluation.py) -- pass this only after "
                        "you have reviewed the Pareto table and approved a pick.")
    args = p.parse_args()

    agg_csv = STUDY_ROOT / "reports" / f"aggregated_{args.dataset}_stage{args.stage}.csv"
    if not agg_csv.exists():
        raise SystemExit(f"{agg_csv} not found -- run collect_results.py first.")
    agg = pd.read_csv(agg_csv)
    if "history_label" in agg.columns:
        agg["history_label"] = agg["history_label"].astype(object).where(agg["history_label"].notna(), None)

    pareto = build_pareto_table(args.dataset, agg)
    reports_dir = STUDY_ROOT / "reports"
    pareto_csv = reports_dir / f"pareto_{args.dataset}_stage{args.stage}.csv"
    pareto.to_csv(pareto_csv, index=False)

    print(f"Pareto/hierarchical ranking written to {pareto_csv}")
    for flag in pareto.attrs.get("review_flags", []):
        print(f"REVIEW FLAG: {flag}")

    top = pareto.iloc[0] if len(pareto) else None
    if top is None:
        print("No candidates to select from.")
        return

    print(f"\nTop-ranked (hierarchical_rank=1): H={int(top['hidden_size'])} "
          f"L={int(top['num_layers'])} seq_len={int(top['seq_len'])} "
          f"history_label={top.get('history_label')}")

    if args.freeze:
        seeds = top["seeds"] if isinstance(top["seeds"], list) else eval(str(top["seeds"]))
        assert sorted(seeds) == sorted(REQUIRED_SEEDS), (
            f"Selected configuration has seeds {sorted(seeds)}, expected exactly "
            f"{sorted(REQUIRED_SEEDS)} -- refusing to freeze a manifest for an "
            f"incomplete configuration (this would let a single favorable "
            f"seed stand in for the configuration).")

        run_dirs = []
        base = STUDY_ROOT / "results" / args.dataset / f"stage{args.stage}"
        for seed in seeds:
            tag = f"H{int(top['hidden_size'])}_L{int(top['num_layers'])}_seq{int(top['seq_len'])}_seed{seed}"
            if top.get("history_label"):
                tag = f"{top['history_label']}_{tag}"
            run_dirs.append(str(base / tag))

        selected_configuration = {
            "hidden_size": int(top["hidden_size"]),
            "num_layers": int(top["num_layers"]),
        }
        if args.stage == 2:
            selected_configuration["sequence_samples"] = int(top["seq_len"])
            selected_configuration["physical_duration_s"] = float(top["sequence_duration_s_median"])
            selected_configuration["history_label"] = top.get("history_label")

        manifest_path = STUDY_ROOT / "manifests" / f"selection_manifest_{args.dataset}_stage{args.stage}.json"
        manifest = {
            "frozen": True,
            "selection_complete": bool(args.stage == 2),
            "dataset": args.dataset,
            "stage": args.stage,
            "git_commit": git_commit(),
            "selected_configuration": selected_configuration,
            "hierarchical_rank": 1,
            "selection_basis": "validation partition only (open-loop retention gate + "
                                "full-duration closed-loop ranking); see pareto csv",
            "pareto_csv": str(pareto_csv),
            "selected_run_dirs": run_dirs,
            "seeds": sorted(seeds),
            "review_flags": pareto.attrs.get("review_flags", []),
        }
        write_json(manifest_path, manifest)
        print(f"\nFROZEN manifest written to {manifest_path}")
        print(f"Selected CONFIGURATION (not a single run): {selected_configuration}")
        print(f"All 3 seed run_dirs: {run_dirs}")
        print("Test-partition evaluation for this configuration can now be run via:\n"
              f"  python unlock_test_evaluation.py --selection-manifest {manifest_path}")
    else:
        print("\n(not frozen -- pass --freeze once you have reviewed and approved "
              "this ranking; Stage 2 job generation requires a frozen Stage 1 manifest.)")


if __name__ == "__main__":
    main()
