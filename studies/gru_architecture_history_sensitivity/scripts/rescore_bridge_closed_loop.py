"""One-off: rebuild the bridge status_aware_report.csv of existing closed-loop runs without rerunning them."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import FULL_DURATION_S, STUDY_ROOT, load_json, write_json
from viv_analysis.reference_quality import build_status_aware_report


def rescore_run_dir(run_dir: Path) -> dict:
    out_dir = run_dir / "closed_loop_eval"
    csv_path = out_dir / "sweep_results.csv"
    receipt = load_json(run_dir / "study_receipt.json")
    run_config = load_json(run_dir / "run_config.json")
    dataset = run_config["cfd_dataset"]
    assert dataset == "bridge", f"{run_dir}: not a bridge run"

    sweep_df = pd.read_csv(csv_path)
    label = f"{dataset}_{run_dir.name}"

    summary = {
        "run_dir": str(run_dir), "dataset": dataset,
        "total_time_s": FULL_DURATION_S[dataset],
        "n_sweep_cases": len(sweep_df),
        "sweep_cases_are_canonical_validation_partition": True,
        "trainable_parameter_count": receipt["trainable_parameter_count"],
        "mean_inference_time_s": (
            float(np.nanmean(sweep_df["inference_time_s"]))
            if "inference_time_s" in sweep_df and sweep_df["inference_time_s"].notna().any()
            else None),
    }

    freq_errs = sweep_df[f"{label}_f_osc_rel_error"].dropna()
    summary["validation_median_abs_f_osc_rel_error"] = (
        float(freq_errs.abs().median()) if len(freq_errs) else None)

    report = build_status_aware_report(str(out_dir), label)
    report.to_csv(out_dir / "status_aware_report.csv", index=False)

    is_settled = report.get("scoring_method", pd.Series(dtype=object)) == "lco_gate"
    settled = report[is_settled]
    non_lco = report[~is_settled & ~report.get("unscored", pd.Series(dtype=bool)).fillna(False)]
    settled_errs = settled["A_star_rel_error"].dropna() if "A_star_rel_error" in settled else pd.Series(dtype=float)

    summary.update(
        status_aware_report_csv=str(out_dir / "status_aware_report.csv"),
        validation_n=len(report),
        validation_unscored=int(report.get("unscored", pd.Series(dtype=bool)).sum())
            if "unscored" in report else None,
        validation_n_settled_lco=int(is_settled.sum()),
        validation_stable_count_among_settled_lco=(
            int(settled["pass_"].sum()) if "pass_" in settled else None),
        validation_median_abs_rel_amp_error_among_settled_lco=(
            float(settled_errs.abs().median()) if len(settled_errs) else None),
        validation_worst_abs_rel_amp_error_among_settled_lco=(
            float(settled_errs.abs().max()) if len(settled_errs) else None),
        validation_p90_abs_rel_amp_error_among_settled_lco=(
            float(settled_errs.abs().quantile(0.90)) if len(settled_errs) else None),
        validation_n_non_lco=int(len(non_lco)),
        validation_mean_rms_ratio_among_non_lco=(
            float(non_lco["mean_rms_ratio"].dropna().mean())
            if "mean_rms_ratio" in non_lco and len(non_lco) else None),
    )

    write_json(out_dir / "closed_loop_summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, default=1, choices=[1, 2])
    args = p.parse_args()

    base = STUDY_ROOT / "results" / "bridge" / f"stage{args.stage}"
    run_dirs = sorted(d for d in base.iterdir() if (d / "study_receipt.json").exists())
    print(f"Rescoring {len(run_dirs)} bridge stage{args.stage} run(s)...")
    for run_dir in run_dirs:
        if not (run_dir / "closed_loop_eval" / "sweep_results.csv").exists():
            print(f"  [skip] {run_dir.name}: no sweep_results.csv")
            continue
        summary = rescore_run_dir(run_dir)
        print(f"  {run_dir.name}: n_settled_lco={summary['validation_n_settled_lco']} "
              f"stable_among_settled={summary['validation_stable_count_among_settled_lco']} "
              f"n_non_lco={summary['validation_n_non_lco']}")


if __name__ == "__main__":
    main()
