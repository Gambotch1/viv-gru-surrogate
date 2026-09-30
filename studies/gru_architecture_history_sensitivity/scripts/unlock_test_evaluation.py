"""Evaluate the test cases for the frozen, selected configuration only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import (
    CANONICAL_TEST_CASE_COUNT, STUDY_ROOT, assert_frozen_and_get_selected_run_dirs,
    compute_open_loop_metrics, load_json, write_json,
)


def _iqr(values: list[float]) -> float:
    q75, q25 = np.percentile(values, [75, 25])
    return float(q75 - q25)


def main() -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection-manifest", type=Path, required=True)
    args = p.parse_args()

    run_dirs = assert_frozen_and_get_selected_run_dirs(args.selection_manifest)
    manifest = load_json(args.selection_manifest)
    dataset = manifest["dataset"]
    canonical_n = CANONICAL_TEST_CASE_COUNT[dataset]

    per_seed = {}
    for run_dir_str in run_dirs:
        run_dir = Path(run_dir_str)
        run_config = load_json(run_dir / "run_config.json")
        test_cases = run_config["test_cases"]
        seed = load_json(run_dir / "study_receipt.json")["seed"]

        assert run_config["skip_test_eval"] is True, (
            f"{run_dir} was trained WITHOUT --skip_test_eval -- this study "
            f"never evaluates test metrics for a run that could have leaked "
            f"test information into checkpoint selection earlier.")
        assert len(test_cases) == canonical_n, (
            f"{run_dir}: {len(test_cases)} test case(s), expected exactly "
            f"{canonical_n} for {dataset}.")
        audit = load_json(STUDY_ROOT / "manifests" / "stage0_audit.json")
        canonical_test_cases = set(audit[dataset]["test_cases"])
        assert set(test_cases) == canonical_test_cases, (
            f"{run_dir}: run_config['test_cases']={sorted(test_cases)} does "
            f"not match the Stage 0 audit's canonical test set "
            f"{sorted(canonical_test_cases)} for {dataset}.")

        print(f"Computing test metrics for seed={seed} ({run_dir}) "
              f"-- FIRST TIME this run's test partition is evaluated.")
        result = compute_open_loop_metrics(run_dir, test_cases, case_kind="test")
        write_json(run_dir / "test_eval_unlocked.json", result)
        per_seed[seed] = result
        agg = result["aggregate_test_metrics"]
        print(f"  seed={seed}: R2={agg['r2']:.4f}  RMSE={agg['rmse']:.4f}  "
              f"NRMSE={agg['nrmse']:.4f}")

    r2_values = [per_seed[s]["aggregate_test_metrics"]["r2"] for s in per_seed]
    nrmse_values = [per_seed[s]["aggregate_test_metrics"]["nrmse"] for s in per_seed]

    report = {
        "selection_manifest": str(args.selection_manifest),
        "dataset": dataset,
        "selected_configuration": manifest["selected_configuration"],
        "seeds": sorted(per_seed.keys()),
        "per_seed_test_metrics": {
            str(s): per_seed[s]["aggregate_test_metrics"] for s in per_seed
        },
        "median_test_r2": float(np.median(r2_values)),
        "iqr_test_r2": _iqr(r2_values),
        "median_test_nrmse": float(np.median(nrmse_values)),
        "iqr_test_nrmse": _iqr(nrmse_values),
        "note": "All 3 seeds are reported above; none is singled out as "
                "'the' result. Do not report only the best-performing seed.",
    }
    out_path = (STUDY_ROOT / "reports" /
                f"test_unlock_{dataset}_stage{manifest['stage']}.json")
    write_json(out_path, report)
    print(f"\nWrote {out_path}")
    print(f"Median test R2 = {report['median_test_r2']:.4f}  "
          f"(IQR={report['iqr_test_r2']:.4f})  across seeds {report['seeds']}")
    return report


if __name__ == "__main__":
    main()
