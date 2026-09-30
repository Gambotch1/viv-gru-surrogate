"""Closed-loop run of selected training cases (e.g. Ur 6.7385 for the bridge) as a mechanistic check. Not used for selection."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import FULL_DURATION_S, load_json, run_coupled_sweep, write_json
from viv_analysis.utils import parse_ur_label


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--ur", type=float, default=None,
                   help="Sweep a single Ur (must be one of this run's "
                        "train_cases) instead of the full training set, "
                        "e.g. --ur 6.7385 for the bridge mechanistic control.")
    p.add_argument("--handoff_offset", type=int, default=2000)
    p.add_argument("--window_frac", type=float, default=0.5)
    p.add_argument("--pass_amp_rel_error_threshold", type=float, default=0.20)
    p.add_argument("--timeout_s", type=int, default=1800)
    return p.parse_args()


def main() -> dict:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_config = load_json(run_dir / "run_config.json")
    dataset = run_config["cfd_dataset"]
    train_cases = run_config["train_cases"]

    if args.ur is not None:
        matches = [c for c in train_cases if abs(parse_ur_label(c) - args.ur) < 1e-6]
        if not matches:
            raise SystemExit(
                f"--ur {args.ur} is not one of this run's train_cases "
                f"({sorted(train_cases)}) -- this diagnostic path only "
                f"sweeps TRAINING cases, never val/test.")
        sweep_cases = matches
    else:
        sweep_cases = train_cases

    assert not (set(sweep_cases) & set(run_config["val_cases"])), \
        "val case leaked into the training diagnostic sweep -- refusing to run"
    assert not (set(sweep_cases) & set(run_config["test_cases"])), \
        "test case leaked into the training diagnostic sweep -- refusing to run"

    out_dir = run_dir / "closed_loop_train_diagnostic"
    print(f"[TRAINING-CASE DIAGNOSTIC -- excluded from selection] "
          f"Dataset={dataset}  total_time={FULL_DURATION_S[dataset]}s  "
          f"sweep={len(sweep_cases)} case(s): {sorted(sweep_cases)}")

    sweep_df, timings, label = run_coupled_sweep(
        run_dir, sweep_cases, out_dir,
        handoff_offset=args.handoff_offset,
        window_frac=args.window_frac,
        pass_amp_rel_error_threshold=args.pass_amp_rel_error_threshold,
        timeout_s=args.timeout_s,
    )
    sweep_df.to_csv(out_dir / "sweep_results_DIAGNOSTIC_ONLY.csv", index=False)

    summary = {
        "run_dir": str(run_dir), "dataset": dataset,
        "diagnostic_only": True,
        "excluded_from_selection": True,
        "note": "This output is a training-case mechanistic diagnostic. It "
                "is never read by collect_results.py or select_configuration.py "
                "and must not be cited as evidence for architecture/history "
                "selection.",
        "swept_cases": sorted(sweep_cases),
        "n_sweep_cases": len(sweep_cases),
    }
    write_json(out_dir / "closed_loop_train_diagnostic_summary.json", summary)
    print(f"\nWrote {out_dir / 'sweep_results_DIAGNOSTIC_ONLY.csv'} "
          f"(diagnostic-only, excluded from selection)")
    return summary


if __name__ == "__main__":
    main()
