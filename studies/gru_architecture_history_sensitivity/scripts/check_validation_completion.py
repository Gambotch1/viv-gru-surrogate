"""Check that every run of a stage has its open- and closed-loop validation results."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

from _common import STUDY_ROOT, load_json


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_run_dir(run_dir: Path, canonical_val: set[str], canonical_test: set[str]) -> dict:
    from viv_analysis.utils import parse_ur_label

    result = {"run_dir": str(run_dir), "problems": []}

    run_config = load_json(run_dir / "run_config.json")
    val_cases = set(run_config["val_cases"])
    test_cases = set(run_config["test_cases"])
    result["val_cases"] = sorted(val_cases)

    if val_cases != canonical_val:
        result["problems"].append(
            f"run_config.json val_cases {sorted(val_cases)} != canonical "
            f"{sorted(canonical_val)}")
    if val_cases & test_cases:
        result["problems"].append(f"val/test overlap in run_config.json: {sorted(val_cases & test_cases)}")

    canonical_val_ur = {round(parse_ur_label(c), 6) for c in canonical_val}
    canonical_test_ur = {round(parse_ur_label(c), 6) for c in canonical_test}

    ol_path = run_dir / "open_loop_val_metrics.json"
    result["open_loop_complete"] = ol_path.exists()
    if ol_path.exists():
        ol = load_json(ol_path)
        ol_val_cases = set(ol.get("val_cases", []))
        if ol_val_cases != canonical_val:
            result["problems"].append(
                f"open_loop_val_metrics.json val_cases {sorted(ol_val_cases)} "
                f"!= canonical {sorted(canonical_val)}")
    else:
        result["problems"].append("open_loop_val_metrics.json missing")

    cl_summary_path = run_dir / "closed_loop_eval" / "closed_loop_summary.json"
    result["closed_loop_complete"] = cl_summary_path.exists()
    if cl_summary_path.exists():
        sweep_csv = run_dir / "closed_loop_eval" / "sweep_results.csv"
        if not sweep_csv.exists():
            result["problems"].append("closed_loop_summary.json exists but sweep_results.csv is missing")
        else:
            swept_ur = {round(float(u), 6) for u in pd.read_csv(sweep_csv)["Ur"]}
            extra_test = swept_ur & canonical_test_ur
            if extra_test:
                result["problems"].append(
                    f"sweep_results.csv contains TEST-case Ur value(s) {sorted(extra_test)} "
                    f"-- this should be structurally impossible (evaluate_closed_loop.py "
                    f"asserts val/test have no overlap before sweeping); treat as a hard failure")
            if swept_ur != canonical_val_ur:
                result["problems"].append(
                    f"sweep_results.csv Ur values {sorted(swept_ur)} != canonical "
                    f"validation Ur values {sorted(canonical_val_ur)}")
    else:
        result["problems"].append("closed_loop_eval/closed_loop_summary.json missing")

    result["sha256"] = {
        "gru_best.pt": _sha256(run_dir / "gru_best.pt"),
        "x_scaler.pkl": _sha256(run_dir / "x_scaler.pkl"),
        "y_scaler.pkl": _sha256(run_dir / "y_scaler.pkl"),
    }
    for name, digest in result["sha256"].items():
        if digest is None:
            result["problems"].append(f"{name} missing -- cannot hash (but study_receipt.json exists?)")

    result["ok"] = not result["problems"]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["cylinder200", "bridge", "all"], default="all")
    p.add_argument("--stage", type=int, default=1, choices=[1, 2])
    p.add_argument("--expect", type=int, default=None,
                   help="Assert exactly this many trained run_dirs are found "
                        "(across the selected dataset(s)) -- omit to just report "
                        "whatever is currently on disk.")
    args = p.parse_args()

    audit = load_json(STUDY_ROOT / "manifests" / "stage0_audit.json")
    datasets = ["cylinder200", "bridge"] if args.dataset == "all" else [args.dataset]

    all_results = []
    for dataset in datasets:
        canonical_val = set(audit[dataset]["val_cases"])
        canonical_test = set(audit[dataset]["test_cases"])
        base = STUDY_ROOT / "results" / dataset / f"stage{args.stage}"
        if not base.exists():
            continue
        for run_dir in sorted(base.iterdir()):
            if not (run_dir / "study_receipt.json").exists():
                continue
            r = check_run_dir(run_dir, canonical_val, canonical_test)
            r["dataset"] = dataset
            all_results.append(r)

    n_trained = len(all_results)
    n_ol_ok = sum(r["open_loop_complete"] for r in all_results)
    n_cl_ok = sum(r["closed_loop_complete"] for r in all_results)
    n_clean = sum(r["ok"] for r in all_results)

    print(f"Trained run_dirs found: {n_trained}")
    print(f"  open_loop completion receipts:   {n_ol_ok}/{n_trained}")
    print(f"  closed_loop completion receipts: {n_cl_ok}/{n_trained}")
    print(f"  fully clean (no problems):       {n_clean}/{n_trained}")

    for r in all_results:
        if r["problems"]:
            print(f"\nPROBLEM(S) in {r['run_dir']}:")
            for prob in r["problems"]:
                print(f"  - {prob}")

    if args.expect is not None and n_trained != args.expect:
        print(f"\nFAIL: expected exactly {args.expect} trained run_dir(s), found {n_trained}.")
        sys.exit(1)

    if n_clean != n_trained:
        print(f"\nFAIL: {n_trained - n_clean}/{n_trained} run_dir(s) have unresolved problems "
              f"-- do not run collect_results.py / select_configuration.py --freeze yet.")
        sys.exit(1)

    print(f"\nOK: all {n_trained} trained run_dir(s) have clean open-loop + closed-loop "
          f"validation completion, exact canonical validation-case identities, and no "
          f"test-case contamination. sha256 hash-of-record for gru_best.pt/x_scaler.pkl/"
          f"y_scaler.pkl has been computed for each (see per-run 'sha256' field if you "
          f"re-run with output captured) -- note this is a freshly-established hash-of-"
          f"record, not a comparison against a training-time hash, since none was "
          f"previously recorded anywhere in this study.")


if __name__ == "__main__":
    main()
