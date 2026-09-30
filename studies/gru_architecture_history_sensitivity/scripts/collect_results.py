"""Collect the per-run results of a stage into one table and aggregate over seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import REQUIRED_SEEDS, STUDY_ROOT, write_json


def _iqr(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None
    q75, q25 = np.percentile(vals, [75, 25])
    return float(q75 - q25)


def collect(dataset: str, stage: int) -> pd.DataFrame:
    base = STUDY_ROOT / "results" / dataset / f"stage{stage}"
    rows = []
    if not base.exists():
        return pd.DataFrame()

    for run_dir in sorted(base.iterdir()):
        receipt_path = run_dir / "study_receipt.json"
        if not receipt_path.exists():
            continue
        receipt = json.loads(receipt_path.read_text())

        row = {
            "run_dir": str(run_dir),
            "dataset": dataset,
            "stage": stage,
            "hidden_size": receipt["hidden_size"],
            "num_layers": receipt["num_layers"],
            "seq_len": receipt["sequence_samples"],
            "sequence_duration_s": receipt["sequence_duration_s"],
            "history_label": receipt.get("history_label"),
            "seed": receipt["seed"],
            "trainable_parameter_count": receipt["trainable_parameter_count"],
            "elapsed_training_time_s": receipt["elapsed_training_time_s"],
            "peak_gpu_memory_mb": receipt["peak_gpu_memory_mb"],
            "valid": True,
        }

        ol_path = run_dir / "open_loop_val_metrics.json"
        if ol_path.exists():
            ol = json.loads(ol_path.read_text())
            row["open_loop_val_r2"] = ol["aggregate_val_metrics"]["r2"]
            row["open_loop_val_rmse"] = ol["aggregate_val_metrics"]["rmse"]
            row["open_loop_val_nrmse"] = ol["aggregate_val_metrics"]["nrmse"]
            row["open_loop_val_mae"] = ol["aggregate_val_metrics"]["mae"]
            row["open_loop_median_case_r2"] = ol["median_val_r2"]
        else:
            row["open_loop_val_r2"] = None

        cl_path = run_dir / "closed_loop_eval" / "closed_loop_summary.json"
        if cl_path.exists():
            cl = json.loads(cl_path.read_text())
            for k, v in cl.items():
                if k not in row:
                    row[f"closed_loop_{k}"] = v
        else:
            row["closed_loop_validation_n"] = None

        if (not np.isfinite(row.get("open_loop_val_r2") or np.nan)
                and row.get("open_loop_val_r2") is not None):
            row["valid"] = False

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = ["dataset", "stage", "hidden_size", "num_layers", "seq_len", "history_label"]
    numeric_cols = [c for c in df.columns if df[c].dtype.kind in "fi"
                    and c not in ("seed",)]
    agg_rows = []
    for key, grp in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row["n_seeds"] = len(grp)
        row["seeds"] = sorted(grp["seed"].tolist())
        has_all_required_seeds = set(grp["seed"].tolist()) == set(REQUIRED_SEEDS)
        row["all_valid"] = bool(grp["valid"].all()) and has_all_required_seeds
        for c in numeric_cols:
            vals = grp[c].tolist()
            row[f"{c}_median"] = float(np.nanmedian(vals)) if len(vals) else None
            row[f"{c}_iqr"] = _iqr(vals)
        agg_rows.append(row)
    return pd.DataFrame(agg_rows)


def _fmt_median_iqr(median, iqr, decimals=3) -> str:
    if median is None or (isinstance(median, float) and np.isnan(median)):
        return "n/a"
    if iqr is None or (isinstance(iqr, float) and np.isnan(iqr)):
        return f"{median:.{decimals}f}"
    return f"{median:.{decimals}f} [{iqr:.{decimals}f}]"


def format_collection_report(dataset: str, agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return pd.DataFrame()
    if dataset == "bridge":
        stable_col, denom_col = ("closed_loop_validation_stable_count_among_settled_lco",
                                  "closed_loop_validation_n_settled_lco")
        amp_col, worst_col = ("closed_loop_validation_median_abs_rel_amp_error_among_settled_lco",
                               "closed_loop_validation_worst_abs_rel_amp_error_among_settled_lco")
    else:
        stable_col, denom_col = ("closed_loop_validation_stable_count",
                                  "closed_loop_validation_n")
        amp_col, worst_col = ("closed_loop_validation_median_abs_rel_amp_error",
                               "closed_loop_validation_worst_abs_rel_amp_error")

    rows = []
    for _, r in agg.iterrows():
        stable = r.get(f"{stable_col}_median")
        denom = r.get(f"{denom_col}_median")
        params = r.get("trainable_parameter_count_median")
        worst = r.get(f"{worst_col}_median")
        rows.append({
            "Dataset": r["dataset"],
            "Hidden size": int(r["hidden_size"]),
            "Layers": int(r["num_layers"]),
            "Parameters": int(params) if pd.notna(params) else None,
            "Open-loop R2 median[IQR]": _fmt_median_iqr(
                r.get("open_loop_val_r2_median"), r.get("open_loop_val_r2_iqr")),
            "Stable runs": (f"{stable:.0f}/{denom:.0f}"
                             if pd.notna(stable) and pd.notna(denom) else "n/a"),
            "Closed-loop amplitude error median[IQR]": _fmt_median_iqr(
                r.get(f"{amp_col}_median"), r.get(f"{amp_col}_iqr")),
            "Worst error": f"{worst:.3f}" if pd.notna(worst) else "n/a",
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["cylinder200", "bridge"])
    p.add_argument("--stage", type=int, required=True, choices=[1, 2])
    args = p.parse_args()

    per_run = collect(args.dataset, args.stage)
    reports_dir = STUDY_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    per_run_csv = reports_dir / f"per_run_{args.dataset}_stage{args.stage}.csv"
    per_run.to_csv(per_run_csv, index=False)
    print(f"Wrote {len(per_run)} run(s) to {per_run_csv}")

    agg = aggregate(per_run)
    agg_csv = reports_dir / f"aggregated_{args.dataset}_stage{args.stage}.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"Wrote {len(agg)} aggregated config(s) to {agg_csv}")

    report = format_collection_report(args.dataset, agg)
    if not report.empty:
        report_csv = reports_dir / f"collection_report_{args.dataset}_stage{args.stage}.csv"
        report.to_csv(report_csv, index=False)
        print(f"\nCollection report (one row per architecture, not per seed) "
              f"written to {report_csv}:\n")
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()
