"""Train one configuration of the grid (hidden size, layers, sequence length, seed) with train_gru.py, without touching the test cases."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from _common import (
    REPO_ROOT, STUDY_ROOT, STUDY_VERSION, dataset_cache_version,
    exp_subdir_arg, git_commit, load_json, run_output_dir, write_json,
)

import torch  # noqa: E402
from viv_analysis import config as config_module  # noqa: E402
from viv_analysis.models.gru import VIV_GRU  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["cylinder200", "bridge"])
    p.add_argument("--hidden_size", type=int, required=True)
    p.add_argument("--num_layers", type=int, required=True)
    p.add_argument("--seq_len", type=int, required=True,
                   help="Stage 1: dataset baseline (1000 cylinder200 / 2500 bridge). "
                        "Stage 2: the history-grid point under test.")
    p.add_argument("--seed", type=int, required=True, choices=[123, 456, 789])
    p.add_argument("--stage", type=int, required=True, choices=[1, 2])
    p.add_argument("--history_label", default=None,
                   help="Stage 2 only, e.g. '0.5Tn', '4Tn' (from "
                        "configs/{dataset}_history_grid.json).")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                   help="Validate config/paths and print the resolved "
                        "train_gru invocation; do not actually train.")
    p.add_argument("--smoke", action="store_true",
                   help="Write to smoke/ instead of results/ -- invisible "
                        "to collect_results.py / select_configuration.py by "
                        "construction. For temporary end-to-end validation "
                        "runs only; never used by the Stage 1/2 job arrays.")
    p.add_argument("--epochs", type=int, default=100,
                   help="Default 100 (the audited production baseline). "
                        "Smoke runs pass --epochs 1.")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Overrides configs/architecture_grid.json's "
                        "fixed_hyperparameters.batch_size (512) for THIS run "
                        "only. The grid is fixed-hyperparameters-by-construction "
                        "specifically so no config gets this override silently -- "
                        "use only as a disclosed, deliberate exception (e.g. a "
                        "history point whose backward-pass memory footprint "
                        "does not fit at batch_size=512 on available hardware), "
                        "never to tune away a bad early result. Recorded "
                        "verbatim in study_receipt.json's batch_size field, "
                        "with fixed_hyperparameters_deviation set, so it is "
                        "never mistaken for the grid default downstream.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    fixed = load_json(STUDY_ROOT / "configs" / "architecture_grid.json")["fixed_hyperparameters"]

    output_dir = run_output_dir(args.dataset, args.stage, args.hidden_size,
                                 args.num_layers, args.seq_len, args.seed,
                                 args.history_label, smoke=args.smoke)
    if output_dir.exists() and not args.overwrite and not args.dry_run:
        existing = list(output_dir.glob("gru_best.pt"))
        if existing:
            raise SystemExit(
                f"{output_dir} already has a checkpoint; pass --overwrite "
                f"to retrain, or --dry_run to just validate.")

    exp_subdir = exp_subdir_arg(output_dir)

    batch_size = args.batch_size if args.batch_size is not None else fixed["batch_size"]
    if args.batch_size is not None:
        print(f"WARNING: --batch_size {args.batch_size} overrides the fixed grid "
              f"value ({fixed['batch_size']}) for this run only -- disclosed "
              f"exception, recorded in study_receipt.json.")

    train_gru_argv = [
        "train_gru.py",
        "--cfd_dataset", args.dataset,
        "--seq_len", str(args.seq_len),
        "--epochs", str(args.epochs),
        "--batch_size", str(batch_size),
        "--seed", str(args.seed),
        "--num_workers", "0",
        "--noise_std", str(fixed["noise_std"]),
        "--input_cols", *fixed["input_cols"],
        "--use_ur_context",
        "--skip_test_eval",
        "--exp_subdir", exp_subdir,
    ]
    if fixed["nd_inputs"]:
        train_gru_argv.append("--nd_inputs")
    if args.overwrite:
        train_gru_argv.append("--overwrite")

    print("=" * 60)
    print(f"Study:        gru_architecture_history_sensitivity (v{STUDY_VERSION})")
    print(f"Stage:        {args.stage}")
    print(f"Dataset:      {args.dataset}")
    print(f"Architecture: hidden_size={args.hidden_size} num_layers={args.num_layers}")
    print(f"seq_len:      {args.seq_len}  (history_label={args.history_label})")
    print(f"Seed:         {args.seed}")
    print(f"Output dir:   {output_dir}  (smoke={args.smoke})")
    print(f"exp_subdir:   {exp_subdir}  (resolves to {output_dir})")
    print("train_gru.py argv:")
    print(" ", " ".join(train_gru_argv))
    print("=" * 60)

    param_count = sum(
        p.numel() for p in VIV_GRU(
            input_size=len(fixed["input_cols"]) + 1,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=fixed["dropout"],
        ).parameters() if p.requires_grad
    )
    print(f"Trainable parameter count (H={args.hidden_size}, L={args.num_layers}): "
          f"{param_count:,}")
    if args.num_layers == 1:
        print("Note: num_layers=1 -> VIV_GRU passes dropout=0.0 to nn.GRU "
              "regardless of the configured 0.1 (PyTorch inter-layer dropout "
              "is inactive for a single-layer GRU).")

    if args.dry_run:
        print("[DRY RUN] Not training. Config/paths validated OK.")
        return

    config_module.config["hidden_size"] = args.hidden_size
    config_module.config["num_layers"] = args.num_layers

    sys.argv = train_gru_argv
    t0 = time.time()

    from viv_analysis import train_gru
    train_gru.main()

    elapsed_s = time.time() - t0
    peak_mem_mb = (torch.cuda.max_memory_allocated() / 1e6
                   if torch.cuda.is_available() else None)

    run_config = load_json(output_dir / "run_config.json")
    assert run_config["skip_test_eval"] is True, (
        "train_gru.py reported skip_test_eval=False in run_config.json even "
        "though train_sensitivity.py always passes --skip_test_eval -- "
        "refusing to write a receipt for a run that may have touched test data.")
    metrics = load_json(output_dir / "metrics_gru.json")
    assert metrics["test_metrics"] is None and metrics["tf_results"] == {}, (
        "metrics_gru.json contains test-derived content despite "
        "--skip_test_eval -- refusing to write a receipt.")
    receipt = {
        "study": "gru_architecture_history_sensitivity",
        "study_version": STUDY_VERSION,
        "git_commit": git_commit(),
        "dataset_cache_version": dataset_cache_version(args.dataset),
        "dataset": args.dataset,
        "stage": args.stage,
        "history_label": args.history_label,
        "train_cases": run_config["train_cases"],
        "val_cases": run_config["val_cases"],
        "test_cases": run_config["test_cases"],
        "input_cols": run_config["input_cols"],
        "acceleration_present": "acc" in run_config["input_cols"],
        "coordinate_mode": run_config["coordinate_mode"],
        "transform_formula": run_config["transform_formula"],
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "sequence_samples": args.seq_len,
        "dt_s": (0.005 if args.dataset == "cylinder200" else 0.002),
        "sequence_duration_s": args.seq_len * (0.005 if args.dataset == "cylinder200" else 0.002),
        "seed": args.seed,
        "batch_size": batch_size,
        "fixed_hyperparameters_deviation": (
            None if args.batch_size is None else
            f"batch_size={batch_size} overrides the grid's fixed value "
            f"({fixed['batch_size']}) -- disclosed exception, not the study default."
        ),
        "optimizer": "Adam", "lr": fixed["lr"], "weight_decay": fixed["weight_decay"],
        "scaler_source": "sklearn StandardScaler fit_scalers()",
        "scaler_fit_partition": run_config["scaler_fit_cases"],
        "trainable_parameter_count": param_count,
        "checkpoint_path": str(output_dir / "gru_best.pt"),
        "elapsed_training_time_s": elapsed_s,
        "peak_gpu_memory_mb": peak_mem_mb,
        "evaluation_status": "trained; open/closed-loop VALIDATION evaluation "
                              "pending (run evaluate_open_loop.py / "
                              "evaluate_closed_loop.py). No test evaluation "
                              "has been performed or will be performed for "
                              "this run outside of unlock_test_evaluation.py.",
        "skip_test_eval": True,
        "test_partition_evaluated": False,
        "smoke": args.smoke,
    }
    write_json(output_dir / "study_receipt.json", receipt)
    print(f"\nStudy receipt written to {output_dir / 'study_receipt.json'}")
    print(f"Elapsed: {elapsed_s:.1f}s  Peak GPU mem: {peak_mem_mb} MB")


if __name__ == "__main__":
    main()
