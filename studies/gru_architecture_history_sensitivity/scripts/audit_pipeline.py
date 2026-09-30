"""Stage 0: record the production settings this study keeps fixed and the trainable-parameter count of every grid configuration (manifests/stage0_audit.json)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from viv_analysis.config import (  # noqa: E402
    config, prepare_gru_config, bridge_structural_params,
    cylinder200_structural_params,
)
from viv_analysis.train_gru import (  # noqa: E402
    split_cases, CYLINDER200_ALIASES,
)
from viv_analysis.preprocess import merge_dataframes, downsample, compute_kinematics  # noqa: E402
from viv_analysis.models.gru import VIV_GRU  # noqa: E402

STUDY_INPUT_SIZE = 3


def compute_parameter_counts() -> list[dict]:
    grid = json.loads((STUDY_ROOT / "configs" / "architecture_grid.json").read_text())
    rows = []
    for H in grid["hidden_sizes"]:
        for L in grid["num_layers"]:
            model = VIV_GRU(input_size=STUDY_INPUT_SIZE, hidden_size=H,
                             num_layers=L, dropout=grid["fixed_hyperparameters"]["dropout"])
            count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            rows.append({
                "hidden_size": H, "num_layers": L,
                "trainable_parameter_count": count,
                "dropout_active": L > 1,
            })
    return rows


def _load_cases(dataset: str) -> tuple[list[str], dict]:
    cfg = prepare_gru_config(dataset, config).copy()
    if dataset == "bridge":
        raw_df = merge_dataframes(dataset="bridge", fn_hz=cfg["bridge_fn_hz"],
                                   d_ref=cfg["bridge_D_ref"])
        raw_df = downsample(raw_df, cfg["bridge_downsample"])
        params_bridge = bridge_structural_params()
        raw_df = compute_kinematics(raw_df, dataset=dataset,
                                     bridge_structural_params=params_bridge)
        sizes = raw_df.groupby("case").size().sort_values()
        med = float(sizes.median())
        drop = set(sizes[sizes < 0.05 * med].index)
        if drop:
            raw_df = raw_df[~raw_df["case"].isin(drop)].copy()
    else:
        raw_df = merge_dataframes(dataset=dataset)
        params = cylinder200_structural_params()
        raw_df = compute_kinematics(raw_df, dataset=dataset, structural_params=params)

    all_cases = sorted(str(c) for c in raw_df["case"].drop_duplicates())
    return all_cases, cfg


def audit_dataset(dataset: str) -> dict:
    all_cases, cfg = _load_cases(dataset)
    train, val, test, release_time = split_cases(all_cases, dataset, cfg)

    result = {
        "dataset": dataset,
        "n_retained_cases": len(all_cases),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "train_cases": sorted(train),
        "val_cases": sorted(val),
        "test_cases": sorted(test),
        "seq_len_baseline": cfg["seq_len"],
        "stride_train": cfg["stride_train"],
        "use_ur_context_default": cfg["use_ur_context"],
    }
    return result


def main() -> dict:
    cyl = audit_dataset("cylinder200")
    bridge = audit_dataset("bridge")

    assert cyl["n_train"] == 13, f"cylinder200 train count {cyl['n_train']} != 13"
    assert cyl["n_val"] == 4, f"cylinder200 val count {cyl['n_val']} != 4"
    assert cyl["n_test"] == 4, f"cylinder200 test count {cyl['n_test']} != 4"

    assert bridge["n_retained_cases"] == 27, \
        f"bridge retained count {bridge['n_retained_cases']} != 27"
    assert bridge["n_train"] == 17, f"bridge train count {bridge['n_train']} != 17"
    assert bridge["n_val"] == 5, f"bridge val count {bridge['n_val']} != 5"
    assert bridge["n_test"] == 5, f"bridge test count {bridge['n_test']} != 5"
    all_bridge = set(bridge["train_cases"]) | set(bridge["val_cases"]) | set(bridge["test_cases"])
    assert "Ur8.2126" not in all_bridge, \
        "excluded bridge case Ur8.2126 (U=19.5 m/s) is present in the retained partition"

    audit = {
        "study": "gru_architecture_history_sensitivity",
        "stage": 0,
        "optimizer": "torch.optim.Adam",
        "initial_lr": config["lr"],
        "weight_decay": config["weight_decay"],
        "scheduler": {"type": "ReduceLROnPlateau", "mode": "min", "factor": 0.5,
                      "patience": 5, "note": "applied on top of, and separate from, "
                      "the early-stopping patience below"},
        "dropout": config["dropout"],
        "dropout_note": "VIV_GRU (src/viv_analysis/models/gru.py) passes "
                         "dropout=0.0 to nn.GRU whenever num_layers==1 -- "
                         "PyTorch only applies inter-layer dropout when "
                         "num_layers>1, so this is already handled correctly "
                         "by production code, not something the study needs "
                         "to special-case.",
        "batch_size_default": config["batch_size"],
        "training_stride": {"cylinder200": cyl["stride_train"],
                             "bridge": bridge["stride_train"]},
        "early_stopping_patience": config["patience"],
        "max_epochs": config["n_epochs"],
        "gradient_clip_max_norm": 1.0,
        "target_scaling": "sklearn StandardScaler fit on target_col ('cl'), "
                           "training cases only (fit_scalers)",
        "input_scaling": "sklearn StandardScaler fit on input_cols, "
                          "training cases only (fit_scalers); noise (if any) "
                          "is injected AFTER scaling, in standardized space",
        "cylinder200_dt_s": config["cylinder200_dt"],
        "bridge_dt_eff_s": 0.002,
        "bridge_downsample_factor": config["bridge_downsample"],
        "cylinder200_seq_len_baseline": cyl["seq_len_baseline"],
        "bridge_seq_len_baseline": bridge["seq_len_baseline"],
        "cylinder200_release_rule": "t_release(Ur) = t_star_release(=40) * D / U(Ur), "
                                     "per-case (cylinder200_release_time)",
        "bridge_release_rule": "t_release(Ur) = t_star_release(=20) * D_ref(=7.42) / U(Ur), "
                                "per-case (_bridge_split)",
        "handoff_offset_steps_default": 2000,
        "closed_loop_full_duration_convention": {
            "cylinder200_s": 500.0,
            "bridge_s": 300.0,
            "note": "cylinder200 changed from 700s to 500s in production "
                    "convention as of an earlier session; bridge changed from "
                    "700s to 300s later, matching the actual max CFD reference "
                    "duration (every bridge case tops out at t=300s, several "
                    "earlier). This study's closed-loop protocol uses these "
                    "same per-dataset full-duration values so results are "
                    "directly comparable to current production evaluations.",
        },
        "reference_quality_classifiers": {
            "module": "src/viv_analysis/reference_quality.py",
            "classify_reference_status": "per-Ur -> settled_lco | "
                "statistically_stationary_les | transient_or_slowly_evolving | "
                "insufficient_duration | numerically_suspect",
            "build_status_aware_report": "routes settled_lco cases to the "
                "amplitude-based lco_gate scoring, statistically_stationary_les "
                "and transient_or_slowly_evolving cases to compute_non_lco_summary "
                "(scoring_method='non_lco_block_energy'), and leaves "
                "insufficient_duration/numerically_suspect cases unscored.",
            "reference_status_csv": "results/bridge_reference_status.csv",
        },
        "cylinder200": cyl,
        "bridge": bridge,
        "parameter_counts": compute_parameter_counts(),
        "discrepancies": [
            {
                "id": "train_gru_always_evaluates_test_partition",
                "severity": "high",
                "status": "RESOLVED (production code change applied)",
                "description": "train_gru.py's main() previously "
                    "unconditionally built test_df/test_loader and wrote "
                    "'test_metrics' plus per-case test-split entries inside "
                    "'tf_results' into metrics_gru.json on every run, with "
                    "no CLI flag to skip it. This still applies to every "
                    "PRE-EXISTING production checkpoint on disk (P0, the "
                    "ablation matrix, etc.) trained before this change.",
                "resolution_applied": "Added an opt-in --skip_test_eval flag "
                    "to train_gru.py's argparse (default False, i.e. "
                    "unchanged historical behavior for every existing "
                    "caller). When set: the test dataframe/loader is never "
                    "constructed (only case LABELS survive, in "
                    "run_config.json's test_cases field), test inference "
                    "never runs, test_metrics stays None (not merely "
                    "omitted -- never computed), tf_results stays {} "
                    "(test cases never enter that loop), and "
                    "amplitude_comparison's own per-case rollout+CSV never "
                    "sees a test case either. Every training job in this "
                    "study passes --skip_test_eval unconditionally "
                    "(train_sensitivity.py). No _test_locked/ redaction "
                    "step or directory exists anywhere in this study -- "
                    "prevention at the source, not concealment after the "
                    "fact. See tests/test_skip_test_eval.py in the main "
                    "repo test suite for the regression tests proving the "
                    "test-only functions (run_validation w/ test_loader, "
                    "teacher_forcing_rollout w/ a test case, "
                    "plot_tf_result) are never called when the flag is set.",
            },
        ],
    }

    manifests_dir = STUDY_ROOT / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    out_path = manifests_dir / "stage0_audit.json"
    with open(out_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"Stage 0 audit written to {out_path}")
    print(json.dumps({k: v for k, v in audit.items()
                       if k not in ("cylinder200", "bridge")}, indent=2))
    return audit


if __name__ == "__main__":
    main()
