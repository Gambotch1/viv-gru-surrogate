#!/usr/bin/env python3
r"""Rollout-informed refinement of a trained bridge GRU (thesis Sec. 6.6.1, Fig. 6.10-6.12).

Fine-tunes a one-step model with   L = L_TF + lambda_roll * L_roll:
    L_TF    teacher-forced C_L error (keeps the one-step accuracy)
    L_roll  displacement/velocity error after a short closed-loop rollout
            (--rollout_steps, default 250 steps = 0.5 s)
lambda_roll = 0.08064 (Eq. 6.2) and is ramped up over the first updates.
After every pass the model is validated at several rollout horizons; the pass
with the best score whose teacher-forced loss is within --tf_loss_tolerance
of the baseline is kept.

Writes --output_dir in the same format as train_gru.py, so evaluate_all.py
can use it directly, plus run_manifest.json and train_history.json.

Example:
    PYTHONPATH=src python -m viv_analysis.train_rollout \
        --baseline results/gru_bridge_p0_nd_context_noacc --output_dir results/<new folder>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from viv_analysis.config import bridge_structural_params, config
from viv_analysis.models.gru import VIV_GRU
from viv_analysis.preprocess import load_bridge_df_cached
from viv_analysis.rollout_training import (
    ScalerConstants,
    build_batch_from_case,
    build_tf_batch_from_case,
    loss_cl,
    loss_roll,
    rollout_chunk,
    sample_batch_starts,
)
from viv_analysis.utils import PROJECT_ROOT, parse_ur_label


DEFAULT_BASELINE = PROJECT_ROOT / "results" / "gru_bridge_p0_nd_context_noacc"
DEFAULT_HORIZONS_S = (0.5, 3.125, 6.5, 10.0, 20.0)


def sha256(path: Path) -> str:
    """SHA-256 hash of a file (recorded for the baseline checkpoint)."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_baseline(artifact_dir: Path, device: str) -> dict:
    """Load the one-step model folder: model, scalers, Ur statistics, split and settings."""
    with (artifact_dir / "run_config.json").open() as stream:
        run_config = json.load(stream)
    with (artifact_dir / "metrics_gru.json").open() as stream:
        metrics = json.load(stream)
    with (artifact_dir / "x_scaler.pkl").open("rb") as stream:
        x_scaler = pickle.load(stream)
    with (artifact_dir / "y_scaler.pkl").open("rb") as stream:
        y_scaler = pickle.load(stream)
    with (artifact_dir / "ur_stats.pkl").open("rb") as stream:
        ur_payload = pickle.load(stream)

    if run_config.get("cfd_dataset") != "bridge":
        raise ValueError("The refinement driver accepts only a bridge checkpoint.")
    if not bool(run_config.get("nd_inputs")):
        raise ValueError("The proposed refinement requires the nondimensional baseline.")

    input_cols = list(run_config["input_cols"])
    if "disp" not in input_cols or "vel" not in input_cols:
        raise ValueError("The state loss requires both displacement and velocity inputs.")

    gru_config = metrics["gru_config"]
    use_ur_context = bool(run_config["use_ur_context"])
    model = VIV_GRU(
        input_size=len(input_cols) + int(use_ur_context),
        hidden_size=int(gru_config["hidden_size"]),
        num_layers=int(gru_config["num_layers"]),
        dropout=float(gru_config["dropout"]),
    ).to(device)

    checkpoint_path = artifact_dir / "gru_best.pt"
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict)

    train_cases = list(run_config["train_cases"])
    val_cases = list(run_config["val_cases"])
    test_cases = list(run_config["test_cases"])
    if set(train_cases) & set(val_cases) or set(train_cases) & set(test_cases) or set(val_cases) & set(test_cases):
        raise ValueError("The baseline partition contains overlapping cases.")
    if sorted(run_config.get("scaler_fit_cases", train_cases)) != sorted(train_cases):
        raise ValueError("The recorded scaler-fit cases do not match the training partition.")

    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "ur_stats": (float(ur_payload["mean"]), float(ur_payload["std"])),
        "input_cols": input_cols,
        "use_ur_context": use_ur_context,
        "nd_inputs": True,
        "seq_len": int(run_config["seq_len"]),
        "train_cases": sorted(train_cases, key=parse_ur_label),
        "val_cases": sorted(val_cases, key=parse_ur_label),
        "test_cases": sorted(test_cases, key=parse_ur_label),
        "run_config": run_config,
        "gru_config": gru_config,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": sha256(checkpoint_path),
    }


def load_raw_bridge_data():
    """Cached bridge dataframe (see preprocess.load_bridge_df_cached)."""
    D = float(config["bridge_D_ref"])
    fn = float(config["bridge_fn_hz"])
    structural = bridge_structural_params()
    raw_df = load_bridge_df_cached(
        fn_hz=fn,
        d_ref=D,
        bridge_structural_params=structural,
    )

    sizes = raw_df.groupby("case").size().sort_values()
    minimum = 0.05 * float(sizes.median())
    short_cases = set(sizes[sizes < minimum].index)
    if short_cases:
        raw_df = raw_df[~raw_df["case"].isin(short_cases)].copy()
    return raw_df


def release_times(cases: list[str]) -> dict[str, float]:
    """Release time t = t* D / U of each bridge case."""
    D = float(config["bridge_D_ref"])
    fn = float(config["bridge_fn_hz"])
    t_star = float(config["bridge_t_star_release"])
    return {
        case: t_star * D / (parse_ur_label(case) * fn * D)
        for case in cases
    }


def case_dt(case_df) -> float:
    """Time step of one case (median of the time differences)."""
    times = case_df.sort_values("time")["time"].to_numpy(dtype=np.float64)
    differences = np.diff(times)
    if len(differences) == 0 or not np.isfinite(differences).all():
        raise ValueError("Invalid time axis in bridge case.")
    dt = float(np.median(differences))
    if dt <= 0:
        raise ValueError(f"Non-positive effective timestep: {dt}")
    return dt


def build_tf_batch_for_starts(
    case_df,
    case_name: str,
    starts: list[int],
    n_steps: int,
    baseline: dict,
    D: float,
    fn: float,
    device: str,
) -> dict:
    """Teacher-forced windows and C_L targets starting at the given indices."""
    batches = [
        build_tf_batch_from_case(
            case_df=case_df,
            case_name=case_name,
            start_idx=start,
            n_steps=1,
            seq_len=baseline["seq_len"],
            input_cols=baseline["input_cols"],
            x_scaler=baseline["x_scaler"],
            nd_inputs=baseline["nd_inputs"],
            D=D,
            fn=fn,
            use_ur_context=baseline["use_ur_context"],
            ur_stats=baseline["ur_stats"],
            device=device,
        )
        for start in starts
    ]
    return {
        "x": torch.cat([batch["x"] for batch in batches], dim=0),
        "cl_cfd": torch.cat([batch["cl_cfd"] for batch in batches], dim=0),
    }


def two_branch_losses(
    model,
    case_df,
    case_name: str,
    starts: list[int],
    n_steps: int,
    baseline: dict,
    scaler_constants: ScalerConstants,
    D: float,
    fn: float,
    m: float,
    c: float,
    k: float,
    rho: float,
    B_ref: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced loss L_TF and rollout state loss L_roll for one batch."""
    rollout_batch = build_batch_from_case(
        case_df=case_df,
        case_name=case_name,
        starts=starts,
        seq_len=baseline["seq_len"],
        input_cols=baseline["input_cols"],
        x_scaler=baseline["x_scaler"],
        nd_inputs=baseline["nd_inputs"],
        D=D,
        fn=fn,
        use_ur_context=baseline["use_ur_context"],
        ur_stats=baseline["ur_stats"],
        max_future_steps=n_steps,
        device=device,
    )
    U = float(rollout_batch["U"])
    q = 0.5 * rho * U**2 * B_ref
    out = rollout_chunk(
        model=model,
        window=rollout_batch["window"],
        h_state=rollout_batch["h_state"],
        hdot_state=rollout_batch["hdot_state"],
        hddot_state=rollout_batch["hddot_state"],
        n_steps=n_steps,
        dt=float(rollout_batch["dt"]),
        m=m,
        c=c,
        k=k,
        q=q,
        U=U,
        D=D,
        input_cols=baseline["input_cols"],
        nd_inputs=baseline["nd_inputs"],
        sc=scaler_constants,
        use_ur_context=baseline["use_ur_context"],
        ur_scaled=float(rollout_batch["ur_scaled"]),
    )

    tf_batch = build_tf_batch_for_starts(
        case_df, case_name, starts, n_steps, baseline, D, fn, device
    )
    tf_pred_scaled, _ = model(tf_batch["x"])
    l_tf = loss_cl(
        tf_pred_scaled,
        tf_batch["cl_cfd"],
        scaler_constants.y_mean,
        scaler_constants.y_scale,
    )

    disp_idx = baseline["input_cols"].index("disp")
    vel_idx = baseline["input_cols"].index("vel")
    l_state = loss_roll(
        out["h"],
        out["hdot"],
        rollout_batch["cfd_h"][:, :n_steps],
        rollout_batch["cfd_hdot"][:, :n_steps],
        D=D,
        U=U,
        x_mean=scaler_constants.x_mean,
        x_scale=scaler_constants.x_scale,
        disp_idx=disp_idx,
        vel_idx=vel_idx,
    )
    return l_tf, l_state


def deterministic_starts(
    case_df,
    release_t: float,
    seq_len: int,
    n_steps: int,
    batch_size: int,
    seed: int,
) -> list[int]:
    """Fixed, seeded rollout starts for validation."""
    return sample_batch_starts(
        case_df,
        release_t,
        seq_len,
        n_steps,
        batch_size,
        np.random.default_rng(seed),
    )


def validate(
    model,
    raw_df,
    val_cases: list[str],
    releases: dict[str, float],
    horizons_s: tuple[float, ...],
    batch_size: int,
    baseline: dict,
    scaler_constants: ScalerConstants,
    physics: dict,
    device: str,
    lambda_roll: float,
    seed: int,
) -> dict:
    """Validation losses at each rollout horizon and the selection score."""
    was_training = model.training
    model.eval()
    per_horizon: dict[str, dict] = {}

    with torch.no_grad():
        for horizon_i, horizon_s in enumerate(horizons_s):
            case_rows = []
            for case_i, case_name in enumerate(val_cases):
                case_df = raw_df[raw_df["case"] == case_name].sort_values("time").reset_index(drop=True)
                n_steps = max(2, int(round(horizon_s / case_dt(case_df))))
                starts = deterministic_starts(
                    case_df,
                    releases[case_name],
                    baseline["seq_len"],
                    n_steps,
                    batch_size,
                    seed + 1000 * horizon_i + case_i,
                )
                l_tf, l_state = two_branch_losses(
                    model=model,
                    case_df=case_df,
                    case_name=case_name,
                    starts=starts,
                    n_steps=n_steps,
                    baseline=baseline,
                    scaler_constants=scaler_constants,
                    device=device,
                    **physics,
                )
                case_rows.append({
                    "case": case_name,
                    "l_tf": float(l_tf),
                    "l_state": float(l_state),
                    "combined": float(l_tf + lambda_roll * l_state),
                    "n_steps": n_steps,
                })
            key = f"{horizon_s:g}s"
            per_horizon[key] = {
                "mean_l_tf": float(np.mean([row["l_tf"] for row in case_rows])),
                "mean_l_state": float(np.mean([row["l_state"] for row in case_rows])),
                "median_l_state": float(np.median([row["l_state"] for row in case_rows])),
                "mean_combined": float(np.mean([row["combined"] for row in case_rows])),
                "per_case": case_rows,
            }

    if was_training:
        model.train()

    return {
        "horizons": per_horizon,
        "selection_score": float(np.mean([
            values["median_l_state"] for values in per_horizon.values()
        ])),
        "mean_l_tf": float(np.mean([
            values["mean_l_tf"] for values in per_horizon.values()
        ])),
    }


def save_compatible_artifact(output_dir: Path, baseline: dict, state_dict: dict, manifest: dict) -> None:
    """Save the selected model in the train_gru.py folder format."""
    torch.save(state_dict, output_dir / "gru_best.pt")
    for name, obj in (
        ("x_scaler.pkl", baseline["x_scaler"]),
        ("y_scaler.pkl", baseline["y_scaler"]),
        ("ur_stats.pkl", {
            "mean": baseline["ur_stats"][0],
            "std": baseline["ur_stats"][1],
            "use_ur_context": baseline["use_ur_context"],
            "nd_inputs": baseline["nd_inputs"],
            "coordinate_mode": "nondimensional",
            "cfd_dataset": "bridge",
            "exp_subdir": output_dir.name,
        }),
    ):
        with (output_dir / name).open("wb") as stream:
            pickle.dump(obj, stream)

    run_config = dict(baseline["run_config"])
    run_config.update({
        "output_directory": str(output_dir),
        "source": "train_rollout_refinement.py two-branch refinement",
        "warm_start_checkpoint": str(baseline["checkpoint_path"]),
        "refinement": manifest,
    })
    with (output_dir / "run_config.json").open("w") as stream:
        json.dump(run_config, stream, indent=2)

    with (output_dir / "metrics_gru.json").open("w") as stream:
        json.dump({
            "dataset": "bridge",
            "coordinate_mode": "nondimensional",
            "nd_inputs": True,
            "use_ur_context": baseline["use_ur_context"],
            "gru_config": baseline["gru_config"],
            "case_split": {
                "train": baseline["train_cases"],
                "val": baseline["val_cases"],
                "test": baseline["test_cases"],
            },
            "scaler_fit_cases": baseline["train_cases"],
            "ur_stats_fit_cases": baseline["train_cases"],
        }, stream, indent=2)


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--rollout_steps", type=int, default=250)
    parser.add_argument("--passes", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lambda_roll", type=float, default=0.08064)
    parser.add_argument("--lambda_ramp_updates", type=int, default=17)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument(
        "--validation_horizons_s",
        default=",".join(str(value) for value in DEFAULT_HORIZONS_S),
    )
    parser.add_argument(
        "--tf_loss_tolerance",
        type=float,
        default=0.05,
        help="Maximum relative increase in validation TF loss allowed during selection.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the refinement passes and keep the best eligible model."""
    args = parse_args()
    if args.rollout_steps < 2 or args.passes < 1 or args.batch_size < 1:
        raise ValueError("rollout_steps>=2, passes>=1 and batch_size>=1 are required.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    baseline = load_baseline(args.baseline, device)
    raw_df = load_raw_bridge_data()

    available = set(raw_df["case"].astype(str).unique())
    required = set(baseline["train_cases"] + baseline["val_cases"])
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"Training/validation cases missing from the live bridge cache: {missing}")

    D = float(config["bridge_D_ref"])
    fn = float(config["bridge_fn_hz"])
    physics = {
        "D": D,
        "fn": fn,
        "m": float(bridge_structural_params()["m"]),
        "c": float(bridge_structural_params()["c"]),
        "k": float(bridge_structural_params()["k"]),
        "rho": float(config["bridge_rho"]),
        "B_ref": float(config["bridge_B_ref"]),
    }
    releases = release_times(baseline["train_cases"] + baseline["val_cases"])
    scaler_constants = ScalerConstants.from_sklearn(
        baseline["x_scaler"], baseline["y_scaler"]
    )
    horizons_s = tuple(float(value) for value in args.validation_horizons_s.split(","))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    dt_values = {
        case: case_dt(raw_df[raw_df["case"] == case])
        for case in baseline["train_cases"] + baseline["val_cases"]
    }
    manifest = {
        "objective": "L_TF + lambda_roll * L_state_roll",
        "rollout_steps": args.rollout_steps,
        "nominal_rollout_duration_s": args.rollout_steps * float(np.median(list(dt_values.values()))),
        "passes": args.passes,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lambda_roll_final": args.lambda_roll,
        "lambda_ramp_updates": args.lambda_ramp_updates,
        "grad_clip_norm": args.grad_clip_norm,
        "validation_horizons_s": horizons_s,
        "tf_loss_tolerance": args.tf_loss_tolerance,
        "seed": args.seed,
        "baseline_checkpoint": str(baseline["checkpoint_path"]),
        "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "train_cases": baseline["train_cases"],
        "val_cases": baseline["val_cases"],
        "test_cases_recorded_but_not_loaded": baseline["test_cases"],
        "scalers_refit": False,
        "structural_parameters_trainable": False,
        "graph_truncation_within_rollout": False,
        "effective_dt_by_case_s": dt_values,
    }
    with (args.output_dir / "run_manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2)

    model = baseline["model"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)

    baseline_validation = validate(
        model=model,
        raw_df=raw_df,
        val_cases=baseline["val_cases"],
        releases=releases,
        horizons_s=horizons_s,
        batch_size=args.val_batch_size,
        baseline=baseline,
        scaler_constants=scaler_constants,
        physics=physics,
        device=device,
        lambda_roll=args.lambda_roll,
        seed=args.seed,
    )
    baseline_tf_limit = baseline_validation["mean_l_tf"] * (1.0 + args.tf_loss_tolerance)
    history = {"manifest": manifest, "baseline_validation": baseline_validation, "passes": []}

    selected_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    selected_score = baseline_validation["selection_score"]
    selected_pass = 0
    global_update = 0

    for pass_index in range(1, args.passes + 1):
        model.train()
        cases = list(baseline["train_cases"])
        rng.shuffle(cases)
        training_rows = []

        for case_name in cases:
            case_df = raw_df[raw_df["case"] == case_name].sort_values("time").reset_index(drop=True)
            starts = sample_batch_starts(
                case_df,
                releases[case_name],
                baseline["seq_len"],
                args.rollout_steps,
                args.batch_size,
                rng,
            )
            l_tf, l_state = two_branch_losses(
                model=model,
                case_df=case_df,
                case_name=case_name,
                starts=starts,
                n_steps=args.rollout_steps,
                baseline=baseline,
                scaler_constants=scaler_constants,
                device=device,
                **physics,
            )

            global_update += 1
            ramp = min(1.0, global_update / max(1, args.lambda_ramp_updates))
            lambda_now = args.lambda_roll * ramp
            total = l_tf + lambda_now * l_state

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip_norm
            )
            optimizer.step()

            training_rows.append({
                "case": case_name,
                "global_update": global_update,
                "lambda_roll": lambda_now,
                "l_tf": float(l_tf.detach()),
                "l_state": float(l_state.detach()),
                "total": float(total.detach()),
                "grad_norm_pre_clip": float(grad_norm),
            })

            if args.smoke_test:
                break

        validation = validate(
            model=model,
            raw_df=raw_df,
            val_cases=baseline["val_cases"],
            releases=releases,
            horizons_s=horizons_s,
            batch_size=args.val_batch_size,
            baseline=baseline,
            scaler_constants=scaler_constants,
            physics=physics,
            device=device,
            lambda_roll=args.lambda_roll,
            seed=args.seed,
        )
        eligible = validation["mean_l_tf"] <= baseline_tf_limit
        score = validation["selection_score"]
        if eligible and score < selected_score:
            selected_score = score
            selected_pass = pass_index
            selected_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        checkpoint_path = args.output_dir / f"refinement_pass{pass_index}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        history["passes"].append({
            "pass": pass_index,
            "training": training_rows,
            "validation": validation,
            "tf_eligible": eligible,
            "checkpoint": str(checkpoint_path),
        })
        with (args.output_dir / "train_history.json").open("w") as stream:
            json.dump(history, stream, indent=2)

        print(
            f"pass={pass_index}/{args.passes} "
            f"train_tf={np.mean([row['l_tf'] for row in training_rows]):.6g} "
            f"train_state={np.mean([row['l_state'] for row in training_rows]):.6g} "
            f"val_tf={validation['mean_l_tf']:.6g} "
            f"val_state_score={score:.6g} eligible={eligible}"
        )
        if args.smoke_test:
            break

    manifest["selected_pass"] = selected_pass
    manifest["selected_validation_score"] = selected_score
    save_compatible_artifact(args.output_dir, baseline, selected_state, manifest)
    with (args.output_dir / "train_history.json").open("w") as stream:
        json.dump(history, stream, indent=2)

    print(f"Selected refinement pass: {selected_pass}")
    print(f"Evaluator-compatible checkpoint: {args.output_dir / 'gru_best.pt'}")


if __name__ == "__main__":
    main()
