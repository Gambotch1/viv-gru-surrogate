r"""Train the one-step (teacher-forced) GRU surrogate for C_L (thesis Sec. 4.5).

Steps: load the CFD cases -> compute vel/acc -> split into train/val/test ->
optionally nondimensionalise the inputs -> fit scalers on the training cases ->
train with early stopping -> evaluate -> save the model folder.

Writes results/<exp_subdir>/:
    gru_best.pt, x_scaler.pkl, y_scaler.pkl, ur_stats.pkl  (everything needed for inference)
    run_config.json, metrics_gru.json                       (settings, split, metrics)
    learning curve, amplitude_comparison.csv, open-loop plots

Example (cylinder model without acceleration, nondimensional inputs, Ur context):
    PYTHONPATH=src python -m viv_analysis.train_gru --cfd_dataset cylinder200 \
        --input_cols disp vel --nd_inputs --exp_subdir gru_cylinder200_nd_context_noacc
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
import os

import matplotlib

from viv_analysis.utils import parse_ur_label
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from viv_analysis.plotting.plot_style import apply_thesis_style
apply_thesis_style()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from viv_analysis.config import (
    bridge_structural_params, config, prepare_gru_config,
    CYLINDER200_ALIASES, cylinder200_release_time, cylinder200_structural_params,
)
from viv_analysis.preprocess import merge_dataframes, compute_kinematics, downsample
from viv_analysis.models.gru import VIV_GRU, VIVSequenceDataset, apply_scalers_to_df, fit_scalers
from viv_analysis.evaluate import evaluate
from viv_analysis.utils import PROJECT_ROOT, present_model_label

ROOT_DIR = PROJECT_ROOT

import argparse
import random


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and make cuDNN deterministic, so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    seed = int(torch.initial_seed() % 2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)

def resolve_nd_reference_scales(dataset: str, cfg: dict) -> tuple[float, float]:
    """Reference length D [m] and natural frequency fn [Hz] used to nondimensionalise a dataset."""
    ds = dataset.strip().lower()

    if ds == "bridge":
        D = float(cfg["bridge_D_ref"])
        fn = float(cfg["bridge_fn_hz"])
    elif ds in CYLINDER200_ALIASES:
        D = float(cfg["cylinder200_D_ref"])
        fn = float(cfg["cylinder200_fn"])
    else:
        raise ValueError(
            f"resolve_nd_reference_scales: unsupported dataset '{dataset}'. "
            f"Supported: 'bridge', cylinder200 (aliases: "
            f"{sorted(CYLINDER200_ALIASES)})."
        )

    if not (np.isfinite(D) and D > 0):
        raise ValueError(f"resolve_nd_reference_scales: D must be finite and positive, got {D}")
    if not (np.isfinite(fn) and fn > 0):
        raise ValueError(f"resolve_nd_reference_scales: fn must be finite and positive, got {fn}")

    return D, fn


_ND_TRANSFORM_COLUMNS = ("disp", "vel", "acc")


def apply_nd_transform(df: pd.DataFrame, nd_inputs: bool, D: float, fn: float,
                       input_cols: list[str]) -> pd.DataFrame:
    """Nondimensionalise the kinematic inputs, case by case.

    disp / D,  vel / U,  acc / (U^2 / D),  with U = Ur fn D of that case.
    Returns a copy; with nd_inputs=False the data are returned unchanged.
    """
    if not nd_inputs:
        return df.copy()

    unsupported = [c for c in input_cols if c not in _ND_TRANSFORM_COLUMNS]
    if unsupported:
        raise ValueError(
            f"apply_nd_transform: unsupported input column(s) {unsupported}; "
            f"supported columns are {list(_ND_TRANSFORM_COLUMNS)}"
        )

    if D <= 0 or fn <= 0:
        raise ValueError(f"apply_nd_transform requires D>0, fn>0 (got D={D}, fn={fn})")

    df_out = df.copy()

    for case_name, case_df_idx in df_out.groupby("case", sort=False).groups.items():
        ur = parse_ur_label(str(case_name))
        U_case = float(ur * fn * D)

        if not (U_case > 0 and np.isfinite(U_case)):
            raise ValueError(
                f"apply_nd_transform: invalid U_case={U_case} for case={case_name}, "
                f"Ur={ur}, fn={fn}, D={D}")

        divisor = {"disp": D, "vel": U_case, "acc": (U_case ** 2) / D}

        for col_name in input_cols:
            df_out.loc[case_df_idx, col_name] = (
                df_out.loc[case_df_idx, col_name].astype(np.float32) / divisor[col_name]
            )

    return df_out


def enforce_holdout(train_cases: set, val_cases: set, test_cases: set,
                    all_cases: set, holdout_ur: float | None) -> tuple:
    """Move the case at holdout_ur into the test set (--holdout_ur)."""
    if holdout_ur is None:
        return train_cases, val_cases, test_cases

    holdout_label = format_ur_label(holdout_ur)

    if holdout_label not in all_cases:
        raise ValueError(f"Holdout Ur={holdout_ur} (label={holdout_label}) not found in data.")

    train_cases = train_cases - {holdout_label}
    val_cases = val_cases - {holdout_label}
    test_cases = test_cases | {holdout_label}

    overlap_tv = train_cases & val_cases
    overlap_tt = train_cases & test_cases
    overlap_vt = val_cases & test_cases

    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            f"Split overlap detected: train∩val={overlap_tv}, "
            f"train∩test={overlap_tt}, val∩test={overlap_vt}")

    return train_cases, val_cases, test_cases


def enforce_force_train(train_cases: set, val_cases: set, test_cases: set,
                        all_cases: set, force_train_ur: list[float] | None) -> tuple:
    """Move the cases in force_train_ur into the training set (--force_train_ur)."""
    if not force_train_ur:
        return train_cases, val_cases, test_cases

    for ur in force_train_ur:
        label = format_ur_label(ur)
        if label not in all_cases:
            raise ValueError(f"force_train Ur={ur} (label={label}) not found in data.")
        train_cases = train_cases | {label}
        val_cases = val_cases - {label}
        test_cases = test_cases - {label}

    overlap_tv = train_cases & val_cases
    overlap_tt = train_cases & test_cases
    overlap_vt = val_cases & test_cases
    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            f"Split overlap detected: train∩val={overlap_tv}, "
            f"train∩test={overlap_tt}, val∩test={overlap_vt}")

    return train_cases, val_cases, test_cases


def format_ur_label(ur: float) -> str:
    from viv_analysis.utils import format_ur_label as original_format
    return original_format(ur)


def resolve_use_ur_context(dataset_default: bool, use_ur_context_flag: bool,
                           no_ur_context_flag: bool) -> bool:
    """Decide whether Ur is used as an input: command-line flag if given, else the dataset default."""
    if use_ur_context_flag:
        return True
    if no_ur_context_flag:
        return False
    return bool(dataset_default)


def resolve_dataset(dataset_pos: str | None, dataset_cli: str | None) -> str:
    """Dataset name from the positional argument or --cfd_dataset (default: cylinder200)."""
    if dataset_pos is not None and dataset_cli is not None:
        if dataset_pos != dataset_cli:
            raise ValueError(
                f"Dataset mismatch: positional='{dataset_pos}' vs --cfd_dataset='{dataset_cli}'"
            )
    if dataset_cli is not None:
        return dataset_cli
    if dataset_pos is not None:
        return dataset_pos
    return "cylinder200"


def check_artifact_collision(output_dir: Path, overwrite: bool) -> None:
    """Refuse to overwrite an existing model folder unless --overwrite is given."""
    if overwrite:
        return
    existing_artifacts = [
        output_dir / "gru_best.pt",
        output_dir / "metrics_gru.json",
    ]
    if any(f.exists() for f in existing_artifacts):
        raise FileExistsError(
            f"Artifacts already exist in {output_dir}. "
            f"Use --overwrite to replace them."
        )


def _cylinder200_split(cases: list[str]) -> tuple:
    """Fixed cylinder split used in the thesis (Fig. 5.6).

    test: Ur 3.5, 5.5, 7, 11;  val: Ur 4.25, 6.25, 9, 10;  train: the other 13 cases.
    Also returns the release time of every case.
    """
    test  = {"Ur3.5", "Ur5.5", "Ur7", "Ur11"}
    val   = {"Ur4.25", "Ur6.25", "Ur9", "Ur10"}
    train = set(cases) - test - val

    missing = (test | val) - set(cases)
    if missing:
        print(f"WARNING: split references cases not in data: {missing}")

    rt = {c: cylinder200_release_time(parse_ur_label(c)) for c in cases}
    return train, val, test, rt


def _bridge_split(cases: list[str], fn_hz: float,
                  d_ref: float, t_star_release: float) -> tuple:
    """Bridge split: cases sorted by Ur, about 20 % test and 20 % validation spread
    evenly over the Ur range, the rest for training. Also returns release times.
    """
    for c in cases:
        try:
            parse_ur_label(c)
        except ValueError as e:
            raise ValueError(
                f"_bridge_split got non-Ur label '{c}'. Cases must be Ur-labelled "
                f"(merge_dataframes(dataset='bridge', ..., convert_bridge_to_ur=True)). "
                f"A raw-speed label here double-converts U and corrupts release times."
            ) from e

    ordered = sorted(cases, key=parse_ur_label)
    n       = len(ordered)
    n_test  = max(1, round(0.20 * n))
    n_val   = max(1, round(0.20 * n))

    test_idx  = list(range(0, n, max(1, n // n_test)))[:n_test]
    remaining = [c for i, c in enumerate(ordered) if i not in test_idx]
    val_idx   = list(range(0, len(remaining),
                           max(1, len(remaining) // n_val)))[:n_val]

    test  = {ordered[i] for i in test_idx}
    val   = {remaining[i] for i in val_idx}
    train = set(cases) - test - val

    rt = {}
    for case in cases:
        ur = parse_ur_label(case)
        U  = ur * fn_hz * d_ref
        if not (1.0 < U < 100.0):
            raise ValueError(
                f"Recovered U={U:.2f} m/s for '{case}' is outside the plausible "
                f"bridge sweep — likely a unit/double-convert error.")
        rt[case] = float(t_star_release * d_ref / U)
        print(f"  case={case}  Ur={ur:.4f}  U={U:.4f} m/s  t_release={rt[case]:.4f}s")
    return train, val, test, rt


def split_cases(cases: list[str], dataset: str,
                cfg: dict) -> tuple[set, set, set, dict]:
    """Train/val/test case sets and release times for the given dataset."""
    ds = dataset.strip().lower()
    if ds in CYLINDER200_ALIASES:
        return _cylinder200_split(cases)
    if ds == "bridge":
        return _bridge_split(
            cases,
            fn_hz=cfg["bridge_fn_hz"],
            d_ref=cfg["bridge_D_ref"],
            t_star_release=cfg["bridge_t_star_release"],
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def train_one_epoch(model, loader, optimizer, criterion, device, input_noise_std=0.0,
                    n_kinematic_cols=3, case_weights=None):
    """One training epoch; returns the mean loss.

    input_noise_std adds Gaussian noise to the standardised kinematic columns
    (not to the Ur column). Gradients are clipped to norm 1.
    """
    model.train()
    total = 0.0
    for x_b, y_b, case_b in loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        if input_noise_std > 0.0:
            x_b = x_b.clone()
            x_b[..., :n_kinematic_cols] = (
                x_b[..., :n_kinematic_cols]
                + torch.randn_like(x_b[..., :n_kinematic_cols]) * input_noise_std
            )
        optimizer.zero_grad()
        pred, _ = model(x_b)
        if case_weights is not None:
            w_b = torch.tensor([case_weights.get(str(c), 1.0) for c in case_b],
                               dtype=torch.float32, device=device)
            loss = (w_b * (pred - y_b) ** 2).mean()
        else:
            loss = criterion(pred, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def run_validation(model, loader, criterion, device):
    """Loss and predictions on a data loader. Returns (loss, predictions, targets, case names)."""
    model.eval()
    total, preds, trues, cases = 0.0, [], [], []
    with torch.no_grad():
        for x_b, y_b, case_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            pred, _ = model(x_b)
            total += criterion(pred, y_b).item()
            preds.append(pred.cpu().numpy())
            trues.append(y_b.cpu().numpy())
            cases.extend(case_b)
    return (
        total / len(loader),
        np.concatenate(preds),
        np.concatenate(trues),
        np.array(cases, dtype=object),
    )


def compute_case_loss_weights(
    df_scaled: pd.DataFrame, target_col: str, cases: list[str],
    min_var_frac_of_median: float = 0.01,
) -> dict[str, float]:
    """Per-case loss weights 1/var(C_L), normalised to mean 1 (--amplitude_aware_loss)."""
    variances = {}
    for case in cases:
        vals = df_scaled.loc[df_scaled["case"] == case, target_col].to_numpy(dtype=np.float64)
        variances[case] = float(np.var(vals)) if len(vals) > 0 else 0.0

    median_var = float(np.median(list(variances.values()))) if variances else 0.0
    floor = max(median_var * min_var_frac_of_median, 1e-12)

    raw_weights = {c: 1.0 / max(v, floor) for c, v in variances.items()}
    mean_w = float(np.mean(list(raw_weights.values()))) if raw_weights else 1.0
    mean_w = mean_w if mean_w > 0 else 1.0
    return {c: w / mean_w for c, w in raw_weights.items()}


def teacher_forcing_rollout(
        model, case_df, input_cols, seq_len, release_t,
        y_scaler, case_name, device,
        use_ur_context=False, ur_stats=None,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open-loop prediction over one whole case using the true CFD kinematics as input. Returns C_L predicted, C_L true, time."""
    model.eval()
    ordered   = case_df.sort_values("time").reset_index(drop=True)
    signal    = ordered[input_cols].to_numpy(dtype=np.float32)

    if use_ur_context:
        ur_mean, ur_std = ur_stats if ur_stats is not None else (0.0, 1.0)
        ur_std = float(ur_std) if abs(float(ur_std)) > 0 else 1.0
        ur_val = parse_ur_label(str(case_name))
        ur_scaled = (ur_val - float(ur_mean)) / ur_std
        ur_col = np.full((signal.shape[0], 1), ur_scaled, dtype=np.float32)
        signal = np.hstack([signal, ur_col])

    cl_true_s = ordered["cl"].to_numpy(dtype=np.float32)
    times     = ordered["time"].to_numpy(dtype=np.float32)

    release_idx = int(np.searchsorted(times, release_t))
    start       = max(seq_len, release_idx + seq_len)

    preds = []
    with torch.no_grad():
        for i in range(start, len(ordered)):
            w = signal[i - seq_len : i]
            w = np.array(w, copy=True)
            x = torch.from_numpy(w).unsqueeze(0).to(device)
            p, _ = model(x)
            preds.append(p.item())

    cl_pred_s   = np.array(preds, dtype=np.float32)
    cl_true_s_s = cl_true_s[start:]
    times_s     = times[start:]

    cl_pred = y_scaler.inverse_transform(cl_pred_s.reshape(-1, 1)).ravel()
    cl_true = y_scaler.inverse_transform(cl_true_s_s.reshape(-1, 1)).ravel()
    return cl_pred, cl_true, times_s


def plot_tf_result(cl_pred, cl_true, times, case_name, output_dir, dataset: str = ""):
    """Quick-look plot of an open-loop prediction and its residual (ar_<case>.png)."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    ax = axes[0]
    ax.plot(times, cl_true, lw=0.8, color="black",    label="CFD (ground truth)")
    ax.plot(times, cl_pred, lw=0.8, color="tab:blue", alpha=0.85,
            label=present_model_label(dataset, "GRU (teacher forcing)"))
    ax.set_ylabel("$C_L$", fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(times, cl_pred - cl_true, lw=0.6, color="tab:red", alpha=0.8)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel(r"Residual ($\hat{C}_L - C_L$)", fontsize=13)
    ax2.set_xlabel("Time [s]", fontsize=13)
    ax2.grid(True, alpha=0.3)

    fig.savefig(output_dir / f"ar_{case_name}.png", dpi=600)
    plt.close(fig)


def amplitude_comparison(model, all_df_s, release_time,
                          input_cols, seq_len, y_scaler,
                          device, output_dir,
                          train_cases, val_cases, test_cases,
                          use_ur_context=False, ur_stats=None):
    """Compare predicted and CFD C_L amplitude (last 30 % of each case) for all cases; writes amplitude_comparison.csv."""
    results = []
    for case_name, case_df in all_df_s.groupby("case"):
        ur        = parse_ur_label(str(case_name))
        release_t = release_time[case_name]
        cl_pred, cl_true, _ = teacher_forcing_rollout(
            model,
            case_df,
            input_cols,
            seq_len,
            release_t,
            y_scaler,
            case_name,
            device,
            use_ur_context=use_ur_context,
            ur_stats=ur_stats,
        )

        ss       = int(0.7 * len(cl_true))
        amp_cfd  = (cl_true[ss:].max() - cl_true[ss:].min()) / 2
        amp_gru  = (cl_pred[ss:].max() - cl_pred[ss:].min()) / 2
        amp_abs_err = abs(amp_gru - amp_cfd)
        amp_rel_err_pct = 100.0 * amp_abs_err / (abs(amp_cfd) + 1e-12)

        results.append({
            "Ur": ur,
            "CL_amp_CFD": amp_cfd,
            "CL_amp_GRU": amp_gru,
            "CL_amp_abs_error": amp_abs_err,
            "CL_amp_rel_error_pct": amp_rel_err_pct,
            "ar_r2": r2_score(cl_true, cl_pred),
            "split": ("test" if case_name in test_cases
                    else "val" if case_name in val_cases
                    else "train"),
        })
        print(
            f"  {case_name}: CFD={amp_cfd:.4f}  GRU={amp_gru:.4f}  "
            f"amp_err={amp_rel_err_pct:.2f}%  "
            f"R²={results[-1]['ar_r2']:.4f}  [{results[-1]['split']}]"
        )

    df_res = pd.DataFrame(results).sort_values("Ur")
    df_res.to_csv(output_dir / "amplitude_comparison.csv", index=False)
    return df_res


def setup_argparse() -> argparse.ArgumentParser:
    """Command-line options of the training script."""
    parser = argparse.ArgumentParser(
        description="Train GRU model for VIV lift prediction (cylinder200 and bridge)."
    )

    parser.add_argument(
        "dataset_pos", nargs="?", default=None,
        help="Dataset: cylinder200 or bridge (same as --cfd_dataset)"
    )

    parser.add_argument(
        "--cfd_dataset", type=str, default=None,
        help="Dataset: cylinder200 or bridge"
    )
    parser.add_argument(
        "--nd_inputs", action="store_true",
        help="Use nondimensional inputs [h/D, hdot/U, hddot/(U²/D)]"
    )

    context_group = parser.add_mutually_exclusive_group()
    context_group.add_argument(
        "--use_ur_context", action="store_true",
        help="Include Ur as a context feature (overrides dataset default)"
    )
    context_group.add_argument(
        "--no_ur_context", action="store_true",
        help="Disable Ur context (overrides dataset default)"
    )

    parser.add_argument(
        "--holdout_ur", type=float, default=None,
        help="Hold out a specific Ur from training (e.g., 5.5)"
    )
    parser.add_argument(
        "--exclude_ur", type=float, nargs="+", default=None,
        help="Drop specific Ur case(s) from the dataset entirely, before "
             "train/val/test splitting -- e.g. --exclude_ur 8.4232 8.6338 "
             "8.8443 9.2655 10.1078 for the in-scope 22-case bridge subset "
             "(excludes every case at/above 20 m/s). Different from "
             "--holdout_ur (which keeps the case, just forces it into "
             "test): this removes it from all three partitions, so the "
             "model never trains, validates, or tests against it. Recorded "
             "verbatim in run_config.json's exclude_ur field."
    )
    parser.add_argument(
        "--force_train_ur", type=float, nargs="+", default=None,
        help="Move specific Ur case(s) into the training set, out of "
             "whichever of val/test split_cases() put them in (e.g. "
             "--force_train_ur 6.7385 7.1597). Symmetric to --holdout_ur "
             "but multi-valued and the opposite direction: an explicit, "
             "disclosed deviation from the dataset's normal automatic "
             "split, not a silent one -- recorded verbatim in "
             "run_config.json's force_train_ur field. Raises if a value "
             "is not found in the retained dataset, or overlaps "
             "--holdout_ur."
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (default: dataset config)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size (default: dataset config)"
    )
    parser.add_argument(
        "--seq_len", type=int, default=None,
        help="Sequence length (default: dataset config)"
    )
    parser.add_argument(
        "--input_cols", type=str, nargs="+", default=None,
        choices=["disp", "vel", "acc"],
        help="Input kinematic columns, e.g. --input_cols disp for a "
             "displacement-only ablation (default: dataset config, "
             "normally all three: disp vel acc)"
    )
    parser.add_argument(
        "--noise_std", type=float, default=0.05,
        help="Input noise std after standardization (default: 0.05)"
    )
    parser.add_argument(
        "--amplitude_aware_loss", action="store_true",
        help="Weight the training loss per-case by 1/variance of that case's "
             "own (scaled) target, so weak-amplitude cases aren't drowned out "
             "by a strong-response case (e.g. a lock-in peak) under the "
             "single global y_scaler. See compute_case_loss_weights."
    )
    parser.add_argument(
        "--acc_source", type=str, default="force_residual",
        choices=["force_residual", "savgol_vel"],
        help="How 'acc' is derived in compute_kinematics. 'force_residual' "
             "(default) = (F_fluid-c*v-k*y)/m, using the SAME m,c,k as the "
             "closed-loop Newmark integrator -- makes C_L an exact linear "
             "function of [disp,vel,acc] per Ur case. 'savgol_vel' = "
             "genuine numerical differentiation of the recorded velocity "
             "signal, with no algebraic tie to C_L."
    )
    parser.add_argument(
        "--seed", type=int, default=123,
        help="Random seed for reproducibility (default: 123)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader workers (default: 0)"
    )

    parser.add_argument(
        "--exp_subdir", type=str, default=None,
        help="Artifact subdirectory in results/; if omitted use default gru_{dataset}"
    )
    parser.add_argument(
        "--preflight_only", action="store_true",
        help="Stop after preflight (data load, split, scaler fit); do not train"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow overwriting existing artifacts"
    )
    parser.add_argument(
        "--skip_test_eval", action="store_true",
        help="Opt-in; default False (unchanged historical behavior). When set, "
             "the test partition is never loaded into a dataframe/loader, never "
             "run through the model, and no test metrics/plots/artifacts are "
             "produced -- only the test case LABELS are retained (in "
             "run_config.json's 'test_cases' list, for provenance). Intended "
             "for architecture/hyperparameter selection studies that must not "
             "touch the test partition before a final configuration is frozen."
    )

    return parser


def main() -> None:
    """Run the full training pipeline described at the top of this file."""
    parser = setup_argparse()
    args = parser.parse_args()

    dataset = resolve_dataset(args.dataset_pos, args.cfd_dataset)
    print(f"Dataset: {dataset}")

    try:
        from viv_analysis.preprocess import _resolve_data_dirs
        _resolve_data_dirs(dataset)
    except ValueError as e:
        raise ValueError(f"Invalid dataset '{dataset}': {e}")

    cfg = prepare_gru_config(dataset, config).copy()

    nd_inputs = bool(args.nd_inputs)
    use_ur_context = resolve_use_ur_context(
        cfg.get("use_ur_context", False), args.use_ur_context, args.no_ur_context)

    cfg["use_ur_context"] = use_ur_context
    cfg["nd_inputs"] = nd_inputs

    D_nd, fn_nd = resolve_nd_reference_scales(dataset, cfg) if nd_inputs else (None, None)

    if args.epochs is not None:
        cfg["n_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.seq_len is not None:
        cfg["seq_len"] = args.seq_len
    if args.input_cols is not None:
        cfg["input_cols"] = args.input_cols

    seed_everything(args.seed)

    if args.exp_subdir:
        output_dir = ROOT_DIR / "results" / args.exp_subdir
    else:
        coord_suffix = "_nd" if nd_inputs else ""
        ctx_suffix = "_ctx" if use_ur_context else "_noctx"
        output_dir = ROOT_DIR / "results" / f"gru_{dataset}{coord_suffix}{ctx_suffix}"

    output_dir.mkdir(parents=True, exist_ok=True)

    check_artifact_collision(output_dir, args.overwrite)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  seq_len={cfg['seq_len']}  "
          f"coord_mode={'nondimensional' if nd_inputs else 'dimensional'}  "
          f"use_ur_context={use_ur_context}")

    from viv_analysis.preprocess import _resolve_data_dirs, downsample

    if dataset == "bridge":
        raw_df = merge_dataframes(
            dataset="bridge",
            fn_hz=cfg["bridge_fn_hz"],
            d_ref=cfg["bridge_D_ref"],
        )
        raw_df = downsample(raw_df, cfg["bridge_downsample"])
    else:
        raw_df = merge_dataframes(dataset=dataset)

    if raw_df.empty:
        raise RuntimeError("No data found. Check data directories.")

    params_Re200 = None
    params_bridge = None

    if dataset.strip().lower() in CYLINDER200_ALIASES:
        params_Re200 = cylinder200_structural_params()
        raw_df = compute_kinematics(raw_df, dataset=dataset,
                                    structural_params=params_Re200,
                                    acc_source=args.acc_source)
    elif dataset == "bridge":
        params_bridge = bridge_structural_params()
        raw_df = compute_kinematics(raw_df, dataset=dataset,
                                    bridge_structural_params=params_bridge,
                                    acc_source=args.acc_source)
    else:
        raw_df = compute_kinematics(raw_df, dataset=dataset)

    if dataset == "bridge":
        # Drop bridge cases shorter than 5 % of the median case length (unusable runs).
        sizes = raw_df.groupby("case").size().sort_values()
        BRIDGE_MIN_FRAC = 0.05
        med = float(sizes.median())
        drop = set(sizes[sizes < BRIDGE_MIN_FRAC * med].index)
        if drop:
            print(f"Quarantining {len(drop)} bridge case(s): {sorted(drop)}")
            raw_df = raw_df[~raw_df['case'].isin(drop)].copy()

    exclude_labels = ([format_ur_label(u) for u in args.exclude_ur]
                      if args.exclude_ur else [])
    if exclude_labels:
        present = set(str(c) for c in raw_df["case"].drop_duplicates())
        missing = set(exclude_labels) - present
        if missing:
            raise ValueError(f"--exclude_ur label(s) {missing} not found in data.")
        overlap = set(exclude_labels) & {
            format_ur_label(u) for u in (args.force_train_ur or [])
        }
        if overlap:
            raise ValueError(
                f"--exclude_ur and --force_train_ur both reference {overlap} "
                f"-- contradictory (drop entirely vs force into training).")
        print(f"Excluding {len(exclude_labels)} case(s) entirely: {exclude_labels}")
        raw_df = raw_df[~raw_df["case"].isin(exclude_labels)].copy()

    all_cases_unsplit = sorted(str(c) for c in raw_df["case"].drop_duplicates())
    print(f"Cases loaded (before split): {all_cases_unsplit}")

    train_cases, val_cases, test_cases, release_time = split_cases(
        all_cases_unsplit, dataset, cfg)

    holdout_label = format_ur_label(args.holdout_ur) if args.holdout_ur is not None else None
    print(f"Holdout: ur={args.holdout_ur}  label={holdout_label}")
    if args.holdout_ur is not None:
        train_cases, val_cases, test_cases = enforce_holdout(
            train_cases, val_cases, test_cases, set(all_cases_unsplit),
            args.holdout_ur
        )

    if args.force_train_ur is not None and args.holdout_ur is not None:
        overlap = {format_ur_label(u) for u in args.force_train_ur} & {format_ur_label(args.holdout_ur)}
        if overlap:
            raise ValueError(
                f"--force_train_ur and --holdout_ur both reference {overlap} "
                f"-- contradictory (hold out of training vs force into it).")
    if args.force_train_ur is not None:
        print(f"Force-train: ur={args.force_train_ur}  "
              f"label(s)={[format_ur_label(u) for u in args.force_train_ur]}")
        train_cases, val_cases, test_cases = enforce_force_train(
            train_cases, val_cases, test_cases, set(all_cases_unsplit),
            args.force_train_ur
        )

    print(f"Train: {sorted(train_cases)}")
    print(f"Val:   {sorted(val_cases)}")
    print(f"Test:  {sorted(test_cases)}")

    if (train_cases & val_cases) or (train_cases & test_cases) or (val_cases & test_cases):
        raise ValueError("Split overlap detected after holdout enforcement.")

    input_cols = list(cfg.get("input_cols", ["disp", "vel", "acc"]))

    if nd_inputs:
        print(f"\nApplying nondimensional transform "
              f"(dataset={dataset}, D={D_nd}, fn={fn_nd}):")
        raw_df = apply_nd_transform(raw_df, nd_inputs=True, D=D_nd, fn=fn_nd,
                                    input_cols=input_cols)

        available_cases = sorted(
            (str(c) for c in raw_df["case"].drop_duplicates()), key=parse_ur_label
        )
        if available_cases:
            case_name = "Ur6.7385" if "Ur6.7385" in available_cases else available_cases[0]
            ur = parse_ur_label(case_name)
            U_case = ur * fn_nd * D_nd
            divisor = {"disp": D_nd, "vel": U_case, "acc": (U_case ** 2) / D_nd}
            print(f"  [ND preflight] dataset={dataset}  D={D_nd}  fn={fn_nd}")
            print(f"  [ND preflight] representative case={case_name}  Ur={ur}  "
                  f"recovered U=Ur*fn*D={U_case:.4f}")
            print(f"  [ND preflight] divisors: disp={divisor['disp']:.6f}  "
                  f"vel={divisor['vel']:.6f}  acc={divisor['acc']:.6f}")

    train_df = raw_df[raw_df["case"].isin(train_cases)].copy()
    val_df = raw_df[raw_df["case"].isin(val_cases)].copy()
    test_df = None if args.skip_test_eval else raw_df[raw_df["case"].isin(test_cases)].copy()

    if args.holdout_ur is not None:
        holdout_label = format_ur_label(args.holdout_ur)
        if holdout_label in train_df["case"].values:
            raise ValueError(f"Holdout {holdout_label} still in train_df!")
        if holdout_label in val_df["case"].values:
            raise ValueError(f"Holdout {holdout_label} still in val_df!")
        if test_df is not None and holdout_label not in test_df["case"].values:
            raise ValueError(f"Holdout {holdout_label} not in test_df!")

    target_col = str(cfg.get("target_col", "cl"))
    seq_len = int(cfg["seq_len"])
    batch_size = int(cfg["batch_size"])

    # Scalers and Ur statistics are fitted on the training cases only.
    x_scaler, y_scaler = fit_scalers(train_df, input_cols, target_col)

    print(f"\nx_scaler: mean={x_scaler.mean_} scale={x_scaler.scale_}")
    print(f"y_scaler ({target_col}): "
          f"mean={float(y_scaler.mean_[0]):.6f} scale={float(y_scaler.scale_[0]):.6f}")

    train_ur = np.array([parse_ur_label(c) for c in sorted(train_cases)],
                        dtype=np.float32)
    ur_mean = float(train_ur.mean())
    ur_std = float(train_ur.std()) + 1e-6

    if use_ur_context:
        print(f"Using Ur context feature: mean={ur_mean:.4f}, std={ur_std:.4f} "
              f"(fitted from {len(train_cases)} training cases)")

    coord_mode = "nondimensional" if nd_inputs else "dimensional"
    if nd_inputs:
        transform_formula = "disp/D,  vel/U,  acc*D/U²  (case-dependent U)"
    else:
        transform_formula = "identity (physical units)"

    run_config = {
        "cfd_dataset": dataset,
        "coordinate_mode": coord_mode,
        "nd_inputs": nd_inputs,
        "transform_formula": transform_formula,
        "D": float(D_nd) if D_nd is not None else None,
        "fn": float(fn_nd) if fn_nd is not None else None,
        "use_ur_context": use_ur_context,
        "holdout_ur": args.holdout_ur,
        "holdout_label": holdout_label,
        "exclude_ur": args.exclude_ur,
        "exclude_labels": exclude_labels or None,
        "force_train_ur": args.force_train_ur,
        "force_train_labels": ([format_ur_label(u) for u in args.force_train_ur]
                               if args.force_train_ur else None),
        "input_cols": input_cols,
        "acc_source": args.acc_source,
        "amplitude_aware_loss": args.amplitude_aware_loss,
        "target_col": target_col,
        "epochs": cfg["n_epochs"],
        "batch_size": batch_size,
        "seq_len": seq_len,
        "noise_std": float(args.noise_std),
        "noise_coordinate_space": "standardized model inputs",
        "skip_test_eval": args.skip_test_eval,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "output_directory": str(output_dir),
        "train_cases": sorted(train_cases),
        "val_cases": sorted(val_cases),
        "test_cases": sorted(test_cases),
        "scaler_fit_cases": sorted(train_cases),
        "ur_stats_fit_cases": sorted(train_cases),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"\nRun config saved to {output_dir / 'run_config.json'}")

    print(f"\nScaler-fit cases:   {sorted(train_cases)}")
    print(f"Ur-stats-fit cases: {sorted(train_cases)}")
    print(f"Noise std: {args.noise_std}  "
          f"(applied to standardized model inputs, after scaler transform)")

    if args.preflight_only:
        print("\n[PREFLIGHT MODE] Stopping before training.")
        print(f"CFD dataset: {dataset}")
        print(f"Coordinate mode: {coord_mode}")
        print(f"Use Ur context: {use_ur_context}")
        print(f"Holdout: ur={args.holdout_ur}  label={holdout_label}")
        print(f"Output directory: {output_dir}")
        test_rows_str = "SKIPPED (--skip_test_eval)" if test_df is None else f"{len(test_df)} test rows"
        print(f"Ready for training: {len(train_df)} train rows, "
              f"{len(val_df)} val rows, {test_rows_str}")
        return

    train_df_s = apply_scalers_to_df(train_df, x_scaler, y_scaler,
                                      input_cols, target_col)
    val_df_s = apply_scalers_to_df(val_df, x_scaler, y_scaler,
                                    input_cols, target_col)
    test_df_s = None if test_df is None else apply_scalers_to_df(
        test_df, x_scaler, y_scaler, input_cols, target_col)

    case_weights = None
    if args.amplitude_aware_loss:
        case_weights = compute_case_loss_weights(train_df_s, target_col, sorted(train_cases))
        print("\nAmplitude-aware loss weights (per training case, mean=1.0):")
        for c, w in sorted(case_weights.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {c:12s}  weight={w:.3f}")

    stride_train = int(cfg["stride_train"])
    common_ds_args = dict(
        seq_len=seq_len, target_col=target_col, input_cols=input_cols,
        release_time=release_time, use_ur_context=use_ur_context,
        ur_mean=ur_mean, ur_std=ur_std
    )

    train_ds = VIVSequenceDataset(train_df_s, stride=stride_train, **common_ds_args)
    val_ds = VIVSequenceDataset(val_df_s, stride=1, **common_ds_args)
    test_ds = None if test_df_s is None else VIVSequenceDataset(test_df_s, stride=1, **common_ds_args)

    test_ds_len_str = "SKIPPED (--skip_test_eval)" if test_ds is None else str(len(test_ds))
    print(f"\nDataset sizes — train: {len(train_ds)} val: {len(val_ds)} test: {test_ds_len_str}")

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = None if test_ds is None else DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    input_size = len(input_cols) + (1 if use_ur_context else 0)
    model = VIV_GRU(
        input_size=input_size,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    print(f"GRU parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Random seed: {args.seed} (controlled reproducibility)")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # Halve the learning rate after 5 epochs without validation improvement.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_count = 0
    train_losses, val_losses = [], []

    n_epochs = cfg["n_epochs"]
    print(f"\nTraining for up to {n_epochs} epochs "
          f"(noise_std={args.noise_std} in standardized space)...")

    for epoch in range(1, n_epochs + 1):
        tl = train_one_epoch(model, train_loader, optimizer, criterion, device,
                             input_noise_std=args.noise_std,
                             n_kinematic_cols=len(input_cols),
                             case_weights=case_weights)
        vl, vp, vt, _ = run_validation(model, val_loader, criterion, device)
        scheduler.step(vl)
        train_losses.append(tl)
        val_losses.append(vl)

        ss_res = np.sum((vt - vp) ** 2)
        ss_tot = np.sum((vt - vt.mean()) ** 2)
        val_r2 = 1 - ss_res / (ss_tot + 1e-10)

        print(f"Epoch {epoch:3d}/{n_epochs}  "
              f"train={tl:.5f}  val={vl:.5f}  "
              f"R²={val_r2:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a best model state.")

    model.load_state_dict(best_state)
    torch.save(best_state, output_dir / "gru_best.pt")
    print(f"\nBest model saved (val_loss={best_val_loss:.5f})")

    with open(output_dir / "x_scaler.pkl", "wb") as f:
        pickle.dump(x_scaler, f)
    with open(output_dir / "y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)

    ur_stats_dict = {
        "mean": ur_mean,
        "std": ur_std,
        "use_ur_context": use_ur_context,
        "nd_inputs": nd_inputs,
        "coordinate_mode": coord_mode,
        "cfd_dataset": dataset,
        "holdout_ur": args.holdout_ur,
        "exclude_ur": args.exclude_ur,
        "force_train_ur": args.force_train_ur,
        "exp_subdir": output_dir.name,
    }
    with open(output_dir / "ur_stats.pkl", "wb") as f:
        pickle.dump(ur_stats_dict, f)

    if test_loader is not None:
        _, tp, tt, _ = run_validation(model, test_loader, criterion, device)
        test_pred = y_scaler.inverse_transform(tp.reshape(-1, 1)).ravel()
        test_true = y_scaler.inverse_transform(tt.reshape(-1, 1)).ravel()
        test_metrics = evaluate(test_true, test_pred)
    else:
        test_metrics = None
        print("\nTest:       SKIPPED (--skip_test_eval) -- no test inference performed")

    _, vp, vt, _ = run_validation(model, val_loader, criterion, device)
    val_pred = y_scaler.inverse_transform(vp.reshape(-1, 1)).ravel()
    val_true = y_scaler.inverse_transform(vt.reshape(-1, 1)).ravel()
    val_metrics = evaluate(val_true, val_pred)

    print(f"\nValidation: {json.dumps(val_metrics, indent=2)}")
    if test_metrics is not None:
        print(f"Test:       {json.dumps(test_metrics, indent=2)}")

    tf_results = {}
    if test_df_s is not None:
        print("\nTeacher forcing rollout on test cases:")
        for case_name in sorted(test_cases):
            case_df_s = test_df_s[test_df_s["case"] == case_name].copy()
            if case_df_s.empty:
                continue

            cl_pred, cl_true, times_ar = teacher_forcing_rollout(
                model, case_df_s, input_cols, seq_len,
                release_time[case_name], y_scaler, case_name, device,
                use_ur_context=use_ur_context,
                ur_stats=(ur_mean, ur_std),
            )

            if len(cl_true) < 20:
                print(f"  {case_name}: too short, skipping")
                continue

            tf_m = evaluate(cl_true, cl_pred)
            tf_results[case_name] = tf_m
            print(f"  {case_name}: R²={tf_m['r2']:.4f}  RMSE={tf_m['rmse']:.4f}")
            plot_tf_result(cl_pred, cl_true, times_ar, case_name, output_dir, dataset=dataset)
    else:
        print("\nTeacher forcing rollout on test cases: SKIPPED (--skip_test_eval)")

    amplitude_dfs = [train_df_s, val_df_s] + ([test_df_s] if test_df_s is not None else [])
    all_df_s = pd.concat(amplitude_dfs, ignore_index=True)
    amplitude_comparison(
        model, all_df_s, release_time, input_cols, seq_len, y_scaler,
        device, output_dir, train_cases, val_cases,
        (test_cases if test_df_s is not None else set()),
        use_ur_context=use_ur_context, ur_stats=(ur_mean, ur_std),
    )

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(train_losses, color="black", label="train")
    ax.plot(val_losses, color="tab:blue", label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (scaled)")
    ax.legend()
    fig.savefig(output_dir / "learning_curve.png", dpi=150)
    plt.close(fig)

    metrics = {
        "dataset": dataset,
        "coordinate_mode": coord_mode,
        "nd_inputs": nd_inputs,
        "use_ur_context": use_ur_context,
        "holdout_ur": args.holdout_ur,
        "holdout_label": holdout_label,
        "exclude_ur": args.exclude_ur,
        "force_train_ur": args.force_train_ur,
        "gru_config": {k: v for k, v in cfg.items() if not callable(v)},
        "case_split": {
            "train": sorted(train_cases),
            "val": sorted(val_cases),
            "test": sorted(test_cases),
        },
        "scaler_fit_cases": sorted(train_cases),
        "ur_stats_fit_cases": sorted(train_cases),
        "skip_test_eval": args.skip_test_eval,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "tf_results": tf_results,
    }
    with open(output_dir / "metrics_gru.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
