r"""Closed-loop GRU-structure simulation for one reduced velocity (thesis Sec. 4.6).

The CFD solver is replaced by the trained GRU:
    1. Start from a window of CFD kinematics (warm-start) and hand off to the
       surrogate at  handoff = release + seq_len + handoff_offset  steps.
    2. At every step the GRU predicts C_L from the last seq_len states.
    3. F = 0.5 rho U^2 B C_L  (B = deck width for the bridge, D for the cylinder).
    4. Newmark-beta (average acceleration) advances displacement, velocity, acceleration.
    5. The new state is appended to the input window.

Reads the model folder written by train_gru.py (results/<model_subdir>/) and
the CFD case at the requested Ur.

Writes to results/ or --output_dir:
    coupled_<dataset>_Ur<Ur>_<tags>.npz   trajectory (t, h, cl, h_dot, h_ddot, e)
                                         plus the CFD reference h_cfd, cl_cfd
    .receipt.json                        settings and git commit of the run
    coupled_viv_Ur<Ur>_<tags>.png        quick-look plot

Options:
    --residual_npz + --noise_mode surrogate|white   residual forcing test (Sec. 6.6.2)
    --run_replay_diag   drive Newmark with the CFD lift instead of the GRU (Fig. 6.8)

Example:
    PYTHONPATH=src python -m viv_analysis.coupled_inference --cfd_dataset cylinder200 \
        --model_subdir gru_cylinder200_nd_context_noacc --Ur 5 --total_time 300
"""

import argparse
import hashlib
import json
import subprocess
from math import gamma
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
from viv_analysis.models.gru import VIV_GRU
import pickle
import matplotlib.pyplot as plt
matplotlib.use("Agg")

from viv_analysis.preprocess import compute_kinematics, merge_dataframes, downsample
from viv_analysis.utils import PROJECT_ROOT, format_ur_label, present_model_label
from viv_analysis.config import config, CYLINDER200_ALIASES, cylinder200_structural_params


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
})


def to_model_coords(kin: np.ndarray, nd_inputs: bool, D: float, U: float,
                    input_cols: tuple[str, ...] = ("disp", "vel", "acc")) -> np.ndarray:
    """Convert physical kinematics to model inputs: unchanged, or disp/D, vel/U, acc/(U^2/D) if nd_inputs."""
    if not nd_inputs:
        return kin
    if D <= 0 or U <= 0:
        raise ValueError(f"to_model_coords requires D>0, U>0 (got D={D}, U={U})")
    divisor_by_name = {"disp": D, "vel": U, "acc": (U * U) / D}
    unsupported = [c for c in input_cols if c not in divisor_by_name]
    if unsupported:
        raise ValueError(f"to_model_coords: unsupported input_cols {unsupported}")
    divisors = np.array([divisor_by_name[c] for c in input_cols], dtype=np.float32)
    return kin / divisors


def positive_finite_float(value: str) -> float:
    """argparse type: a finite number > 0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected a float, got {value!r}")
    if not np.isfinite(v):
        raise argparse.ArgumentTypeError(f"expected a finite value, got {v}")
    if v <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive value, got {v}")
    return v


def get_git_dirty_and_patch_hash(cwd: Optional[Path] = None) -> tuple[bool, str]:
    """Whether the working tree has uncommitted changes, and a hash of `git diff HEAD` (stored in the receipt)."""
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd or PROJECT_ROOT,
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
        dirty = len(status.strip()) > 0
        diff = subprocess.run(
            ["git", "diff", "HEAD", "--binary"], cwd=cwd or PROJECT_ROOT,
            capture_output=True, timeout=15, check=True,
        ).stdout
        patch_hash = hashlib.sha256(diff).hexdigest() if diff else ""
        return dirty, patch_hash
    except Exception:
        return True, ""


def get_git_commit(cwd: Optional[Path] = None) -> Optional[str]:
    """Current git commit hash, or None if git is unavailable (stored in the receipt)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd or PROJECT_ROOT,
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


LEGACY_COORD_SOURCE = "legacy assumption: dimensional"


def load_artifact_coordinate_mode(artifact_dir: Path, cli_nd_inputs: bool | None) -> bool:
    """Decide whether the model expects nondimensional inputs.

    Reads the mode recorded in run_config.json (or ur_stats.pkl). A command-line
    choice that contradicts the recorded mode is an error. Old model folders
    without a record are treated as dimensional and cannot be run with --nd_inputs.
    """
    artifact_nd_inputs = None
    metadata_source = None

    run_config_path = artifact_dir / "run_config.json"
    if run_config_path.exists():
        try:
            with open(run_config_path, "r") as f:
                run_config = json.load(f)
            recorded = run_config.get("nd_inputs", None)
            if recorded is not None:
                artifact_nd_inputs = bool(recorded)
                metadata_source = f"run_config.json (coordinate_mode={run_config.get('coordinate_mode', '?')})"
        except Exception as e:
            print(f"[warn] Could not read run_config.json: {e}")

    if artifact_nd_inputs is None:
        ur_stats_path = artifact_dir / "ur_stats.pkl"
        if ur_stats_path.exists():
            try:
                with open(ur_stats_path, "rb") as f:
                    ur_stats_dict = pickle.load(f)
                recorded = ur_stats_dict.get("nd_inputs", None)
                if recorded is not None:
                    artifact_nd_inputs = bool(recorded)
                    metadata_source = "ur_stats.pkl"
            except Exception as e:
                print(f"[warn] Could not read ur_stats.pkl: {e}")

    if artifact_nd_inputs is None:
        metadata_source = LEGACY_COORD_SOURCE
        print(f"[warn] [coord] No recorded coordinate mode in {artifact_dir} "
              f"(no run_config.json, and ur_stats.pkl has no nd_inputs key). "
              f"Assuming dimensional mode for backward compatibility "
              f"({metadata_source}); this is an ASSUMPTION, not a recorded fact.")
        if cli_nd_inputs:
            raise ValueError(
                f"--nd_inputs was requested but {artifact_dir} has no recorded "
                f"coordinate mode (no run_config.json, and ur_stats.pkl has no "
                f"nd_inputs key). This artifact predates coordinate-mode "
                f"tracking, so its true training coordinate space is unknown; "
                f"refusing to silently assume nondimensional. Retrain with "
                f"recorded metadata (or add a run_config.json with "
                f"nd_inputs=true) before using --nd_inputs on this artifact."
            )
        return False

    if cli_nd_inputs is None:
        print(f"[coord] Using coordinate mode from artifact ({metadata_source}): "
              f"{'nondimensional' if artifact_nd_inputs else 'dimensional'}")
        return artifact_nd_inputs

    if cli_nd_inputs != artifact_nd_inputs:
        raise ValueError(
            f"Coordinate mode mismatch: CLI specifies "
            f"{'--nd_inputs' if cli_nd_inputs else '--dim_inputs'} "
            f"but artifact was trained in "
            f"{'nondimensional' if artifact_nd_inputs else 'dimensional'} mode "
            f"({metadata_source}). "
            f"This would silently corrupt inference. Use matching coordinate mode."
        )
    return cli_nd_inputs


def normalize_cfd_dataset(dataset: str) -> str:
    """Canonical dataset name ('cylinder200' or 'bridge')."""
    ds = dataset.strip().lower()
    if ds == "bridge":
        return "bridge"
    if ds in CYLINDER200_ALIASES:
        return "cylinder200"
    return ds


def check_artifact_dataset_compatibility(artifact_dir: Path, requested_cfd_dataset: str) -> None:
    """Error if the model was trained on a different dataset than the one requested."""
    recorded_dataset = None
    metadata_source = None

    run_config_path = artifact_dir / "run_config.json"
    if run_config_path.exists():
        try:
            with open(run_config_path, "r") as f:
                run_config = json.load(f)
            recorded = run_config.get("cfd_dataset", None)
            if recorded:
                recorded_dataset = recorded
                metadata_source = "run_config.json"
        except Exception as e:
            print(f"[warn] Could not read run_config.json for dataset check: {e}")

    if recorded_dataset is None:
        ur_stats_path = artifact_dir / "ur_stats.pkl"
        if ur_stats_path.exists():
            try:
                with open(ur_stats_path, "rb") as f:
                    ur_stats_dict = pickle.load(f)
                recorded = ur_stats_dict.get("cfd_dataset", None)
                if recorded:
                    recorded_dataset = recorded
                    metadata_source = "ur_stats.pkl"
            except Exception as e:
                print(f"[warn] Could not read ur_stats.pkl for dataset check: {e}")

    if recorded_dataset is None:
        print(f"[warn] [dataset] {artifact_dir} has no recorded cfd_dataset "
              f"(no run_config.json, and ur_stats.pkl has no cfd_dataset key); "
              f"dataset compatibility could not be verified. Proceeding with "
              f"requested dataset '{requested_cfd_dataset}' unchecked.")
        return

    recorded_canonical = normalize_cfd_dataset(recorded_dataset)
    requested_canonical = normalize_cfd_dataset(requested_cfd_dataset)

    if recorded_canonical != requested_canonical:
        raise ValueError(
            f"Dataset mismatch: requested CFD dataset does not match the "
            f"artifact's recorded training dataset. "
            f"requested CFD dataset: '{requested_cfd_dataset}' (canonical: '{requested_canonical}'). "
            f"recorded artifact dataset: '{recorded_dataset}' (canonical: '{recorded_canonical}', from {metadata_source}). "
            f"artifact directory: {artifact_dir}. "
            f"Using the wrong dataset's physical constants (D/fn/B/m/c/k) "
            f"would silently corrupt inference."
        )

    print(f"[dataset] Verified: requested '{requested_cfd_dataset}' matches "
          f"artifact's recorded '{recorded_dataset}' ({metadata_source})")


def Newmark_beta( F, h, h_dot, h_ddot, dt, m, c, k, beta=0.25, gamma=0.5):
    """One Newmark-beta step for m h'' + c h' + k h = F.

    beta = 1/4, gamma = 1/2 (average acceleration, unconditionally stable).
    Takes the state at step n and the force F, returns the state at step n+1.
    Works on floats and on torch tensors (used by the rollout training).
    """
    a1 = m / (beta * dt**2) + gamma * c / (beta * dt)
    a2 = m / (beta * dt) + (gamma / beta - 1.0) * c
    a3 = (0.5 / beta - 1.0) * m + dt * (gamma / (2.0 * beta) - 1.0) * c
    kbar = k + a1

    pbar = F + a1 * h + a2 * h_dot + a3 * h_ddot
    h_new = pbar / kbar

    h_dot_new = (gamma / (beta * dt)) * (h_new - h) + (1.0 - gamma / beta) * h_dot \
                + dt * (1.0 - gamma / (2.0 * beta)) * h_ddot
    h_ddot_new = (1.0 / (beta * dt**2)) * (h_new - h) - (1.0 / (beta * dt)) * h_dot \
                 - (0.5 / beta - 1.0) * h_ddot

    return h_new, h_dot_new, h_ddot_new

def warmup_history(
    cfd_case_df,
    release_t,
    seq_len,
    input_cols,
    x_scaler,
    nd_inputs: bool,
    D: float,
    U: float,
    use_ur_context,
    ur_value,
    ur_stats,
    handoff_offset_steps: int = 0,
    cfd_scale: float = 1.0,
):
    """Build the initial input window and state from the CFD record.

    The window is the seq_len CFD rows before the handoff index, converted to model
    coordinates and scaled. The initial state is the CFD disp/vel/acc at the
    handoff index. Returns (window, state, handoff time, handoff index).
    """
    ordered = cfd_case_df.sort_values("time").reset_index(drop=True)
    times = ordered["time"].to_numpy(dtype=np.float32)

    release_idx = int(np.searchsorted(times, release_t))

    handoff_idx = release_idx + seq_len + int(handoff_offset_steps)

    if handoff_idx >= len(ordered):
        raise ValueError(
            f"CFD trajectory too short: handoff_idx={handoff_idx}, "
            f"trajectory length={len(ordered)}."
        )

    win_start = handoff_idx - seq_len
    win_end = handoff_idx

    if win_start < 0:
        raise ValueError(
            f"Invalid warmup window: win_start={win_start}, seq_len={seq_len}."
        )

    win = ordered.iloc[win_start:win_end]
    if len(win) != seq_len:
        raise ValueError(
            f"Warmup window length mismatch: got {len(win)}, expected {seq_len}."
        )

    kinematics = win[input_cols].to_numpy(dtype=np.float32)
    kinematics = kinematics * float(cfd_scale)
    kinematics = to_model_coords(kinematics, nd_inputs, D, U, input_cols=input_cols)
    kinematics_scaled = x_scaler.transform(kinematics)

    if use_ur_context:
        ur_mean, ur_std = ur_stats
        ur_std_safe = float(ur_std) if abs(float(ur_std)) > 0 else 1.0
        ur_scaled = (float(ur_value) - float(ur_mean)) / ur_std_safe
        ur_column = np.full((seq_len, 1), ur_scaled, dtype=np.float32)
        history = np.hstack([kinematics_scaled, ur_column])
    else:
        history = kinematics_scaled

    initial_state = {
        "h": float(ordered["disp"].iloc[handoff_idx]) * float(cfd_scale),
        "h_dot": float(ordered["vel"].iloc[handoff_idx]) * float(cfd_scale),
        "h_ddot": float(ordered["acc"].iloc[handoff_idx]) * float(cfd_scale),
    }

    handoff_time = float(times[handoff_idx])

    return history.astype(np.float32), initial_state, handoff_time, handoff_idx

def diagnostic_true_force_newmark_replay(
    case_df,
    handoff_idx,
    n_steps,
    m,
    c,
    k,
    rho,
    U,
    D,
    B=None,
    dt=None,
    force_timing: str = "current",
):
    """Replay the structural response with the CFD lift instead of the GRU (thesis Fig. 6.8).

    If Newmark driven by the CFD force reproduces the CFD displacement, the
    structural part of the coupling is correct. force_timing selects whether the
    force of the current or the next sample drives each step.
    """
    ordered = case_df.sort_values("time").reset_index(drop=True)

    if handoff_idx + n_steps + 1 >= len(ordered):
        n_steps = len(ordered) - handoff_idx - 2

    cfd_h = ordered["disp"].to_numpy(dtype=np.float64)
    cfd_v = ordered["vel"].to_numpy(dtype=np.float64)
    cfd_a = ordered["acc"].to_numpy(dtype=np.float64)
    cfd_cl = ordered["cl"].to_numpy(dtype=np.float64)
    times = ordered["time"].to_numpy(dtype=np.float64)

    B = D if B is None else B
    if dt is None:
        raise ValueError("dt must be provided for diagnostic_true_force_newmark_replay")

    h = np.zeros(n_steps + 1, dtype=np.float64)
    v = np.zeros(n_steps + 1, dtype=np.float64)
    a = np.zeros(n_steps + 1, dtype=np.float64)

    h[0] = cfd_h[handoff_idx]
    v[0] = cfd_v[handoff_idx]
    a[0] = cfd_a[handoff_idx]

    qD = 0.5 * rho * U**2 * B

    for j in range(n_steps):
        if force_timing == "current":
            cl_used = cfd_cl[handoff_idx + j]
        elif force_timing == "next":
            cl_used = cfd_cl[handoff_idx + j + 1]
        else:
            raise ValueError("force_timing must be 'current' or 'next'")

        F = qD * cl_used

        h[j + 1], v[j + 1], a[j + 1] = Newmark_beta(
            F=F,
            h=h[j],
            h_dot=v[j],
            h_ddot=a[j],
            dt=dt,
            m=m,
            c=c,
            k=k,
        )

        if not np.isfinite([h[j+1], v[j+1], a[j+1]]).all():
            raise FloatingPointError(f"Non-finite Newmark replay at step={j}")

    cfd_h_cmp = cfd_h[handoff_idx + 1 : handoff_idx + n_steps + 1]
    cfd_v_cmp = cfd_v[handoff_idx + 1 : handoff_idx + n_steps + 1]
    cfd_a_cmp = cfd_a[handoff_idx + 1 : handoff_idx + n_steps + 1]
    t_cmp = times[handoff_idx + 1 : handoff_idx + n_steps + 1]

    def rmse(x, y):
        return float(np.sqrt(np.mean((np.asarray(x) - np.asarray(y)) ** 2)))

    result = {
        "time": t_cmp,
        "h_replay": h[1:],
        "v_replay": v[1:],
        "a_replay": a[1:],
        "h_cfd": cfd_h_cmp,
        "v_cfd": cfd_v_cmp,
        "a_cfd": cfd_a_cmp,
        "rmse_h": rmse(h[1:], cfd_h_cmp),
        "rmse_v": rmse(v[1:], cfd_v_cmp),
        "rmse_a": rmse(a[1:], cfd_a_cmp),
        "max_abs_h_over_D": float(np.max(np.abs(h[1:] - cfd_h_cmp)) / D),
        "force_timing": force_timing,
    }

    print(f"\nTrue-force Newmark replay [{force_timing}]")
    print("-----------------------------------------")
    print(f"RMSE h       = {result['rmse_h']:.6e} m")
    print(f"RMSE v       = {result['rmse_v']:.6e} m/s")
    print(f"RMSE a       = {result['rmse_a']:.6e} m/s²")
    print(f"Max |h err|/D= {result['max_abs_h_over_D']:.6f}")

    return result


def run_coupled_viv(
    model:        VIV_GRU,
    x_scaler,
    y_scaler,
    seq_len:      int,
    input_cols:   list[str],
    initial_history: np.ndarray,
    initial_state: dict,
    m:            float,
    c:            float,
    k:            float,
    rho:          float,
    U:            float,
    D:            float,
    n_steps:      int,
    B:          float = None,
    dt:           float = None,
    use_ur_context: bool = False,
    ur_value:     float = 0.0,
    ur_stats:     tuple   = (0.0, 1.0),
    device:       str = "cpu",
    nd_inputs:    bool = False,
    e_forcing:    np.ndarray = None,
    track_hidden: bool = False,
) -> dict:
    """The closed loop: GRU lift -> aerodynamic force -> Newmark step -> new input row.

    initial_history and initial_state come from warmup_history. e_forcing is an
    optional additive C_L forcing of length n_steps. Returns a dict with the time
    series (time, displacement, velocity, acceleration, CL, CL_det = GRU part,
    e_forcing) and simple diagnostics.
    """
    model.eval()
    B = D if B is None else B
    if dt is None:
        raise ValueError("dt must be provided for run_coupled_viv")

    h      = np.zeros(n_steps + 1, dtype=np.float32)
    h_dot  = np.zeros(n_steps + 1, dtype=np.float32)
    h_ddot = np.zeros(n_steps + 1, dtype=np.float32)
    CL     = np.zeros(n_steps + 1, dtype=np.float32)


    h[0]      = initial_state["h"]
    h_dot[0]  = initial_state["h_dot"]
    h_ddot[0] = initial_state["h_ddot"]
    history   = initial_history.copy()

    expected_features = len(input_cols) + (1 if use_ur_context else 0)
    if history.ndim != 2 or history.shape[1] != expected_features:
        raise ValueError(
            f"History feature mismatch: got {history.shape}, "
            f"expected (*, {expected_features})."
        )

    max_abs_z_seen = 0.0
    n_ood_warnings = 0

    if e_forcing is None or len(e_forcing) == 0:
        _e = np.zeros(n_steps, dtype=np.float32)
    else:
        _e = np.asarray(e_forcing, dtype=np.float32)
        if len(_e) < n_steps:
            raise ValueError(
                f"e_forcing length {len(_e)} < n_steps {n_steps}; pad or tile before passing."
            )
        _e = _e[:n_steps]

    ur_mean, ur_std = ur_stats
    ur_std_safe = float(ur_std) if abs(float(ur_std)) > 0 else 1.0
    ur_scaled = (ur_value - float(ur_mean)) / ur_std_safe

    x_mean = x_scaler.mean_.astype(np.float32)
    x_scale = x_scaler.scale_.astype(np.float32)
    y_mean = float(y_scaler.mean_[0])
    y_scale = float(y_scaler.scale_[0])
    _nd_divisor_by_name = {"disp": D, "vel": U, "acc": (U * U) / D} if nd_inputs else None
    _state_names = ("disp", "vel", "acc")

    CL_det_arr = np.zeros(n_steps, dtype=np.float32)
    hidden_norms = [] if track_hidden else None
    hidden_vecs = [] if track_hidden else None
    zmax_trace = [] if track_hidden else None

    with torch.inference_mode():
        for i in range(n_steps):
            x = torch.from_numpy(history).unsqueeze(0).to(device)
            cl_scaled, hn = model(x)
            if track_hidden:
                hidden_norms.append(float(torch.linalg.norm(hn[-1]).item()))
                hidden_vecs.append(hn[-1].detach().cpu().numpy().ravel().copy())
            cl_det = float(cl_scaled.item()) * y_scale + y_mean
            # e is the residual forcing (zero unless --residual_npz and --noise_mode are given).
            cl = cl_det + _e[i]

            CL_det_arr[i] = cl_det
            CL[i] = cl

            F_aero = 0.5 * rho * U**2 * B * cl

            h[i+1], h_dot[i+1], h_ddot[i+1] = Newmark_beta(
                F=F_aero,
                h=h[i],
                h_dot=h_dot[i],
                h_ddot=h_ddot[i],
                dt=dt,
                m=m,
                c=c,
                k=k,
            )
            # Append the state that drove this step, so the window ends at step i and the
            # next prediction is C_L at i+1 (same convention as VIVSequenceDataset).
            _state_by_name = dict(zip(_state_names, (h[i], h_dot[i], h_ddot[i])))
            new_kinematics_raw = np.array(
                [_state_by_name[c] for c in input_cols], dtype=np.float32)
            if nd_inputs:
                new_kinematics_t = np.array(
                    [_state_by_name[c] / _nd_divisor_by_name[c] for c in input_cols],
                    dtype=np.float32)
            else:
                new_kinematics_t = new_kinematics_raw
            new_kinematics_scaled = (new_kinematics_t - x_mean) / x_scale

            if use_ur_context:
                new_row = np.append(new_kinematics_scaled, ur_scaled).astype(np.float32)
            else:
                new_row = new_kinematics_scaled.astype(np.float32)

            if not np.isfinite([h[i+1], h_dot[i+1], h_ddot[i+1], cl]).all():
                raise FloatingPointError(
                    f"Non-finite state at step={i}: "
                    f"h={h[i+1]}, v={h_dot[i+1]}, a={h_ddot[i+1]}, CL={cl}"
                )

            zmax = float(max(abs(v) for v in new_kinematics_scaled))
            max_abs_z_seen = max(max_abs_z_seen, zmax)
            if track_hidden:
                zmax_trace.append(zmax)

            if zmax > 6.0 and n_ood_warnings < 10:
                n_ood_warnings += 1
                print(
                    f"WARNING OOD step={i}: max|z|={zmax:.2f}, "
                    f"h={h[i]:.4e}, v={h_dot[i]:.4e}, "
                    f"a={h_ddot[i]:.4e}, CL={cl:.4e}"
                )

            history = np.roll(history, -1, axis=0)
            history[-1] = new_row


    t = np.arange(n_steps + 1) * dt

    return {
        "time": t[:n_steps],
        "displacement": h[:n_steps],
        "velocity": h_dot[:n_steps],
        "acceleration": h_ddot[:n_steps],
        "CL": CL[:n_steps],
        "CL_det": CL_det_arr,
        "e_forcing": _e[:n_steps].copy(),
        "max_abs_scaled_kinematics": float(max_abs_z_seen),
        "n_ood_warnings": int(n_ood_warnings),
        "hidden_norm": np.array(hidden_norms, dtype=np.float64) if track_hidden else None,
        "hidden_state": np.stack(hidden_vecs, axis=0) if track_hidden else None,
        "max_abs_z_trace": np.array(zmax_trace, dtype=np.float64) if track_hidden else None,
    }


def main(
    Ur: float = 6.0,
    cfd_dataset: str = "cylinder200",
    model_dataset: str = "cylinder200",
    total_time: float = 300.0,
    checkpoint: str = "gru_best.pt",
    model_subdir: Optional[str] = None,
    handoff_offset_steps: int = 2000,
    cfd_scale: float = 1.0,
    nd_inputs: bool = False,
    residual_npz: Optional[str] = None,
    noise_mode: str = "none",
    noise_scale: float = 1.0,
    noise_seed: int = 0,
    run_replay_diag: bool = False,
    replay_duration_s: Optional[float] = None,
    output_dir: Optional[str] = None,
    ):
    """Load the model and CFD case, run the closed loop and save npz, receipt and plot."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if output_dir is None:
        results_out_dir = PROJECT_ROOT / "results"
    elif Path(output_dir).is_absolute():
        results_out_dir = Path(output_dir)
    else:
        results_out_dir = PROJECT_ROOT / "results" / output_dir
    results_out_dir.mkdir(parents=True, exist_ok=True)

    if model_subdir is not None:
        artifact_dir = PROJECT_ROOT / "results" / model_subdir
    else:
        artifact_dir = PROJECT_ROOT / "results" / f"gru_{model_dataset}"
        artifact_dir_base = PROJECT_ROOT / "results" / f"gru_{cfd_dataset}"


    with open(artifact_dir / "x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open(artifact_dir / "y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)
    if model_subdir is not None:
        with open(artifact_dir / "ur_stats.pkl", "rb") as f:
            ur_info = pickle.load(f)
    else:
        with open(artifact_dir_base / "ur_stats.pkl", "rb") as f:
            ur_info = pickle.load(f)

    cli_coord_choice = None
    if args.nd_inputs:
        cli_coord_choice = True
    elif args.dim_inputs:
        cli_coord_choice = False

    nd_inputs = load_artifact_coordinate_mode(artifact_dir, cli_coord_choice)

    check_artifact_dataset_compatibility(artifact_dir, cfd_dataset)


    use_ur_context = bool(ur_info["use_ur_context"])
    ur_mean   = float(ur_info["mean"])
    ur_std    = float(ur_info["std"]) + 1e-8
    print(f"Loaded scalers and UR stats: use_ur_context={use_ur_context}"
          f"mean={ur_mean:.4f}, std={ur_std:.4f}")
    print(f"y_scaler: mean={float(y_scaler.mean_[0]):.6f}  "
          f"scale={float(y_scaler.scale_[0]):.6f}")


    ds = cfd_dataset.strip().lower()
    params_Re1000 = None; bsp = None
    if ds == "bridge":
        from viv_analysis.config import bridge_structural_params
        bsp = bridge_structural_params()
        m, c, k = bsp["m"], bsp["c"], bsp["k"]
        rho = config["bridge_rho"]
        fn  = config["bridge_fn_hz"]
        D   = config["bridge_D_ref"]
        B   = config["bridge_B_ref"]
        t_star_release = config["bridge_t_star_release"]
        dt  = None
    elif ds in CYLINDER200_ALIASES:
        rho = config["cylinder200_rho"]
        D   = config["cylinder200_D_ref"]
        B   = D
        fn  = config["cylinder200_fn"]
        params_Re1000 = cylinder200_structural_params()
        m, c, k = params_Re1000["m"], params_Re1000["c"], params_Re1000["k"]
        t_star_release = config["cylinder200_t_star_release"]
        dt = config["cylinder200_dt"]
    else:
        raise ValueError(
            f"coupled_inference: unsupported cfd_dataset '{cfd_dataset}'. "
            f"Supported: 'bridge', cylinder200 (aliases: {sorted(CYLINDER200_ALIASES)})."
        )

    U = Ur * fn * D
    t_release = t_star_release * D / U
    print(f"[{ds}] Ur={Ur}  U={U:.4f} m/s  fn={fn}  D(Ur,rel,h/D)={D}  B(force)={B}")
    print(f"  m={m:.6e}  c={c:.6e}  k={k:.6e}  rho={rho}  t_release={t_release:.4f}s")


    input_cols = ["disp", "vel", "acc"]
    hidden_size, num_layers = 64, 2
    if model_subdir is not None:
        metrics_path = artifact_dir / "metrics_gru.json"
    else:
        metrics_path = artifact_dir_base / "metrics_gru.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            saved_metrics = json.load(f)
        hidden_size = saved_metrics["gru_config"].get("hidden_size", hidden_size)
        num_layers  = saved_metrics["gru_config"].get("num_layers", num_layers)
        seq_len = saved_metrics["gru_config"].get("seq_len", 1000)
        input_cols = saved_metrics["gru_config"].get("input_cols", input_cols)
    input_size = len(input_cols) + (1 if use_ur_context else 0)


    print(f"\nLoading CFD trajectory at Ur={Ur} for warm-start...")
    if ds == "bridge":
        from viv_analysis.preprocess import load_bridge_df_cached

        raw_df = load_bridge_df_cached(
            fn_hz=fn,
            d_ref=D,
            bridge_structural_params=bsp,
        )
    else:
        raw_df = merge_dataframes(dataset=cfd_dataset)
        if raw_df.empty:
            raise RuntimeError("Could not load CFD data for warmup.")
        raw_df = compute_kinematics(
            raw_df,
            dataset=cfd_dataset,
            structural_params=params_Re1000,
        )
    if raw_df.empty:
        raise RuntimeError("Could not load CFD data for warmup.")

    case_label = format_ur_label(Ur)
    case_df = raw_df[raw_df["case"] == case_label].copy()
    if case_df.empty:
        available_cases = sorted(raw_df["case"].unique().tolist())
        raise ValueError(f"No CFD data found for case '{case_label}'."
                        f"Available cases: {available_cases}")
    print(f"  CFD trajectory has {len(case_df)} steps")

    if dt is None:
        tt = np.sort(np.unique(case_df["time"].to_numpy(dtype=np.float64)))
        dt = float(np.median(np.diff(tt)))
        print(f"  [bridge] Newmark dt = {dt:.6f}s  ({(1/fn)/dt:.0f} steps/cycle)")


    initial_history, initial_state, t_handoff, handoff_idx = warmup_history(
        cfd_case_df=case_df,
        release_t=t_release,
        seq_len=seq_len,
        input_cols=input_cols,
        x_scaler=x_scaler,
        nd_inputs=nd_inputs,
        D=D,
        U=U,
        use_ur_context=use_ur_context,
        ur_value=Ur,
        ur_stats=(ur_mean, ur_std),
        handoff_offset_steps=handoff_offset_steps,
        cfd_scale=cfd_scale,
    )
    print(f"  Warm-start window: CFD steps "
          f"{handoff_idx}]  (post-release)")
    print(f"  Handoff at t={t_handoff:.4f}s   "
          f"({seq_len*dt:.2f}s after release)")
    print(f"  Initial state at handoff: "
          f"h={initial_state['h']:.6f}  h_dot={initial_state['h_dot']:.6f}  "
          f"h_ddot={initial_state['h_ddot']:.6f}")


    raw_kin = case_df.iloc[handoff_idx - seq_len : handoff_idx][input_cols].to_numpy()
    raw_kin = raw_kin * float(cfd_scale)
    raw_kin = to_model_coords(raw_kin, nd_inputs, D, U, input_cols=input_cols)
    print(f"  Warm-start raw stats (scaled by {cfd_scale}):")
    for _i, _col in enumerate(input_cols):
        print(f"    {_col:4s} range: [{raw_kin[:,_i].min():.5f}, {raw_kin[:,_i].max():.5f}]")


    model = VIV_GRU(input_size=input_size, hidden_size=hidden_size,
                    num_layers=num_layers, dropout=0.1).to(device)

    ckpt = torch.load(artifact_dir / checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    print(f"Model loaded: input_size={input_size}  hidden_size={hidden_size}")

    expected_features = len(input_cols) + (1 if use_ur_context else 0)

    if initial_history.shape != (seq_len, expected_features):
        raise ValueError(
            f"Initial history shape mismatch: got {initial_history.shape}, "
            f"expected ({seq_len}, {expected_features})."
        )

    print(
        f"Model/input check: input_size={input_size}, "
        f"history_shape={initial_history.shape}"
    )

    if run_replay_diag:
        replay_n_steps = (int(round(replay_duration_s / dt))
                           if replay_duration_s is not None else 5000)
        replay_current = diagnostic_true_force_newmark_replay(
            case_df=case_df,
            handoff_idx=handoff_idx,
            n_steps=replay_n_steps,
            m=m,
            c=c,
            k=k,
            rho=rho,
            U=U,
            D=D,
            B=B,
            dt=dt,
            force_timing="current",
        )

        replay_next = diagnostic_true_force_newmark_replay(
            case_df=case_df,
            handoff_idx=handoff_idx,
            n_steps=replay_n_steps,
            m=m,
            c=c,
            k=k,
            rho=rho,
            U=U,
            D=D,
            B=B,
            dt=dt,
            force_timing="next",
        )

        best_replay = (
            replay_current
            if replay_current["max_abs_h_over_D"] <= replay_next["max_abs_h_over_D"]
            else replay_next
        )

        from viv_analysis.plotting.plot_style import (
            CFD_STYLE, MODEL_STYLE, TEXT_WIDTH_IN, apply_thesis_style,
        )
        apply_thesis_style()
        INCLUDE_WIDTH_FRAC = 0.8
        t_star_replay = best_replay["time"] * U / D

        fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * INCLUDE_WIDTH_FRAC, 4.0 * INCLUDE_WIDTH_FRAC),
                               constrained_layout=True)
        ax.plot(t_star_replay, best_replay["h_cfd"] / D, **{**CFD_STYLE, "label": "CFD"})
        ax.plot(t_star_replay, best_replay["h_replay"] / D,
                **{**MODEL_STYLE, "label": f"Newmark replay ({best_replay['force_timing']})"})
        ax.set_xlabel(r"$t^*=tU/D$")
        ax.set_ylabel("$h/D$")
        ax.legend()
        ax.grid(True, alpha=0.3)

        replay_stem_name = f"diagnostic_newmark_replay_Ur_{Ur}_{checkpoint}_handoff_{handoff_offset_steps}_{model_subdir}"
        replay_png = results_out_dir / f"{replay_stem_name}.png"
        fig.savefig(replay_png, dpi=150)
        fig.savefig(results_out_dir / f"{replay_stem_name}.pdf")
        plt.close(fig)

        print(f"Saved Newmark replay diagnostic to {replay_png}_{checkpoint}")
    else:
        print("[info] --run_replay_diag not set; skipping Newmark replay diagnostic")

    n_steps = int((total_time - t_handoff) / dt)
    if n_steps <= 0:
        raise ValueError(
            f"total_time={total_time}s is before handoff t={t_handoff:.2f}s "
            f"(total_time is absolute sim end-time, not duration); need > {t_handoff:.1f}"
        )
    print(f"\nRunning coupled inference for {total_time}s ({n_steps} steps)...")
    print(f"{n_steps} steps) ...")

    e_forcing = np.zeros(n_steps, dtype=np.float32)
    if noise_mode != "none" and residual_npz is not None:
        from viv_analysis.residual_forcing import make_forcing
        d_npz = np.load(residual_npz)
        if "Ur" in d_npz and abs(float(d_npz["Ur"]) - Ur) > 1e-6:
            raise ValueError(f"residual_npz is Ur={float(d_npz['Ur'])} but run is Ur={Ur}")
        resid = np.asarray(d_npz["cl_true"], float) - np.asarray(d_npz["cl_tf"], float)
        # Drop the first 5 s of the residual record (same convention as build_pooled_tf_residual --skip_s).
        skip = int(5.0 / float(d_npz["dt"]))
        resid = resid[skip:]
        e_forcing = make_forcing(resid, n_steps, mode=noise_mode, scale=noise_scale, seed=noise_seed)
        print(f"[stochastic] mode={noise_mode} scale={noise_scale} "
              f"resid_std={resid.std():.4f} e_std={e_forcing.std():.4f}")
    result = run_coupled_viv(
        model        = model,
        x_scaler     = x_scaler,
        y_scaler     = y_scaler,
        initial_history = initial_history,
        initial_state = initial_state,
        seq_len      = seq_len,
        input_cols   = input_cols,
        m            = m,
        c            = c,
        k            = k,
        rho          = rho,
        U            = U,
        D            = D,
        B            = B,
        dt           = dt,
        n_steps      = n_steps,
        device       = device,
        nd_inputs    = nd_inputs,
        use_ur_context = use_ur_context,
        ur_value     = Ur,
        ur_stats     = (ur_mean, ur_std),
        e_forcing    = e_forcing,
    )

    t    = result["time"] + t_handoff
    h    = result["displacement"]
    CL   = result["CL"]

    _case_df_sorted = case_df.sort_values("time")
    h_cfd_tail = _case_df_sorted["disp"].to_numpy(dtype=np.float32)
    h_cfd_tail = h_cfd_tail[handoff_idx : handoff_idx + n_steps]
    cl_cfd_tail = _case_df_sorted["cl"].to_numpy(dtype=np.float32)
    cl_cfd_tail = cl_cfd_tail[handoff_idx : handoff_idx + n_steps]
    _sub = model_subdir or f"gru_{model_dataset}"
    _nd_tag = "_nd" if nd_inputs else ""
    _noise_scale_tag = f"s{noise_scale:.6g}"
    _cfd_scale_tag = f"_scale{cfd_scale:g}"
    # The '_forc-v1_additive' and '_muNone' parts of the file name are kept so
    # new runs have the same names as the result files used in the thesis.
    _exp_tag = f"{_sub}_forc-v1_additive_noise-{noise_mode}{_nd_tag}{_cfd_scale_tag}_{_noise_scale_tag}_seed{noise_seed}_handoff_{handoff_offset_steps}"

    coordinate_mode = "nondimensional" if nd_inputs else "dimensional"
    git_commit = get_git_commit()
    git_dirty, worktree_patch_hash = get_git_dirty_and_patch_hash()
    if git_dirty:
        print(f"[provenance] [WARNING] worktree is DIRTY (uncommitted changes present). "
              f"For reproducible experiments, commit before running coupled inference. "
              f"worktree_patch_hash={worktree_patch_hash[:12] if worktree_patch_hash else 'n/a'}")

    npz_out = results_out_dir / f"coupled_{cfd_dataset}_Ur{Ur}_{_exp_tag}_muNone.npz"
    np.savez(npz_out, t=t, h=h, cl=CL, h_cfd=h_cfd_tail, cl_cfd=cl_cfd_tail, D=D, Ur=float(Ur),
             model_subdir=_sub, noise_mode=noise_mode,
             noise_scale=float(noise_scale), noise_seed=int(noise_seed),
             cl_det=result.get("CL_det"), e=result.get("e_forcing"),
             h_dot=result["velocity"], h_ddot=result["acceleration"],
             coordinate_mode=coordinate_mode, checkpoint=checkpoint,
             handoff_offset=int(handoff_offset_steps), git_commit=git_commit or "unknown",
             git_dirty=bool(git_dirty), worktree_patch_hash=worktree_patch_hash or "")
    print(f"Saved coupled trajectory -> {npz_out}")

    receipt = {
        "cfd_dataset": cfd_dataset,
        "Ur": float(Ur),
        "D": float(D),
        "noise_mode": noise_mode,
        "coordinate_mode": coordinate_mode,
        "model_subdir": _sub,
        "checkpoint": checkpoint,
        "handoff_offset": int(handoff_offset_steps),
        "handoff_time_s": float(t_handoff),
        "total_time_s": float(total_time),
        "git_commit": git_commit or "unknown",
        "git_dirty": bool(git_dirty),
        "worktree_patch_hash": worktree_patch_hash or "",
        "npz_path": str(npz_out),
    }
    receipt_out = npz_out.with_suffix(".receipt.json")
    with open(receipt_out, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"Saved receipt -> {receipt_out}")

    CFD_t = case_df["time"].values
    CFD_h = case_df["disp"].values
    CFD_cl = case_df["cl"].values


    fig, axes = plt.subplots(2, 1, figsize=(13, 6), constrained_layout=True)

    axes[0].plot(CFD_t, CFD_h / D, lw=0.8, color="black", alpha=0.7, label="CFD")
    axes[0].plot(t, h / D, lw=1, color="tab:blue", alpha=0.9, label=present_model_label(ds, "GRU coupled"))
    axes[0].axvline(t_handoff, color="green", ls="--", lw=1, alpha=0.7,
                    label=f"handoff t={t_handoff:.1f}s")
    axes[0].axhline(0, color="0.8", lw=1, ls=":")
    axes[0].set_ylabel(r"$h/D$")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(CFD_t, CFD_cl, lw=0.8, color="black", alpha=0.7, label="CFD")
    axes[1].plot(t, CL, lw=1, color="tab:orange", alpha=0.9, label=present_model_label(ds, "GRU coupled"))
    axes[1].axvline(t_handoff, color="green", ls="--", lw=1, alpha=0.7)
    axes[1].axhline(0, color="0.8", lw=1, ls=":")
    axes[1].set_ylabel("$C_L$")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)


    out_png = results_out_dir / f"coupled_viv_Ur{Ur}_{_exp_tag}_muNone.png"
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved coupled VIV plot to {out_png}")

    ss_start = int(0.7 * len(h))
    amp      = (h[ss_start:].max() - h[ss_start:].min()) / (2 * D)
    print(f"\nSteady-state A/D = {amp:.4f}")
    print(f"Displacement range: {h.min():.4f} to {h.max():.4f} m")
    print(f"CL range: {CL.min():.4f} to {CL.max():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ur",type=float, default=6.0)
    parser.add_argument("--total_time", type=float, default=300.0)
    parser.add_argument("--cfd_dataset", type=str, default="cylinder200",
                    help="Dataset key for loading CFD data")
    parser.add_argument("--model_dataset", type=str, default="cylinder200",
                    help="Dataset key for loading model artifacts")
    parser.add_argument("--checkpoint", type=str, default="gru_best.pt",
                    help="Checkpoint filename within model artifact_dir")
    parser.add_argument("--model_subdir", type=str, default=None,
                    help="Override: full subdir name like 'gru_cylinder200'")
    parser.add_argument("--handoff_offset", type=int, default=2000,
                    help="CFD steps past (release+seq_len) for handoff. "
                         "Default 2000 = existing sweep; vary for noise floor.")
    parser.add_argument("--cfd_scale", type=float, default=1.0,
                    help="Scale factor for physical CFD kinematics at handoff")

    coord_group = parser.add_mutually_exclusive_group()
    coord_group.add_argument("--nd_inputs", action="store_true",
                    help="Use non-dimensional physical inputs [h/D, hdot/U, hddot/(U^2/D)]")
    coord_group.add_argument("--dim_inputs", action="store_true",
                    help="Use dimensional (physical) inputs [h, hdot, hddot]; overrides artifact mode")

    parser.add_argument("--residual_npz", default=None,
                    help="npz with cl_true, cl_tf (the TF-residual you measured)")
    parser.add_argument("--noise_mode", default="none",
                    choices=["surrogate", "white", "none"])
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--noise_seed", type=int, default=0)
    parser.add_argument("--run_replay_diag", action="store_true",
                    help="Run Newmark replay diagnostic (oracle for force/timing validation). Safe to run on demand.")
    parser.add_argument("--replay_duration_s", type=positive_finite_float, default=None,
                    help="Duration in seconds of the --run_replay_diag comparison "
                         "window, starting at handoff. Default (omitted): the "
                         "original fixed 5000-step window. Only used when "
                         "--run_replay_diag is also passed.")
    parser.add_argument("--output_dir", type=str, default=None,
                    help="Where to write this run's own outputs (npz/png/receipt). "
                         "Relative paths are resolved under results/; absolute paths "
                         "are used verbatim. Default: results/ (unchanged behavior). "
                         "Does NOT affect where model artifacts are read from "
                         "(--model_subdir / --model_dataset, always under results/).")
    args = parser.parse_args()
    main(
        Ur=args.Ur,
        total_time=args.total_time,
        cfd_dataset=args.cfd_dataset,
        model_dataset=args.model_dataset,
        checkpoint=args.checkpoint,
        model_subdir=args.model_subdir,
        handoff_offset_steps=args.handoff_offset,
        cfd_scale=args.cfd_scale,
        nd_inputs=args.nd_inputs,
        residual_npz=args.residual_npz,
        noise_mode=args.noise_mode,
        noise_scale=args.noise_scale,
        noise_seed=args.noise_seed,
        run_replay_diag=args.run_replay_diag,
        replay_duration_s=args.replay_duration_s,
        output_dir=args.output_dir,
    )
