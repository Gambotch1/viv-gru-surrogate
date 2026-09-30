"""The coupled GRU-Newmark loop with Ur changing in time. For a constant schedule it gives the same result as coupled_inference.run_coupled_viv (tested)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

import sys
_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viv_analysis.coupled_inference import Newmark_beta  # noqa: E402

CYLINDER200_RE = 200.0


def run_coupled_viv_time_varying_ur(
    model,
    x_scaler,
    y_scaler,
    seq_len: int,
    input_cols: list[str],
    initial_history: np.ndarray,
    initial_state: dict,
    m: float,
    c: float,
    k: float,
    rho: float,
    D: float,
    fn: float,
    Ur_schedule: np.ndarray,
    n_steps: int,
    B: Optional[float] = None,
    dt: Optional[float] = None,
    use_ur_context: bool = False,
    ur_stats: tuple = (0.0, 1.0),
    device: str = "cpu",
    nd_inputs: bool = False,
    re: float = CYLINDER200_RE,
    track_history_provenance: bool = False,
) -> dict:
    model.eval()
    B = D if B is None else B
    if dt is None:
        raise ValueError("dt must be provided")
    Ur_schedule = np.asarray(Ur_schedule, dtype=np.float64)
    if len(Ur_schedule) != n_steps:
        raise ValueError(f"Ur_schedule length {len(Ur_schedule)} != n_steps {n_steps}")
    if "acc" in input_cols:
        raise ValueError(
            "input_cols must not contain 'acc' -- this study's GRU inputs "
            "are restricted to displacement/velocity (+ Ur context) by "
            "explicit requirement.")

    h = np.zeros(n_steps + 1, dtype=np.float32)
    h_dot = np.zeros(n_steps + 1, dtype=np.float32)
    h_ddot = np.zeros(n_steps + 1, dtype=np.float32)
    CL = np.zeros(n_steps + 1, dtype=np.float32)
    U_arr = np.zeros(n_steps, dtype=np.float64)
    mu_arr = np.zeros(n_steps, dtype=np.float64)
    F_L_arr = np.zeros(n_steps, dtype=np.float32)

    h[0] = initial_state["h"]
    h_dot[0] = initial_state["h_dot"]
    h_ddot[0] = initial_state["h_ddot"]
    history = initial_history.copy()

    expected_features = len(input_cols) + (1 if use_ur_context else 0)
    if history.ndim != 2 or history.shape[1] != expected_features:
        raise ValueError(
            f"History feature mismatch: got {history.shape}, expected "
            f"(*, {expected_features}).")

    ur_mean, ur_std = ur_stats
    ur_std_safe = float(ur_std) if abs(float(ur_std)) > 0 else 1.0

    x_mean = x_scaler.mean_.astype(np.float32)
    x_scale = x_scaler.scale_.astype(np.float32)
    y_mean = float(y_scaler.mean_[0])
    y_scale = float(y_scaler.scale_[0])
    _state_names = ("disp", "vel", "acc")

    if track_history_provenance:
        hist_U_prov = np.full(history.shape[0], np.nan, dtype=np.float64)
        hist_Ur_prov = np.full(history.shape[0], np.nan, dtype=np.float64)
    else:
        hist_U_prov = hist_Ur_prov = None

    max_abs_z_seen = 0.0
    n_ood_warnings = 0

    with torch.inference_mode():
        for i in range(n_steps):
            Ur_i = float(Ur_schedule[i])
            U_i = Ur_i * fn * D
            mu_i = rho * U_i * D / re
            U_arr[i] = U_i
            mu_arr[i] = mu_i

            x = torch.from_numpy(history.astype(np.float32)).unsqueeze(0).to(device)
            cl_scaled, _hn = model(x)
            cl = float(cl_scaled.item()) * y_scale + y_mean
            CL[i] = cl

            F_aero = 0.5 * rho * U_i**2 * B * cl
            F_L_arr[i] = F_aero

            h[i + 1], h_dot[i + 1], h_ddot[i + 1] = Newmark_beta(
                F=F_aero, h=h[i], h_dot=h_dot[i], h_ddot=h_ddot[i],
                dt=dt, m=m, c=c, k=k,
            )

            if not np.isfinite([h[i + 1], h_dot[i + 1], h_ddot[i + 1], cl]).all():
                raise FloatingPointError(f"Non-finite state at step={i}")

            _nd_divisor_by_name = (
                {"disp": D, "vel": U_i, "acc": (U_i * U_i) / D} if nd_inputs else None
            )
            _state_by_name = dict(zip(_state_names, (h[i], h_dot[i], h_ddot[i])))
            if nd_inputs:
                new_kinematics_t = np.array(
                    [_state_by_name[c] / _nd_divisor_by_name[c] for c in input_cols],
                    dtype=np.float32)
            else:
                new_kinematics_t = np.array(
                    [_state_by_name[c] for c in input_cols], dtype=np.float32)
            new_kinematics_scaled = (new_kinematics_t - x_mean) / x_scale

            ur_scaled_i = (Ur_i - float(ur_mean)) / ur_std_safe
            if use_ur_context:
                new_row = np.append(new_kinematics_scaled, ur_scaled_i).astype(np.float32)
            else:
                new_row = new_kinematics_scaled.astype(np.float32)

            zmax = float(np.max(np.abs(new_kinematics_scaled)))
            max_abs_z_seen = max(max_abs_z_seen, zmax)
            if zmax > 6.0 and n_ood_warnings < 10:
                n_ood_warnings += 1
                print(f"WARNING OOD step={i}: max|z|={zmax:.2f}")

            history = np.roll(history, -1, axis=0)
            history[-1] = new_row
            if track_history_provenance:
                hist_U_prov = np.roll(hist_U_prov, -1)
                hist_Ur_prov = np.roll(hist_Ur_prov, -1)
                hist_U_prov[-1] = U_i
                hist_Ur_prov[-1] = Ur_i

    t = np.arange(n_steps + 1) * dt

    power = F_L_arr * h_dot[:n_steps]
    cumulative_work = np.concatenate([[0.0], np.cumsum(0.5 * (power[:-1] + power[1:]) * dt)]) \
        if n_steps > 1 else np.zeros(n_steps)

    result = {
        "time": t[:n_steps],
        "Ur": Ur_schedule.copy(),
        "U": U_arr,
        "mu": mu_arr,
        "displacement": h[:n_steps],
        "velocity": h_dot[:n_steps],
        "acceleration": h_ddot[:n_steps],
        "CL": CL[:n_steps],
        "F_L": F_L_arr,
        "aero_power": power,
        "cumulative_work": cumulative_work,
        "max_abs_scaled_kinematics": float(max_abs_z_seen),
        "n_ood_warnings": int(n_ood_warnings),
        "final_history": history,
        "final_state": {"h": float(h[n_steps]), "h_dot": float(h_dot[n_steps]),
                         "h_ddot": float(h_ddot[n_steps])},
    }
    if track_history_provenance:
        result["history_U_provenance"] = hist_U_prov
        result["history_Ur_provenance"] = hist_Ur_prov
    return result
