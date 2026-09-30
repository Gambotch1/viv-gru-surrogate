#!/usr/bin/env python3
"""Differentiable closed-loop building blocks for the rollout refinement (thesis Sec. 6.6.1).

The same loop as coupled_inference.run_coupled_viv, written in torch so that
gradients flow through GRU -> force -> Newmark -> next input window over
several steps. Used by train_rollout.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from viv_analysis.coupled_inference import Newmark_beta, warmup_history
from viv_analysis.utils import parse_ur_label


@dataclass
class ScalerConstants:
    """Means and scales of the fitted input/output scalers, as plain arrays."""
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float

    @classmethod
    def from_sklearn(cls, x_scaler, y_scaler) -> "ScalerConstants":
        return cls(
            x_mean=x_scaler.mean_.astype(np.float64),
            x_scale=x_scaler.scale_.astype(np.float64),
            y_mean=float(y_scaler.mean_[0]),
            y_scale=float(y_scaler.scale_[0]),
        )


def build_next_row_torch(
    h_i: torch.Tensor, hdot_i: torch.Tensor, hddot_i: torch.Tensor,
    input_cols: list[str], nd_inputs: bool, D: float, U: float,
    x_mean: np.ndarray, x_scale: np.ndarray,
    use_ur_context: bool, ur_scaled: float,
) -> torch.Tensor:
    """Next input row from the current state (model coordinates, scaled, optional Ur column)."""
    state_by_name = {"disp": h_i, "vel": hdot_i, "acc": hddot_i}
    kin = torch.stack([state_by_name[c] for c in input_cols], dim=-1)

    if nd_inputs:
        divisor_by_name = {"disp": D, "vel": U, "acc": (U * U) / D}
        divisors = torch.tensor([divisor_by_name[c] for c in input_cols],
                                dtype=kin.dtype, device=kin.device)
        kin = kin / divisors

    x_mean_t = torch.as_tensor(x_mean, dtype=kin.dtype, device=kin.device)
    x_scale_t = torch.as_tensor(x_scale, dtype=kin.dtype, device=kin.device)
    kin_scaled = (kin - x_mean_t) / x_scale_t

    if use_ur_context:
        ur_col = torch.full((kin_scaled.shape[0], 1), float(ur_scaled),
                            dtype=kin.dtype, device=kin.device)
        return torch.cat([kin_scaled, ur_col], dim=-1)
    return kin_scaled


def rollout_chunk(
    model, window: torch.Tensor,
    h_state: torch.Tensor, hdot_state: torch.Tensor, hddot_state: torch.Tensor,
    n_steps: int, dt: float, m: float, c: float, k: float, q: float, U: float, D: float,
    input_cols: list[str], nd_inputs: bool, sc: ScalerConstants,
    use_ur_context: bool, ur_scaled: float,
) -> dict:
    """Run the coupled loop for n_steps with gradients.

    q = 0.5 rho U^2 B, so F = q C_L. Returns the predicted C_L (scaled and
    physical), the displacement and velocity at each step, and the final window
    and state so a longer rollout can be continued.
    """
    win = window
    h_i, hdot_i, hddot_i = h_state, hdot_state, hddot_state
    cl_scaled_steps, cl_phys_steps, h_steps, hdot_steps = [], [], [], []

    for _ in range(n_steps):
        pred_scaled, _ = model(win)
        cl_phys = pred_scaled * sc.y_scale + sc.y_mean
        F = q * cl_phys

        h_next, hdot_next, hddot_next = Newmark_beta(
            F=F, h=h_i, h_dot=hdot_i, h_ddot=hddot_i, dt=dt, m=m, c=c, k=k,
        )

        new_row = build_next_row_torch(
            h_i, hdot_i, hddot_i, input_cols, nd_inputs, D, U,
            sc.x_mean, sc.x_scale, use_ur_context, ur_scaled,
        )
        win = torch.cat([win[:, 1:, :], new_row.unsqueeze(1)], dim=1)

        cl_scaled_steps.append(pred_scaled)
        cl_phys_steps.append(cl_phys)
        h_steps.append(h_i)
        hdot_steps.append(hdot_i)

        h_i, hdot_i, hddot_i = h_next, hdot_next, hddot_next

    return dict(
        cl_scaled=torch.stack(cl_scaled_steps, dim=1),
        cl_phys=torch.stack(cl_phys_steps, dim=1),
        h=torch.stack(h_steps, dim=1),
        hdot=torch.stack(hdot_steps, dim=1),
        window=win, h_state=h_i, hdot_state=hdot_i, hddot_state=hddot_i,
    )


def sample_batch_starts(
    case_df: pd.DataFrame, release_t: float, seq_len: int, max_future_steps: int,
    batch_size: int, rng: np.random.Generator,
) -> list[int]:
    """Random start indices for rollouts inside one case, spread evenly over the usable range after release."""
    ordered_times = case_df["time"].to_numpy(dtype=np.float64)
    release_idx = int(np.searchsorted(ordered_times, release_t))
    lo = release_idx + seq_len
    hi = len(ordered_times) - max_future_steps - 1
    if hi <= lo:
        raise ValueError(
            f"Case too short for {max_future_steps}-step rollout starting "
            f"after release+seq_len: usable range [{lo},{hi}]."
        )
    edges = np.linspace(lo, hi, batch_size + 1)
    starts = [int(rng.integers(int(edges[i]), int(edges[i + 1]) + 1))
              for i in range(batch_size)]
    return starts


def build_batch_from_case(
    case_df: pd.DataFrame, case_name: str, starts: list[int], seq_len: int,
    input_cols: list[str], x_scaler, nd_inputs: bool, D: float, fn: float,
    use_ur_context: bool, ur_stats: tuple[float, float],
    max_future_steps: int, device: str,
) -> dict:
    """Initial windows, states and CFD targets for a batch of rollouts from one case."""
    ur_value = parse_ur_label(str(case_name))
    U = ur_value * fn * D
    ordered = case_df.sort_values("time").reset_index(drop=True)
    times = ordered["time"].to_numpy(dtype=np.float64)

    histories, h0s, hdot0s, hddot0s = [], [], [], []
    cfd_h, cfd_hdot, cfd_cl = [], [], []
    for start_idx in starts:
        release_t_eq = times[start_idx - seq_len]
        hist, init_state, _, handoff_idx = warmup_history(
            ordered, release_t_eq, seq_len, input_cols, x_scaler,
            nd_inputs=nd_inputs, D=D, U=U, use_ur_context=use_ur_context,
            ur_value=ur_value, ur_stats=ur_stats, handoff_offset_steps=0,
        )
        assert handoff_idx == start_idx, (handoff_idx, start_idx)
        histories.append(hist)
        h0s.append(init_state["h"]); hdot0s.append(init_state["h_dot"])
        hddot0s.append(init_state["h_ddot"])
        sl = slice(start_idx, start_idx + max_future_steps)
        cfd_h.append(ordered["disp"].to_numpy(dtype=np.float64)[sl])
        cfd_hdot.append(ordered["vel"].to_numpy(dtype=np.float64)[sl])
        cfd_cl.append(ordered["cl"].to_numpy(dtype=np.float64)[sl])

    dt = float(np.median(np.diff(times)))
    ur_mean, ur_std = ur_stats
    ur_std_safe = float(ur_std) if abs(float(ur_std)) > 0 else 1.0
    ur_scaled = (ur_value - float(ur_mean)) / ur_std_safe

    return dict(
        window=torch.tensor(np.stack(histories), dtype=torch.float32, device=device),
        h_state=torch.tensor(h0s, dtype=torch.float32, device=device),
        hdot_state=torch.tensor(hdot0s, dtype=torch.float32, device=device),
        hddot_state=torch.tensor(hddot0s, dtype=torch.float32, device=device),
        cfd_h=torch.tensor(np.stack(cfd_h), dtype=torch.float32, device=device),
        cfd_hdot=torch.tensor(np.stack(cfd_hdot), dtype=torch.float32, device=device),
        cfd_cl=torch.tensor(np.stack(cfd_cl), dtype=torch.float32, device=device),
        dt=dt, U=U, ur_scaled=ur_scaled, ur_value=ur_value,
    )


def build_tf_batch_from_case(
    case_df: pd.DataFrame, case_name: str, start_idx: int, n_steps: int, seq_len: int,
    input_cols: list[str], x_scaler, nd_inputs: bool, D: float, fn: float,
    use_ur_context: bool, ur_stats: tuple[float, float], device: str,
) -> dict:
    """Teacher-forced input windows and C_L targets for n_steps consecutive steps."""
    from numpy.lib.stride_tricks import sliding_window_view

    ur_value = parse_ur_label(str(case_name))
    U = ur_value * fn * D
    ordered = case_df.sort_values("time").reset_index(drop=True)

    kin_phys = ordered[input_cols].to_numpy(dtype=np.float64).copy()
    if nd_inputs:
        divisor = {"disp": D, "vel": U}
        for k, col in enumerate(input_cols):
            kin_phys[:, k] = kin_phys[:, k] / divisor[col]
    x_mean = x_scaler.mean_.astype(np.float64)
    x_scale = x_scaler.scale_.astype(np.float64)
    kin_scaled = (kin_phys - x_mean) / x_scale

    if use_ur_context:
        ur_mean, ur_std = ur_stats
        ur_std_safe = float(ur_std) if abs(float(ur_std)) > 0 else 1.0
        ur_scaled = (ur_value - float(ur_mean)) / ur_std_safe
        ur_col = np.full((kin_scaled.shape[0], 1), ur_scaled, dtype=np.float64)
        signal = np.hstack([kin_scaled, ur_col])
    else:
        signal = kin_scaled

    all_windows = sliding_window_view(signal, window_shape=seq_len, axis=0)
    all_windows = np.transpose(all_windows, (0, 2, 1))
    tf_windows = all_windows[start_idx - seq_len: start_idx - seq_len + n_steps]
    if tf_windows.shape[0] != n_steps:
        raise ValueError(f"TF batch out of bounds: got {tf_windows.shape[0]} windows, "
                        f"expected {n_steps} (start_idx={start_idx}, case len={len(ordered)}).")

    cl_true = ordered["cl"].to_numpy(dtype=np.float64)[start_idx: start_idx + n_steps]

    return dict(
        x=torch.tensor(np.ascontiguousarray(tf_windows), dtype=torch.float32, device=device),
        cl_cfd=torch.tensor(cl_true, dtype=torch.float32, device=device),
    )


def loss_cl(cl_scaled_pred: torch.Tensor, cl_cfd_phys: torch.Tensor,
           y_mean: float, y_scale: float) -> torch.Tensor:
    """Mean squared error of C_L in scaled units."""
    cl_cfd_scaled = (cl_cfd_phys - y_mean) / y_scale

    return torch.mean((cl_scaled_pred - cl_cfd_scaled) ** 2)


def loss_roll(h_pred: torch.Tensor, hdot_pred: torch.Tensor,
             h_cfd: torch.Tensor, hdot_cfd: torch.Tensor,
             D: float, U: float, x_mean: np.ndarray, x_scale: np.ndarray,
             disp_idx: int, vel_idx: int) -> torch.Tensor:
    """Mean squared error of displacement and velocity along the rollout, in standardised model coordinates."""
    h_pred_std = (h_pred / D - x_mean[disp_idx]) / x_scale[disp_idx]
    h_cfd_std = (h_cfd / D - x_mean[disp_idx]) / x_scale[disp_idx]
    hdot_pred_std = (hdot_pred / U - x_mean[vel_idx]) / x_scale[vel_idx]
    hdot_cfd_std = (hdot_cfd / U - x_mean[vel_idx]) / x_scale[vel_idx]

    return torch.mean((h_pred_std - h_cfd_std) ** 2) + torch.mean((hdot_pred_std - hdot_cfd_std) ** 2)
