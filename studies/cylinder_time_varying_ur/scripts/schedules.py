"""Reduced-velocity schedules Ur(t) for the continuous sweep: ascending, ascending with cosine ramps, and triangular 2 -> 12 -> 2 (thesis Sec. 5.7, Appendix F)."""

from __future__ import annotations

import numpy as np

CYLINDER_UR_LIST = [
    2.00, 2.50, 3.00, 3.50, 4.00,
    4.25, 4.50, 4.75, 5.00, 5.25, 5.50,
    5.75, 6.00, 6.25, 6.50,
    7.00, 8.00, 9.00, 10.00, 11.00, 12.00,
]


def build_ascending_schedule(Ur_list: list[float], dwell_s: float, dt: float) -> dict:
    dwell_steps = int(round(dwell_s / dt))
    if abs(dwell_steps * dt - dwell_s) > 1e-9:
        raise ValueError(f"dwell_s={dwell_s} is not an exact multiple of dt={dt}")

    Ur_schedule = np.concatenate([np.full(dwell_steps, ur, dtype=np.float64) for ur in Ur_list])
    transition_step_indices = [i * dwell_steps for i in range(1, len(Ur_list))]
    transition_times = [idx * dt for idx in transition_step_indices]
    return {
        "kind": "ascending_instantaneous",
        "Ur_list": list(Ur_list),
        "dwell_s": dwell_s,
        "dwell_steps": dwell_steps,
        "transition_duration_convention": "instantaneous_zero_added_time",
        "dt": dt,
        "n_steps": len(Ur_schedule),
        "Ur_schedule": Ur_schedule,
        "transition_step_indices": transition_step_indices,
        "transition_times": transition_times,
        "transition_ur_values": list(Ur_list[1:]),
    }


def build_ascending_cosine_schedule(Ur_list: list[float], dwell_s: float,
                                     transition_s: float, dt: float) -> dict:
    dwell_steps = int(round(dwell_s / dt))
    transition_steps = int(round(transition_s / dt))
    if abs(dwell_steps * dt - dwell_s) > 1e-9:
        raise ValueError(f"dwell_s={dwell_s} is not an exact multiple of dt={dt}")
    if abs(transition_steps * dt - transition_s) > 1e-9:
        raise ValueError(f"transition_s={transition_s} is not an exact multiple of dt={dt}")

    segments = [np.full(dwell_steps, Ur_list[0], dtype=np.float64)]
    transition_step_indices = []
    transition_times = []
    cursor = dwell_steps
    for prev_ur, next_ur in zip(Ur_list[:-1], Ur_list[1:]):
        tau = np.arange(1, transition_steps + 1, dtype=np.float64) / transition_steps
        ramp = prev_ur + (next_ur - prev_ur) * 0.5 * (1.0 - np.cos(np.pi * tau))
        segments.append(ramp)
        transition_step_indices.append(cursor)
        transition_times.append(cursor * dt)
        cursor += transition_steps
        segments.append(np.full(dwell_steps, next_ur, dtype=np.float64))
        cursor += dwell_steps

    Ur_schedule = np.concatenate(segments)
    return {
        "kind": "ascending_cosine",
        "Ur_list": list(Ur_list),
        "dwell_s": dwell_s,
        "dwell_steps": dwell_steps,
        "transition_s": transition_s,
        "transition_steps": transition_steps,
        "transition_duration_convention": "added_to_dwell",
        "dt": dt,
        "n_steps": len(Ur_schedule),
        "Ur_schedule": Ur_schedule,
        "transition_step_indices": transition_step_indices,
        "transition_times": transition_times,
        "transition_ur_values": list(Ur_list[1:]),
    }


def build_triangular_schedule(Ur_list: list[float], dwell_s: float, dt: float,
                               transition: str = "instantaneous",
                               transition_s: float | None = None) -> dict:
    full_list = list(Ur_list) + list(reversed(Ur_list[:-1]))
    if transition == "instantaneous":
        sched = build_ascending_schedule(full_list, dwell_s, dt)
    elif transition == "cosine":
        if transition_s is None:
            raise ValueError("transition_s required for transition='cosine'")
        sched = build_ascending_cosine_schedule(full_list, dwell_s, transition_s, dt)
    else:
        raise ValueError(f"unknown transition kind: {transition}")
    sched["kind"] = f"triangular_{transition}"
    sched["reversal_index_in_Ur_list"] = len(Ur_list) - 1
    sched["reversal_ur_value"] = Ur_list[-1]
    return sched
