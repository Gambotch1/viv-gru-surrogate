"""Short check of the time-varying pipeline on a development checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "scripts"))

import run_time_varying_sweep as rtvs  # noqa: E402
import schedules  # noqa: E402
from time_varying_coupled import run_coupled_viv_time_varying_ur  # noqa: E402
from viv_analysis.config import config, cylinder200_structural_params  # noqa: E402

SMOKE_UR_LIST = [2.0, 2.5]
SMOKE_DWELL_S = 10.0
SMOKE_TRANSITION_S = 5.0


def _run(schedule: dict, artifacts, warmup, sp, dt, D, fn, rho, device):
    return run_coupled_viv_time_varying_ur(
        model=artifacts["model"], x_scaler=artifacts["x_scaler"],
        y_scaler=artifacts["y_scaler"], seq_len=artifacts["seq_len"],
        input_cols=artifacts["input_cols"],
        initial_history=warmup["initial_history"].copy(), initial_state=dict(warmup["initial_state"]),
        m=sp["m"], c=sp["c"], k=sp["k"], rho=rho, D=D, fn=fn,
        Ur_schedule=schedule["Ur_schedule"], n_steps=schedule["n_steps"],
        dt=dt, use_ur_context=artifacts["use_ur_context"],
        ur_stats=(artifacts["ur_mean"], artifacts["ur_std"]),
        device=device, nd_inputs=artifacts["nd_inputs"],
        track_history_provenance=True,
    )


def report_for(label: str, out: dict, schedule: dict, out_const: dict,
                artifacts, warmup, sp, dt, D, fn, rho, device) -> dict:
    tr = schedule["transition_step_indices"][0]
    dwell_steps = schedule["dwell_steps"]

    h_step_change = float(np.max(np.abs(np.diff(out["displacement"])[max(0, tr - 3):tr + 3])))
    hdot_step_change = float(np.max(np.abs(np.diff(out["velocity"])[max(0, tr - 3):tr + 3])))

    n_compare = min(dwell_steps, len(out_const["displacement"]))
    const_regression_err_h = float(np.max(np.abs(
        out["displacement"][:n_compare] - out_const["displacement"][:n_compare])))
    const_regression_err_cl = float(np.max(np.abs(
        out["CL"][:n_compare] - out_const["CL"][:n_compare])))

    sched_before = dict(schedule, Ur_schedule=schedule["Ur_schedule"][:tr], n_steps=tr)
    sched_after = dict(schedule, Ur_schedule=schedule["Ur_schedule"][:tr + 1], n_steps=tr + 1)
    out_before = _run(sched_before, artifacts, warmup, sp, dt, D, fn, rho, device)
    out_after = _run(sched_after, artifacts, warmup, sp, dt, D, fn, rho, device)
    model_input_before = out_before["final_history"][-1].tolist()
    model_input_after = out_after["final_history"][-1].tolist()

    U_before, U_after = out["U"][max(0, tr - 1)], out["U"][min(tr, schedule["n_steps"] - 1)]
    force_scale_ratio = float((U_after / U_before) ** 2)

    all_finite = bool(np.isfinite(out["displacement"]).all()
                       and np.isfinite(out["velocity"]).all()
                       and np.isfinite(out["CL"]).all())

    return {
        "label": label,
        "max_h_adjacent_step_change_near_transition": h_step_change,
        "max_hdot_adjacent_step_change_near_transition": hdot_step_change,
        "constant_ur_regression_error_h": const_regression_err_h,
        "constant_ur_regression_error_cl": const_regression_err_cl,
        "model_input_row_immediately_before_transition": model_input_before,
        "model_input_row_immediately_after_transition": model_input_after,
        "model_input_columns": artifacts["input_cols"] + (["Ur_context"] if artifacts["use_ur_context"] else []),
        "U_before_transition": float(U_before),
        "U_after_transition": float(U_after),
        "force_scale_ratio_U_squared": force_scale_ratio,
        "expected_force_scale_ratio": float((SMOKE_UR_LIST[1] / SMOKE_UR_LIST[0]) ** 2),
        "n_ood_warnings": out["n_ood_warnings"],
        "max_abs_scaled_kinematics": out["max_abs_scaled_kinematics"],
        "all_finite": all_finite,
    }


def main():
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    D = config["cylinder200_D_ref"]
    fn = config["cylinder200_fn"]
    dt = config["cylinder200_dt"]
    rho = config["cylinder200_rho"]
    sp = cylinder200_structural_params()

    print("Loading model + performing CFD warm-up (once)...")
    artifacts = rtvs.load_model_and_artifacts(device)
    warmup = rtvs.warmup_once(artifacts, dt=dt, D=D, fn=fn)

    sched_inst = schedules.build_ascending_schedule(SMOKE_UR_LIST, SMOKE_DWELL_S, dt)
    sched_cos = schedules.build_ascending_cosine_schedule(
        SMOKE_UR_LIST, SMOKE_DWELL_S, SMOKE_TRANSITION_S, dt)
    sched_const = schedules.build_ascending_schedule([SMOKE_UR_LIST[0]], SMOKE_DWELL_S, dt)

    print(f"Running constant-Ur={SMOKE_UR_LIST[0]} baseline ({sched_const['n_steps']} steps)...")
    out_const = _run(sched_const, artifacts, warmup, sp, dt, D, fn, rho, device)

    print(f"Running instantaneous-transition schedule ({sched_inst['n_steps']} steps)...")
    out_inst = _run(sched_inst, artifacts, warmup, sp, dt, D, fn, rho, device)

    print(f"Running cosine-transition schedule ({sched_cos['n_steps']} steps)...")
    out_cos = _run(sched_cos, artifacts, warmup, sp, dt, D, fn, rho, device)

    common = (artifacts, warmup, sp, dt, D, fn, rho, device)
    report = {
        "instantaneous": report_for("instantaneous", out_inst, sched_inst, out_const, *common),
        "cosine": report_for("cosine", out_cos, sched_cos, out_const, *common),
        "warmup_handoff_time_s": warmup["t_handoff"],
        "warmup_case": warmup["warmup_case"],
    }

    results_dir = STUDY_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez(results_dir / "smoke_instantaneous.npz",
              **{k: v for k, v in out_inst.items() if isinstance(v, np.ndarray)})
    np.savez(results_dir / "smoke_cosine.npz",
              **{k: v for k, v in out_cos.items() if isinstance(v, np.ndarray)})
    np.savez(results_dir / "smoke_constant_baseline.npz",
              **{k: v for k, v in out_const.items() if isinstance(v, np.ndarray)})

    import json
    with open(results_dir / "smoke_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("SMOKE TEST REPORT")
    print("=" * 70)
    for kind in ("instantaneous", "cosine"):
        r = report[kind]
        print(f"\n--- {kind} ---")
        for k, v in r.items():
            print(f"  {k}: {v}")
    print("=" * 70)
    return report


if __name__ == "__main__":
    main()
