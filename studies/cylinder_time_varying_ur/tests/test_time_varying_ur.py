from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
SRC_DIR = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC_DIR))

from viv_analysis.models.gru import VIV_GRU  # noqa: E402
from viv_analysis.coupled_inference import run_coupled_viv, Newmark_beta  # noqa: E402
from viv_analysis.config import cylinder200_structural_params, config as viv_config  # noqa: E402

import time_varying_coupled as tvc  # noqa: E402
import schedules  # noqa: E402

D = 0.2
FN = 0.2
DT = 0.005
SEQ_LEN = 10
INPUT_COLS = ["disp", "vel"]
RHO = 1.0
SP = cylinder200_structural_params()


def _fixed_model():
    torch = __import__("torch")
    torch.manual_seed(0)
    return VIV_GRU(input_size=3, hidden_size=8, num_layers=1, dropout=0.0)


def _fixed_scalers():
    rng = np.random.RandomState(0)
    x = rng.randn(200, 2) * np.array([0.2, 0.05]) + np.array([0.0, 0.0])
    y = rng.randn(200, 1) * 0.3
    x_scaler = StandardScaler().fit(x)
    y_scaler = StandardScaler().fit(y)
    return x_scaler, y_scaler


def _dummy_artifact_paths() -> dict:
    self_path = Path(__file__)
    return {"checkpoint_path": self_path, "x_scaler_path": self_path,
            "y_scaler_path": self_path}


def _initial_history_and_state(use_ur_context=True, ur_scaled=0.0):
    rng = np.random.RandomState(1)
    n_feat = len(INPUT_COLS) + (1 if use_ur_context else 0)
    history = rng.randn(SEQ_LEN, n_feat).astype(np.float32) * 0.1
    if use_ur_context:
        history[:, -1] = ur_scaled
    initial_state = {"h": 0.01, "h_dot": 0.001, "h_ddot": 0.0}
    return history, initial_state


def test_constant_ur_matches_fixed_ur_run_coupled_viv():
    model = _fixed_model()
    x_scaler, y_scaler = _fixed_scalers()
    ur_stats = (5.0, 2.0)
    Ur_const = 5.5
    n_steps = 300
    U_const = Ur_const * FN * D
    ur_scaled = (Ur_const - ur_stats[0]) / ur_stats[1]
    history, state = _initial_history_and_state(ur_scaled=ur_scaled)

    fixed = run_coupled_viv(
        model=model, x_scaler=x_scaler, y_scaler=y_scaler, seq_len=SEQ_LEN,
        input_cols=INPUT_COLS, initial_history=history.copy(), initial_state=dict(state),
        m=SP["m"], c=SP["c"], k=SP["k"], rho=RHO, U=U_const, D=D, n_steps=n_steps,
        dt=DT, use_ur_context=True, ur_value=Ur_const, ur_stats=ur_stats,
        nd_inputs=True,
    )

    schedule = np.full(n_steps, Ur_const, dtype=np.float64)
    varying = tvc.run_coupled_viv_time_varying_ur(
        model=model, x_scaler=x_scaler, y_scaler=y_scaler, seq_len=SEQ_LEN,
        input_cols=INPUT_COLS, initial_history=history.copy(), initial_state=dict(state),
        m=SP["m"], c=SP["c"], k=SP["k"], rho=RHO, D=D, fn=FN,
        Ur_schedule=schedule, n_steps=n_steps, dt=DT, use_ur_context=True,
        ur_stats=ur_stats, nd_inputs=True,
    )

    np.testing.assert_allclose(varying["displacement"], fixed["displacement"], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(varying["velocity"], fixed["velocity"], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(varying["CL"], fixed["CL"], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(varying["U"], np.full(n_steps, U_const), rtol=1e-9)


def test_h_and_hdot_continuous_across_transition():
    model = _fixed_model()
    x_scaler, y_scaler = _fixed_scalers()
    ur_stats = (5.0, 2.0)
    history, state = _initial_history_and_state(ur_scaled=(2.0 - 5.0) / 2.0)
    sched = schedules.build_ascending_schedule([2.0, 2.5], dwell_s=50 * DT, dt=DT)

    out = tvc.run_coupled_viv_time_varying_ur(
        model=model, x_scaler=x_scaler, y_scaler=y_scaler, seq_len=SEQ_LEN,
        input_cols=INPUT_COLS, initial_history=history, initial_state=state,
        m=SP["m"], c=SP["c"], k=SP["k"], rho=RHO, D=D, fn=FN,
        Ur_schedule=sched["Ur_schedule"], n_steps=sched["n_steps"], dt=DT,
        use_ur_context=True, ur_stats=ur_stats, nd_inputs=True,
    )

    tr = sched["transition_step_indices"][0]
    h_step_diffs = np.abs(np.diff(out["displacement"]))
    hdot_step_diffs = np.abs(np.diff(out["velocity"]))
    local_h_scale = np.median(h_step_diffs[max(0, tr - 5):tr + 5]) + 1e-12
    local_hdot_scale = np.median(hdot_step_diffs[max(0, tr - 5):tr + 5]) + 1e-12
    assert h_step_diffs[tr] < 20 * local_h_scale
    assert hdot_step_diffs[tr] < 20 * local_hdot_scale


def test_force_scaling_uses_current_U():
    model = _fixed_model()
    x_scaler, y_scaler = _fixed_scalers()
    ur_stats = (5.0, 2.0)
    history, state = _initial_history_and_state(ur_scaled=(2.0 - 5.0) / 2.0)
    sched = schedules.build_ascending_schedule([2.0, 2.5], dwell_s=20 * DT, dt=DT)
    B = D

    out = tvc.run_coupled_viv_time_varying_ur(
        model=model, x_scaler=x_scaler, y_scaler=y_scaler, seq_len=SEQ_LEN,
        input_cols=INPUT_COLS, initial_history=history, initial_state=state,
        m=SP["m"], c=SP["c"], k=SP["k"], rho=RHO, D=D, fn=FN,
        Ur_schedule=sched["Ur_schedule"], n_steps=sched["n_steps"], dt=DT,
        use_ur_context=True, ur_stats=ur_stats, nd_inputs=True,
    )

    expected_U = sched["Ur_schedule"] * FN * D
    np.testing.assert_allclose(out["U"], expected_U, rtol=1e-10)
    expected_F = 0.5 * RHO * out["U"]**2 * B * out["CL"]
    np.testing.assert_allclose(out["F_L"], expected_F, rtol=1e-5)
    expected_mu = RHO * out["U"] * D / tvc.CYLINDER200_RE
    np.testing.assert_allclose(out["mu"], expected_mu, rtol=1e-10)
    tr = sched["transition_step_indices"][0]
    assert out["U"][tr] != out["U"][tr - 1]
    assert not np.isclose(out["U"][tr], out["U"][tr - 1])


def test_history_rows_retain_original_U_and_Ur():
    model = _fixed_model()
    x_scaler, y_scaler = _fixed_scalers()
    ur_stats = (5.0, 2.0)
    history, state = _initial_history_and_state(ur_scaled=(2.0 - 5.0) / 2.0)
    sched = schedules.build_ascending_schedule([2.0, 2.5, 3.0], dwell_s=15 * DT, dt=DT)

    tr = sched["transition_step_indices"][0]

    out_before = tvc.run_coupled_viv_time_varying_ur(
        model=model, x_scaler=x_scaler, y_scaler=y_scaler, seq_len=SEQ_LEN,
        input_cols=INPUT_COLS, initial_history=history.copy(), initial_state=dict(state),
        m=SP["m"], c=SP["c"], k=SP["k"], rho=RHO, D=D, fn=FN,
        Ur_schedule=sched["Ur_schedule"][:tr], n_steps=tr, dt=DT,
        use_ur_context=True, ur_stats=ur_stats, nd_inputs=True,
        track_history_provenance=True,
    )
    out_after = tvc.run_coupled_viv_time_varying_ur(
        model=model, x_scaler=x_scaler, y_scaler=y_scaler, seq_len=SEQ_LEN,
        input_cols=INPUT_COLS, initial_history=history.copy(), initial_state=dict(state),
        m=SP["m"], c=SP["c"], k=SP["k"], rho=RHO, D=D, fn=FN,
        Ur_schedule=sched["Ur_schedule"][:tr + 1], n_steps=tr + 1, dt=DT,
        use_ur_context=True, ur_stats=ur_stats, nd_inputs=True,
        track_history_provenance=True,
    )

    window_before = out_before["history_Ur_provenance"]
    window_after = out_after["history_Ur_provenance"]

    assert np.all(np.isclose(window_before, 2.0)), (
        "window is entirely pre-transition (Ur=2.0) rows before the transition step")
    assert window_after[-1] == pytest.approx(2.5), (
        "the single newly-appended row at the transition step must use the NEW Ur")
    assert np.all(np.isclose(window_after[:-1], 2.0)), (
        "a pre-transition history row's stored Ur_j was overwritten with the new Ur")


def test_acceleration_absent_from_gru_input():
    model = _fixed_model()
    x_scaler, y_scaler = _fixed_scalers()
    history, state = _initial_history_and_state(ur_scaled=0.0)
    sched = schedules.build_ascending_schedule([2.0], dwell_s=10 * DT, dt=DT)

    with pytest.raises(ValueError, match="acc"):
        tvc.run_coupled_viv_time_varying_ur(
            model=model, x_scaler=x_scaler, y_scaler=y_scaler, seq_len=SEQ_LEN,
            input_cols=["disp", "vel", "acc"], initial_history=history, initial_state=state,
            m=SP["m"], c=SP["c"], k=SP["k"], rho=RHO, D=D, fn=FN,
            Ur_schedule=sched["Ur_schedule"], n_steps=sched["n_steps"], dt=DT,
            use_ur_context=True, ur_stats=(5.0, 2.0), nd_inputs=True,
        )
    assert INPUT_COLS == ["disp", "vel"]


def test_cfd_warmup_called_exactly_once(monkeypatch, tmp_path):
    import run_time_varying_sweep as rtvs

    n = 50000
    t = np.arange(n) * DT
    df = pd.DataFrame({
        "case": "Ur2", "time": t, "step": np.arange(n),
        "disp": 0.01 * np.sin(0.2 * t), "vel": 0.01 * 0.2 * np.cos(0.2 * t),
        "acc": -0.01 * 0.2**2 * np.sin(0.2 * t), "cl": 0.1 * np.sin(0.2 * t),
    })
    monkeypatch.setattr(rtvs, "merge_dataframes", lambda **kw: df.copy())

    call_count = {"n": 0}
    real_compute_kinematics = rtvs.compute_kinematics

    def counting_compute_kinematics(raw_df, **kw):
        call_count["n"] += 1
        return raw_df

    monkeypatch.setattr(rtvs, "compute_kinematics", counting_compute_kinematics)

    model = _fixed_model()
    x_scaler, y_scaler = _fixed_scalers()
    artifacts = dict(
        model=model, x_scaler=x_scaler, y_scaler=y_scaler,
        nd_inputs=True, use_ur_context=True, ur_mean=5.0, ur_std=2.0,
        seq_len=SEQ_LEN, input_cols=INPUT_COLS,
        **_dummy_artifact_paths(),
    )
    rtvs.warmup_once(artifacts, dt=DT, D=D, fn=FN)
    assert call_count["n"] == 1

    monkeypatch.setattr(rtvs, "load_model_and_artifacts", lambda device, artifact_dir=None: artifacts)
    sched = schedules.build_ascending_schedule([2.0, 2.5, 3.0], dwell_s=10 * DT, dt=DT)
    call_count["n"] = 0
    rtvs.run_sweep(sched, device="cpu")
    assert call_count["n"] == 1, (
        f"compute_kinematics (part of the CFD warm-up path) was called "
        f"{call_count['n']} times for a 3-Ur schedule -- expected exactly 1")


def test_production_newmark_implementation_reused():
    import inspect
    src = inspect.getsource(tvc)
    assert "from viv_analysis.coupled_inference import Newmark_beta" in src
    assert "def Newmark_beta" not in src, "must not redefine Newmark_beta locally"

    h1, v1, a1 = Newmark_beta(F=1.0, h=0.0, h_dot=0.0, h_ddot=0.0, dt=DT,
                               m=SP["m"], c=SP["c"], k=SP["k"])
    h2, v2, a2 = tvc.Newmark_beta(F=1.0, h=0.0, h_dot=0.0, h_ddot=0.0, dt=DT,
                                   m=SP["m"], c=SP["c"], k=SP["k"])
    assert h1 == h2 and v1 == v2 and a1 == a2


def test_schedule_durations_and_transitions_exact():
    Ur_list = schedules.CYLINDER_UR_LIST
    dwell_s = 100.0
    sched = schedules.build_ascending_schedule(Ur_list, dwell_s, DT)
    assert sched["n_steps"] == len(Ur_list) * round(dwell_s / DT)
    assert sched["transition_step_indices"] == [
        i * round(dwell_s / DT) for i in range(1, len(Ur_list))]
    for idx, expected_ur in zip(sched["transition_step_indices"], Ur_list[1:]):
        assert sched["Ur_schedule"][idx] == expected_ur
        assert sched["Ur_schedule"][idx - 1] == Ur_list[Ur_list.index(expected_ur) - 1]

    cos_sched = schedules.build_ascending_cosine_schedule(Ur_list, dwell_s, 5.0, DT)
    expected_n = len(Ur_list) * round(dwell_s / DT) + (len(Ur_list) - 1) * round(5.0 / DT)
    assert cos_sched["n_steps"] == expected_n

    tri = schedules.build_triangular_schedule(Ur_list, dwell_s, DT)
    assert tri["n_steps"] == (2 * len(Ur_list) - 1) * round(dwell_s / DT)
    assert tri["reversal_ur_value"] == Ur_list[-1]


def test_existing_fixed_ur_results_untouched(monkeypatch, tmp_path):
    import run_time_varying_sweep as rtvs

    fixed_ur_dir = Path(__file__).resolve().parents[3] / "results" / "gru_cylinder200_nd_context_noacc_coupled_eval"
    if not fixed_ur_dir.exists():
        pytest.skip("no existing fixed-Ur coupled_eval directory to check on this machine")
    before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in fixed_ur_dir.glob("*.npz")}

    n = 50000
    t = np.arange(n) * DT
    df = pd.DataFrame({
        "case": "Ur2", "time": t, "step": np.arange(n),
        "disp": 0.01 * np.sin(0.2 * t), "vel": 0.01 * 0.2 * np.cos(0.2 * t),
        "acc": -0.01 * 0.2**2 * np.sin(0.2 * t), "cl": 0.1 * np.sin(0.2 * t),
    })
    monkeypatch.setattr(rtvs, "merge_dataframes", lambda **kw: df.copy())
    monkeypatch.setattr(rtvs, "compute_kinematics", lambda raw_df, **kw: raw_df)
    model = _fixed_model()
    x_scaler, y_scaler = _fixed_scalers()
    artifacts = dict(
        model=model, x_scaler=x_scaler, y_scaler=y_scaler,
        nd_inputs=True, use_ur_context=True, ur_mean=5.0, ur_std=2.0,
        seq_len=SEQ_LEN, input_cols=INPUT_COLS,
        **_dummy_artifact_paths(),
    )
    monkeypatch.setattr(rtvs, "load_model_and_artifacts", lambda device, artifact_dir=None: artifacts)
    monkeypatch.setattr(rtvs, "STUDY_ROOT", tmp_path)

    sched = schedules.build_ascending_schedule([2.0, 2.5], dwell_s=10 * DT, dt=DT)
    result, receipt = rtvs.run_sweep(sched, device="cpu")
    rtvs.save_result(result, receipt, tag="test_isolation")

    assert (tmp_path / "results" / "time_varying_ur_test_isolation.npz").exists()
    after = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in fixed_ur_dir.glob("*.npz")}
    assert before == after


def _write_stage2_manifest(path, frozen=True, selection_complete=True, n_run_dirs=3):
    import json as _json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps({
        "frozen": frozen,
        "selection_complete": selection_complete,
        "selected_run_dirs": [f"/fake/run_dir_{i}" for i in range(n_run_dirs)],
    }))


def test_gate_fails_if_manifest_missing(tmp_path, monkeypatch):
    import run_time_varying_sweep as rtvs
    monkeypatch.setattr(rtvs, "STAGE2_SELECTION_MANIFEST", tmp_path / "nope.json")
    with pytest.raises(SystemExit):
        rtvs.assert_stage2_selection_frozen()


def test_gate_fails_if_not_frozen(tmp_path, monkeypatch):
    import run_time_varying_sweep as rtvs
    manifest_path = tmp_path / "selection_manifest_stage2.json"
    _write_stage2_manifest(manifest_path, frozen=False)
    monkeypatch.setattr(rtvs, "STAGE2_SELECTION_MANIFEST", manifest_path)
    with pytest.raises(SystemExit):
        rtvs.assert_stage2_selection_frozen()


def test_gate_fails_if_selection_not_complete(tmp_path, monkeypatch):
    import run_time_varying_sweep as rtvs
    manifest_path = tmp_path / "selection_manifest_stage2.json"
    _write_stage2_manifest(manifest_path, frozen=True, selection_complete=False)
    monkeypatch.setattr(rtvs, "STAGE2_SELECTION_MANIFEST", manifest_path)
    with pytest.raises(SystemExit):
        rtvs.assert_stage2_selection_frozen()


def test_gate_fails_if_not_exactly_3_run_dirs(tmp_path, monkeypatch):
    import run_time_varying_sweep as rtvs
    manifest_path = tmp_path / "selection_manifest_stage2.json"
    _write_stage2_manifest(manifest_path, n_run_dirs=2)
    monkeypatch.setattr(rtvs, "STAGE2_SELECTION_MANIFEST", manifest_path)
    with pytest.raises(SystemExit):
        rtvs.assert_stage2_selection_frozen()


def test_gate_succeeds_with_frozen_complete_3_seed_manifest(tmp_path, monkeypatch):
    import run_time_varying_sweep as rtvs
    manifest_path = tmp_path / "selection_manifest_stage2.json"
    _write_stage2_manifest(manifest_path)
    monkeypatch.setattr(rtvs, "STAGE2_SELECTION_MANIFEST", manifest_path)
    run_dirs = rtvs.assert_stage2_selection_frozen()
    assert len(run_dirs) == 3


def test_single_seed_production_bypasses_gate_and_labels_receipt(tmp_path, monkeypatch):
    import sys

    import run_time_varying_sweep as rtvs
    monkeypatch.setattr(rtvs, "STAGE2_SELECTION_MANIFEST", tmp_path / "nope.json")

    captured = {}

    def fake_run_sweep(schedule, track_history_provenance=False, artifact_dir=None):
        assert artifact_dir is None
        return {"h": None}, {"study": "cylinder_time_varying_ur"}

    def fake_save_result(result, receipt, tag):
        captured["receipt"] = receipt
        captured["tag"] = tag
        return tmp_path / f"{tag}.npz"

    monkeypatch.setattr(rtvs, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(rtvs, "save_result", fake_save_result)

    old_argv = sys.argv
    sys.argv = ["run_time_varying_sweep.py", "--schedule", "ascending",
                "--single_seed_production"]
    try:
        rtvs.main()
    finally:
        sys.argv = old_argv

    assert captured["receipt"]["single_seed_production"] is True
    assert captured["receipt"]["n_seeds"] == 1
    assert captured["receipt"]["stage2_gate_bypassed"] is True
    assert "stage2_gate_bypass_reason" in captured["receipt"]
    assert "single_seed_production" in captured["tag"]
    assert "smoke" not in captured["tag"]


def test_smoke_and_single_seed_production_are_mutually_exclusive():
    import sys

    import run_time_varying_sweep as rtvs
    old_argv = sys.argv
    sys.argv = ["run_time_varying_sweep.py", "--smoke", "--single_seed_production"]
    try:
        with pytest.raises(SystemExit):
            rtvs.main()
    finally:
        sys.argv = old_argv


def test_envelope_aggregation_not_attenuated_by_seed_phase_differences():
    import plot_time_varying_ur as ptv
    dt = 0.005
    D, fn = 0.2, 0.2
    n = 12000
    t = np.arange(n) * dt
    amp_true = 0.45
    phases = {123: 0.0, 456: 0.35, 789: -0.5}
    disp_by_seed = {s: amp_true * D * np.sin(2 * np.pi * fn * t + p) for s, p in phases.items()}

    naive_median = np.median(np.stack(list(disp_by_seed.values())), axis=0)
    assert np.max(np.abs(naive_median)) / D < 0.999 * amp_true

    agg = ptv.aggregate_envelope_across_seeds(disp_by_seed, D, dt, fn)
    np.testing.assert_allclose(agg["envelope_median"], amp_true, rtol=1e-9)
    assert agg["seeds"] == [123, 456, 789]


def test_scalar_metric_aggregation_median_iqr():
    import plot_time_varying_ur as ptv
    metrics_by_seed = {
        123: {"A_star": 0.40, "f_osc": 0.201},
        456: {"A_star": 0.42, "f_osc": 0.199},
        789: {"A_star": 0.38, "f_osc": 0.200},
    }
    out = ptv.aggregate_scalar_metrics_across_seeds(metrics_by_seed)
    assert out["A_star_median"] == pytest.approx(0.40)
    assert out["f_osc_median"] == pytest.approx(0.200)
    assert out["A_star_iqr"] > 0


def test_representative_seed_is_predeclared_not_data_dependent():
    import plot_time_varying_ur as ptv
    assert ptv.REPRESENTATIVE_SEED == 123


def test_plot_multiseed_panels_rejects_representative_seed_not_present():
    import plot_time_varying_ur as ptv
    dt = 0.005
    D, fn = 0.2, 0.2
    n = 4000
    t = np.arange(n) * dt
    sched = {"transition_times": [], "transition_ur_values": []}
    results_by_seed = {
        456: {"time": t, "Ur": np.full(n, 5.0), "displacement": 0.1 * np.sin(t),
              "velocity": 0.1 * np.cos(t)},
        789: {"time": t, "Ur": np.full(n, 5.0), "displacement": 0.1 * np.sin(t),
              "velocity": 0.1 * np.cos(t)},
    }
    with pytest.raises(ValueError, match="representative_seed"):
        ptv.plot_multiseed_panels(results_by_seed, sched, D, fn, Path("/tmp/unused"))


def test_find_transition_time_instantaneous_matches_first_post_jump_sample():
    import schedules
    import plot_time_varying_ur as ptv

    dt = 0.005
    Ur_list = [2.0, 2.5, 3.0]
    dwell_s = 1.0
    sched = schedules.build_ascending_schedule(Ur_list, dwell_s, dt)

    t = ptv.find_transition_time(sched, 2.0, 2.5)
    Ur_arr = sched["Ur_schedule"]
    idx = int(np.where(np.isclose(Ur_arr, 2.5))[0][0])
    assert t == pytest.approx(idx * dt)


def test_find_transition_time_cosine_is_end_of_ramp_not_start():
    import schedules
    import plot_time_varying_ur as ptv

    dt = 0.005
    Ur_list = [2.0, 2.5, 3.0]
    dwell_s = 1.0
    transition_s = 0.25
    sched = schedules.build_ascending_cosine_schedule(Ur_list, dwell_s, transition_s, dt)

    t = ptv.find_transition_time(sched, 2.0, 2.5)
    ramp_start_idx = sched["transition_step_indices"][0]
    ramp_end_idx = ramp_start_idx + sched["transition_steps"]
    assert t == pytest.approx(ramp_end_idx * dt)
    assert t > ramp_start_idx * dt

    Ur_arr = sched["Ur_schedule"]
    assert Ur_arr[ramp_end_idx] == pytest.approx(2.5)
    assert Ur_arr[ramp_start_idx] < 2.1


def test_find_transition_time_triangular_matches_ascending_leg_first():
    import schedules
    import plot_time_varying_ur as ptv

    dt = 0.005
    Ur_list = [2.0, 2.5, 3.0]
    dwell_s = 1.0
    sched = schedules.build_triangular_schedule(Ur_list, dwell_s, dt, transition="instantaneous")

    t = ptv.find_transition_time(sched, 2.0, 2.5)
    Ur_arr = sched["Ur_schedule"]
    idx = int(np.where(np.isclose(Ur_arr, 2.5))[0][0])
    assert t == pytest.approx(idx * dt)


def test_find_transition_time_raises_for_missing_transition():
    import schedules
    import plot_time_varying_ur as ptv

    sched = schedules.build_ascending_schedule([2.0, 2.5, 3.0], 1.0, 0.005)
    with pytest.raises(ValueError, match="No .* transition found"):
        ptv.find_transition_time(sched, 2.0, 99.0)


def test_plot_transition_window_end_to_end_on_cosine_schedule(tmp_path):
    import schedules
    import plot_time_varying_ur as ptv

    dt = 0.005
    D, fn = 0.2, 0.2
    Ur_list = [2.0, 2.5, 3.0]
    sched = schedules.build_ascending_cosine_schedule(Ur_list, dwell_s=2.0, transition_s=0.5, dt=dt)
    n = sched["n_steps"]
    t = np.arange(n) * dt
    rng = np.random.default_rng(0)
    result = {
        "time": t,
        "Ur": sched["Ur_schedule"],
        "displacement": 0.01 * np.sin(2 * np.pi * 1.0 * t) + 1e-4 * rng.standard_normal(n),
        "velocity": 0.01 * np.cos(2 * np.pi * 1.0 * t),
        "CL": 0.1 * np.sin(2 * np.pi * 1.0 * t),
    }

    pdf, png = ptv.plot_transition_window(
        result, sched, D, fn, ur_before=2.0, ur_after=2.5,
        out_path_stem=tmp_path / "transition_test",
    )
    assert pdf.exists()
    assert png.exists()


def _make_constant_ur_result(D, fn, dt, n_steps, amplitude, growing=False):
    t = np.arange(n_steps) * dt
    if growing:
        env = np.linspace(0.05 * amplitude, amplitude, n_steps)
    else:
        env = np.full(n_steps, amplitude)
    h = env * np.sin(2 * np.pi * fn * t)
    return h


def test_plateau_summary_flags_constant_amplitude_as_stabilised():
    import plot_time_varying_ur as ptv

    D, fn, dt = 0.2, 0.2, 0.005
    dwell_s = 30.0
    dwell_steps = int(round(dwell_s / dt))
    Ur_list = [5.0]
    schedule = {
        "Ur_list": Ur_list, "dwell_steps": dwell_steps,
    }
    h = _make_constant_ur_result(D, fn, dt, dwell_steps, amplitude=0.05, growing=False)
    result = {"displacement": h}

    df, fig = ptv.plateau_summary(result, schedule, D, fn, dt, None, None)
    import matplotlib.pyplot as plt
    plt.close(fig)
    assert bool(df.iloc[0]["stabilised"]) is True


def test_plateau_summary_flags_growing_amplitude_as_not_stabilised():
    import plot_time_varying_ur as ptv

    D, fn, dt = 0.2, 0.2, 0.005
    dwell_s = 30.0
    dwell_steps = int(round(dwell_s / dt))
    Ur_list = [5.0]
    schedule = {
        "Ur_list": Ur_list, "dwell_steps": dwell_steps,
    }
    h = _make_constant_ur_result(D, fn, dt, dwell_steps, amplitude=0.05, growing=True)
    result = {"displacement": h}

    df, fig = ptv.plateau_summary(result, schedule, D, fn, dt, None, None)
    import matplotlib.pyplot as plt
    plt.close(fig)
    assert bool(df.iloc[0]["stabilised"]) is False


def test_plot_transitions_composite_covers_every_transition_with_2_columns(tmp_path):
    import schedules
    import plot_time_varying_ur as ptv

    dt = 0.005
    D, fn = 0.2, 0.2
    Ur_list = [3.5, 4.0, 5.25, 5.5, 6.0, 6.25]
    sched = schedules.build_ascending_cosine_schedule(Ur_list, dwell_s=2.0, transition_s=0.5, dt=dt)
    n = sched["n_steps"]
    t = np.arange(n) * dt
    rng = np.random.default_rng(0)
    result = {
        "time": t,
        "Ur": sched["Ur_schedule"],
        "displacement": 0.01 * np.sin(2 * np.pi * 1.0 * t) + 1e-4 * rng.standard_normal(n),
        "velocity": 0.01 * np.cos(2 * np.pi * 1.0 * t),
        "CL": 0.1 * np.sin(2 * np.pi * 1.0 * t),
    }
    transitions = {
        "onset": {"ur_before": 3.5, "ur_after": 4.0},
        "lockin": {"ur_before": 5.25, "ur_after": 5.5},
        "departure_from_lockin": {"ur_before": 6.0, "ur_after": 6.25},
    }

    pdf, png = ptv.plot_transitions_composite_thesis(
        result, sched, D, fn, transitions, out_path_stem=tmp_path / "composite",
        window_before_Tn=1.0, window_after_Tn=1.0,
    )
    assert pdf.exists()
    assert png.exists()


def test_plot_transitions_composite_applies_fixed_xlim_to_every_panel(tmp_path, monkeypatch):
    import schedules
    import plot_time_varying_ur as ptv

    dt = 0.005
    D, fn = 0.2, 0.2
    Ur_list = [3.5, 4.0, 5.25, 5.5]
    sched = schedules.build_ascending_schedule(Ur_list, dwell_s=2.0, dt=dt)
    n = sched["n_steps"]
    t = np.arange(n) * dt
    result = {
        "time": t,
        "Ur": sched["Ur_schedule"],
        "displacement": 0.01 * np.sin(2 * np.pi * 1.0 * t),
        "velocity": 0.01 * np.cos(2 * np.pi * 1.0 * t),
        "CL": 0.1 * np.sin(2 * np.pi * 1.0 * t),
    }
    transitions = {
        "onset": {"ur_before": 3.5, "ur_after": 4.0},
        "lockin": {"ur_before": 5.25, "ur_after": 5.5},
    }
    fixed_xlim = (-0.7, 0.7)

    captured = {}
    real_subplots = ptv.plt.subplots

    def spying_subplots(*args, **kwargs):
        fig, axes = real_subplots(*args, **kwargs)
        captured["axes"] = axes
        return fig, axes

    monkeypatch.setattr(ptv.plt, "subplots", spying_subplots)

    ptv.plot_transitions_composite_thesis(
        result, sched, D, fn, transitions, out_path_stem=tmp_path / "composite2",
        window_before_Tn=1.0, window_after_Tn=1.0, fixed_xlim=fixed_xlim,
    )

    axes = np.atleast_2d(captured["axes"])
    assert axes.shape == (2, 2)
    for ax in axes.ravel():
        assert ax.get_xlim() == pytest.approx(fixed_xlim)
