"""
Focused CPU tests for coupled_inference.py's nondimensional coordinate
transform -- added by the read-only bridge-ND compatibility audit.

Scope: to_model_coords() and run_coupled_viv()'s per-step transform had NO
existing test coverage (unlike apply_nd_transform in train_gru.py, which is
covered in test_viv_ablation_framework.py). These tests are purely additive;
no production code in coupled_inference.py is modified.

Does not train a model, run GPU jobs, or run real coupled inference against
CFD data -- run_coupled_viv is exercised with a trivial zero-CL stub model
and identity scalers so its internal per-step state transform is directly
observable through max_abs_scaled_kinematics without needing history exposed.
"""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from viv_analysis.coupled_inference import (
    to_model_coords, run_coupled_viv, Newmark_beta,
    load_artifact_coordinate_mode, LEGACY_COORD_SOURCE,
    check_artifact_dataset_compatibility, normalize_cfd_dataset,
    CYLINDER200_ALIASES,
)

# Required numerical example: bridge Ur=6.7385, D=7.42, fn=0.32
D_BRIDGE = 7.42
FN_BRIDGE = 0.32
UR_BRIDGE = 6.7385
U_BRIDGE = UR_BRIDGE * FN_BRIDGE * D_BRIDGE  # ~15.9999


class _IdentityXScaler:
    """StandardScaler-like: mean_=0, scale_=1 -> transform is identity.
    run_coupled_viv only reads .mean_/.scale_ (precomputed once, not
    .transform()), so a bare attribute holder is sufficient."""
    mean_ = np.zeros(3, dtype=np.float32)
    scale_ = np.ones(3, dtype=np.float32)


class _IdentityYScaler:
    mean_ = np.zeros(1, dtype=np.float32)
    scale_ = np.ones(1, dtype=np.float32)


class _ZeroCLModel:
    """Always predicts CL=0 -- isolates the state/coordinate transform from
    any GRU weight behavior. run_coupled_viv only calls .eval() and
    __call__(x) -> (cl_scaled, hidden)."""
    def eval(self):
        return self

    def __call__(self, x):
        return torch.zeros(x.shape[0], dtype=torch.float32), None


class TestToModelCoordsBridgeNumerical:
    """Required numerical test: bridge Ur=6.7385 -> model coords ~[0.01,0.01,0.01]."""

    def test_required_bridge_example(self):
        h = 0.0742
        h_dot = 0.16
        h_ddot = 0.01 * U_BRIDGE ** 2 / D_BRIDGE

        kin = np.array([[h, h_dot, h_ddot]], dtype=np.float32)
        out = to_model_coords(kin, nd_inputs=True, D=D_BRIDGE, U=U_BRIDGE)

        np.testing.assert_allclose(out[0], [0.01, 0.01, 0.01], rtol=2e-4)

    def test_dimensional_mode_is_identity(self):
        """Item 5: dimensional checkpoints must see the raw physical state unchanged."""
        kin = np.array([[0.0742, 0.16, 0.345]], dtype=np.float32)
        out = to_model_coords(kin, nd_inputs=False, D=D_BRIDGE, U=U_BRIDGE)
        np.testing.assert_array_equal(out, kin)

    def test_disp_divided_by_D_only(self):
        kin = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        out = to_model_coords(kin, nd_inputs=True, D=D_BRIDGE, U=U_BRIDGE)
        assert out[0, 0] == pytest.approx(1.0 / D_BRIDGE, rel=1e-6)

    def test_vel_divided_by_U_not_D(self):
        kin = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        out = to_model_coords(kin, nd_inputs=True, D=D_BRIDGE, U=U_BRIDGE)
        assert out[0, 1] == pytest.approx(1.0 / U_BRIDGE, rel=1e-6)
        assert out[0, 1] != pytest.approx(1.0 / D_BRIDGE, rel=1e-3)

    def test_acc_divided_by_U2_over_D(self):
        kin = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        out = to_model_coords(kin, nd_inputs=True, D=D_BRIDGE, U=U_BRIDGE)
        assert out[0, 2] == pytest.approx(1.0 / (U_BRIDGE ** 2 / D_BRIDGE), rel=1e-6)

    def test_rejects_nonpositive_D_or_U_when_nd(self):
        kin = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        with pytest.raises(ValueError):
            to_model_coords(kin, nd_inputs=True, D=0.0, U=U_BRIDGE)
        with pytest.raises(ValueError):
            to_model_coords(kin, nd_inputs=True, D=D_BRIDGE, U=0.0)


class TestRunCoupledVivPerStepTransform:
    """Item 2/4: every Newmark-generated closed-loop state must be
    ND-transformed before being appended to the GRU history window, for
    EVERY step (not just the initial handoff state)."""

    def _run(self, nd_inputs: bool, n_steps: int, h0=0.0742, hd0=0.16, hdd0=None,
              m=1.0, c=0.1, k=1.0, dt=0.001, seq_len=4):
        if hdd0 is None:
            hdd0 = 0.01 * U_BRIDGE ** 2 / D_BRIDGE
        initial_state = {"h": h0, "h_dot": hd0, "h_ddot": hdd0}
        initial_history = np.zeros((seq_len, 3), dtype=np.float32)
        return run_coupled_viv(
            model=_ZeroCLModel(), x_scaler=_IdentityXScaler(), y_scaler=_IdentityYScaler(),
            seq_len=seq_len, input_cols=["disp", "vel", "acc"],
            initial_history=initial_history, initial_state=initial_state,
            m=m, c=c, k=k, rho=1.0, U=U_BRIDGE, D=D_BRIDGE, B=1.0, dt=dt,
            n_steps=n_steps, nd_inputs=nd_inputs, device="cpu",
        )

    def test_required_example_single_step_max_scaled_kinematics(self):
        """Required numerical test, first appended row: with h/h_dot/h_ddot
        set to the required example and IDENTITY scalers, the internal
        per-step transform's max|scaled kinematic| must equal 0.01 (all
        three components individually equal 0.01, so their max is 0.01)."""
        h_ddot = 0.01 * U_BRIDGE ** 2 / D_BRIDGE
        result = self._run(nd_inputs=True, n_steps=1, h0=0.0742, hd0=0.16, hdd0=h_ddot)
        assert result["max_abs_scaled_kinematics"] == pytest.approx(0.01, rel=2e-4)
        # the state that DROVE this step is returned unchanged (dimensional):
        assert result["displacement"][0] == pytest.approx(0.0742)
        assert result["velocity"][0] == pytest.approx(0.16)
        assert result["acceleration"][0] == pytest.approx(h_ddot)

    def test_newmark_generated_state_also_transformed(self):
        """After Newmark produces a NEW dimensional state (step i=1, not the
        initial handoff state), the transform applied to THAT state must
        still match to_model_coords -- verified by independently recomputing
        to_model_coords on the function's own returned dimensional states
        (h, h_dot, h_ddot for every step) and confirming the reported
        max_abs_scaled_kinematics (observable proxy for the internal
        per-step transformed values, exact under identity scalers) matches."""
        n_steps = 4
        result = self._run(nd_inputs=True, n_steps=n_steps)

        h_arr = result["displacement"]
        hd_arr = result["velocity"]
        hdd_arr = result["acceleration"]
        kin = np.column_stack([h_arr, hd_arr, hdd_arr]).astype(np.float32)

        model_coords = to_model_coords(kin, nd_inputs=True, D=D_BRIDGE, U=U_BRIDGE)
        expected_max = float(np.max(np.abs(model_coords)))

        assert result["max_abs_scaled_kinematics"] == pytest.approx(expected_max, rel=1e-4)
        # sanity: state actually evolved under Newmark (not stuck at the seed)
        assert not np.allclose(hd_arr, hd_arr[0])

    def test_dimensional_mode_unaffected_by_D_U(self):
        """Item 5: for nd_inputs=False, max_abs_scaled_kinematics must equal
        the raw physical magnitude -- changing D/U must have NO effect."""
        hdd0 = 0.01 * U_BRIDGE ** 2 / D_BRIDGE
        result_a = self._run(nd_inputs=False, n_steps=3, h0=0.0742, hd0=0.16, hdd0=hdd0)
        # Re-run with wildly different D/U (same exact initial state); the
        # dimensional path must ignore D/U entirely.
        initial_state = {"h": 0.0742, "h_dot": 0.16, "h_ddot": hdd0}
        initial_history = np.zeros((4, 3), dtype=np.float32)
        result_b = run_coupled_viv(
            model=_ZeroCLModel(), x_scaler=_IdentityXScaler(), y_scaler=_IdentityYScaler(),
            seq_len=4, input_cols=["disp", "vel", "acc"],
            initial_history=initial_history, initial_state=initial_state,
            m=1.0, c=0.1, k=1.0, rho=1.0, U=999.0, D=0.001, B=1.0, dt=0.001,
            n_steps=3, nd_inputs=False, device="cpu",
        )
        assert result_a["max_abs_scaled_kinematics"] == pytest.approx(
            result_b["max_abs_scaled_kinematics"], rel=1e-5
        )

    def test_nd_and_dimensional_diverge_for_same_physical_state(self):
        """Sanity: nd_inputs=True vs False on the SAME physical state must
        give DIFFERENT scaled kinematics (D=7.42, U~16 are not both 1.0), so
        a silently-dropped nd_inputs flag would be caught by any regression
        comparing these two paths."""
        result_nd = self._run(nd_inputs=True, n_steps=1)
        result_dim = self._run(nd_inputs=False, n_steps=1)
        assert result_nd["max_abs_scaled_kinematics"] != pytest.approx(
            result_dim["max_abs_scaled_kinematics"], rel=1e-2
        )


class _IdentityXScalerN:
    """Identity scaler sized to N features -- for restricted input_cols
    (e.g. a single-kinematic-quantity model) where the fixed 3-wide
    _IdentityXScaler would broadcast-mismatch."""
    def __init__(self, n: int):
        self.mean_ = np.zeros(n, dtype=np.float32)
        self.scale_ = np.ones(n, dtype=np.float32)


class TestRestrictedInputCols:
    """Regression coverage for the cylinder200 single-input-column models
    (--input_cols disp / vel / acc): to_model_coords and run_coupled_viv's
    per-step window update must select each column's value/divisor BY
    NAME, not by a hardcoded [disp, vel, acc] position -- otherwise a
    restricted or reordered input_cols silently broadcasts the wrong
    divisor (nd_inputs=True) or crashes on a shape mismatch when the
    per-step row is appended to a narrower history window."""

    def test_to_model_coords_single_column_vel(self):
        """A vel-only column must be divided by U, not D (the position-0
        divisor in the old hardcoded [D, U, U^2/D] array)."""
        kin = np.array([[1.0]], dtype=np.float32)
        out = to_model_coords(kin, nd_inputs=True, D=D_BRIDGE, U=U_BRIDGE,
                              input_cols=["vel"])
        assert out.shape == (1, 1)
        assert out[0, 0] == pytest.approx(1.0 / U_BRIDGE, rel=1e-6)

    def test_to_model_coords_reordered_two_columns(self):
        """["acc", "disp"] (reordered, partial) -- each column keeps its
        own divisor regardless of position in the list."""
        acc_val, disp_val = 2.0, 3.0
        kin = np.array([[acc_val, disp_val]], dtype=np.float32)
        out = to_model_coords(kin, nd_inputs=True, D=D_BRIDGE, U=U_BRIDGE,
                              input_cols=["acc", "disp"])
        assert out[0, 0] == pytest.approx(acc_val / (U_BRIDGE**2 / D_BRIDGE), rel=1e-6)
        assert out[0, 1] == pytest.approx(disp_val / D_BRIDGE, rel=1e-6)

    def test_to_model_coords_default_input_cols_unchanged(self):
        """Omitting input_cols must reproduce the original full 3-column
        [disp, vel, acc] behavior exactly (backward compatibility)."""
        kin = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out_default = to_model_coords(kin, nd_inputs=True, D=D_BRIDGE, U=U_BRIDGE)
        out_explicit = to_model_coords(kin, nd_inputs=True, D=D_BRIDGE, U=U_BRIDGE,
                                       input_cols=["disp", "vel", "acc"])
        np.testing.assert_array_equal(out_default, out_explicit)

    def test_run_coupled_viv_single_column_disp_only_does_not_crash(self):
        """The actual bug this class guards against: with input_cols=["disp"]
        the old code built a hardcoded 3-wide [h, h_dot, h_ddot] per-step row
        and divided by a 3-wide x_scale -- crashing (dimensional) or silently
        broadcast-corrupting (nd_inputs=True) against the model's true 1-wide
        history window."""
        seq_len = 4
        initial_state = {"h": 0.01, "h_dot": 0.02, "h_ddot": 0.03}
        initial_history = np.zeros((seq_len, 1), dtype=np.float32)
        result = run_coupled_viv(
            model=_ZeroCLModel(), x_scaler=_IdentityXScalerN(1), y_scaler=_IdentityYScaler(),
            seq_len=seq_len, input_cols=["disp"],
            initial_history=initial_history, initial_state=initial_state,
            m=1.0, c=0.1, k=1.0, rho=1.0, U=U_BRIDGE, D=D_BRIDGE, B=1.0, dt=0.001,
            n_steps=5, nd_inputs=False, device="cpu",
        )
        assert np.isfinite(result["displacement"]).all()

    def test_run_coupled_viv_single_column_vel_only_nd_correct_divisor(self):
        """Same as above but nd_inputs=True with a vel-only model: the
        per-step scaled value must equal h_dot/U (identity scaler), not
        h_dot/D (the position-0 divisor a hardcoded array would apply)."""
        seq_len = 3
        hd0 = 0.16
        initial_state = {"h": 0.0742, "h_dot": hd0, "h_ddot": 0.0}
        initial_history = np.zeros((seq_len, 1), dtype=np.float32)
        result = run_coupled_viv(
            model=_ZeroCLModel(), x_scaler=_IdentityXScalerN(1), y_scaler=_IdentityYScaler(),
            seq_len=seq_len, input_cols=["vel"],
            initial_history=initial_history, initial_state=initial_state,
            m=1.0, c=0.1, k=1.0, rho=1.0, U=U_BRIDGE, D=D_BRIDGE, B=1.0, dt=0.001,
            n_steps=1, nd_inputs=True, device="cpu",
        )
        expected = hd0 / U_BRIDGE
        assert result["max_abs_scaled_kinematics"] == pytest.approx(expected, rel=1e-4)


class TestBridgeForceUsesB:
    """Item 6: bridge lift force must use B=25.9, never D=7.42."""

    def test_force_magnitude_scales_with_B_not_D(self):
        """0.5*rho*U^2*B*cl -- with cl held nonzero via a stub model, the
        resulting Newmark-driven state after one step must differ when B
        changes (25.9 vs 7.42), proving B (not D) drives the force term."""
        class _ConstCLModel:
            def eval(self):
                return self

            def __call__(self, x):
                return torch.full((x.shape[0],), 0.5, dtype=torch.float32), None

        def run_with_B(B):
            # n_steps=2 so index [1] is the Newmark-GENERATED state (index
            # [0] is just the initial condition we seeded, independent of B).
            initial_state = {"h": 0.0, "h_dot": 0.0, "h_ddot": 0.0}
            initial_history = np.zeros((4, 3), dtype=np.float32)
            return run_coupled_viv(
                model=_ConstCLModel(), x_scaler=_IdentityXScaler(), y_scaler=_IdentityYScaler(),
                seq_len=4, input_cols=["disp", "vel", "acc"],
                initial_history=initial_history, initial_state=initial_state,
                m=1.0, c=0.1, k=1.0, rho=1.225, U=U_BRIDGE, D=D_BRIDGE, B=B, dt=0.001,
                n_steps=2, nd_inputs=True, device="cpu",
            )

        result_B_bridge = run_with_B(25.9)   # correct bridge B
        result_B_as_D = run_with_B(7.42)     # the exact bug this item guards against

        # F_aero = 0.5*rho*U^2*B*cl differs by a factor of 25.9/7.42 between
        # these two runs -> the resulting acceleration[1] (first Newmark
        # output) must differ measurably; equal values would mean B was
        # silently replaced by D somewhere in the force path.
        assert result_B_bridge["acceleration"][1] != pytest.approx(
            result_B_as_D["acceleration"][1], rel=1e-2
        )
        assert result_B_bridge["acceleration"][1] != 0.0
        assert result_B_as_D["acceleration"][1] != 0.0


# ── Provenance-safety patch: coordinate-mode legacy-vs-recorded distinction ─

def _write_run_config(artifact_dir: Path, **fields) -> None:
    with open(artifact_dir / "run_config.json", "w") as f:
        json.dump(fields, f)


def _write_ur_stats(artifact_dir: Path, **fields) -> None:
    with open(artifact_dir / "ur_stats.pkl", "wb") as f:
        pickle.dump(fields, f)


class TestCoordinateModeLegacyVsRecorded:
    """Fix 1: a missing nd_inputs key must never be reported as an
    artifact-recorded value."""

    def test_7_legacy_no_coord_metadata_no_cli_request_warns_and_assumes_dimensional(self, capsys):
        """7. legacy artifact with no coordinate metadata and no explicit ND
        request -> warning and dimensional assumption."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)  # no run_config.json, no ur_stats.pkl at all
            result = load_artifact_coordinate_mode(artifact_dir, cli_nd_inputs=None)
            assert result is False
            out = capsys.readouterr().out
            assert "No recorded coordinate mode" in out
            assert LEGACY_COORD_SOURCE in out
            # must NOT claim ur_stats.pkl/run_config.json actually recorded this
            assert "Using coordinate mode from artifact (ur_stats.pkl)" not in out
            assert "Using coordinate mode from artifact (run_config.json" not in out

    def test_8_legacy_no_coord_metadata_explicit_nd_request_raises(self):
        """8. legacy artifact with no coordinate metadata but explicit ND
        request -> ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            with pytest.raises(ValueError, match="no recorded coordinate mode"):
                load_artifact_coordinate_mode(artifact_dir, cli_nd_inputs=True)

    def test_legacy_no_coord_metadata_explicit_dim_request_is_allowed(self):
        """--dim_inputs against a legacy artifact matches the backward-
        compatible assumption, so it must NOT raise."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            assert load_artifact_coordinate_mode(artifact_dir, cli_nd_inputs=False) is False

    def test_9_explicit_recorded_false_distinguished_from_missing_key(self, capsys):
        """9. explicit recorded nd_inputs=False is distinguished from a
        missing key -- reports the REAL source, not the legacy fallback."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            _write_run_config(artifact_dir, nd_inputs=False, coordinate_mode="dimensional")
            result = load_artifact_coordinate_mode(artifact_dir, cli_nd_inputs=None)
            assert result is False
            out = capsys.readouterr().out
            assert "Using coordinate mode from artifact (run_config.json" in out
            assert LEGACY_COORD_SOURCE not in out

            # An explicit --nd_inputs against this recorded-False artifact
            # must raise the ORIGINAL mismatch error, not the "no recorded
            # coordinate mode" legacy error -- it's a real recorded conflict.
            with pytest.raises(ValueError, match="Coordinate mode mismatch"):
                load_artifact_coordinate_mode(artifact_dir, cli_nd_inputs=True)

    def test_ur_stats_missing_key_falls_through_to_legacy_not_recorded_false(self, capsys):
        """A ur_stats.pkl that exists but has no nd_inputs key at all (the
        exact shape of the real legacy results/gru_bridge_noise0.05/ur_stats.pkl)
        must NOT be reported as having recorded dimensional mode."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            _write_ur_stats(artifact_dir, mean=7.3, std=1.5, use_ur_context=True)  # no nd_inputs key
            result = load_artifact_coordinate_mode(artifact_dir, cli_nd_inputs=None)
            assert result is False
            out = capsys.readouterr().out
            assert LEGACY_COORD_SOURCE in out
            assert "Using coordinate mode from artifact (ur_stats.pkl)" not in out

    def test_10_existing_nd_mismatch_guard_still_works(self):
        """10. existing ND mismatch guard continues to work."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            _write_run_config(artifact_dir, nd_inputs=True, coordinate_mode="nondimensional")
            with pytest.raises(ValueError, match="Coordinate mode mismatch"):
                load_artifact_coordinate_mode(artifact_dir, cli_nd_inputs=False)
            # matching CLI must NOT raise
            assert load_artifact_coordinate_mode(artifact_dir, cli_nd_inputs=True) is True

    def test_run_config_recorded_true_via_cli_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            _write_run_config(artifact_dir, nd_inputs=True, coordinate_mode="nondimensional")
            assert load_artifact_coordinate_mode(artifact_dir, cli_nd_inputs=None) is True


# ── Provenance-safety patch: dataset compatibility check ────────────────────

class TestDatasetCompatibility:
    def test_1_run_config_records_bridge_requested_bridge_passes(self):
        """1. run_config records bridge, requested bridge -> pass."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            _write_run_config(artifact_dir, cfd_dataset="bridge")
            check_artifact_dataset_compatibility(artifact_dir, "bridge")  # must not raise

    def test_2_run_config_records_bridge_requested_cylinder200_raises(self):
        """2. run_config records bridge, requested cylinder200 -> ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            _write_run_config(artifact_dir, cfd_dataset="bridge")
            with pytest.raises(ValueError, match="Dataset mismatch") as exc_info:
                check_artifact_dataset_compatibility(artifact_dir, "cylinder200")
            msg = str(exc_info.value)
            assert "requested CFD dataset" in msg
            assert "cylinder200" in msg
            assert "recorded artifact dataset" in msg
            assert "bridge" in msg
            assert "artifact directory" in msg
            assert str(artifact_dir) in msg

    def test_3_ur_stats_records_bridge_requested_bridge_passes(self):
        """3. ur_stats records bridge, requested bridge -> pass."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            _write_ur_stats(artifact_dir, cfd_dataset="bridge")
            check_artifact_dataset_compatibility(artifact_dir, "bridge")  # must not raise

    def test_4_run_config_takes_precedence_over_ur_stats(self, capsys):
        """4. run_config takes precedence over ur_stats when both exist."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            _write_run_config(artifact_dir, cfd_dataset="bridge")
            _write_ur_stats(artifact_dir, cfd_dataset="cylinder200")  # would conflict if used
            check_artifact_dataset_compatibility(artifact_dir, "bridge")  # must not raise
            out = capsys.readouterr().out
            assert "run_config.json" in out
            assert "matches artifact's recorded 'bridge'" in out

    @pytest.mark.parametrize("alias", sorted(CYLINDER200_ALIASES))
    def test_5_cylinder200_aliases_normalize_to_same_canonical_dataset(self, alias):
        """5. cylinder200 aliases normalize to the same canonical dataset."""
        assert normalize_cfd_dataset(alias) == "cylinder200"
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            _write_run_config(artifact_dir, cfd_dataset="cylinder200")
            check_artifact_dataset_compatibility(artifact_dir, alias)  # must not raise

    def test_bridge_and_re200_cylinder_are_distinct(self):
        assert normalize_cfd_dataset("bridge") != normalize_cfd_dataset("cylinder")
        assert normalize_cfd_dataset("cylinder200") != normalize_cfd_dataset("cylinder")
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            _write_run_config(artifact_dir, cfd_dataset="bridge")
            with pytest.raises(ValueError, match="Dataset mismatch"):
                check_artifact_dataset_compatibility(artifact_dir, "cylinder")

    def test_6_legacy_artifact_no_dataset_metadata_warns_and_allows(self, capsys):
        """6. legacy artifact with no dataset metadata -> warning, allowed."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)  # no run_config.json, no ur_stats.pkl
            check_artifact_dataset_compatibility(artifact_dir, "bridge")  # must not raise
            out = capsys.readouterr().out
            assert "no recorded cfd_dataset" in out
            assert "could not be verified" in out

    def test_legacy_ur_stats_present_but_no_dataset_key_warns_and_allows(self, capsys):
        """Mirrors the real results/gru_bridge_noise0.05/ur_stats.pkl shape
        (exists, but no cfd_dataset key)."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            _write_ur_stats(artifact_dir, mean=7.3, std=1.5, use_ur_context=True)
            check_artifact_dataset_compatibility(artifact_dir, "bridge")  # must not raise
            out = capsys.readouterr().out
            assert "could not be verified" in out

    def test_does_not_infer_dataset_from_directory_name(self):
        """A directory literally named 'gru_bridge_noise0.05' with NO
        recorded cfd_dataset metadata must still be treated as unverified
        (warn + allow), never inferred to be 'bridge' from the folder name."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "gru_bridge_noise0.05"
            artifact_dir.mkdir()
            # Requesting a totally different dataset must NOT raise, since
            # nothing was actually recorded -- if the name were used as a
            # fallback, this would incorrectly raise a mismatch.
            check_artifact_dataset_compatibility(artifact_dir, "cylinder200")
