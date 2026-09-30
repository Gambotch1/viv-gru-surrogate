"""
Tests for src/viv_analysis/rollout_training.py -- the differentiable
closed-loop rollout mechanics used for Model A/B fine-tuning.

Scope: correctness of the torch reimplementation against the validated
non-differentiable run_coupled_viv (coupled_inference.py), the loss
functions' sign/zero behavior, gradient flow through Newmark + multiple
rollout steps, and TBPTT detachment actually severing the graph while
preserving numerical continuity.

No GPU, no real CFD data, no training -- purely-synthetic small models and
identity/near-identity scalers, matching the existing lightweight pattern in
tests/test_coupled_inference_nd.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from viv_analysis.coupled_inference import run_coupled_viv, to_model_coords, warmup_history
from viv_analysis.models.gru import VIV_GRU
from viv_analysis.rollout_training import (
    ScalerConstants, build_next_row_torch, rollout_chunk,
    sample_batch_starts, build_batch_from_case, build_tf_batch_from_case,
    loss_cl, loss_roll,
)

D_BRIDGE = 7.42
FN_BRIDGE = 0.32
UR_BRIDGE = 6.7385
U_BRIDGE = UR_BRIDGE * FN_BRIDGE * D_BRIDGE


class _IdentityXScaler:
    def __init__(self, n=2):
        self.mean_ = np.zeros(n, dtype=np.float64)
        self.scale_ = np.ones(n, dtype=np.float64)

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_


class _IdentityYScaler:
    mean_ = np.zeros(1, dtype=np.float64)
    scale_ = np.ones(1, dtype=np.float64)


# ── build_next_row_torch vs run_coupled_viv's inline transform ─────────────

class TestBuildNextRowTorch:
    def test_matches_dimensional_identity_scaler(self):
        h, hd, hdd = 0.05, 0.1, -0.2
        row = build_next_row_torch(
            torch.tensor([h]), torch.tensor([hd]), torch.tensor([hdd]),
            input_cols=["disp", "vel"], nd_inputs=False, D=D_BRIDGE, U=U_BRIDGE,
            x_mean=np.zeros(2), x_scale=np.ones(2),
            use_ur_context=False, ur_scaled=0.0,
        )
        np.testing.assert_allclose(row.numpy(), [[h, hd]], atol=1e-6)

    def test_nd_divisors_by_name(self):
        h, hd, hdd = D_BRIDGE, U_BRIDGE, 0.0
        row = build_next_row_torch(
            torch.tensor([h]), torch.tensor([hd]), torch.tensor([hdd]),
            input_cols=["disp", "vel"], nd_inputs=True, D=D_BRIDGE, U=U_BRIDGE,
            x_mean=np.zeros(2), x_scale=np.ones(2),
            use_ur_context=False, ur_scaled=0.0,
        )
        np.testing.assert_allclose(row.numpy(), [[1.0, 1.0]], atol=1e-6)

    def test_ur_context_appended(self):
        row = build_next_row_torch(
            torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]),
            input_cols=["disp", "vel"], nd_inputs=False, D=D_BRIDGE, U=U_BRIDGE,
            x_mean=np.zeros(2), x_scale=np.ones(2),
            use_ur_context=True, ur_scaled=1.234,
        )
        assert row.shape == (1, 3)
        assert row[0, 2].item() == pytest.approx(1.234)

    def test_batched(self):
        h = torch.tensor([0.0, 1.0, 2.0])
        hd = torch.tensor([0.0, 0.0, 0.0])
        hdd = torch.tensor([0.0, 0.0, 0.0])
        row = build_next_row_torch(
            h, hd, hdd, input_cols=["disp", "vel"], nd_inputs=False,
            D=D_BRIDGE, U=U_BRIDGE, x_mean=np.zeros(2), x_scale=np.ones(2),
            use_ur_context=False, ur_scaled=0.0,
        )
        assert row.shape == (3, 2)
        np.testing.assert_allclose(row[:, 0].numpy(), [0.0, 1.0, 2.0])


# ── rollout_chunk must reproduce run_coupled_viv exactly ────────────────────

def _seeded_model(input_size=3, seed=0):
    torch.manual_seed(seed)
    return VIV_GRU(input_size=input_size, hidden_size=8, num_layers=2, dropout=0.0)


class TestRolloutChunkMatchesRunCoupledViv:
    """The critical correctness test: given identical initial conditions, a
    no-grad rollout_chunk call must produce the SAME trajectory as
    run_coupled_viv to floating-point precision -- both implement the same
    per-step recipe (model call -> inverse-transform -> force -> Newmark ->
    push pre-update state), just numpy-vs-torch."""

    def _compare(self, nd_inputs, use_ur_context, input_cols=("disp", "vel")):
        input_cols = list(input_cols)
        n_kin = len(input_cols)
        seq_len = 6
        n_steps = 15
        m, c, k, rho, B, dt = 24604.0, 989.39, 99463.88, 1.225, 25.9, 0.002
        q = 0.5 * rho * U_BRIDGE ** 2 * B

        model = _seeded_model(input_size=n_kin + (1 if use_ur_context else 0), seed=42)
        model.eval()
        x_scaler = _IdentityXScaler(n_kin)
        y_scaler = _IdentityYScaler()

        rng = np.random.default_rng(0)
        h0, hd0, hdd0 = 0.02, 0.01, -0.001
        # ur_stats=(0.0,1.0) -> ur_scaled = UR_BRIDGE identically; must match
        # what run_coupled_viv computes internally for its generated rows, or
        # the two paths' windows diverge as soon as any generated row enters
        # the seq_len-wide window (both paths' initial warm-up rows may use
        # any consistent placeholder, but new rows generated during the
        # rollout are NOT free parameters -- they must agree).
        ur_scaled = UR_BRIDGE
        init_hist = rng.normal(scale=0.01, size=(seq_len, n_kin + (1 if use_ur_context else 0))).astype(np.float32)
        if use_ur_context:
            init_hist[:, -1] = ur_scaled  # constant Ur-context column

        ref = run_coupled_viv(
            model=model, x_scaler=x_scaler, y_scaler=y_scaler,
            seq_len=seq_len, input_cols=input_cols,
            initial_history=init_hist.copy(), initial_state={"h": h0, "h_dot": hd0, "h_ddot": hdd0},
            m=m, c=c, k=k, rho=rho, U=U_BRIDGE, D=D_BRIDGE, B=B, dt=dt,
            n_steps=n_steps, use_ur_context=use_ur_context, ur_value=UR_BRIDGE,
            ur_stats=(0.0, 1.0), device="cpu", nd_inputs=nd_inputs,
        )

        sc = ScalerConstants.from_sklearn(x_scaler, y_scaler)
        with torch.no_grad():
            out = rollout_chunk(
                model=model,
                window=torch.tensor(init_hist.copy(), dtype=torch.float32).unsqueeze(0),
                h_state=torch.tensor([h0], dtype=torch.float32),
                hdot_state=torch.tensor([hd0], dtype=torch.float32),
                hddot_state=torch.tensor([hdd0], dtype=torch.float32),
                n_steps=n_steps, dt=dt, m=m, c=c, k=k, q=q, U=U_BRIDGE, D=D_BRIDGE,
                input_cols=input_cols, nd_inputs=nd_inputs, sc=sc,
                use_ur_context=use_ur_context, ur_scaled=ur_scaled if use_ur_context else 0.0,
            )

        np.testing.assert_allclose(out["h"][0].numpy(), ref["displacement"], atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(out["hdot"][0].numpy(), ref["velocity"], atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(out["cl_phys"][0].numpy(), ref["CL"], atol=1e-4, rtol=1e-4)

    def test_dimensional_no_context(self):
        self._compare(nd_inputs=False, use_ur_context=False)

    def test_nd_with_ur_context(self):
        self._compare(nd_inputs=True, use_ur_context=True)

    def test_nd_no_context_single_input_col(self):
        self._compare(nd_inputs=True, use_ur_context=False, input_cols=("vel",))


# ── loss functions ──────────────────────────────────────────────────────────

class TestLossCl:
    def test_zero_when_predictions_match_cfd_exactly(self):
        y_mean, y_scale = 0.1, 2.0
        cl_cfd_phys = torch.tensor([[0.1, 0.3, -0.2]])
        cl_scaled_pred = (cl_cfd_phys - y_mean) / y_scale
        loss = loss_cl(cl_scaled_pred, cl_cfd_phys, y_mean, y_scale)
        assert loss.item() == pytest.approx(0.0, abs=1e-8)

    def test_positive_when_predictions_differ(self):
        loss = loss_cl(torch.tensor([[0.0]]), torch.tensor([[1.0]]), 0.0, 1.0)
        assert loss.item() > 0.0


_IDENTITY_X_MEAN = np.zeros(2)
_IDENTITY_X_SCALE = np.ones(2)


class TestLossRoll:
    def test_zero_when_trajectories_match(self):
        h = torch.tensor([[0.1, 0.2, 0.3]])
        hd = torch.tensor([[1.0, 2.0, 3.0]])
        loss = loss_roll(h, hd, h.clone(), hd.clone(), D=D_BRIDGE, U=U_BRIDGE,
                         x_mean=_IDENTITY_X_MEAN, x_scale=_IDENTITY_X_SCALE, disp_idx=0, vel_idx=1)
        assert loss.item() == pytest.approx(0.0, abs=1e-8)

    def test_positive_and_scales_with_identity_standardization(self):
        h_pred = torch.tensor([[0.0]])
        h_cfd = torch.tensor([[D_BRIDGE]])  # one full D of displacement error
        hd = torch.tensor([[0.0]])
        loss = loss_roll(h_pred, hd, h_cfd, hd, D=D_BRIDGE, U=U_BRIDGE,
                         x_mean=_IDENTITY_X_MEAN, x_scale=_IDENTITY_X_SCALE, disp_idx=0, vel_idx=1)
        assert loss.item() == pytest.approx(1.0, rel=1e-6)  # (D/D)^2 = 1 under identity standardization

    def test_nonzero_x_scale_rescales_loss_magnitude(self):
        """The whole point of standardizing L_roll: a smaller x_scale (the
        model's own std for that channel) makes the SAME physical error
        register as a LARGER standardized loss -- this is what keeps
        L_roll comparable in magnitude to L_CL (see the docstring note
        about the 1e-5-vs-1-7 smoke-test discrepancy this fixes)."""
        h_pred = torch.tensor([[0.0]])
        h_cfd = torch.tensor([[D_BRIDGE * 0.01]])  # small physical error
        hd = torch.tensor([[0.0]])
        loss_wide = loss_roll(h_pred, hd, h_cfd, hd, D=D_BRIDGE, U=U_BRIDGE,
                              x_mean=_IDENTITY_X_MEAN, x_scale=np.array([1.0, 1.0]), disp_idx=0, vel_idx=1)
        loss_narrow = loss_roll(h_pred, hd, h_cfd, hd, D=D_BRIDGE, U=U_BRIDGE,
                                x_mean=_IDENTITY_X_MEAN, x_scale=np.array([0.01, 1.0]), disp_idx=0, vel_idx=1)
        assert loss_narrow.item() > loss_wide.item()

# ── gradient flow through Newmark + multiple rollout steps ─────────────────

class TestGradientFlow:
    def test_loss_backward_reaches_model_params_through_chunk(self):
        seq_len, n_steps = 5, 6
        m, c, k, rho, B, dt = 24604.0, 989.39, 99463.88, 1.225, 25.9, 0.002
        q = 0.5 * rho * U_BRIDGE ** 2 * B
        model = _seeded_model(input_size=2, seed=7)
        sc = ScalerConstants.from_sklearn(_IdentityXScaler(2), _IdentityYScaler())

        window = torch.zeros(1, seq_len, 2)
        h0 = torch.tensor([0.01]); hd0 = torch.tensor([0.0]); hdd0 = torch.tensor([0.0])

        out = rollout_chunk(
            model=model, window=window, h_state=h0, hdot_state=hd0, hddot_state=hdd0,
            n_steps=n_steps, dt=dt, m=m, c=c, k=k, q=q, U=U_BRIDGE, D=D_BRIDGE,
            input_cols=["disp", "vel"], nd_inputs=False, sc=sc,
            use_ur_context=False, ur_scaled=0.0,
        )

        cl_cfd = torch.zeros(1, n_steps)
        h_cfd = torch.zeros(1, n_steps)
        hdot_cfd = torch.zeros(1, n_steps)
        total = (loss_cl(out["cl_scaled"], cl_cfd, sc.y_mean, sc.y_scale)
                + loss_roll(out["h"], out["hdot"], h_cfd, hdot_cfd, D=D_BRIDGE, U=U_BRIDGE,
                           x_mean=sc.x_mean, x_scale=sc.x_scale, disp_idx=0, vel_idx=1))
        total.backward()

        grads = [p.grad for p in model.parameters()]
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads), (
            "no finite nonzero gradient reached any model parameter through the rollout chunk"
        )

    def test_detach_severs_graph_but_preserves_values(self):
        seq_len = 4
        model = _seeded_model(input_size=2, seed=3)
        sc = ScalerConstants.from_sklearn(_IdentityXScaler(2), _IdentityYScaler())
        window = torch.zeros(1, seq_len, 2, requires_grad=False)
        h0 = torch.tensor([0.01], requires_grad=True)
        hd0 = torch.tensor([0.0]); hdd0 = torch.tensor([0.0])

        out = rollout_chunk(
            model=model, window=window, h_state=h0, hdot_state=hd0, hddot_state=hdd0,
            n_steps=3, dt=0.002, m=24604.0, c=989.39, k=99463.88, q=100.0,
            U=U_BRIDGE, D=D_BRIDGE, input_cols=["disp", "vel"], nd_inputs=False,
            sc=sc, use_ur_context=False, ur_scaled=0.0,
        )
        assert out["h_state"].requires_grad
        detached_h = out["h_state"].detach()
        detached_window = out["window"].detach()
        assert not detached_h.requires_grad
        assert detached_h.grad_fn is None
        np.testing.assert_allclose(detached_h.numpy(), out["h_state"].detach().numpy())
        np.testing.assert_allclose(detached_window.numpy(), out["window"].detach().numpy())


# ── batch sampling / construction against a small synthetic CFD case ───────

def _synthetic_case_df(case_name="Ur6.7385", n=20000, dt=0.002, release_t=5.0):
    t = np.arange(n) * dt
    f = 0.32
    disp = 0.05 * np.sin(2 * np.pi * f * t) * np.minimum(t / 20.0, 1.0)
    vel = np.gradient(disp, dt)
    cl = 0.3 * np.cos(2 * np.pi * f * t)
    return pd.DataFrame({"case": case_name, "time": t, "disp": disp, "vel": vel,
                        "acc": np.gradient(vel, dt), "cl": cl})


class TestSampleBatchStarts:
    def test_stratified_within_usable_range(self):
        df = _synthetic_case_df()
        rng = np.random.default_rng(0)
        seq_len, max_future = 2500, 3000
        starts = sample_batch_starts(df, release_t=5.0, seq_len=seq_len,
                                     max_future_steps=max_future, batch_size=8, rng=rng)
        release_idx = int(np.searchsorted(df["time"].to_numpy(), 5.0))
        lo, hi = release_idx + seq_len, len(df) - max_future - 1
        assert len(starts) == 8
        assert all(lo <= s <= hi for s in starts)
        assert sorted(starts) == starts or len(set(starts)) > 1  # spread, not all identical

    def test_raises_when_case_too_short(self):
        df = _synthetic_case_df(n=3000)
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="too short"):
            sample_batch_starts(df, release_t=5.0, seq_len=2500, max_future_steps=3000,
                                batch_size=4, rng=rng)


class TestBuildBatchFromCase:
    def test_shapes_and_handoff_alignment(self):
        df = _synthetic_case_df()
        seq_len, max_future = 200, 500
        rng = np.random.default_rng(0)
        starts = sample_batch_starts(df, release_t=5.0, seq_len=seq_len,
                                     max_future_steps=max_future, batch_size=3, rng=rng)
        batch = build_batch_from_case(
            df, "Ur6.7385", starts, seq_len, input_cols=["disp", "vel"],
            x_scaler=_IdentityXScaler(2), nd_inputs=False, D=D_BRIDGE, fn=FN_BRIDGE,
            use_ur_context=False, ur_stats=(0.0, 1.0), max_future_steps=max_future,
            device="cpu",
        )
        assert batch["window"].shape == (3, seq_len, 2)
        assert batch["cfd_h"].shape == (3, max_future)
        assert batch["h_state"].shape == (3,)
        # the FIRST cfd_h sample at each start must equal the true disp there
        for b, s in enumerate(starts):
            assert batch["cfd_h"][b, 0].item() == pytest.approx(df["disp"].iloc[s], abs=1e-5)


class TestExactlyOneNdTransform:
    """Regression coverage for a real bug (found 2026-08-09): feeding
    build_batch_from_case/warmup_history a dataframe that had ALREADY been
    ND-transformed (e.g. by apply_nd_transform, applied upstream for a
    DIFFERENT function's needs) causes the ND transform to be applied a
    SECOND time inside to_model_coords, AND silently corrupts
    initial_state['h']/['h_dot'] -- warmup_history reads those two fields
    straight off the dataframe (ordered['disp'].iloc[...]), bypassing
    to_model_coords entirely, so a pre-transformed 'disp' column is
    returned as the "physical" initial displacement.

    These tests pin the CORRECT, single-application contract: callers must
    always pass RAW (physical) case dataframes to build_batch_from_case /
    build_tf_batch_from_case / warmup_history -- nd_inputs=True makes those
    functions apply the transform internally, exactly once."""

    def test_build_batch_from_case_initial_state_is_raw_physical_not_nd(self):
        """h_state/hdot_state must equal the RAW disp/vel value at the start
        index exactly -- NOT divided by D/U -- regardless of nd_inputs."""
        df = _synthetic_case_df()
        seq_len, max_future = 200, 500
        start = 10000
        batch = build_batch_from_case(
            df, "Ur6.7385", [start], seq_len, input_cols=["disp", "vel"],
            x_scaler=_IdentityXScaler(2), nd_inputs=True, D=D_BRIDGE, fn=FN_BRIDGE,
            use_ur_context=False, ur_stats=(0.0, 1.0), max_future_steps=max_future,
            device="cpu",
        )
        assert batch["h_state"][0].item() == pytest.approx(df["disp"].iloc[start], rel=1e-5)
        assert batch["hdot_state"][0].item() == pytest.approx(df["vel"].iloc[start], rel=1e-5)

    def test_build_batch_from_case_window_matches_single_to_model_coords_application(self):
        """The window's last row (identity scaler) must equal to_model_coords
        applied ONCE to the raw physical kinematics just before the start
        index -- not zero times (nd_inputs silently dropped) and not twice
        (double-transformed)."""
        df = _synthetic_case_df()
        seq_len, max_future = 200, 500
        start = 10000
        U = UR_BRIDGE * FN_BRIDGE * D_BRIDGE
        batch = build_batch_from_case(
            df, "Ur6.7385", [start], seq_len, input_cols=["disp", "vel"],
            x_scaler=_IdentityXScaler(2), nd_inputs=True, D=D_BRIDGE, fn=FN_BRIDGE,
            use_ur_context=False, ur_stats=(0.0, 1.0), max_future_steps=max_future,
            device="cpu",
        )
        raw_last_row = df[["disp", "vel"]].iloc[start - 1: start].to_numpy(dtype=np.float32)
        expected = to_model_coords(raw_last_row, nd_inputs=True, D=D_BRIDGE, U=U,
                                   input_cols=["disp", "vel"])
        np.testing.assert_allclose(batch["window"][0, -1, :].numpy(), expected[0], rtol=1e-4)

    def test_pretransformed_input_would_corrupt_initial_state_by_factor_of_D(self):
        """Pins the exact failure mode: if a caller mistakenly pre-divides
        'disp' by D before calling build_batch_from_case (simulating the
        real bug -- feeding an apply_nd_transform'd dataframe into a
        self-transforming function), h_state comes out D times too small.
        This is a caller-contract test, not a call this project should ever
        make -- it exists so the exact magnitude of the historical bug is
        pinned and any future accidental re-introduction is unambiguous."""
        df = _synthetic_case_df()
        df_pretransformed = df.copy()
        df_pretransformed["disp"] = df_pretransformed["disp"] / D_BRIDGE
        seq_len, max_future = 200, 500
        start = 10000

        batch_raw = build_batch_from_case(
            df, "Ur6.7385", [start], seq_len, input_cols=["disp", "vel"],
            x_scaler=_IdentityXScaler(2), nd_inputs=True, D=D_BRIDGE, fn=FN_BRIDGE,
            use_ur_context=False, ur_stats=(0.0, 1.0), max_future_steps=max_future,
            device="cpu",
        )
        batch_bugged = build_batch_from_case(
            df_pretransformed, "Ur6.7385", [start], seq_len, input_cols=["disp", "vel"],
            x_scaler=_IdentityXScaler(2), nd_inputs=True, D=D_BRIDGE, fn=FN_BRIDGE,
            use_ur_context=False, ur_stats=(0.0, 1.0), max_future_steps=max_future,
            device="cpu",
        )
        ratio = batch_raw["h_state"][0].item() / batch_bugged["h_state"][0].item()
        assert ratio == pytest.approx(D_BRIDGE, rel=1e-4)

    def test_build_tf_batch_from_case_applies_nd_transform_exactly_once(self):
        """build_tf_batch_from_case (the revised Model-A objective's L_TF
        branch) must produce windows identical to a single to_model_coords
        application on the raw kinematics -- this is the exact class of bug
        that made modelA_v2_smoke.py's separate tf_validation() function
        (a DIFFERENT code path, not this one) report nonsense P0 R^2 (~-11)
        by omitting the transform outright; this test pins the opposite
        failure mode (double-application) for the function actually used
        inside training."""
        df = _synthetic_case_df()
        seq_len, n_steps = 200, 50
        start = 10000
        U = UR_BRIDGE * FN_BRIDGE * D_BRIDGE

        tf_batch = build_tf_batch_from_case(
            df, "Ur6.7385", start, n_steps, seq_len, input_cols=["disp", "vel"],
            x_scaler=_IdentityXScaler(2), nd_inputs=True, D=D_BRIDGE, fn=FN_BRIDGE,
            use_ur_context=False, ur_stats=(0.0, 1.0), device="cpu",
        )
        raw_last_row = df[["disp", "vel"]].iloc[start - 1: start].to_numpy(dtype=np.float32)
        expected = to_model_coords(raw_last_row, nd_inputs=True, D=D_BRIDGE, U=U,
                                   input_cols=["disp", "vel"])
        np.testing.assert_allclose(tf_batch["x"][0, -1, :].numpy(), expected[0], rtol=1e-4)
        assert tf_batch["cl_cfd"][0].item() == pytest.approx(df["cl"].iloc[start], rel=1e-5)


class TestWarmupHistoryExactlyOneNdTransform:
    """Same invariant, tested directly on coupled_inference.warmup_history
    (the function whose 'initial_state["h"]/["h_dot"] read straight off the
    dataframe, bypassing to_model_coords' behavior is the actual root
    cause of the historical bug)."""

    def test_initial_state_is_raw_physical(self):
        df = _synthetic_case_df()
        seq_len = 200
        U = UR_BRIDGE * FN_BRIDGE * D_BRIDGE
        _, init_state, _, handoff_idx = warmup_history(
            df, release_t=5.0, seq_len=seq_len, input_cols=["disp", "vel"],
            x_scaler=_IdentityXScaler(2), nd_inputs=True, D=D_BRIDGE, U=U,
            use_ur_context=False, ur_value=UR_BRIDGE, ur_stats=(0.0, 1.0),
            handoff_offset_steps=0,
        )
        assert init_state["h"] == pytest.approx(df["disp"].iloc[handoff_idx], rel=1e-5)
        assert init_state["h_dot"] == pytest.approx(df["vel"].iloc[handoff_idx], rel=1e-5)

    def test_pretransformed_disp_corrupts_initial_state_by_exactly_D(self):
        """The exact bug found in audit_v1.py's Section 2 (component-gradient
        audit): feeding warmup_history a case_df whose 'disp' column was
        already divided by D (e.g. via apply_nd_transform, applied upstream
        for a DIFFERENT function's needs) silently changes init_state['h']
        by a factor of D, because that field is read directly off the
        dataframe rather than through to_model_coords."""
        df = _synthetic_case_df()
        df_pretransformed = df.copy()
        df_pretransformed["disp"] = df_pretransformed["disp"] / D_BRIDGE
        seq_len = 200
        U = UR_BRIDGE * FN_BRIDGE * D_BRIDGE

        _, init_raw, _, _ = warmup_history(
            df, release_t=5.0, seq_len=seq_len, input_cols=["disp", "vel"],
            x_scaler=_IdentityXScaler(2), nd_inputs=True, D=D_BRIDGE, U=U,
            use_ur_context=False, ur_value=UR_BRIDGE, ur_stats=(0.0, 1.0),
            handoff_offset_steps=0,
        )
        _, init_bugged, _, _ = warmup_history(
            df_pretransformed, release_t=5.0, seq_len=seq_len, input_cols=["disp", "vel"],
            x_scaler=_IdentityXScaler(2), nd_inputs=True, D=D_BRIDGE, U=U,
            use_ur_context=False, ur_value=UR_BRIDGE, ur_stats=(0.0, 1.0),
            handoff_offset_steps=0,
        )
        assert init_raw["h"] / init_bugged["h"] == pytest.approx(D_BRIDGE, rel=1e-4)
