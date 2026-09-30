"""
Validation tests for closed_loop_metrics.py against synthetic signals with
known ground truth:

  h(t)  = A0 * sin(2*pi*f0*t)                (LCO displacement)
  cl(t) = F1 * sin(2*pi*f0*t + phi)          (lift, phase-shifted by phi)

For which the analytic results are:
  A_star = A0 / D,  f_osc = f0
  E_f = oint C_L dh = pi * A0 * F1 * sin(phi)   (derived in closed_loop_metrics
                                                  docstring / thesis Sec 2.4.2)
  F1 recovered by the single-bin DFT
  phi recovered via arcsin(E_f / (pi * A_phys * F1))
"""
from __future__ import annotations

import numpy as np
import pytest

from viv_analysis.closed_loop_metrics import (
    cycles_from_displacement,
    cycle_amplitude_and_frequency,
    cycle_average_energy,  # noqa: F401 -- used directly in TestEnergyVariance
    single_bin_dft_amplitude,
    energy_and_phase,
    classify_stability,
    compute_signal_metrics,
    compare_surrogate_vs_cfd,
    compute_case_metrics,
)

F0 = 0.32
DT = 0.005
DURATION = 80.0
D_REF = 7.42
A0 = 0.05


def _t():
    return np.arange(0, DURATION, DT)


class TestCycleAmplitudeAndFrequency:
    def test_pure_sinusoid_recovers_amplitude_and_frequency(self):
        t = _t()
        h = A0 * np.sin(2 * np.pi * F0 * t)
        out = cycle_amplitude_and_frequency(t, h, D_REF, window_frac=0.5)
        assert out["A_phys"] == pytest.approx(A0, rel=1e-3)
        assert out["A_star"] == pytest.approx(A0 / D_REF, rel=1e-3)
        assert out["f_osc"] == pytest.approx(F0, rel=1e-3)
        assert out["n_cycles"] > 5

    def test_cycle_based_differs_from_mean_abs_for_asymmetric_signal(self):
        """The whole point of cycle-based half-range: for an asymmetric
        (non-sinusoidal) waveform, half-range != mean(|h|)."""
        t = _t()
        # Asymmetric: a sinusoid riding on a rectified second harmonic bump
        h = A0 * np.sin(2 * np.pi * F0 * t) + 0.3 * A0 * np.sin(2 * np.pi * F0 * t) ** 2
        out = cycle_amplitude_and_frequency(t, h, D_REF, window_frac=0.5)
        mean_abs_amplitude = float(np.mean(np.abs(h[len(h) // 2:])))
        assert out["A_phys"] != pytest.approx(mean_abs_amplitude, rel=0.05)

    def test_no_cycles_returns_nan(self):
        t = np.array([0.0, 0.1])
        h = np.array([0.0, 0.01])
        out = cycle_amplitude_and_frequency(t, h, D_REF, window_frac=0.5)
        assert out["n_cycles"] == 0
        assert np.isnan(out["A_star"])


class TestSingleBinDftAmplitude:
    def test_recovers_known_amplitude(self):
        t = _t()
        F1 = 0.3
        cl = F1 * np.sin(2 * np.pi * F0 * t + 0.4)
        out = single_bin_dft_amplitude(t[len(t) // 2:], cl[len(t) // 2:], F0)
        assert out == pytest.approx(F1, rel=0.03)

    def test_off_frequency_gives_much_smaller_amplitude(self):
        t = _t()
        cl = 0.3 * np.sin(2 * np.pi * F0 * t)
        at_f0 = single_bin_dft_amplitude(t, cl, F0)
        at_2f0 = single_bin_dft_amplitude(t, cl, 2 * F0)
        assert at_2f0 < 0.1 * at_f0


class TestEnergyAndPhase:
    @pytest.mark.parametrize("phi_true", [0.0, 0.35, -0.4, 1.0])
    def test_energy_matches_analytic_pi_A_F1_sin_phi(self, phi_true):
        t = _t()
        F1 = 0.3
        h = A0 * np.sin(2 * np.pi * F0 * t)
        cl = F1 * np.sin(2 * np.pi * F0 * t + phi_true)
        out = energy_and_phase(t, h, cl, D_REF, window_frac=0.5)
        expected_Ef = np.pi * A0 * F1 * np.sin(phi_true)
        assert out["E_f"] == pytest.approx(expected_Ef, abs=2e-4)

    def test_phase_recovered_within_arcsin_range(self):
        """For phi in (-pi/2, pi/2), arcsin(sin(phi)) = phi directly."""
        t = _t()
        F1 = 0.3
        phi_true = 0.35
        h = A0 * np.sin(2 * np.pi * F0 * t)
        cl = F1 * np.sin(2 * np.pi * F0 * t + phi_true)
        out = energy_and_phase(t, h, cl, D_REF, window_frac=0.5)
        assert out["phi_rad"] == pytest.approx(phi_true, abs=0.03)

    def test_zero_phase_gives_zero_energy(self):
        """CL in phase with h (phi=0): purely elastic, no net work per cycle."""
        t = _t()
        h = A0 * np.sin(2 * np.pi * F0 * t)
        cl = 0.3 * np.sin(2 * np.pi * F0 * t)  # phi=0
        out = energy_and_phase(t, h, cl, D_REF, window_frac=0.5)
        assert out["E_f"] == pytest.approx(0.0, abs=2e-4)

    def test_quadrature_phase_gives_maximal_energy(self):
        """phi=pi/2: CL in phase with h_dot -- maximal energy transfer,
        sin(phi)=1 exactly."""
        t = _t()
        F1 = 0.3
        h = A0 * np.sin(2 * np.pi * F0 * t)
        cl = F1 * np.sin(2 * np.pi * F0 * t + np.pi / 2)
        out = energy_and_phase(t, h, cl, D_REF, window_frac=0.5)
        assert out["E_f"] == pytest.approx(np.pi * A0 * F1, rel=0.02)


class TestEnergyVariance:
    """Cycle-to-cycle spread of E_f -- phi (derived from the mean E_f) is
    only as precise as this spread allows."""

    def test_clean_limit_cycle_has_near_zero_spread(self):
        """A perfectly periodic h/cl (constant amplitude AND phase) should
        have essentially identical energy every cycle."""
        t = _t()
        h = A0 * np.sin(2 * np.pi * F0 * t)
        cl = 0.3 * np.sin(2 * np.pi * F0 * t + 0.3)
        out = energy_and_phase(t, h, cl, D_REF, window_frac=0.5)
        assert out["n_cycles"] > 10
        assert out["E_f_cv"] < 0.01  # <1% cycle-to-cycle variation

    def test_phase_wander_produces_meaningful_spread(self):
        """CL whose phase relative to h drifts cycle-to-cycle (mimicking a
        phase-randomised residual) must show up as real E_f spread, not be
        averaged away silently."""
        t = _t()
        h = A0 * np.sin(2 * np.pi * F0 * t)
        # phase modulated slowly relative to the oscillation itself
        phase_wander = 0.6 * np.sin(2 * np.pi * (F0 / 8) * t)
        cl = 0.3 * np.sin(2 * np.pi * F0 * t + phase_wander)
        out = energy_and_phase(t, h, cl, D_REF, window_frac=0.5)
        assert out["n_cycles"] > 10
        assert out["E_f_cv"] > 0.05  # meaningfully more spread than the clean case

    def test_single_cycle_reports_zero_std_not_nan(self):
        t = np.arange(0, 4.0, 0.005)  # ~1.3 periods at F0
        h = A0 * np.sin(2 * np.pi * F0 * t)
        cl = 0.3 * np.sin(2 * np.pi * F0 * t + 0.2)
        energy = cycle_average_energy(t, h, cl, window_frac=1.0, D=D_REF)
        if energy["n_cycles"] == 1:
            assert energy["E_f_std"] == 0.0
            assert not np.isnan(energy["E_f_std"])

    def test_no_cycles_reports_nan_std(self):
        t = _t()
        h = np.full_like(t, 0.028)  # flat, no oscillation
        cl = np.full_like(t, 0.01)
        energy = cycle_average_energy(t, h, cl, window_frac=0.5, D=D_REF)
        assert energy["n_cycles"] == 0
        assert np.isnan(energy["E_f_std"])
        assert np.isnan(energy["E_f_cv"])


class TestClassifyStability:
    def test_constant_amplitude_is_stationary(self):
        t = _t()
        h = A0 * np.sin(2 * np.pi * F0 * t)
        out = classify_stability(t, h, window_frac=0.5)
        assert out["label"] == "stationary_lco"

    def test_growing_envelope_is_divergence(self):
        t = _t()
        b_true = 0.01
        h = A0 * np.exp(b_true * t) * np.sin(2 * np.pi * F0 * t)
        out = classify_stability(t, h, window_frac=0.5)
        assert out["label"] == "divergence"
        assert out["growth_rate_per_s"] == pytest.approx(b_true, rel=0.1)
        assert out["fractional_envelope_change"] > 0

    def test_decaying_envelope_is_decay_to_rest(self):
        t = _t()
        b_true = 0.01
        h = A0 * np.exp(-b_true * t) * np.sin(2 * np.pi * F0 * t)
        out = classify_stability(t, h, window_frac=0.5)
        assert out["label"] == "decay_to_rest"
        assert out["growth_rate_per_s"] == pytest.approx(-b_true, rel=0.1)
        assert out["fractional_envelope_change"] < 0

    def test_threshold_controls_sensitivity(self):
        """A slow drift that's 'stationary' under a loose threshold becomes
        'divergence'/'decay_to_rest' under a tight one."""
        t = _t()
        b_small = 0.0015  # small growth rate -> small fractional change over the window
        h = A0 * np.exp(b_small * t) * np.sin(2 * np.pi * F0 * t)
        loose = classify_stability(t, h, window_frac=0.5, stationary_frac_threshold=0.5)
        tight = classify_stability(t, h, window_frac=0.5, stationary_frac_threshold=0.01)
        assert loose["label"] == "stationary_lco"
        assert tight["label"] == "divergence"


class TestSurrogateVsCfdComparison:
    def test_identical_signals_give_zero_errors(self):
        t = _t()
        h = A0 * np.sin(2 * np.pi * F0 * t)
        cl = 0.3 * np.sin(2 * np.pi * F0 * t + 0.2)
        m1 = compute_signal_metrics(t, h, cl, D_REF, window_frac=0.5)
        m2 = compute_signal_metrics(t, h, cl, D_REF, window_frac=0.5)
        cmp = compare_surrogate_vs_cfd(m1, m2)
        assert cmp["A_star_error"] == pytest.approx(0.0, abs=1e-9)
        assert cmp["f_osc_error"] == pytest.approx(0.0, abs=1e-9)
        assert cmp["energy_error_eps_E"] == pytest.approx(0.0, abs=1e-6)
        assert cmp["phase_error_rad"] == pytest.approx(0.0, abs=1e-9)

    def test_amplitude_mismatch_detected(self):
        t = _t()
        h_surr = 1.2 * A0 * np.sin(2 * np.pi * F0 * t)
        h_cfd = A0 * np.sin(2 * np.pi * F0 * t)
        cl = 0.3 * np.sin(2 * np.pi * F0 * t + 0.2)
        m_surr = compute_signal_metrics(t, h_surr, cl, D_REF, window_frac=0.5)
        m_cfd = compute_signal_metrics(t, h_cfd, cl, D_REF, window_frac=0.5)
        cmp = compare_surrogate_vs_cfd(m_surr, m_cfd)
        assert cmp["A_star_rel_error"] == pytest.approx(0.2, rel=0.05)


class TestComputeCaseMetricsFromNpz:
    def test_full_pipeline_with_synthetic_npz(self, tmp_path):
        t = _t()
        h = A0 * np.sin(2 * np.pi * F0 * t)
        cl = 0.3 * np.sin(2 * np.pi * F0 * t + 0.2)
        h_cfd = 0.9 * A0 * np.sin(2 * np.pi * F0 * t)
        cl_cfd = 0.28 * np.sin(2 * np.pi * F0 * t + 0.15)

        npz_path = tmp_path / "coupled_test_Ur6.0.npz"
        np.savez(npz_path, t=t, h=h, cl=cl, h_cfd=h_cfd, cl_cfd=cl_cfd,
                D=D_REF, Ur=6.0)

        row = compute_case_metrics(npz_path, window_frac=0.5)
        assert row["Ur"] == pytest.approx(6.0)
        assert "surrogate_A_star" in row
        assert "cfd_A_star" in row
        assert "energy_error_eps_E" in row
        assert "phase_error_rad" in row
        assert row["surrogate_label"] == "stationary_lco"
        assert row["cfd_label"] == "stationary_lco"

    def test_missing_cfd_columns_flagged(self, tmp_path):
        t = _t()
        h = A0 * np.sin(2 * np.pi * F0 * t)
        cl = 0.3 * np.sin(2 * np.pi * F0 * t + 0.2)
        npz_path = tmp_path / "coupled_legacy_Ur6.0.npz"
        np.savez(npz_path, t=t, h=h, cl=cl, D=D_REF, Ur=6.0)

        row = compute_case_metrics(npz_path, window_frac=0.5)
        assert row.get("_missing_cfd_cl") is True
        assert "surrogate_A_star" in row
        assert "cfd_A_star" not in row


class TestFlatSignalGuard:
    """Real bug found against actual coupled-inference output: a response
    decayed to a fixed non-zero offset has float-noise-level range
    (~1e-9), and np.gradient of pure noise crosses zero constantly,
    producing spurious "cycles" and nonsense frequencies (58 Hz was
    observed on a real bridge case settled at equilibrium)."""

    def test_numerically_flat_signal_reports_no_cycles_not_garbage(self):
        t = _t()
        rng = np.random.default_rng(0)
        # settled at a fixed offset, only float-level noise -- range ~1e-9
        h = 0.028 + rng.normal(0, 1e-10, size=len(t))
        cl = 0.01 + rng.normal(0, 1e-10, size=len(t))
        out = cycle_amplitude_and_frequency(t, h, D_REF, window_frac=0.5)
        assert out["n_cycles"] == 0
        assert np.isnan(out["f_osc"])

        ep = energy_and_phase(t, h, cl, D_REF, window_frac=0.5)
        assert np.isnan(ep["f_osc"])
        assert np.isnan(ep["phi_rad"])

    def test_flat_signal_does_not_crash_classify_stability(self):
        t = _t()
        h = np.full_like(t, 0.028)
        out = classify_stability(t, h, window_frac=0.5)
        assert out["label"] in ("stationary_lco", "insufficient_data")

    def test_genuine_small_but_real_oscillation_still_detected(self):
        """The guard must not suppress a real, just small-amplitude LCO --
        only genuine float-noise-level flatness."""
        t = _t()
        A_small = 0.005  # A_star ~= 6.7e-4, ~7x above DEFAULT_A_STAR_NOISE_FLOOR (1e-4)
        h = A_small * np.sin(2 * np.pi * F0 * t)
        out = cycle_amplitude_and_frequency(t, h, D_REF, window_frac=0.5)
        assert out["n_cycles"] > 5
        assert out["f_osc"] == pytest.approx(F0, rel=1e-2)

    def test_large_dc_offset_with_tiny_residual_wobble_is_not_an_oscillation(self):
        """The actual real-world failure this guard was added for: a bridge
        response decayed to a large non-zero equilibrium offset (~0.08 m,
        NOT near zero) with ~1e-4 m of residual numerical wobble on top.
        A self-relative threshold (range vs h's own max) is fooled by the
        huge DC offset; comparing to D catches it -- 12.6 Hz was observed
        for real here before the D-based floor was added."""
        t = _t()
        rng = np.random.default_rng(1)
        dc_offset = 0.08
        wobble_amplitude = 1e-4  # (wobble/2)/D_REF ~= 6.7e-6, well under the 1e-4 floor
        h = dc_offset + wobble_amplitude * np.sin(2 * np.pi * 5.0 * t) \
            + rng.normal(0, wobble_amplitude * 0.05, size=len(t))
        out = cycle_amplitude_and_frequency(t, h, D_REF, window_frac=0.5)
        assert out["n_cycles"] == 0
        assert np.isnan(out["f_osc"])

    def test_d_based_floor_not_applied_without_D(self):
        """Without D, only the self-relative float-noise guard applies --
        documents that a_star_floor is opt-in via D, not a silent global
        amplitude cutoff."""
        t = _t()
        dc_offset = 0.08
        wobble_amplitude = 1e-4
        h = dc_offset + wobble_amplitude * np.sin(2 * np.pi * 5.0 * t)
        cycles_with_D = cycles_from_displacement(t, h, D=D_REF)
        cycles_without_D = cycles_from_displacement(t, h, D=None)
        assert cycles_with_D == []
        assert len(cycles_without_D) > 0


class TestMismatchedCfdWindowLength:
    """Real bug found against actual coupled-inference output: the
    surrogate free-runs for the full requested duration, but h_cfd/cl_cfd
    are capped by however long that Ur case's CFD run actually was --
    shorter than t/h/cl for high-Ur cases in the bridge campaign. Used to
    crash inside np.gradient with a length-mismatch error."""

    def test_short_cfd_reference_does_not_crash(self, tmp_path):
        t = _t()
        h = A0 * np.sin(2 * np.pi * F0 * t)
        cl = 0.3 * np.sin(2 * np.pi * F0 * t + 0.2)
        n_cfd = len(t) // 2  # CFD reference much shorter than the surrogate
        h_cfd = A0 * np.sin(2 * np.pi * F0 * t[:n_cfd])
        cl_cfd = 0.3 * np.sin(2 * np.pi * F0 * t[:n_cfd] + 0.2)

        npz_path = tmp_path / "coupled_short_cfd_Ur6.0.npz"
        np.savez(npz_path, t=t, h=h, cl=cl, h_cfd=h_cfd, cl_cfd=cl_cfd, D=D_REF, Ur=6.0)

        row = compute_case_metrics(npz_path, window_frac=0.5)
        assert row["cfd_window_truncated_to_n"] == n_cfd
        assert "cfd_A_star" in row
        assert np.isfinite(row["cfd_A_star"])
        # surrogate_* columns still reflect the FULL surrogate trajectory,
        # not truncated to the CFD window
        assert row["surrogate_n_cycles"] > row["cfd_n_cycles"]

    def test_equal_length_leaves_no_truncation_flag(self, tmp_path):
        t = _t()
        h = A0 * np.sin(2 * np.pi * F0 * t)
        cl = 0.3 * np.sin(2 * np.pi * F0 * t + 0.2)
        npz_path = tmp_path / "coupled_equal_Ur6.0.npz"
        np.savez(npz_path, t=t, h=h, cl=cl, h_cfd=h, cl_cfd=cl, D=D_REF, Ur=6.0)

        row = compute_case_metrics(npz_path, window_frac=0.5)
        assert "cfd_window_truncated_to_n" not in row


class TestNoiseTailDoesNotDilutePerCycleAverage:
    """Real bug: a window that starts with a few REAL decaying-transient
    cycles and settles into noise for the rest passes the window-level
    gate (there IS real signal somewhere), but thousands of noise-driven
    micro-cycles from the settled tail then dominate a plain mean by sheer
    count. Observed for real: 7819 "cycles" averaging to f_osc~23 Hz, when
    only the first 2 cycles (at the true ~0.29 Hz) were real."""

    def test_real_transient_then_noise_tail_ignores_noise_cycles(self):
        t = _t()
        f_real = 0.29
        A_real = 0.005          # well above the noise floor
        A_noise = 2e-6          # deep in float-noise territory
        n = len(t)
        transient_end = n // 4  # real oscillation only in the first quarter

        h = np.empty(n)
        h[:transient_end] = A_real * np.sin(2 * np.pi * f_real * t[:transient_end])
        rng = np.random.default_rng(2)
        h[transient_end:] = rng.normal(0, A_noise, size=n - transient_end)

        out = cycle_amplitude_and_frequency(t, h, D_REF, window_frac=1.0)
        # only the real cycles should survive filtering
        assert out["f_osc"] == pytest.approx(f_real, rel=0.05)
        assert out["A_star"] == pytest.approx(A_real / D_REF, rel=0.1)
        assert out["n_cycles"] < 20  # nowhere near the thousands a noise tail would add
