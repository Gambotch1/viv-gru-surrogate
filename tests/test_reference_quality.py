import numpy as np
import pytest

from viv_analysis.reference_quality import (
    classify_reference_status,
    _numerically_suspect_reason,
    REFERENCE_QUALITY_CRITERIA,
)


def _base_row(**overrides):
    row = dict(
        n_tau_s_post_release=6.0, n_cycles_post_release=80,
        block1_h_rms=0.1, block2_h_rms=0.2, block3_h_rms=0.2,
        freq_cv_across_blocks=0.01, cfd_label="stationary_lco",
        numerically_suspect_reason=None,
    )
    row.update(overrides)
    return row


def test_numerically_suspect_wins_over_everything():
    row = _base_row(numerically_suspect_reason="fake reason",
                    n_tau_s_post_release=0.1, n_cycles_post_release=1)
    status, reason = classify_reference_status(row)
    assert status == "numerically_suspect"
    assert reason == "fake reason"


def test_insufficient_duration_checked_before_settled_lco():
    row = _base_row(n_tau_s_post_release=1.0, n_cycles_post_release=5)
    status, _ = classify_reference_status(row)
    assert status == "insufficient_duration"


def test_settled_lco_requires_block_and_freq_agreement_not_slope_alone():
    # slope says stationary, but blocks disagree -> must NOT be settled_lco
    row = _base_row(cfd_label="stationary_lco", block2_h_rms=0.1, block3_h_rms=0.5)
    status, _ = classify_reference_status(row)
    assert status != "settled_lco"

    # slope AND blocks AND freq all agree -> settled_lco
    row2 = _base_row(cfd_label="stationary_lco", block2_h_rms=0.2, block3_h_rms=0.21,
                     freq_cv_across_blocks=0.02)
    status2, _ = classify_reference_status(row2)
    assert status2 == "settled_lco"


def test_statistically_stationary_les_catches_beating_but_stable():
    # slope says divergence (beating envelope), but block RMS is still within
    # the wider stationary band -- must NOT fall through to transient.
    row = _base_row(cfd_label="divergence", block2_h_rms=0.2, block3_h_rms=0.3,
                    freq_cv_across_blocks=0.3)
    status, _ = classify_reference_status(row)
    assert status == "statistically_stationary_les"


def test_transient_when_blocks_genuinely_drift():
    row = _base_row(cfd_label="divergence", block2_h_rms=0.1, block3_h_rms=1.0)
    status, _ = classify_reference_status(row)
    assert status == "transient_or_slowly_evolving"


def test_categories_are_mutually_exclusive_by_construction():
    # classify_reference_status always returns exactly one of the five.
    valid = {"settled_lco", "statistically_stationary_les",
             "transient_or_slowly_evolving", "insufficient_duration",
             "numerically_suspect"}
    for row in [_base_row(), _base_row(n_tau_s_post_release=0.5),
                _base_row(numerically_suspect_reason="x"),
                _base_row(cfd_label="decay_to_rest", block2_h_rms=0.1, block3_h_rms=1.0)]:
        status, _ = classify_reference_status(row)
        assert status in valid


def test_numerically_suspect_detects_time_axis_gap():
    dt = 0.002
    t = np.arange(0, 2, dt)  # 1000 samples
    t = np.concatenate([t[:500], t[500:] + 50.0])  # inject a 50s gap partway through
    h = np.sin(2 * np.pi * 0.3 * t) * 0.1
    v = np.gradient(h, t)
    cl = np.sin(2 * np.pi * 0.3 * t) * 0.2
    reason = _numerically_suspect_reason(t, h, v, cl)
    assert reason is not None
    assert "gap" in reason


def test_numerically_suspect_none_for_clean_signal():
    dt = 0.002
    t = np.arange(0, 10, dt)
    h = np.sin(2 * np.pi * 0.3 * t) * 0.1
    v = np.gradient(h, t)
    cl = np.sin(2 * np.pi * 0.3 * t) * 0.2
    assert _numerically_suspect_reason(t, h, v, cl) is None


def test_numerically_suspect_detects_nonfinite():
    t = np.arange(0, 1, 0.01)
    h = np.zeros_like(t)
    h[5] = np.nan
    v = np.zeros_like(t)
    cl = np.zeros_like(t)
    reason = _numerically_suspect_reason(t, h, v, cl)
    assert reason is not None
    assert "non-finite" in reason


def test_criteria_are_versioned():
    assert "version" in REFERENCE_QUALITY_CRITERIA
    assert isinstance(REFERENCE_QUALITY_CRITERIA["version"], int)
