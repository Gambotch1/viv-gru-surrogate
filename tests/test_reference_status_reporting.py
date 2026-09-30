import numpy as np
import pandas as pd
import pytest

from viv_analysis.reference_quality import compute_non_lco_summary, build_status_aware_report


def _write_npz(path, t, h, cl, h_cfd, cl_cfd):
    np.savez(path, t=t, h=h, cl=cl, h_cfd=h_cfd, cl_cfd=cl_cfd, D=7.42, Ur=6.0)


def test_non_lco_summary_identical_signals_gives_ratio_one(tmp_path):
    t = np.linspace(0, 10, 5000)
    h = np.sin(2 * np.pi * 0.3 * t) * 0.1
    cl = np.sin(2 * np.pi * 0.3 * t) * 0.2
    p = tmp_path / "case.npz"
    _write_npz(p, t, h, cl, h.copy(), cl.copy())

    out = compute_non_lco_summary(str(p))
    assert out["unscored"] is False
    assert out["mean_rms_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert out["rms_agreement"] is True
    assert out["cumulative_energy_ratio"] == pytest.approx(1.0, abs=1e-3)


def test_non_lco_summary_scaled_surrogate_detected(tmp_path):
    t = np.linspace(0, 10, 5000)
    h_cfd = np.sin(2 * np.pi * 0.3 * t) * 0.1
    h_sur = h_cfd * 3.0  # surrogate 3x too big
    cl = np.sin(2 * np.pi * 0.3 * t) * 0.2
    p = tmp_path / "case.npz"
    _write_npz(p, t, h_sur, cl, h_cfd, cl.copy())

    out = compute_non_lco_summary(str(p))
    assert out["mean_rms_ratio"] == pytest.approx(3.0, rel=1e-2)
    assert out["rms_agreement"] is False


def test_non_lco_summary_missing_cfd_is_unscored(tmp_path):
    t = np.linspace(0, 10, 100)
    h = np.sin(t)
    cl = np.sin(t)
    p = tmp_path / "case.npz"
    np.savez(p, t=t, h=h, cl=cl, D=7.42, Ur=6.0)  # no h_cfd/cl_cfd

    out = compute_non_lco_summary(str(p))
    assert out["unscored"] is True


def test_status_aware_report_routes_by_category(tmp_path):
    # Minimal reference_status table: one of each interesting category.
    ref = pd.DataFrame([
        dict(Ur=6.0, case="Ur6.0", reference_status="settled_lco",
             reference_status_reason="ok"),
        dict(Ur=7.0, case="Ur7.0", reference_status="statistically_stationary_les",
             reference_status_reason="ok"),
        dict(Ur=8.0, case="Ur8.0", reference_status="insufficient_duration",
             reference_status_reason="too short"),
        dict(Ur=9.0, case="Ur9.0", reference_status="numerically_suspect",
             reference_status_reason="time gap detected"),
    ])
    ref_csv = tmp_path / "ref.csv"
    ref.to_csv(ref_csv, index=False)

    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    sweep = pd.DataFrame([
        dict(Ur=6.0, mymodel_stability_label="stationary_lco",
             mymodel_A_star_rel_error=0.05, mymodel_pass=True),
        dict(Ur=7.0, mymodel_stability_label="decay_to_rest",
             mymodel_A_star_rel_error=-0.9, mymodel_pass=False),
        dict(Ur=8.0, mymodel_stability_label="divergence",
             mymodel_A_star_rel_error=float("nan"), mymodel_pass=False),
        dict(Ur=9.0, mymodel_stability_label="decay_to_rest",
             mymodel_A_star_rel_error=-0.9, mymodel_pass=False),
    ])
    sweep.to_csv(sweep_dir / "sweep_results.csv", index=False)

    t = np.linspace(0, 10, 3000)
    h = np.sin(2 * np.pi * 0.3 * t) * 0.05
    cl = np.sin(2 * np.pi * 0.3 * t) * 0.2
    _write_npz(sweep_dir / "coupled_bridge_Ur7.0_mymodel.npz", t, h, cl, h.copy(), cl.copy())

    report = build_status_aware_report(str(sweep_dir), "mymodel", reference_status_csv=str(ref_csv))
    report = report.set_index("Ur")

    assert report.loc[6.0, "scoring_method"] == "lco_gate"
    assert report.loc[6.0, "unscored"] == False

    assert report.loc[7.0, "scoring_method"] == "non_lco_block_energy"
    assert report.loc[7.0, "unscored"] == False

    assert report.loc[8.0, "unscored"] == True
    assert report.loc[8.0, "scoring_method"] == "none"

    assert report.loc[9.0, "unscored"] == True
    assert "gap" in report.loc[9.0, "unscored_reason"]
