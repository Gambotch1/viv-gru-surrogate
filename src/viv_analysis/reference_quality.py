#!/usr/bin/env python3
"""Classify how usable each bridge CFD case is as a reference (thesis Sec. 4.7, 6.2).

Only U = 16 m/s reached a clearly stationary response; other cases are
slowly evolving or beating. They cannot all be scored with one steady amplitude.
Each case gets one status (criteria in REFERENCE_QUALITY_CRITERIA):
    settled_lco                    steady limit cycle: amplitude, frequency, phase and energy are compared
    statistically_stationary_les   stable but modulated: statistics only, no single amplitude
    transient_or_slowly_evolving   still changing: finite-horizon comparison only
    insufficient_duration          too short after release: not scored
    numerically_suspect            non-finite values or jumps in the record: not scored

Run as a script to write results/bridge_reference_status.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from viv_analysis.closed_loop_metrics import classify_stability, cycles_from_displacement
from viv_analysis.config import config, bridge_structural_params
from viv_analysis.preprocess import load_bridge_df_cached
from viv_analysis.utils import parse_ur_label, segment_by_time_gaps

REFERENCE_QUALITY_CRITERIA_VERSION = 1

REFERENCE_QUALITY_CRITERIA = {
    "version": REFERENCE_QUALITY_CRITERIA_VERSION,
    "min_n_tau_s_post_release": 3.0,
    "min_n_cycles_post_release": 10,
    "rms_ratio_settled_lco_band": (0.7, 1.43),
    "freq_cv_settled_lco_max": 0.15,
    "rms_ratio_stationary_les_band": (0.5, 2.0),
    "discontinuity_jump_factor": 20.0,
    "cfd_slope_classifier": "closed_loop_metrics.classify_stability(window_frac=0.5, stationary_frac_threshold=0.10)",
    "notes": (
        "insufficient_duration is checked FIRST and overrides everything else. "
        "numerically_suspect is checked second and requires direct evidence "
        "(non-finite values or an outlier single-step jump), not judgment. "
        "settled_lco requires the slope-based CFD label to ALSO agree with "
        "blockwise RMS-ratio and frequency-CV stability -- never assigned from "
        "the Hilbert-envelope slope alone. statistically_stationary_les is the "
        "explicit fallback for broadband/modulated-but-stable responses that "
        "fail the tighter settled_lco bands. transient_or_slowly_evolving is "
        "the residual category for anything with sufficient duration that "
        "still shows a systematic blockwise trend. "
        "RMS-ratio uses block2/block3 (NOT block1/block3): block1 is, by "
        "construction, the first third of the post-release window and always "
        "contains genuine release-transient growth for every case (verified "
        "directly -- Ur=6.7385's cleanest-known case has block1->3 ratio 1.68 "
        "purely from that transient, vs. block2->3 ratio 1.04). Comparing "
        "block1 to block3 would flag every case's real transient as "
        "'not settled'; block2-vs-block3 asks the right question, whether the "
        "back half has plateaued, and mirrors classify_stability's own "
        "window_frac=0.5 (last-half) convention."
    ),
}

_EVAL_PROCEDURE_BY_STATUS = {
    "settled_lco": "amplitude, frequency, phase, and cycle-energy balance (closed_loop_metrics full suite)",
    "statistically_stationary_les": "blockwise RMS, PSD, probability distributions, cross-spectral quantities, energy statistics -- NOT single-cycle amplitude/phase",
    "transient_or_slowly_evolving": "finite-horizon envelope evolution, trajectory statistics, cumulative energy transfer -- no steady-amplitude comparison",
    "insufficient_duration": "none -- excluded from steady-amplitude/LCO pass-fail scoring entirely",
    "numerically_suspect": "none -- excluded from quantitative model scoring; numerical reason reported",
}


def _dominant_freq(t: np.ndarray, x: np.ndarray) -> float:
    """Frequency of the largest spectral peak (Hann window, mean removed)."""
    if len(t) < 8:
        return float("nan")
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return float("nan")
    x = x - np.mean(x)
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=dt)
    mag = np.abs(np.fft.rfft(x * np.hanning(n)))
    if len(mag) < 2:
        return float("nan")
    mag[0] = 0.0
    return float(freqs[int(np.argmax(mag))])


def _numerically_suspect_reason(t: np.ndarray, h: np.ndarray, v: np.ndarray,
                                cl: np.ndarray) -> str | None:
    """Reason text if a record has non-finite values, time gaps or single-step displacement jumps; otherwise None."""
    for name, arr in (("disp", h), ("vel", v), ("cl", cl)):
        if not np.all(np.isfinite(arr)):
            return f"non-finite values present in {name}"

    factor = REFERENCE_QUALITY_CRITERIA["discontinuity_jump_factor"]
    segments = segment_by_time_gaps(t, gap_factor=factor)
    if len(segments) > 1:
        dt = np.diff(t)
        dt_pos = dt[dt > 0]
        med_dt = float(np.median(dt_pos)) if len(dt_pos) else float("nan")
        idx = int(np.argmax(dt))
        return (f"time-axis gap of {dt[idx]:.4g}s ({dt[idx] / med_dt:.0f}x the case's "
                f"median timestep {med_dt:.4g}s) between sample {idx} and {idx + 1} "
                f"(t={t[idx]:.3f}s -> t={t[idx + 1]:.3f}s) -- a discontinuous restart: "
                f"data was dropped between an autosave and its resume")

    dh = np.abs(np.diff(h))
    nz = dh[dh > 0]
    if len(nz) == 0:
        return None
    med = float(np.median(nz))
    factor = REFERENCE_QUALITY_CRITERIA["discontinuity_jump_factor"]
    bad = dh > factor * med
    if np.any(bad):
        idx = int(np.argmax(dh))
        return (f"single-step disp jump {dh[idx]:.4g} is {dh[idx] / med:.1f}x the case's "
                f"own median step-to-step change ({med:.4g}) at sample {idx} -- possible "
                f"corrupted report line or discontinuous restart")
    return None


def classify_reference_status(row: dict) -> tuple[str, str]:
    """Status and reason for one case, from its duration, blockwise RMS ratio and frequency stability."""
    crit = REFERENCE_QUALITY_CRITERIA

    if row.get("numerically_suspect_reason"):
        return "numerically_suspect", row["numerically_suspect_reason"]

    if (row["n_tau_s_post_release"] < crit["min_n_tau_s_post_release"]
            or row["n_cycles_post_release"] < crit["min_n_cycles_post_release"]):
        return "insufficient_duration", (
            f"only {row['n_tau_s_post_release']:.2f} structural settling times "
            f"({row['n_cycles_post_release']} cycles) in the post-release window; "
            f"require >= {crit['min_n_tau_s_post_release']} tau_s and "
            f">= {crit['min_n_cycles_post_release']} cycles")

    b2, b3 = row["block2_h_rms"], row["block3_h_rms"]
    rms_ratio = (b3 / b2) if b2 else float("nan")
    freq_cv = row["freq_cv_across_blocks"]
    lo1, hi1 = crit["rms_ratio_settled_lco_band"]
    lo2, hi2 = crit["rms_ratio_stationary_les_band"]

    settled_rms_ok = np.isfinite(rms_ratio) and lo1 <= rms_ratio <= hi1
    freq_ok = np.isfinite(freq_cv) and freq_cv < crit["freq_cv_settled_lco_max"]
    if row["cfd_label"] == "stationary_lco" and settled_rms_ok and freq_ok:
        return "settled_lco", (
            f"Hilbert-slope='stationary_lco' AND blockwise RMS ratio {rms_ratio:.2f} "
            f"in [{lo1},{hi1}] AND freq CV {freq_cv:.3f} < {crit['freq_cv_settled_lco_max']}")

    if np.isfinite(rms_ratio) and lo2 <= rms_ratio <= hi2:
        return "statistically_stationary_les", (
            f"blockwise RMS ratio {rms_ratio:.2f} within the wider stationary band "
            f"[{lo2},{hi2}] even though Hilbert-slope label='{row['cfd_label']}' "
            f"and/or freq CV={freq_cv:.3f} did not meet the tighter settled_lco bands")

    return "transient_or_slowly_evolving", (
        f"blockwise RMS ratio {rms_ratio:.2f} outside the stationary band "
        f"[{lo2},{hi2}], Hilbert-slope label='{row['cfd_label']}' -- systematic "
        f"amplitude/frequency evolution over the retained interval")


def build_reference_status_table() -> pd.DataFrame:
    """Compute the statistics and the status of every bridge CFD case."""
    D = config["bridge_D_ref"]
    fn = config["bridge_fn_hz"]
    zeta = config["bridge_zeta"]
    t_star_release = config["bridge_t_star_release"]
    omega_n = 2 * np.pi * fn
    tau_s = 1.0 / (zeta * omega_n)

    sp = bridge_structural_params()
    df = load_bridge_df_cached(fn_hz=fn, d_ref=D, bridge_structural_params=sp)

    rows = []
    for case, cdf in df.groupby("case"):
        cdf = cdf.sort_values("time")
        Ur = parse_ur_label(str(case))
        U = Ur * fn * D
        t = cdf["time"].to_numpy(dtype=float)
        h = cdf["disp"].to_numpy(dtype=float)
        v = cdf["vel"].to_numpy(dtype=float)
        cl = cdf["cl"].to_numpy(dtype=float)

        release_time = t_star_release * D / U
        post_mask = t >= release_time
        t_p, h_p, v_p, cl_p = t[post_mask], h[post_mask], v[post_mask], cl[post_mask]
        post_duration = float(t_p.max() - t_p.min()) if len(t_p) > 1 else 0.0
        n_tau_s = post_duration / tau_s
        n_cycles = len(cycles_from_displacement(t_p, h_p, D=D))

        n = len(t_p)
        edges = [0, n // 3, 2 * n // 3, n]
        block_rms, block_freq = [], []
        for i in range(3):
            seg_t, seg_h = t_p[edges[i]:edges[i + 1]], h_p[edges[i]:edges[i + 1]]
            if len(seg_h) < 2:
                block_rms.append(float("nan")); block_freq.append(float("nan")); continue
            block_rms.append(float(np.sqrt(np.mean(seg_h ** 2))))
            block_freq.append(_dominant_freq(seg_t, seg_h))
        freq_valid = np.array([f for f in block_freq if np.isfinite(f)])
        freq_cv = (float(np.std(freq_valid) / np.mean(freq_valid))
                   if len(freq_valid) > 1 and np.mean(freq_valid) > 0 else float("nan"))

        stab = classify_stability(t_p, h_p, window_frac=0.5)
        suspect_reason = _numerically_suspect_reason(t_p, h_p, v_p, cl_p)

        row = {
            "Ur": Ur, "U_ms": U, "case": case,
            "post_release_duration_s": post_duration,
            "n_tau_s_post_release": n_tau_s,
            "n_cycles_post_release": n_cycles,
            "block1_h_rms": block_rms[0], "block2_h_rms": block_rms[1], "block3_h_rms": block_rms[2],
            "freq_cv_across_blocks": freq_cv,
            "cfd_label": stab["label"],
            "frac_envelope_change": stab["fractional_envelope_change"],
            "envelope_change_abs": stab["envelope_change_abs"],
            "numerically_suspect_reason": suspect_reason,
        }
        status, reason = classify_reference_status(row)
        row["reference_status"] = status
        row["reference_status_reason"] = reason
        row["evaluation_procedure"] = _EVAL_PROCEDURE_BY_STATUS[status]
        rows.append(row)

    return pd.DataFrame(rows).sort_values("Ur").reset_index(drop=True)


def compute_non_lco_summary(npz_path: str, rms_agreement_band: tuple[float, float] = (0.7, 1.43)) -> dict:
    """Scores for cases without a steady limit cycle: blockwise RMS ratio and cumulative aerodynamic work, surrogate vs CFD."""
    d = np.load(npz_path, allow_pickle=True)
    if "h_cfd" not in d.files or "cl_cfd" not in d.files:
        return {"scoring_method": "non_lco_block_energy", "unscored": True,
                "unscored_reason": "npz has no cl_cfd/h_cfd (predates that being saved)"}

    t, h, cl = d["t"], d["h"], d["cl"]
    h_cfd, cl_cfd = d["h_cfd"], d["cl_cfd"]
    n = min(len(t), len(h_cfd), len(cl_cfd))
    t, h, cl = t[:n], h[:n], cl[:n]
    h_cfd, cl_cfd = h_cfd[:n], cl_cfd[:n]

    edges = [0, n // 3, 2 * n // 3, n]
    block_ratios = []
    for i in range(3):
        s, e = edges[i], edges[i + 1]
        if e - s < 2:
            block_ratios.append(float("nan")); continue
        rms_sur = float(np.sqrt(np.mean(h[s:e] ** 2)))
        rms_cfd = float(np.sqrt(np.mean(h_cfd[s:e] ** 2)))
        block_ratios.append(rms_sur / rms_cfd if rms_cfd else float("nan"))

    mean_ratio = float(np.nanmean(block_ratios)) if any(np.isfinite(block_ratios)) else float("nan")
    lo, hi = rms_agreement_band
    rms_agreement = bool(np.isfinite(mean_ratio) and lo <= mean_ratio <= hi)

    h_dot = np.gradient(h, t)
    h_dot_cfd = np.gradient(h_cfd, t)
    E_sur = float(np.trapezoid(cl * h_dot, t))
    E_cfd = float(np.trapezoid(cl_cfd * h_dot_cfd, t))
    energy_ratio = (E_sur / E_cfd) if E_cfd else float("nan")

    return {
        "scoring_method": "non_lco_block_energy", "unscored": False,
        "block1_rms_ratio": block_ratios[0], "block2_rms_ratio": block_ratios[1],
        "block3_rms_ratio": block_ratios[2], "mean_rms_ratio": mean_ratio,
        "rms_agreement": rms_agreement,
        "cumulative_energy_sur": E_sur, "cumulative_energy_cfd": E_cfd,
        "cumulative_energy_ratio": energy_ratio,
    }


def build_status_aware_report(sweep_dir: str, model_label: str,
                              reference_status_csv: str = "results/bridge_reference_status.csv"
                              ) -> pd.DataFrame:
    """Combine a sweep_results.csv with the reference statuses, scoring each case in the way its status allows."""
    from pathlib import Path

    ref = pd.read_csv(reference_status_csv).set_index("Ur")
    sweep_csv = Path(sweep_dir) / "sweep_results.csv"
    sweep = pd.read_csv(sweep_csv)
    stab_col = next(c for c in sweep.columns if c.endswith("_stability_label"))
    err_col = next(c for c in sweep.columns if c.endswith("_A_star_rel_error"))
    pass_col = next(c for c in sweep.columns if c.endswith("_pass"))

    rows = []
    for _, srow in sweep.iterrows():
        ur = srow["Ur"]
        row = {"Ur": ur, "model": model_label}
        if ur not in ref.index:
            row.update(reference_status="not_in_retained_dataset", unscored=True,
                       unscored_reason="excluded from the retained bridge dataset (e.g. 19.5 m/s)")
            rows.append(row)
            continue

        status = ref.loc[ur, "reference_status"]
        row["reference_status"] = status

        if status == "settled_lco":
            row.update(scoring_method="lco_gate", unscored=False,
                      stability_label=srow[stab_col], A_star_rel_error=srow[err_col],
                      pass_=bool(srow[pass_col]))
        elif status in ("statistically_stationary_les", "transient_or_slowly_evolving"):
            npz_glob = sorted(Path(sweep_dir).glob(f"coupled_bridge_Ur{ur}_*.npz"))
            if not npz_glob:
                row.update(scoring_method="non_lco_block_energy", unscored=True,
                          unscored_reason=f"no npz found for Ur={ur} in {sweep_dir}")
            else:
                row.update(**compute_non_lco_summary(str(npz_glob[0])))
        else:
            row.update(scoring_method="none", unscored=True,
                      unscored_reason=ref.loc[ur, "reference_status_reason"])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("Ur").reset_index(drop=True)


def main():
    """Build the status table and write it to CSV."""
    table = build_reference_status_table()
    out_csv = "results/bridge_reference_status.csv"
    table.to_csv(out_csv, index=False)
    print(f"Saved {len(table)} case(s) to {out_csv}")
    print(table["reference_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
