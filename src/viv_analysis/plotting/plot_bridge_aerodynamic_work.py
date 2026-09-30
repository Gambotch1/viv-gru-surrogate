"""Aerodynamic energy transfer during the coupled bridge prediction at Ur = 6.7385
(thesis Fig. 6.9): (a) lift-velocity phase over time, (b) band-passed lift and
velocity near fn, (c) cumulative net aerodynamic work per cycle.

Reads one coupled npz (default: the final baseline run at Ur = 6.7385) and
writes to <npz folder>/thesis_figures/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import butter, sosfiltfilt

from viv_analysis.config import bridge_structural_params, config
from viv_analysis.plotting.plot_style import (
    MODEL_COLOR, SECONDARY_COLOR, TEXT_WIDTH_IN, apply_thesis_style,
)

DEFAULT_NPZ = (
    "results/gru_bridge_nd_context_noacc_final22_coupled_eval/"
    "coupled_bridge_Ur6.7385_gru_bridge_nd_context_noacc_forc-v1_additive_"
    "noise-none_nd_scale1_s1_seed0_handoff_2000_muNone.npz"
)

STRUCTURAL_BAND_FRAC = (0.5, 1.5)


def single_bin_dft_complex(t: np.ndarray, x: np.ndarray, f: float) -> complex:
    """Complex Fourier coefficient of x at one frequency f."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n == 0:
        return complex(np.nan, np.nan)
    kernel = np.exp(-2j * np.pi * f * np.asarray(t, dtype=float))
    return (2.0 / n) * np.sum(x * kernel)


def phase_difference_rad(t: np.ndarray, x1: np.ndarray, x2: np.ndarray, f: float) -> float:
    """Phase of x1 relative to x2 at frequency f, in [-pi, pi]."""
    c1 = single_bin_dft_complex(t, x1, f)
    c2 = single_bin_dft_complex(t, x2, f)
    if not (np.isfinite(c1.real) and np.isfinite(c2.real)):
        return float("nan")
    diff = np.angle(c1) - np.angle(c2)
    return float(np.angle(np.exp(1j * diff)))


def sliding_phase(t: np.ndarray, F_L: np.ndarray, h_dot: np.ndarray, fn: float,
                   window_periods: float = 2.0, stride_periods: float = 1.0) -> dict:
    """Lift-velocity phase in sliding windows over time."""
    Tn = 1.0 / fn
    win_s, stride_s = window_periods * Tn, stride_periods * Tn
    t_rel = t - t[0]
    t_max = t_rel[-1]

    centers, phases_deg, n_samples = [], [], []
    start = 0.0
    while start + win_s <= t_max:
        m = (t_rel >= start) & (t_rel < start + win_s)
        if m.sum() > 10:
            phi = phase_difference_rad(t[m], F_L[m], h_dot[m], fn)
            centers.append(start + win_s / 2.0)
            phases_deg.append(float(np.degrees(phi)))
            n_samples.append(int(m.sum()))
        start += stride_s
    phases_deg = np.array(phases_deg)
    return {"t_center": np.array(centers), "phase_deg": phases_deg,
            "phase_lag_deg": np.abs(phases_deg), "n_samples": np.array(n_samples)}


def cycle_resolved_work(t: np.ndarray, F_L: np.ndarray, h_dot: np.ndarray, c: float,
                        fn: float) -> dict:
    """Aerodynamic work done in each displacement cycle."""
    Tn = 1.0 / fn
    t_rel = t - t[0]
    n_cycles = int(t_rel[-1] / Tn)

    ends, W_f, W_d = [], [], []

    for k in range(n_cycles):
        m = (t_rel >= k * Tn) & (t_rel < (k + 1) * Tn)
        if m.sum() < 2:
            continue

        t_k, F_k, hd_k = t[m], F_L[m], h_dot[m]
        ends.append((k + 1.0) * Tn)
        W_f.append(float(trapezoid(F_k * hd_k, t_k)))
        W_d.append(float(trapezoid(c * hd_k**2, t_k)))

    return {
        "t_end": np.asarray(ends),
        "W_f": np.asarray(W_f),
        "W_d": np.asarray(W_d),
    }


def compute_aerodynamic_work(npz_path: Path) -> dict:
    """All quantities shown in the figure, from one coupled npz."""
    d = np.load(npz_path, allow_pickle=True)
    receipt = json.loads(npz_path.with_suffix(".receipt.json").read_text())

    t, h_dot, cl = d["t"], d["h_dot"], d["cl"]
    Ur = float(d["Ur"])
    D = float(d["D"])

    t_h_receipt = float(receipt["handoff_time_s"])
    assert abs(float(t[0]) - t_h_receipt) < 1e-6, (
        f"{npz_path.name}: t[0]={t[0]!r} != receipt handoff_time_s="
        f"{t_h_receipt!r} -- the whole-trajectory-is-post-handoff assumption "
        f"this figure relies on does not hold for this npz; integrate from "
        f"the correct handoff index instead of t[0].")

    rho = config["bridge_rho"]
    B = config["bridge_B_ref"]
    fn = config["bridge_fn_hz"]
    c = bridge_structural_params()["c"]
    U = Ur * fn * D
    dt = float(np.median(np.diff(t)))

    F_L = 0.5 * rho * U**2 * B * cl

    phase = sliding_phase(t, F_L, h_dot, fn)

    fs = 1.0 / dt
    lo, hi = STRUCTURAL_BAND_FRAC[0] * fn, STRUCTURAL_BAND_FRAC[1] * fn
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    F_L_bp = sosfiltfilt(sos, F_L)
    h_dot_bp = sosfiltfilt(sos, h_dot)
    c_exc = float(np.mean(F_L_bp * h_dot_bp) / np.mean(h_dot_bp**2))
    c_exc_over_c = c_exc / c

    cyc = cycle_resolved_work(t, F_L, h_dot, c, fn)

    return {
        "t": t, "t_rel": t - t[0], "F_L": F_L, "F_L_bp": F_L_bp,
        "h_dot": h_dot, "h_dot_bp": h_dot_bp,
        "phase": phase, "cycle": cyc,
        "c_exc": c_exc, "c_exc_over_c": c_exc_over_c,
        "Ur": Ur, "U": U, "D": D, "B": B, "rho": rho, "c": c, "fn": fn,
    }


def plot_aerodynamic_work(result: dict, out_path_stem: Path,
                          bandpass_window_periods: tuple[float, float] = (0.0, 10.0),
                          phase_window_periods: tuple[float, float] = (0.0, 30.0),
                          font_scale: float = 1.0):
    """Draw the three-panel figure."""
    apply_thesis_style()
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    if font_scale != 1.0:
        mpl.rcParams.update({
            "axes.labelsize": mpl.rcParams["axes.labelsize"] * font_scale,
            "axes.titlesize": mpl.rcParams["axes.titlesize"] * font_scale,
            "xtick.labelsize": mpl.rcParams["xtick.labelsize"] * font_scale,
            "ytick.labelsize": mpl.rcParams["ytick.labelsize"] * font_scale,
            "legend.fontsize": mpl.rcParams["legend.fontsize"] * font_scale,
        })

    Tn = 1.0 / result["fn"]
    t_periods = result["t_rel"] / Tn
    t_star = result["t_rel"] * result["U"] / result["D"]

    fig, (ax_phase, ax_bp, ax_cyc) = plt.subplots(
        3, 1, figsize=(TEXT_WIDTH_IN, 7.5),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]},
    )
    ph = result["phase"]
    lo_ph, hi_ph = phase_window_periods
    ph_periods = ph["t_center"] / Tn
    ph_star = ph["t_center"] * result["U"] / result["D"]
    ph_mask = (ph_periods >= lo_ph) & (ph_periods <= hi_ph)
    ax_phase.plot(ph_star[ph_mask], ph["phase_lag_deg"][ph_mask], color=MODEL_COLOR,
                  marker="o", ms=3, lw=1.05)
    ax_phase.axhline(0, color="gray", lw=0.6, ls=":")
    ax_phase.axhline(90, color="gray", lw=0.6, ls="--")
    ax_phase.axhline(180, color="gray", lw=0.6, ls=":")
    trans = ax_phase.get_yaxis_transform()
    ax_phase.text(0.995, 2, "aerodynamic excitation", transform=trans, ha="right",
                  va="bottom", fontsize=7 * font_scale, style="italic", color="dimgray")
    ax_phase.text(0.995, 92, "zero coherent work", transform=trans, ha="right",
                  va="bottom", fontsize=7 * font_scale, style="italic", color="dimgray")
    ax_phase.text(0.995, 183, "aerodynamic damping", transform=trans, ha="right",
                  va="bottom", fontsize=7 * font_scale, style="italic", color="dimgray")
    ax_phase.set_ylabel(r"absolute phase difference [deg]")
    ax_phase.set_xlim(lo_ph * Tn * result["U"] / result["D"],
                      hi_ph * Tn * result["U"] / result["D"])
    ax_phase.set_ylim(-10, 195)
    ax_phase.set_yticks([0, 45, 90, 135, 180])
    ax_phase.annotate("(a)", xy=(-0.1 * font_scale, 1.0), xycoords="axes fraction",
                       fontsize=10 * font_scale, fontweight="bold", va="top")


    lo_p, hi_p = bandpass_window_periods
    mask = (t_periods >= lo_p) & (t_periods <= hi_p)
    F_bp_win = result["F_L_bp"][mask]
    hdot_bp_win = result["h_dot_bp"][mask]
    F_norm = F_bp_win / np.sqrt(np.mean(F_bp_win**2))
    hdot_norm = hdot_bp_win / np.sqrt(np.mean(hdot_bp_win**2))
    ax_bp.plot(t_star[mask], F_norm, color=MODEL_COLOR, lw=1.0,
              label=r"$\widetilde{F}_L'$ (band-passed, RMS-normalised)")
    ax_bp.plot(t_star[mask], hdot_norm, color=SECONDARY_COLOR, lw=1.0,
              ls=(0, (4, 2)), label=r"$\dot h$ (band-passed, RMS-normalised)")
    ax_bp.axhline(0, color="black", lw=0.5)
    ax_bp.set_ylabel("Normalised amplitude [-]")
    ax_bp.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1,
                frameon=False, borderaxespad=0.0)
    ax_bp.annotate("(b)", xy=(-0.1 * font_scale, 1.0), xycoords="axes fraction",
                    fontsize=10 * font_scale, fontweight="bold", va="top")


    cyc = result["cycle"]
    W_net = np.cumsum(cyc["W_f"] - cyc["W_d"])

    work_time_star = np.concatenate(([0.0], cyc["t_end"] * result["U"] / result["D"]))
    W_net_plot = np.concatenate(([0.0], W_net))

    ax_cyc.plot(work_time_star, W_net_plot,
                color=MODEL_COLOR, lw=1.2)
    ax_cyc.axhline(0.0, color="black", lw=0.6)

    peak_idx = int(np.argmax(W_net_plot))
    ax_cyc.plot(work_time_star[peak_idx], W_net_plot[peak_idx],
                marker="o", ms=3.5, color=MODEL_COLOR)

    ax_cyc.annotate(
        "maximum following handoff",
        xy=(work_time_star[peak_idx], W_net_plot[peak_idx]),
        xytext=(16, -10),
        textcoords="offset points",
        fontsize=7.5 * font_scale,
        arrowprops={"arrowstyle": "->", "lw": 0.6},
    )
    ax_cyc.text(
    -0.10 * font_scale, 1.0, "(c)",
    transform=ax_cyc.transAxes,
    fontsize=10 * font_scale,
    fontweight="bold",
    va="top",
    clip_on=False,
    )
    ax_cyc.set_ylabel(r"$W_{\mathrm{net}}=W_f-W_d$ [J/m]")
    ax_cyc.set_xlabel(r"$(t-t_{\mathrm{h}})U/D$")

    fig.align_ylabels([ax_phase, ax_bp, ax_cyc])
    fig.tight_layout()
    fig.savefig(out_path_stem.with_suffix(".pdf"))
    fig.savefig(out_path_stem.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return out_path_stem.with_suffix(".pdf"), out_path_stem.with_suffix(".png")


def main():
    """Compute and plot for the given npz."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz_path", default=DEFAULT_NPZ)
    p.add_argument("--out_dir", default=None,
                    help="Default: <npz's own model dir>/thesis_figures/")
    p.add_argument("--bandpass_window_periods", type=float, nargs=2, default=(0.0, 10.0),
                    help="Structural-period range shown in panel (b), e.g. 0 10 "
                         "to span both the pre-flip and post-flip regimes.")
    p.add_argument("--phase_window_periods", type=float, nargs=2, default=(0.0, 30.0),
                    help="Structural-period range shown in panel (a). Default 0-30: "
                         "the band-passed oscillation amplitude decays ~600 N/m to "
                         "~5-8 N/m over this range as CL saturates, and phase "
                         "estimates beyond it are on a signal at the residual noise "
                         "floor rather than a resolvable oscillation.")
    p.add_argument("--font_scale", type=float, default=1.0,
                    help="Font-size multiplier for the figure. Pass "
                         "1/display_fraction, e.g. 1/0.8 for 0.8\\textwidth.")
    args = p.parse_args()

    npz_path = Path(args.npz_path)
    out_dir = Path(args.out_dir) if args.out_dir else npz_path.parent / "thesis_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = compute_aerodynamic_work(npz_path)
    ur_tag = f"{result['Ur']:g}".replace(".", "p")
    stem = out_dir / f"aerodynamic_work_Ur{ur_tag}"
    pdf, png = plot_aerodynamic_work(result, stem,
                                     bandpass_window_periods=tuple(args.bandpass_window_periods),
                                     phase_window_periods=tuple(args.phase_window_periods),
                                     font_scale=args.font_scale)

    ph = result["phase"]
    Tn = 1.0 / result["fn"]

    post_flip = ph["phase_lag_deg"] > 150
    flip_t_star = float(ph["t_center"][post_flip][0] / Tn) if post_flip.any() else float("nan")

    lo_ph, hi_ph = tuple(args.phase_window_periods)
    cyc = result["cycle"]
    W_f0, W_d0 = cyc["W_f"][0], cyc["W_d"][0]
    W_net = np.cumsum(cyc["W_f"]) - np.cumsum(cyc["W_d"])
    W_net_plot = np.concatenate(([0.0], W_net))
    work_time_star = np.concatenate(([0.0], cyc["t_end"] / Tn))
    peak_idx = int(np.argmax(W_net_plot))

    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    print(f"c_exc = {result['c_exc']:.6g} N*s/m^2   c = {result['c']:.6g} N*s/m^2   "
          f"c_exc/c = {result['c_exc_over_c']:.6g}")
    print(f"Phase lag reaches antiphase by (t-t_h)/Tn = {flip_t_star:.2f}  "
          f"(t-t_h = {flip_t_star*Tn:.2f}s)")
    print(f"Cycle 0 (handoff transient): W_f={W_f0:.4g} J/m, W_d={W_d0:.4g} J/m")
    print(
        f"Cumulative net work: peak {W_net_plot[peak_idx]:.4g} J/m at "
        f"(t-t_h)/Tn={work_time_star[peak_idx]:.2f}, "
        f"final {W_net_plot[-1]:.4g} J/m"
    )
    print(f"Cycle-resolved work (cycles 1+): W_f range "
          f"[{cyc['W_f'][1:].min():.4g}, {cyc['W_f'][1:].max():.4g}] J/m, "
          f"W_d range [{cyc['W_d'][1:].min():.4g}, {cyc['W_d'][1:].max():.4g}] J/m")


if __name__ == "__main__":
    main()
