#!/usr/bin/env python3
"""Metrics for closed-loop trajectories (thesis Sec. 4.7).

All metrics use the last `window_frac` of the trajectory (default: the last half).
    amplitude A* = A/D and frequency  - averaged over complete oscillation cycles
    aerodynamic work per cycle        - E_f = integral of C_L h' dt over one cycle
    lift-velocity phase               - sin(phi) = E_f / (pi A F1), F1 = lift amplitude at f_osc
    stability label                   - stationary_lco, divergence or decay_to_rest,
                                        from the growth of the Hilbert envelope

Run as a script to score a folder of coupled_*.npz files:
    PYTHONPATH=src python -m viv_analysis.closed_loop_metrics --npz_dir <folder>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import hilbert


def _same_direction_crossings(x: np.ndarray) -> np.ndarray:
    """Indices where x crosses zero upwards."""
    sign = np.sign(x)
    sign[sign == 0] = 1.0
    return np.where((sign[:-1] < 0) & (sign[1:] >= 0))[0] + 1


_FLAT_SIGNAL_REL_TOL = 1e-5
# Oscillations with A/D below this are treated as noise, not as cycles.
DEFAULT_A_STAR_NOISE_FLOOR = 1e-4


def cycles_from_displacement(t: np.ndarray, h: np.ndarray, D: float | None = None,
                             a_star_floor: float = DEFAULT_A_STAR_NOISE_FLOOR
                             ) -> list[tuple[int, int]]:
    """Split a displacement signal into cycles (between upward zero crossings of the velocity)."""
    if len(t) < 2:
        return []
    h_range = float(h.max() - h.min())
    if D is not None and D > 0 and (h_range / 2.0) / D < a_star_floor:
        return []
    h_scale = float(np.abs(h).max()) + 1e-30
    if h_range < _FLAT_SIGNAL_REL_TOL * h_scale:
        return []
    h_dot = np.gradient(h, t)
    idx = _same_direction_crossings(h_dot)
    return list(zip(idx[:-1], idx[1:]))


def _windowed(t: np.ndarray, *arrays: np.ndarray, window_frac: float):
    """Keep the last window_frac of each array."""
    n = len(t)
    start = int((1.0 - window_frac) * n)
    return (t[start:],) + tuple(a[start:] for a in arrays)


def _trustworthy_cycles(t_w: np.ndarray, h_w: np.ndarray, cycles: list[tuple[int, int]],
                        D: float | None, a_star_floor: float = DEFAULT_A_STAR_NOISE_FLOOR,
                        min_period_frac_of_median: float = 0.5
                        ) -> list[tuple[int, int]]:
    """Drop cycles below the amplitude floor or shorter than half the median period."""
    if D is None or D <= 0:
        return cycles
    amp_ok = []
    for i0, i1 in cycles:
        seg = h_w[i0:i1 + 1]
        amp = (float(seg.max()) - float(seg.min())) / 2.0
        if amp / D >= a_star_floor:
            amp_ok.append((i0, i1))
    if len(amp_ok) < 2:
        return amp_ok
    periods = np.array([t_w[i1] - t_w[i0] for i0, i1 in amp_ok])
    median_T = float(np.median(periods))
    return [c for c, T in zip(amp_ok, periods) if T >= min_period_frac_of_median * median_T]


def cycle_amplitude_and_frequency(t: np.ndarray, h: np.ndarray, D: float,
                                  window_frac: float = 0.5) -> dict:
    """Mean cycle amplitude (A, A* = A/D) and frequency in the analysis window."""
    t_w, h_w = _windowed(t, h, window_frac=window_frac)
    cycles = _trustworthy_cycles(t_w, h_w, cycles_from_displacement(t_w, h_w, D=D), D=D)
    if not cycles:
        return {"A_phys": float("nan"), "A_star": float("nan"),
                "f_osc": float("nan"), "n_cycles": 0}
    amps, periods = [], []
    for i0, i1 in cycles:
        seg = h_w[i0:i1 + 1]
        amps.append((float(seg.max()) - float(seg.min())) / 2.0)
        periods.append(float(t_w[i1] - t_w[i0]))
    A = float(np.mean(amps))
    T = float(np.mean(periods))
    return {
        "A_phys": A,
        "A_star": A / D if D else float("nan"),
        "f_osc": 1.0 / T if T > 0 else float("nan"),
        "n_cycles": len(cycles),
    }


def cycle_average_energy(t: np.ndarray, h: np.ndarray, cl: np.ndarray,
                         window_frac: float = 0.5, D: float | None = None
                         ) -> dict:
    """Aerodynamic work per cycle, E_f = integral of C_L h' dt: mean, std and coefficient of variation."""
    t_w, h_w, cl_w = _windowed(t, h, cl, window_frac=window_frac)
    cycles = _trustworthy_cycles(t_w, h_w, cycles_from_displacement(t_w, h_w, D=D), D=D)
    if not cycles:
        return {"E_f_mean": float("nan"), "E_f_std": float("nan"),
                "E_f_cv": float("nan"), "n_cycles": 0}
    h_dot_w = np.gradient(h_w, t_w)
    energies = np.array([
        float(np.trapezoid(cl_w[i0:i1 + 1] * h_dot_w[i0:i1 + 1], t_w[i0:i1 + 1]))
        for i0, i1 in cycles
    ])
    E_f_mean = float(np.mean(energies))
    E_f_std = float(np.std(energies, ddof=1)) if len(energies) > 1 else 0.0
    E_f_cv = E_f_std / abs(E_f_mean) if E_f_mean else float("nan")
    return {"E_f_mean": E_f_mean, "E_f_std": E_f_std, "E_f_cv": E_f_cv,
            "n_cycles": len(cycles)}


def single_bin_dft_amplitude(t: np.ndarray, x: np.ndarray, f: float) -> float:
    """Amplitude of x at one frequency f (single-bin Fourier transform)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n == 0:
        return float("nan")
    kernel = np.exp(-2j * np.pi * f * np.asarray(t, dtype=float))
    return float(2.0 / n * np.abs(np.sum(x * kernel)))


def energy_and_phase(t: np.ndarray, h: np.ndarray, cl: np.ndarray, D: float,
                     window_frac: float = 0.5) -> dict:
    """Work per cycle plus the lift-velocity phase derived from it (sin phi = E_f / (pi A F1))."""
    energy = cycle_average_energy(t, h, cl, window_frac, D=D)
    E_f, E_f_std, E_f_cv, n_cycles = (
        energy["E_f_mean"], energy["E_f_std"], energy["E_f_cv"], energy["n_cycles"]
    )
    amp = cycle_amplitude_and_frequency(t, h, D, window_frac)
    A_phys, f_osc = amp["A_phys"], amp["f_osc"]

    if n_cycles == 0 or not np.isfinite(f_osc) or A_phys <= 0:
        return {"E_f": E_f, "E_f_std": E_f_std, "E_f_cv": E_f_cv,
                "A_phys": A_phys, "f_osc": f_osc,
                "F1": float("nan"), "sin_phi": float("nan"),
                "sin_phi_clipped": float("nan"), "phi_rad": float("nan"),
                "n_cycles": n_cycles}

    t_w, cl_w = _windowed(t, cl, window_frac=window_frac)
    F1 = single_bin_dft_amplitude(t_w, cl_w, f_osc)
    denom = np.pi * A_phys * F1
    sin_phi = E_f / denom if denom > 0 else float("nan")
    sin_phi_clipped = float(np.clip(sin_phi, -1.0, 1.0)) if np.isfinite(sin_phi) else float("nan")
    phi_rad = float(np.arcsin(sin_phi_clipped)) if np.isfinite(sin_phi_clipped) else float("nan")

    return {
        "E_f": E_f, "E_f_std": E_f_std, "E_f_cv": E_f_cv,
        "A_phys": A_phys, "f_osc": f_osc, "F1": F1,
        "sin_phi": float(sin_phi) if np.isfinite(sin_phi) else float("nan"),
        "sin_phi_clipped": sin_phi_clipped, "phi_rad": phi_rad,
        "n_cycles": n_cycles,
    }


def classify_stability(t: np.ndarray, h: np.ndarray,
                       window_frac: float = 0.5,
                       stationary_frac_threshold: float = 0.10) -> dict:
    """Label a trajectory from its envelope growth over the analysis window.

    The Hilbert envelope is fitted with an exponential. If it changes by less
    than stationary_frac_threshold (10 %) over the window: stationary_lco;
    otherwise divergence (growing) or decay_to_rest (decaying).
    """
    t_w, h_w = _windowed(t, h, window_frac=window_frac)
    if len(t_w) < 4:
        return {"label": "insufficient_data", "growth_rate_per_s": float("nan"),
                "fractional_envelope_change": float("nan"),
                "envelope_change_abs": float("nan"),
                "envelope_start": float("nan"), "envelope_end": float("nan"),
                "envelope_start_fitted": float("nan"), "envelope_end_fitted": float("nan")}

    h_centered = np.asarray(h_w, dtype=float) - float(np.mean(h_w))
    envelope = np.abs(hilbert(h_centered))
    envelope_safe = np.clip(envelope, 1e-12, None)

    b, log_a = np.polyfit(t_w, np.log(envelope_safe), 1)
    T = float(t_w[-1] - t_w[0])
    frac_change = float(np.exp(b * T) - 1.0) if np.isfinite(b) else float("nan")

    if np.isfinite(b) and np.isfinite(log_a):
        env_start_fitted = float(np.exp(log_a + b * t_w[0]))
        env_end_fitted = float(np.exp(log_a + b * t_w[-1]))
        env_change_abs = env_end_fitted - env_start_fitted
    else:
        env_start_fitted = env_end_fitted = env_change_abs = float("nan")

    if not np.isfinite(frac_change):
        label = "insufficient_data"
    elif abs(frac_change) < stationary_frac_threshold:
        label = "stationary_lco"
    elif b > 0:
        label = "divergence"
    else:
        label = "decay_to_rest"

    return {
        "label": label,
        "growth_rate_per_s": float(b),
        "fractional_envelope_change": frac_change,
        "envelope_change_abs": env_change_abs,
        "envelope_start": float(envelope[0]),
        "envelope_end": float(envelope[-1]),
        "envelope_start_fitted": env_start_fitted,
        "envelope_end_fitted": env_end_fitted,
    }


def compute_signal_metrics(t: np.ndarray, h: np.ndarray, cl: np.ndarray, D: float,
                           window_frac: float = 0.5) -> dict:
    """Amplitude, frequency, energy, phase and stability of one trajectory."""
    amp = cycle_amplitude_and_frequency(t, h, D, window_frac)
    energy = energy_and_phase(t, h, cl, D, window_frac)
    stab = classify_stability(t, h, window_frac)
    return {**amp, **energy, **stab}


def compare_surrogate_vs_cfd(surrogate: dict, cfd: dict) -> dict:
    """Errors of the surrogate against CFD: amplitude, frequency, energy and phase."""
    out = {}
    out["A_star_error"] = surrogate["A_star"] - cfd["A_star"]
    out["A_star_rel_error"] = (out["A_star_error"] / cfd["A_star"]
                               if cfd.get("A_star") else float("nan"))
    out["f_osc_error"] = surrogate["f_osc"] - cfd["f_osc"]
    out["f_osc_rel_error"] = (out["f_osc_error"] / cfd["f_osc"]
                              if cfd.get("f_osc") else float("nan"))
    Ef_cfd = cfd.get("E_f")
    out["energy_error_eps_E"] = ((surrogate["E_f"] - Ef_cfd) / abs(Ef_cfd)
                                  if Ef_cfd not in (None, 0) and np.isfinite(Ef_cfd)
                                  else float("nan"))
    if np.isfinite(surrogate.get("phi_rad", float("nan"))) and np.isfinite(cfd.get("phi_rad", float("nan"))):
        out["phase_error_rad"] = surrogate["phi_rad"] - cfd["phi_rad"]
    else:
        out["phase_error_rad"] = float("nan")
    return out


def compute_case_metrics(npz_path: str | Path, window_frac: float = 0.5) -> dict:
    """All metrics for one coupled npz, for the surrogate and (if stored) the CFD reference."""
    npz_path = Path(npz_path)
    d = np.load(npz_path, allow_pickle=True)
    t, h, cl = d["t"], d["h"], d["cl"]
    D = float(d["D"])
    Ur = float(d["Ur"])

    row = {"npz": str(npz_path), "Ur": Ur}
    surrogate = compute_signal_metrics(t, h, cl, D, window_frac)
    row.update({f"surrogate_{k}": v for k, v in surrogate.items()})

    if "cl_cfd" in d.files and "h_cfd" in d.files:
        h_cfd, cl_cfd = d["h_cfd"], d["cl_cfd"]
        n_common = min(len(t), len(h_cfd), len(cl_cfd))
        if n_common < len(t):
            row["cfd_window_truncated_to_n"] = n_common
        t_c, h_c, cl_c = t[:n_common], h[:n_common], cl[:n_common]
        h_cfd_c, cl_cfd_c = h_cfd[:n_common], cl_cfd[:n_common]

        surrogate_matched = compute_signal_metrics(t_c, h_c, cl_c, D, window_frac)
        cfd = compute_signal_metrics(t_c, h_cfd_c, cl_cfd_c, D, window_frac)
        row.update({f"cfd_{k}": v for k, v in cfd.items()})
        row.update(compare_surrogate_vs_cfd(surrogate_matched, cfd))
    else:
        row["_missing_cfd_cl"] = True

    return row


def parse_args():
    """Command-line options."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz_dir", required=True,
                   help="Directory of coupled_*.npz files (an evaluate_all.py "
                        "--output_dir sweep directory)")
    p.add_argument("--window_frac", type=float, default=0.5,
                   help="Fraction of the trajectory (from the end) treated as "
                        "the analysis window for all four metrics (default 0.5)")
    p.add_argument("--out_csv", default=None,
                   help="Output CSV path (default: <npz_dir>/closed_loop_metrics.csv)")
    return p.parse_args()


def main():
    """Score every coupled_*.npz in a folder and write closed_loop_metrics.csv."""
    args = parse_args()
    npz_dir = Path(args.npz_dir)
    npz_files = sorted(npz_dir.glob("coupled_*.npz"))
    if not npz_files:
        raise SystemExit(f"No coupled_*.npz files found in {npz_dir}")

    rows = []
    n_missing_cfd = 0
    for npz_path in npz_files:
        try:
            row = compute_case_metrics(npz_path, window_frac=args.window_frac)
        except Exception as e:
            print(f"  FAILED {npz_path.name}: {e}")
            continue
        if row.pop("_missing_cfd_cl", False):
            n_missing_cfd += 1
        rows.append(row)
        stab = row.get("surrogate_label", "?")
        print(f"  Ur={row['Ur']:.4f}  stability={stab:14s}  "
              f"A*={row.get('surrogate_A_star', float('nan')):.4f}  "
              f"f_osc={row.get('surrogate_f_osc', float('nan')):.4f}")

    if n_missing_cfd:
        print(f"\nWARNING: {n_missing_cfd}/{len(rows)} npz file(s) have no cl_cfd "
              f"(pre-date this module's addition to coupled_inference.py's save "
              f"call) -- surrogate-only metrics were computed for those; "
              f"re-run evaluate_all.py for that model to get comparison metrics.")

    out_csv = Path(args.out_csv) if args.out_csv else npz_dir / "closed_loop_metrics.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    front = ["npz", "Ur", "surrogate_label", "cfd_label"]
    fieldnames = [c for c in front if c in fieldnames] + \
                 [c for c in fieldnames if c not in front]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} case(s) to {out_csv}")


if __name__ == "__main__":
    main()
