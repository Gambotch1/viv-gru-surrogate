"""Figures of the time-varying Ur runs (thesis Fig. 5.11-5.13 and F.1): Ur(t), displacement and amplitude envelope, transitions, and plateau amplitudes vs fixed-Ur runs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

STUDY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = STUDY_ROOT.parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from viv_analysis.plotting.plot_style import TEXT_WIDTH_IN, apply_thesis_style  # noqa: E402
MIN_ENVELOPE_WINDOWS = 5


def t_star_of(t_arr: np.ndarray, ur_arr: np.ndarray, fn: float) -> np.ndarray:
    return fn * np.concatenate(([0.0], np.cumsum(
        0.5 * (ur_arr[1:] + ur_arr[:-1]) * np.diff(t_arr))))


def amplitude_envelope(h, D, dt, fn, win_cycles=3.0, step_frac=0.5):
    h = np.asarray(h, float)
    win = max(int(win_cycles / fn / dt), 8)
    step = max(int(step_frac * win), 1)
    centers, env = [], []
    for s in range(0, len(h) - win, step):
        seg = h[s:s + win]
        centers.append((s + win / 2) * dt)
        env.append(np.sqrt(2.0) * np.std(seg) / D)
    return np.asarray(centers), np.asarray(env)


def envelope_or_unavailable(h: np.ndarray, D: float, dt: float, fn: float,
                             win_cycles: float = 3.0, step_frac: float = 0.5,
                             min_windows: int = MIN_ENVELOPE_WINDOWS):
    env_t, env = amplitude_envelope(h, D, dt, fn, win_cycles=win_cycles, step_frac=step_frac)
    if len(env_t) >= min_windows:
        return env_t, env, None
    win_s = max(win_cycles / fn, 8 * dt)
    needed_s = win_s + (min_windows - 1) * step_frac * win_s
    have_s = len(h) * dt
    reason = (f"envelope unavailable: signal spans {have_s:.1f}s, needs >= "
              f"{needed_s:.1f}s for {min_windows} envelope windows "
              f"(win_cycles={win_cycles}, fn={fn})")
    return None, None, reason


def plot_time_varying_panels(result: dict, schedule: dict, D: float, fn: float,
                              out_path_stem: Path, font_scale: float = 1.0,
                              h_linewidth: float = 0.3):
    apply_thesis_style()
    if font_scale != 1.0:
        matplotlib.rcParams.update({
            "axes.labelsize": matplotlib.rcParams["axes.labelsize"] * font_scale,
            "xtick.labelsize": matplotlib.rcParams["xtick.labelsize"] * font_scale,
            "ytick.labelsize": matplotlib.rcParams["ytick.labelsize"] * font_scale,
            "legend.fontsize": matplotlib.rcParams["legend.fontsize"] * font_scale,
        })

    t = result["time"]
    Ur_t = result["Ur"]
    h_star = result["displacement"] / D
    dt = float(np.median(np.diff(t)))
    env_t, env, env_unavailable_reason = envelope_or_unavailable(result["displacement"], D, dt, fn)

    t_star = t_star_of(t, Ur_t, fn)
    env_t_star = (np.interp(env_t, t, t_star) if env_t is not None else None)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(TEXT_WIDTH_IN, 5.5))
    axes[0].plot(t_star, Ur_t, color="black", lw=0.3)
    axes[0].set_ylabel(r"$U_r(t)$")
    axes[1].plot(t_star, h_star, color="tab:blue", lw=h_linewidth)
    axes[1].set_ylabel(r"$h/D$")
    if env_t_star is not None:
        axes[2].plot(env_t_star, env, color="tab:red", lw=0.3)
    else:
        axes[2].text(0.5, 0.5, "envelope unavailable\n(insufficient cycles in window)",
                      ha="center", va="center", transform=axes[2].transAxes,
                      fontsize=8, color="gray")
        print(f"[plot_time_varying_panels] {env_unavailable_reason}")
    axes[2].set_ylabel(r"$A/D$")
    axes[2].set_xlabel(r"$t^*=tU/D$")

    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig(out_path_stem.with_suffix(".pdf"))
    fig.savefig(out_path_stem.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return out_path_stem.with_suffix(".pdf"), out_path_stem.with_suffix(".png")


def plateau_summary(result: dict, schedule: dict, D: float, fn: float, dt: float,
                     fixed_ur_cfd: dict[float, float] | None,
                     fixed_ur_gru: dict[float, float] | None,
                     n_cycles: int = 5, stat_tol: float = 0.10,
                     font_scale: float = 1.0) -> "tuple":
    import pandas as pd

    Ur_list = schedule["Ur_list"]
    dwell_steps = schedule["dwell_steps"]
    Tn = 1.0 / fn
    cycle_steps = int(round(Tn / dt))

    rows = []
    cursor = 0
    for ur in Ur_list:
        seg_end = cursor + dwell_steps
        seg_h = result["displacement"][cursor:seg_end]
        cycles_available = len(seg_h) / cycle_steps
        sufficient_data = cycles_available >= n_cycles
        tail = seg_h[-n_cycles * cycle_steps:] if sufficient_data else seg_h
        a_star = np.sqrt(2.0) * np.std(tail) / D

        n_full_cycles = max(1, len(tail) // cycle_steps)
        per_cycle_a_star = np.array([
            np.sqrt(2.0) * np.std(tail[c * cycle_steps:(c + 1) * cycle_steps]) / D
            for c in range(n_full_cycles)
        ])
        cv = (float(np.std(per_cycle_a_star) / (np.mean(per_cycle_a_star) + 1e-30))
              if n_full_cycles >= 2 else float("nan"))
        stabilised = bool(cv < stat_tol and sufficient_data)
        rows.append({
            "Ur": ur,
            "continuous_plateau_A_star": float(a_star),
            "cycles_available": float(cycles_available),
            "sufficient_data_for_n_cycles": bool(sufficient_data),
            "stabilised": stabilised,
            "cfd_fixed_ur_A_star": (fixed_ur_cfd or {}).get(ur),
            "gru_fixed_ur_A_star": (fixed_ur_gru or {}).get(ur),
        })
        cursor = seg_end

    df = pd.DataFrame(rows)

    apply_thesis_style()
    if font_scale != 1.0:
        matplotlib.rcParams.update({
            "axes.labelsize": matplotlib.rcParams["axes.labelsize"] * font_scale,
            "xtick.labelsize": matplotlib.rcParams["xtick.labelsize"] * font_scale,
            "ytick.labelsize": matplotlib.rcParams["ytick.labelsize"] * font_scale,
            "legend.fontsize": matplotlib.rcParams["legend.fontsize"] * font_scale,
        })
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    unstab = ~df["stabilised"]
    ax.plot(df.loc[df["stabilised"], "Ur"], df.loc[df["stabilised"], "continuous_plateau_A_star"],
            "o-", color="tab:blue", label="Continuous sweep: stationary plateau")
    if unstab.any():
        ax.plot(df.loc[unstab, "Ur"], df.loc[unstab, "continuous_plateau_A_star"],
                "x", color="tab:blue", mfc="none",
                label="Continuous sweep: stationarity not reached")
    if fixed_ur_cfd:
        ax.plot(df["Ur"], df["cfd_fixed_ur_A_star"], "s--", color="black", label="Fixed-$U_r$ CFD")
    if fixed_ur_gru:
        ax.plot(df["Ur"], df["gru_fixed_ur_A_star"], "^--", color="tab:red", label="Independently initialised surrogate")
    ax.set_xlabel("$U_r$")
    ax.set_ylabel(r"$A^*=A/D$")
    ax.legend(fontsize=7 * font_scale)
    fig.tight_layout()

    return df, fig


LOCKIN_REGION_TRANSITIONS = {
    "onset": {"ur_before": 3.50, "ur_after": 4.00},
    "lockin": {"ur_before": 5.25, "ur_after": 5.50},
    "departure_from_lockin": {"ur_before": 6.00, "ur_after": 6.25},
}
POST_LOCKIN_APPENDIX_TRANSITION = {"ur_before": 7.00, "ur_after": 8.00}


def find_transition_time(schedule: dict, ur_before: float, ur_after: float) -> float:
    Ur_list = schedule["Ur_list"]
    transition_step_indices = schedule["transition_step_indices"]
    transition_steps = schedule.get("transition_steps", 0)
    dt = schedule["dt"]

    for i, (frm, to) in enumerate(zip(Ur_list[:-1], Ur_list[1:])):
        if np.isclose(frm, ur_before) and np.isclose(to, ur_after):
            return (transition_step_indices[i] + transition_steps) * dt

    raise ValueError(f"No {ur_before}->{ur_after} transition found in this "
                      f"schedule's Ur_list={Ur_list}.")


def plot_transition_window(result: dict, schedule: dict, D: float, fn: float,
                            ur_before: float, ur_after: float,
                            out_path_stem: Path,
                            window_before_Tn: float = 5.0, window_after_Tn: float = 15.0,
                            include_velocity: bool = False,
                            fixed_xlim: tuple[float, float] | None = None):
    apply_thesis_style()
    Tn = 1.0 / fn
    dt = float(np.median(np.diff(result["time"])))

    t_transition = find_transition_time(schedule, ur_before, ur_after)
    Ur_arr = result["Ur"]

    t0 = t_transition - window_before_Tn * Tn
    t1 = t_transition + window_after_Tn * Tn
    mask = (result["time"] >= t0) & (result["time"] <= t1)
    t_rel = result["time"][mask] - t_transition

    n_panels = 4 if include_velocity else 3
    fig, axes = plt.subplots(n_panels, 1, sharex=True, figsize=(6.0, 1.8 * n_panels + 1.0))

    axes[0].plot(t_rel, Ur_arr[mask], color="black")
    axes[0].axvline(0.0, color="gray", lw=0.8, ls="--")
    axes[0].annotate(f"{ur_before:g}$\\to${ur_after:g}", xy=(0.0, Ur_arr[mask].max()),
                      xytext=(2, -8), textcoords="offset points", fontsize=7, color="gray")
    axes[0].set_ylabel(r"$U_r(t)$")

    axes[1].plot(t_rel, result["displacement"][mask] / D, color="tab:blue", lw=0.7)
    axes[1].axvline(0.0, color="gray", lw=0.5, ls="--")
    axes[1].set_ylabel(r"$h/D$")

    panel_idx = 2
    if include_velocity:
        axes[panel_idx].plot(t_rel, result["velocity"][mask] / (fn * D), color="tab:orange", lw=0.7)
        axes[panel_idx].axvline(0.0, color="gray", lw=0.5, ls="--")
        axes[panel_idx].set_ylabel(r"$\dot h/(f_n D)$")
        panel_idx += 1

    env_t, env, reason = envelope_or_unavailable(
        result["displacement"][mask], D, dt, fn)
    if env_t is not None:
        axes[panel_idx].plot(env_t + t_rel[0], env, color="tab:red", lw=1.2)
    else:
        axes[panel_idx].text(0.5, 0.5, "envelope unavailable\n(insufficient cycles in window)",
                              ha="center", va="center", transform=axes[panel_idx].transAxes,
                              fontsize=8, color="gray")
        print(f"[plot_transition_window] {reason}")
    axes[panel_idx].axvline(0.0, color="gray", lw=0.5, ls="--")
    axes[panel_idx].set_ylabel(r"$A/D$ (envelope)")
    axes[panel_idx].set_xlabel(f"Time from transition [s]  ($T_n$={Tn:.3g}s)")

    if fixed_xlim is not None:
        for ax in axes:
            ax.set_xlim(fixed_xlim)

    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig(out_path_stem.with_suffix(".pdf"))
    fig.savefig(out_path_stem.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return out_path_stem.with_suffix(".pdf"), out_path_stem.with_suffix(".png")


def plot_transitions_composite_thesis(result: dict, schedule: dict, D: float, fn: float,
                                       transitions: dict[str, dict],
                                       out_path_stem: Path,
                                       window_before_Tn: float = 5.0, window_after_Tn: float = 15.0,
                                       fixed_xlim: tuple[float, float] | None = None,
                                       font_scale: float = 1.0):
    apply_thesis_style()
    if font_scale != 1.0:
        matplotlib.rcParams.update({
            "axes.labelsize": matplotlib.rcParams["axes.labelsize"] * font_scale,
            "axes.titlesize": matplotlib.rcParams["axes.titlesize"] * font_scale,
            "xtick.labelsize": matplotlib.rcParams["xtick.labelsize"] * font_scale,
            "ytick.labelsize": matplotlib.rcParams["ytick.labelsize"] * font_scale,
            "legend.fontsize": matplotlib.rcParams["legend.fontsize"] * font_scale,
        })
    Tn = 1.0 / fn
    dt = float(np.median(np.diff(result["time"])))
    t_star_full = t_star_of(result["time"], result["Ur"], fn)

    labels = list(transitions.keys())
    n = len(labels)
    fig, axes = plt.subplots(n, 2, figsize=(TEXT_WIDTH_IN, 2.0 * n + 0.3),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for row, label in enumerate(labels):
        tr = transitions[label]
        ur_before, ur_after = tr["ur_before"], tr["ur_after"]
        t_transition = find_transition_time(schedule, ur_before, ur_after)
        t_star_transition = float(np.interp(t_transition, result["time"], t_star_full))

        t0 = t_transition - window_before_Tn * Tn
        t1 = t_transition + window_after_Tn * Tn
        mask = (result["time"] >= t0) & (result["time"] <= t1)
        t_rel = result["time"][mask] - t_transition
        t_star_rel = t_star_full[mask] - t_star_transition

        title = f"{label.replace('_', ' ')}: $U_r={ur_before:g}\\to{ur_after:g}$"
        ax_h, ax_env = axes[row, 0], axes[row, 1]

        ax_h.plot(t_star_rel, result["displacement"][mask] / D, color="tab:blue", lw=0.7)
        ax_h.axvline(0.0, color="gray", lw=0.5, ls="--")
        ax_h.set_ylabel(r"$h/D$")
        ax_h.set_title(title, fontsize=8 * font_scale)

        env_t, env, reason = envelope_or_unavailable(result["displacement"][mask], D, dt, fn)
        if env_t is not None:
            env_seconds_rel = env_t + t_rel[0]
            env_star_rel = np.interp(env_seconds_rel, t_rel, t_star_rel)
            ax_env.plot(env_star_rel, env, color="tab:red", lw=1.2)
        else:
            ax_env.text(0.5, 0.5, "envelope unavailable\n(insufficient cycles in window)",
                        ha="center", va="center", transform=ax_env.transAxes,
                        fontsize=7, color="gray")
            print(f"[plot_transitions_composite_thesis] {label}: {reason}")
        ax_env.axvline(0.0, color="gray", lw=0.5, ls="--")
        ax_env.set_ylabel(r"$A/D$")
        ax_env.set_title(title, fontsize=8 * font_scale)

        ax_h.set_xlim(t_star_rel[0], t_star_rel[-1])
        ax_env.set_xlim(t_star_rel[0], t_star_rel[-1])

        if row == n - 1:
            xlabel = r"$\Delta t^*=(t-t_{\mathrm{transition}})U/D$"
            ax_h.set_xlabel(xlabel)
            ax_env.set_xlabel(xlabel)

    fig.savefig(out_path_stem.with_suffix(".pdf"))
    fig.savefig(out_path_stem.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return out_path_stem.with_suffix(".pdf"), out_path_stem.with_suffix(".png")


REPRESENTATIVE_SEED = 123


def aggregate_envelope_across_seeds(displacement_by_seed: dict[int, np.ndarray],
                                     D: float, dt: float, fn: float) -> dict:
    seeds = sorted(displacement_by_seed.keys())
    env_list, common_env_t = [], None
    for s in seeds:
        env_t, env, reason = envelope_or_unavailable(displacement_by_seed[s], D, dt, fn)
        if env_t is None:
            raise ValueError(f"seed {s}: {reason}")
        if common_env_t is None:
            common_env_t = env_t
        elif len(env_t) != len(common_env_t) or not np.allclose(env_t, common_env_t):
            raise ValueError(
                "seeds produced different envelope time grids -- did they "
                "all run the same schedule/dt?")
        env_list.append(env)

    env_stack = np.stack(env_list, axis=0)
    q25, q75 = np.percentile(env_stack, [25, 75], axis=0)
    return {
        "seeds": seeds,
        "envelope_time": common_env_t,
        "envelope_median": np.median(env_stack, axis=0),
        "envelope_q25": q25,
        "envelope_q75": q75,
    }


def aggregate_scalar_metrics_across_seeds(metrics_by_seed: dict[int, dict]) -> dict:
    seeds = sorted(metrics_by_seed.keys())
    keys = set(metrics_by_seed[seeds[0]].keys())
    for s in seeds[1:]:
        if set(metrics_by_seed[s].keys()) != keys:
            raise ValueError(f"seed {s} has different metric keys than seed {seeds[0]}")
    out = {"seeds": seeds}
    for k in sorted(keys):
        vals = [metrics_by_seed[s][k] for s in seeds]
        q25, q75 = np.percentile(vals, [25, 75])
        out[f"{k}_median"] = float(np.median(vals))
        out[f"{k}_iqr"] = float(q75 - q25)
    return out


def plot_multiseed_panels(results_by_seed: dict[int, dict], schedule: dict,
                           D: float, fn: float, out_path_stem: Path,
                           representative_seed: int = REPRESENTATIVE_SEED,
                           transition_labels: list[float] | None = None):
    apply_thesis_style()
    seeds = sorted(results_by_seed.keys())
    if representative_seed not in seeds:
        raise ValueError(f"representative_seed={representative_seed} not among {seeds}")

    rep = results_by_seed[representative_seed]
    t = rep["time"]
    dt = float(np.median(np.diff(t)))
    agg = aggregate_envelope_across_seeds(
        {s: results_by_seed[s]["displacement"] for s in seeds}, D, dt, fn)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7.0, 6.0))
    axes[0].plot(t, rep["Ur"], color="black")
    axes[0].set_ylabel(r"$U_r(t)$")

    for s in seeds:
        if s == representative_seed:
            continue
        axes[1].plot(results_by_seed[s]["time"], results_by_seed[s]["displacement"] / D,
                     color="gray", lw=0.4, alpha=0.35, zorder=1)
    axes[1].plot(t, rep["displacement"] / D, color="tab:blue", lw=0.8, zorder=2,
                label=f"seed {representative_seed} (predeclared representative)")
    axes[1].set_ylabel(r"$h/D$")
    axes[1].legend(fontsize=6, loc="upper right")

    axes[2].plot(agg["envelope_time"], agg["envelope_median"], color="tab:red", lw=1.2,
                label=f"median over {len(seeds)} seeds")
    axes[2].fill_between(agg["envelope_time"], agg["envelope_q25"], agg["envelope_q75"],
                        color="tab:red", alpha=0.2, label="IQR")
    axes[2].set_ylabel(r"$A/D$ (envelope)")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(fontsize=6)

    transition_times = schedule.get("transition_times", [])
    transition_ur_values = schedule.get("transition_ur_values", [])
    label_set = set(transition_labels) if transition_labels is not None else None
    for ax in axes:
        for tt in transition_times:
            ax.axvline(tt, color="gray", lw=0.5, alpha=0.5, ls="--")
    for tt, ur_val in zip(transition_times, transition_ur_values):
        if label_set is None or ur_val in label_set:
            axes[0].annotate(f"{ur_val:g}", xy=(tt, axes[0].get_ylim()[1]),
                            xytext=(2, -8), textcoords="offset points",
                            fontsize=7, color="gray")

    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig(out_path_stem.with_suffix(".pdf"))
    fig.savefig(out_path_stem.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return out_path_stem.with_suffix(".pdf"), out_path_stem.with_suffix(".png")
