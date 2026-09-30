"""Figures for the rollout-informed refinement (thesis Sec. 6.6.1, Fig. 6.11 and 6.12):
training progression, closed-loop error before/after refinement, and the
long displacement trace at Ur = 6.7385.

Reads the fixed result folders named at the top of this file and writes to
results/gru_bridge_p0_nd_context_noacc/modelA_v2/thesis_figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from viv_analysis.plotting.plot_style import (
    CFD_COLOR, ERROR_COLOR, MODEL_COLOR, SECONDARY_COLOR, TEXT_WIDTH_IN,
    apply_thesis_style,
)
from viv_analysis.config import config
from viv_analysis.utils import PROJECT_ROOT

MODELA_DIR = PROJECT_ROOT / "results" / "gru_bridge_p0_nd_context_noacc" / "modelA_v2"
P0_DIR = PROJECT_ROOT / "results" / "gru_bridge_p0_nd_context_noacc_pass5cmp_P0"
PASS5_DIR = PROJECT_ROOT / "results" / "gru_bridge_p0_nd_context_noacc_pass5cmp_pass5"
P0_SWEEP = P0_DIR / "sweep_results.csv"
PASS5_SWEEP = PASS5_DIR / "sweep_results.csv"
OUT_DIR = MODELA_DIR / "thesis_figures"

HORIZONS = ["0.5s", "3.125s", "6.5s", "10s", "20s"]


def plot_training_progression() -> tuple[Path, Path]:
    """Validation errors per refinement pass."""
    apply_thesis_style()
    import matplotlib.pyplot as plt

    history = json.loads(MODELA_DIR.joinpath("modelA_v2_history.json").read_text())
    baseline_r2 = history["baseline_p0"]["tf"]["mean_r2"]
    passes = history["passes"]
    pass_idx = [p["pass_idx"] for p in passes]
    tf_r2 = [p["tf_val_mean_r2"] for p in passes]

    fig, (ax_r2, ax_roll) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.4))

    ax_r2.axhline(baseline_r2, color=CFD_COLOR, lw=0.6, ls=":", label="P0 baseline")
    ax_r2.plot([0] + pass_idx, [baseline_r2] + tf_r2, color=MODEL_COLOR, marker="o", ms=4, lw=1.1)
    ax_r2.set_xlabel("Fine-tune pass")
    ax_r2.set_ylabel(r"Open-loop (teacher-forced) $R^2$")
    ax_r2.set_xticks([0] + pass_idx)
    ax_r2.legend(loc="lower left", frameon=False, fontsize=8)

    colors = [SECONDARY_COLOR, MODEL_COLOR, ERROR_COLOR, "#CC79A7", "#E69F00"]
    for h, color in zip(HORIZONS, colors):
        vals = [p["rollout_summary"][h]["disp_nrmse_median"] for p in passes]
        ax_roll.plot(pass_idx, vals, color=color, marker="s", ms=3.5, lw=0.6, label=h)
    ax_roll.set_xlabel("Fine-tune pass")
    ax_roll.set_ylabel("Rollout displacement NRMSE (median)")
    ax_roll.set_yscale("log")
    ax_roll.set_xticks(pass_idx)
    ax_roll.legend(loc="upper right", frameon=False, fontsize=7, title="Horizon", title_fontsize=7.5)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "rollout_refinement_training_progression"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return stem.with_suffix(".pdf"), stem.with_suffix(".png")


def plot_closed_loop_comparison() -> tuple[Path, Path]:
    """Closed-loop amplitude error, baseline vs refined model."""
    apply_thesis_style()
    import matplotlib.pyplot as plt

    p0 = pd.read_csv(P0_SWEEP).sort_values("Ur")
    p5 = pd.read_csv(PASS5_SWEEP).sort_values("Ur")

    cases = [f"Ur{u}" for u in p0["Ur"]]
    x = np.arange(len(cases))
    width = 0.35

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.4))
    p0_err = p0["P0_A_star_rel_error"].fillna(-1.0).to_numpy()
    p5_err = p5["pass5_A_star_rel_error"].fillna(-1.0).to_numpy()
    ax.bar(x - width / 2, p0_err, width, color=MODEL_COLOR, label="P0 baseline")
    ax.bar(x + width / 2, p5_err, width, color=ERROR_COLOR, label="Pass 5 (Model A refined)")
    ax.axhline(0.0, color="gray", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(cases, fontsize=8)
    ax.set_ylabel(r"$A^*$ relative error")
    ax.set_ylim(-1.35, 0.0)
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    for i, (l0, l5) in enumerate(zip(p0["P0_stability_label"], p5["pass5_stability_label"])):
        ax.annotate(l0.replace("_", " "), xy=(x[i] - width / 2, -1.33), rotation=90,
                    ha="center", va="bottom", fontsize=6.2, color=MODEL_COLOR)
        ax.annotate(l5.replace("_", " "), xy=(x[i] + width / 2, -1.33), rotation=90,
                    ha="center", va="bottom", fontsize=6.2, color=ERROR_COLOR)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "rollout_refinement_closed_loop"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)

    return stem.with_suffix(".pdf"), stem.with_suffix(".png")


def _load_coupled_case(case_dir: Path):
    """Load one coupled npz and its receipt."""
    npz_path = next(case_dir.glob("coupled_bridge_Ur6.7385_*.npz"))
    receipt = json.loads(npz_path.with_suffix(".receipt.json").read_text())
    d = np.load(npz_path, allow_pickle=True)
    return dict(
        t=d["t"], h=d["h"], h_cfd=d["h_cfd"], D=float(d["D"]),
        t_handoff=receipt["handoff_time_s"],
    )


def plot_representative_trace(font_scale: float = 1.0) -> tuple[Path, Path]:
    """Displacement at Ur = 6.7385: CFD, baseline and refined model."""
    apply_thesis_style()
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    if font_scale != 1.0:
        mpl.rcParams.update({
            "axes.labelsize": mpl.rcParams["axes.labelsize"] * font_scale,
            "xtick.labelsize": mpl.rcParams["xtick.labelsize"] * font_scale,
            "ytick.labelsize": mpl.rcParams["ytick.labelsize"] * font_scale,
            "legend.fontsize": mpl.rcParams["legend.fontsize"] * font_scale,
        })

    p0 = _load_coupled_case(P0_DIR)
    p5 = _load_coupled_case(PASS5_DIR)

    t_h = p0["t_handoff"]
    assert np.isclose(p0["t"][0], p0["t_handoff"], atol=1e-6)
    assert np.isclose(p5["t"][0], p5["t_handoff"], atol=1e-6)
    assert np.isclose(p5["t_handoff"], t_h, atol=1e-6), (
        f"P0 t_handoff={t_h} != pass5 t_handoff={p5['t_handoff']}"
    )
    D = p0["D"]
    Ur = 6.7385
    fn = config["bridge_fn_hz"]
    U = Ur * fn * D

    plot_duration = 200.0
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.6))

    n_cfd = len(p0["h_cfd"])
    t_cfd_rel = p0["t"][:n_cfd] - t_h
    cfd_mask = t_cfd_rel <= plot_duration
    ax.plot(t_cfd_rel[cfd_mask] * U / D, p0["h_cfd"][cfd_mask] / D,
            color=CFD_COLOR, lw=0.6, label="CFD reference")

    for label, case, color in [("Baseline model", p0, MODEL_COLOR),
                                ("Rollout-refined model", p5, ERROR_COLOR)]:
        t_rel = case["t"] - case["t_handoff"]
        mask = t_rel <= plot_duration
        ax.plot(t_rel[mask] * U / D, case["h"][mask] / D,
                color=color, lw=0.6, ls=(0, (4, 2)), label=label)

    ax.axvline(0.0, color="gray", lw=0.4, ls=":")
    ax.set_ylabel(r"$h/D$")
    ax.set_xlabel(r"$t^*=(t-t_{\mathrm{h}})U/D$")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False,
              borderaxespad=0.0)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "rollout_refinement_extended_Ur6p7385"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return stem.with_suffix(".pdf"), stem.with_suffix(".png")


def main():
    """Make all refinement figures."""
    p1 = plot_training_progression()
    print(f"Wrote {p1[0]}\nWrote {p1[1]}")
    p2 = plot_closed_loop_comparison()
    print(f"Wrote {p2[0]}\nWrote {p2[1]}")
    p3 = plot_representative_trace(font_scale=1 / 0.8)
    print(f"Wrote {p3[0]}\nWrote {p3[1]}")


if __name__ == "__main__":
    main()
