"""Figures for the residual forcing test (thesis Fig. 6.13): displacement traces at
Ur = 6.7385 for the deterministic GRU, frequency-matched residual forcing and
white noise, compared with CFD.

Reads the fixed result folders named at the top of this file and writes to
<residual forcing folder>/thesis_figures/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from viv_analysis.config import config
from viv_analysis.evaluate_all import load_full_cfd_df
from viv_analysis.plotting.plot_style import (
    CFD_COLOR, ERROR_COLOR, MODEL_COLOR, SECONDARY_COLOR, TEXT_WIDTH_IN,
    apply_thesis_style,
)
from viv_analysis.utils import PROJECT_ROOT, format_ur_label

EVAL_DIR = PROJECT_ROOT / "results" / "gru_bridge_nd_context_noacc_stochastic_closure_coupled_eval"
BASELINE_DIR = PROJECT_ROOT / "results" / "gru_bridge_nd_context_noacc_v1_coupled_eval"
OUT_DIR = EVAL_DIR / "thesis_figures"

INCLUDE_WIDTH_FRAC = 0.8
FIG_WIDTH_IN = TEXT_WIDTH_IN * INCLUDE_WIDTH_FRAC

MODE_COLOR = {
    "none": MODEL_COLOR,
    "surrogate": SECONDARY_COLOR,
    "white": ERROR_COLOR,
    "replay": "#CC79A7",
}
MODE_LABEL = {
    "none": "Deterministic\n(no noise)",
    "surrogate": "Surrogate\nnoise (5 seeds)",
    "white": "White\nnoise (5 seeds)",
    "replay": "Replay\n(1 trace)",
}


def _seed_from_name(npz_name: str) -> int | None:
    """Forcing seed from a result file name."""
    m = re.search(r"seed(\d+)", npz_name)
    return int(m.group(1)) if m else None


def plot_amplitude_error_by_mode() -> tuple[Path, Path]:
    """Amplitude and stability outcome per forcing mode."""
    apply_thesis_style()
    import matplotlib.pyplot as plt

    df = pd.read_csv(EVAL_DIR / "stochastic_closure_summary.csv")
    df["seed"] = df["npz"].apply(_seed_from_name)
    cfd_a_star = float(df["cfd_A_star"].dropna().iloc[0])

    modes = ["none", "replay", "surrogate", "white"]
    fig, (ax_a, ax_stab) = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 3.4 * INCLUDE_WIDTH_FRAC),
                                         gridspec_kw={"width_ratios": [1.3, 1.0]})

    x_positions, x_labels = [], []
    for i, mode in enumerate(modes):
        sub = df[df["noise_mode"] == mode]
        vals = sub["A_star"].dropna().to_numpy()
        x_positions.append(i)
        x_labels.append(MODE_LABEL[mode])
        color = MODE_COLOR[mode]
        if len(vals) == 0:
            continue
        if len(vals) == 1:
            ax_a.scatter([i], vals, color=color, s=28, zorder=3)
        else:
            med = np.median(vals)
            ax_a.scatter([i] * len(vals), vals, color=color, s=14, alpha=0.55, zorder=2)
            ax_a.scatter([i], [med], color=color, marker="_", s=260, linewidths=2.2, zorder=3)

    ax_a.axhline(cfd_a_star, color=CFD_COLOR, lw=1.0, ls="-", label="CFD reference")
    ax_a.set_xticks(x_positions)
    ax_a.set_xticklabels(x_labels, fontsize=7.8)
    ax_a.set_ylabel(r"$A^*$ (closed-loop response amplitude)")
    ax_a.legend(loc="upper right", frameon=False, fontsize=8)

    stab_order = ["divergence", "decay_to_rest", "stationary_lco"]
    stab_colors = {"divergence": ERROR_COLOR, "decay_to_rest": MODEL_COLOR, "stationary_lco": SECONDARY_COLOR}
    counts = np.zeros((len(modes), len(stab_order)))
    for i, mode in enumerate(modes):
        sub = df[df["noise_mode"] == mode]
        for j, lab in enumerate(stab_order):
            counts[i, j] = (sub["stability_label"] == lab).sum()

    bottom = np.zeros(len(modes))
    for j, lab in enumerate(stab_order):
        ax_stab.bar(x_positions, counts[:, j], bottom=bottom, color=stab_colors[lab],
                    width=0.6, label=lab.replace("_", " "))
        bottom += counts[:, j]
    ax_stab.set_xticks(x_positions)
    ax_stab.set_xticklabels(x_labels, fontsize=7.8)
    ax_stab.set_ylabel("Number of runs")
    ax_stab.legend(loc="upper right", frameon=False, fontsize=7.5)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "stochastic_closure_amplitude_and_stability"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)

    return stem.with_suffix(".pdf"), stem.with_suffix(".png")


def _load_case(npz_path: Path, cfd_df):
    """Load one coupled npz."""
    d = np.load(npz_path, allow_pickle=True)
    receipt = json.loads(npz_path.with_suffix(".receipt.json").read_text())
    Ur = float(d["Ur"])
    case_label = format_ur_label(Ur)
    case_df = cfd_df[cfd_df["case"].astype(str) == case_label]
    return dict(
        t=d["t"], h=d["h"], D=float(d["D"]),
        t_handoff=receipt.get("handoff_time_s"),
        CFD_t=case_df["time"].to_numpy(), CFD_h=case_df["disp"].to_numpy(),
    )


def plot_representative_traces() -> tuple[Path, Path]:
    """Representative displacement traces per forcing mode."""
    apply_thesis_style()
    import matplotlib.pyplot as plt

    print("Loading bridge CFD dataset...")
    cfd_df = load_full_cfd_df("bridge")

    files = {
        "none": next(BASELINE_DIR.glob("*Ur6.7385*noise-none*.npz")),
        "surrogate": next(EVAL_DIR.glob("*noise-surrogate*seed0*.npz")),
        "white": next(EVAL_DIR.glob("*noise-white*seed0*.npz")),
    }
    cases = {mode: _load_case(p, cfd_df) for mode, p in files.items()}
    D = cases["none"]["D"]
    Ur = 6.7385
    fn = config["bridge_fn_hz"]
    U = Ur * fn * D

    t_h = cases["none"]["t_handoff"]
    for mode, c in cases.items():
        assert np.isclose(c["t"][0], c["t_handoff"], atol=1e-6), (
            f"mode={mode}: t[0]={c['t'][0]} != t_handoff={c['t_handoff']}"
        )
        assert np.isclose(c["t_handoff"], t_h, atol=1e-6), (
            f"mode={mode}: t_handoff={c['t_handoff']} != baseline t_handoff={t_h}"
        )

    plot_duration = 200.0

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.6 * INCLUDE_WIDTH_FRAC))

    cfd = cases["none"]
    t_cfd_rel = cfd["CFD_t"] - t_h
    cfd_mask = (t_cfd_rel >= 0.0) & (t_cfd_rel <= plot_duration)
    ax.plot(t_cfd_rel[cfd_mask] * U / D, (cfd["CFD_h"][cfd_mask] - np.mean(cfd["CFD_h"][cfd_mask])) / D,
            color=CFD_COLOR, lw=1.0, label="CFD reference")

    legend_label = dict(MODE_LABEL, surrogate="Frequency-matched\nresidual")

    for mode in ["none", "surrogate", "white"]:
        c = cases[mode]
        t_model_rel = c["t"] - c["t_handoff"]
        mask = t_model_rel <= plot_duration
        ax.plot(t_model_rel[mask] * U / D, (c["h"][mask] - np.mean(c["h"][mask])) / D,
                color=MODE_COLOR[mode], lw=0.95,
                ls=("-" if mode == "none" else (0, (4, 2))),
                label=legend_label[mode].replace("\n", " "))

    ax.axvline(0.0, color="gray", lw=0.6, ls=":")
    ax.set_ylabel(r"$(h-\bar h)/D$")
    ax.set_xlabel(r"$t^*=(t-t_{\mathrm{h}})U/D$")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False,
              borderaxespad=0.0, fontsize=7.8)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "stochastic_closure_representative_traces"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return stem.with_suffix(".pdf"), stem.with_suffix(".png")


def main():
    """Make the residual forcing figures."""
    p1 = plot_amplitude_error_by_mode()
    print(f"Wrote {p1[0]}\nWrote {p1[1]}")
    p2 = plot_representative_traces()
    print(f"Wrote {p2[0]}\nWrote {p2[1]}")


if __name__ == "__main__":
    main()
