#!/usr/bin/env python3
"""Figure functions used by the regenerate_* scripts. Nothing here reads data files
or runs a model; every function takes arrays or dataframes and writes a figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from viv_analysis.plotting.plot_style import (
    CFD_COLOR, CFD_STYLE, ERROR_STYLE, GRID_COLOR, MODEL_COLOR, MODEL_STYLE,
    ORANGE_COLOR, SECONDARY_COLOR, TEXT_WIDTH_IN, apply_thesis_style,
)
from viv_analysis.utils import segment_by_time_gaps

HANDOFF_COLOR = "#777777"


def break_at_gaps(t: np.ndarray, *ys: np.ndarray):
    """Insert NaN at time gaps so a line plot does not connect separate pieces."""
    t = np.asarray(t)
    segs = segment_by_time_gaps(t)
    if len(segs) <= 1:
        return (t, *ys)
    break_idxs = [seg_end for seg_start, seg_end in segs[:-1]]
    t_out = np.insert(t.astype(float), break_idxs, np.nan)
    ys_out = tuple(np.insert(np.asarray(y).astype(float), break_idxs, np.nan) for y in ys)
    return (t_out, *ys_out)


def ur_tag(ur: float) -> str:
    """Ur as a file-name-safe tag with two decimals, e.g. 6.7385 -> 'Ur6p74'."""
    return f"Ur{ur:.2f}".replace(".", "p")


def downsample_for_display(x: np.ndarray, y: np.ndarray, max_points: int = 8000):
    """Thin a long time series for plotting (keeps the figure file small)."""
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    if n <= max_points:
        return x, y
    n_bins = max(1, max_points // 2)
    bin_edges = np.linspace(0, n, n_bins + 1).astype(int)
    xs, ys = [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if hi <= lo:
            continue
        seg_x, seg_y = x[lo:hi], y[lo:hi]
        i_min, i_max = int(np.argmin(seg_y)), int(np.argmax(seg_y))
        order = (i_min, i_max) if i_min <= i_max else (i_max, i_min)
        xs.extend(seg_x[list(order)]); ys.extend(seg_y[list(order)])
    return np.array(xs), np.array(ys)


def build_caption(condition_label: str, r2: float, zoom_duration: float,
                  quantity: str = "lift-coefficient") -> str:
    """Default caption text for an open-loop figure."""
    return (
        f"Open-loop {quantity} prediction for {condition_label}. "
        f"The upper panel shows the complete analysed interval, while the "
        f"lower panel enlarges the first {zoom_duration:g} seconds. "
        f"The model achieved $R^2={r2:.4f}$."
    )


def plot_tf_result_thesis(
    cl_pred: np.ndarray,
    cl_true: np.ndarray,
    times: np.ndarray,
    case_label: str,
    output_dir: Path,
    condition_label: str | None = None,
    physical_params_note: str | None = None,
    zoom_duration: float = 20.0,
    include_residual: bool = False,
    fn: float | None = None,
    U: float | None = None,
    D: float | None = None,
    font_scale: float = 1.0,
    linewidth_scale: float = 1.0,
    linewidth: float | None = None,
) -> dict:
    """Open-loop (teacher-forced) C_L prediction vs CFD, with a zoom panel and the residual."""
    apply_thesis_style()
    if font_scale != 1.0:
        mpl.rcParams.update({
            "axes.labelsize": mpl.rcParams["axes.labelsize"] * font_scale,
            "xtick.labelsize": mpl.rcParams["xtick.labelsize"] * font_scale,
            "ytick.labelsize": mpl.rcParams["ytick.labelsize"] * font_scale,
            "legend.fontsize": mpl.rcParams["legend.fontsize"] * font_scale,
        })
    if linewidth is not None:
        cfd_lw, model_lw = linewidth, linewidth
    else:
        cfd_lw = CFD_STYLE["linewidth"] * linewidth_scale
        model_lw = MODEL_STYLE["linewidth"] * linewidth_scale
    cfd_kwargs = {**CFD_STYLE, "linestyle": "-", "linewidth": cfd_lw}
    model_kwargs = {**MODEL_STYLE, "linestyle": "-", "linewidth": model_lw}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t = np.asarray(times) - times[0]
    cl_pred = np.asarray(cl_pred)
    cl_true = np.asarray(cl_true)
    residual = cl_pred - cl_true

    if U is not None and D is not None:
        t_disp, xlabel = t * U / D, r"$t^*=(t-t_0)U/D$"
    elif fn is not None:
        Tn = 1.0 / fn
        t_disp, xlabel = t / Tn, r"$(t-t_0)/T_n$"
    else:
        t_disp, xlabel = t, r"Time after evaluation start, $t-t_0$ [s]"

    n_rows = 3 if include_residual else 2
    height_ratios = [1.6, 1.0, 0.7] if include_residual else [1.6, 1.0]
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(TEXT_WIDTH_IN, 4.6 if not include_residual else 5.6),
        gridspec_kw={"height_ratios": height_ratios}, constrained_layout=True,
    )
    ax_full, ax_zoom = axes[0], axes[1]

    t_full, cl_true_full, cl_pred_full = break_at_gaps(t_disp, cl_true, cl_pred)
    ax_full.plot(t_full, cl_true_full, **cfd_kwargs)
    ax_full.plot(t_full, cl_pred_full, **model_kwargs)
    ax_full.set_ylabel(r"$C_L$")
    ax_full.grid(True, which="major")

    ax_full.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
    frameon=False,
    borderaxespad=0.0,
    columnspacing=1.2,
    handlelength=1.8,
    handletextpad=0.5,
    )

    zoom_mask = t <= zoom_duration
    ax_zoom.plot(t_disp[zoom_mask], cl_true[zoom_mask], **{**cfd_kwargs, "label": "_nolegend_"})
    ax_zoom.plot(t_disp[zoom_mask], cl_pred[zoom_mask], **{**model_kwargs, "label": "_nolegend_"})
    ax_zoom.set_ylabel(r"$C_L$")
    ax_zoom.grid(True, which="major")
    if not include_residual:
        ax_zoom.set_xlabel(xlabel)

    if include_residual:
        ax_res = axes[2]
        ax_res.plot(t_disp[zoom_mask], residual[zoom_mask], **ERROR_STYLE)
        ax_res.axhline(0.0, color=CFD_STYLE["color"], linewidth=0.6)
        ax_res.set_ylabel(r"Residual, $e_{C_L}=\widehat{C}_L-C_L^{\mathrm{CFD}}$")
        ax_res.set_xlabel(xlabel)
        ax_res.grid(True, which="major")

    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"open_loop_{case_label}.{suffix}")
    plt.close(fig)

    from sklearn.metrics import r2_score
    r2 = float(r2_score(cl_true, cl_pred))
    caption = build_caption(condition_label or case_label, r2, zoom_duration)
    if physical_params_note:
        caption += f" Physical parameters: {physical_params_note}."
    caption_path = output_dir / f"open_loop_{case_label}.caption.txt"
    caption_path.write_text(caption + "\n")

    return dict(r2=r2, caption=caption,
               pdf_path=output_dir / f"open_loop_{case_label}.pdf",
               png_path=output_dir / f"open_loop_{case_label}.png",
               caption_path=caption_path)


def plot_coupled_thesis(
    t: np.ndarray, h: np.ndarray, CL: np.ndarray, D: float,
    CFD_t: np.ndarray, CFD_h: np.ndarray, CFD_cl: np.ndarray,
    case_label: str,
    output_dir: Path,
    condition_label: str | None = None,
    t_handoff: float | None = None,
    h_mode: str = "raw",
    dataset_note: str | None = None,
    max_display_points: int = 8000,
    fn: float | None = None,
    U: float | None = None,
    font_scale: float = 1.0,
    linewidth: float = 0.6,
) -> dict:
    """Closed-loop displacement vs CFD for one case."""
    apply_thesis_style()
    if font_scale != 1.0:
        mpl.rcParams.update({
            "axes.labelsize": mpl.rcParams["axes.labelsize"] * font_scale,
            "xtick.labelsize": mpl.rcParams["xtick.labelsize"] * font_scale,
            "ytick.labelsize": mpl.rcParams["ytick.labelsize"] * font_scale,
            "legend.fontsize": mpl.rcParams["legend.fontsize"] * font_scale,
        })
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t, h, CL = np.asarray(t), np.asarray(h), np.asarray(CL)
    CFD_t, CFD_h, CFD_cl = np.asarray(CFD_t), np.asarray(CFD_h), np.asarray(CFD_cl)

    if h_mode == "mean_removed":
        h_disp = h - np.mean(h)
        CFD_h_disp = CFD_h - np.mean(CFD_h)
        h_ylabel = r"$(h-\bar{h})/D$"
    elif h_mode == "raw":
        h_disp, CFD_h_disp = h, CFD_h
        h_ylabel = r"$h/D$"
    else:
        raise ValueError(f"h_mode must be 'raw' or 'mean_removed', got {h_mode!r}")

    if U is not None:
        t_plot, CFD_t_plot = t * U / D, CFD_t * U / D
        t_handoff_plot = t_handoff * U / D if t_handoff is not None else None
        time_xlabel = r"$t^*=tU/D$"
    elif fn is not None:
        Tn = 1.0 / fn
        t0 = t_handoff if t_handoff is not None else 0.0
        t_plot, CFD_t_plot = (t - t0) / Tn, (CFD_t - t0) / Tn
        t_handoff_plot = 0.0 if t_handoff is not None else None
        time_xlabel = r"$(t-t_{\mathrm{h}})/T_n$"
    else:
        t_plot, CFD_t_plot = t, CFD_t
        t_handoff_plot = t_handoff
        time_xlabel = r"Time, $t$ [s]"

    t_d, h_d = downsample_for_display(t_plot, h_disp / D, max_display_points)
    CFD_t_d, CFD_h_d = downsample_for_display(CFD_t_plot, CFD_h_disp / D, max_display_points)
    t_d2, CL_d = downsample_for_display(t_plot, CL, max_display_points)
    CFD_t_d2, CFD_cl_d = downsample_for_display(CFD_t_plot, CFD_cl, max_display_points)

    cfd_kwargs = {**CFD_STYLE, "linestyle": "-", "linewidth": linewidth}
    model_kwargs = {**MODEL_STYLE, "linestyle": "-", "linewidth": linewidth, "label": "GRU-coupled response"}

    fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH_IN, 4.5), sharex=True,
                             constrained_layout=True)

    axes[0].plot(CFD_t_d, CFD_h_d, **cfd_kwargs)
    axes[0].plot(t_d, h_d, **model_kwargs)
    axes[0].set_ylabel(h_ylabel)
    axes[0].legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
    frameon=False,
    borderaxespad=0.0,
    columnspacing=1.2,
    handlelength=1.8,
    handletextpad=0.5,
    )

    axes[1].plot(CFD_t_d2, CFD_cl_d, **{**cfd_kwargs, "label": "_nolegend_"})
    axes[1].plot(t_d2, CL_d, **{**model_kwargs, "label": "_nolegend_"})
    axes[1].set_ylabel(r"$C_L$")
    axes[1].set_xlabel(time_xlabel)

    for ax in axes:
        ax.grid(True, which="major")
        if t_handoff_plot is not None:
            ax.axvline(t_handoff_plot, color=HANDOFF_COLOR, linestyle=":", linewidth=0.9)

    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"closed_loop_{case_label}.{suffix}")
    plt.close(fig)

    caption = (
        f"Closed-loop (coupled GRU-structural) response for {condition_label or case_label}"
        + (f" ({dataset_note})" if dataset_note else "") + ". "
        + ("Displacement is shown with the post-handoff mean removed. "
           if h_mode == "mean_removed" else "")
        + ("The handoff from CFD-history warm-start to fully coupled rollout is marked "
           "with a dotted vertical line. " if t_handoff is not None else "")
        + "CFD reference in black, GRU-coupled response in blue."
    )
    caption_path = output_dir / f"closed_loop_{case_label}.caption.txt"
    caption_path.write_text(caption + "\n")

    return dict(caption=caption,
               pdf_path=output_dir / f"closed_loop_{case_label}.pdf",
               png_path=output_dir / f"closed_loop_{case_label}.png",
               caption_path=caption_path)


AMPLITUDE_DEFINITION = (
    r"$A^*=A/D$, defined as half the peak-to-peak envelope of the steady-state "
    r"response (final 30\% of the simulated trajectory), computed identically "
    r"for CFD and every model series."
)


def plot_amplitude_response_thesis(
    df,
    model_column: str,
    output_dir: Path,
    model_label: str = "GRU-coupled response",
    cfd_column: str = "CFD",
    ur_column: str = "Ur",
    literature_columns: dict | None = None,
    dataset_note: str | None = None,
    out_name: str = "amplitude_response",
    font_scale: float = 1.0,
) -> dict:
    """Closed-loop amplitude A* over Ur, surrogate vs CFD (and literature data if given)."""
    apply_thesis_style()
    if font_scale != 1.0:
        mpl.rcParams.update({
            "axes.labelsize": mpl.rcParams["axes.labelsize"] * font_scale,
            "xtick.labelsize": mpl.rcParams["xtick.labelsize"] * font_scale,
            "ytick.labelsize": mpl.rcParams["ytick.labelsize"] * font_scale,
            "legend.fontsize": mpl.rcParams["legend.fontsize"] * font_scale,
        })
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.sort_values(ur_column)

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.7), constrained_layout=True)

    ax.plot(df[ur_column], df[cfd_column], color=CFD_COLOR, marker="o",
           linestyle="-", linewidth=1.2, markersize=4.5, label="CFD reference")
    ax.plot(df[ur_column], df[model_column], color=MODEL_COLOR, marker="s",
           markerfacecolor="white", markeredgecolor=MODEL_COLOR,
           linestyle="--", linewidth=1.1, markersize=4.5, label=model_label)

    _lit_color_by_name = {"orange": ORANGE_COLOR, "green": SECONDARY_COLOR}
    for col, spec in (literature_columns or {}).items():
        if col not in df.columns:
            raise KeyError(f"literature_columns references missing column '{col}'")
        color = _lit_color_by_name.get(spec.get("color", "orange"), ORANGE_COLOR)
        ax.plot(df[ur_column], df[col], color=color, marker="^",
               markerfacecolor="white", markeredgecolor=color,
               linestyle="-.", linewidth=1.0, markersize=4.5, label=spec["label"])

    ax.set_xlabel(r"Reduced velocity, $U_r$")
    ax.set_ylabel(r"$A^*=A/D$")
    ax.grid(True, which="major")
    ax.legend(loc="best")

    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{out_name}.{suffix}")
    plt.close(fig)

    caption = (
        f"Closed-loop amplitude response"
        + (f" ({dataset_note})" if dataset_note else "") + ". "
        + AMPLITUDE_DEFINITION
    )
    caption_path = output_dir / f"{out_name}.caption.txt"
    caption_path.write_text(caption + "\n")

    return dict(caption=caption,
               pdf_path=output_dir / f"{out_name}.pdf",
               png_path=output_dir / f"{out_name}.png",
               caption_path=caption_path)
PARTITION_MARKERS = {
    "train": {"marker": "o", "markersize": 5.0},
    "val": {"marker": "s", "markersize": 4.5},
    "test": {"marker": "^", "markersize": 5.0},
}
PARTITION_LABELS = {"train": "Training case", "val": "Validation case", "test": "Test case"}
REFERENCE_STATUS_EXCLUDED = ("insufficient_duration", "numerically_suspect")


def plot_amplitude_response_status_aware_thesis(
    df,
    model_column: str,
    status_column: str,
    partition_column: str,
    output_dir: Path,
    model_label: str = "GRU-coupled response",
    cfd_column: str = "CFD",
    ur_column: str = "Ur",
    dataset_note: str | None = None,
    out_name: str = "amplitude_response_status_aware",
    font_scale: float = 1.0,
) -> dict:
    """Bridge amplitude response where only cases with a settled CFD limit cycle get an amplitude point."""
    apply_thesis_style()
    if font_scale != 1.0:
        mpl.rcParams.update({
            "axes.labelsize": mpl.rcParams["axes.labelsize"] * font_scale,
            "xtick.labelsize": mpl.rcParams["xtick.labelsize"] * font_scale,
            "ytick.labelsize": mpl.rcParams["ytick.labelsize"] * font_scale,
            "legend.fontsize": mpl.rcParams["legend.fontsize"] * font_scale,
        })
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.sort_values(ur_column).reset_index(drop=True)
    excluded_mask = df[status_column].isin(REFERENCE_STATUS_EXCLUDED)
    scored = df[~excluded_mask]
    is_settled = scored[status_column] == "settled_lco"

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.9), constrained_layout=True)

    ax.plot(scored[ur_column], scored[cfd_column], color=CFD_COLOR,
           linestyle="-", linewidth=1.1, zorder=2, label="_nolegend_")
    ax.plot(scored[ur_column], scored[model_column], color=MODEL_COLOR,
           linestyle="--", linewidth=1.0, zorder=2, label="_nolegend_")

    for partition, style in PARTITION_MARKERS.items():
        sub = scored[scored[partition_column] == partition]
        if sub.empty:
            continue
        ax.plot(sub[ur_column], sub[cfd_column], color=CFD_COLOR, linestyle="none",
               markerfacecolor=CFD_COLOR, markeredgecolor=CFD_COLOR, zorder=3, **style)
        ax.plot(sub[ur_column], sub[model_column], color=MODEL_COLOR, linestyle="none",
               markerfacecolor="white", markeredgecolor=MODEL_COLOR, zorder=3, **style)

    if excluded_mask.any():
        for ur in df.loc[excluded_mask, ur_column]:
            ax.axvline(ur, color=GRID_COLOR, linestyle=":", linewidth=0.9,
                      ymax=0.045, zorder=1)

    legend_handles = [
        Line2D([0], [0], color=CFD_COLOR, linestyle="-", label="CFD reference"),
        Line2D([0], [0], color=MODEL_COLOR, linestyle="--", label=model_label),
    ]
    for partition, style in PARTITION_MARKERS.items():
        if (df[partition_column] == partition).any():
            legend_handles.append(Line2D([0], [0], color="0.3", linestyle="none",
                                        markerfacecolor="0.3", markeredgecolor="0.3",
                                        label=PARTITION_LABELS[partition], **style))

    ax.set_xlabel(r"Reduced velocity, $U_r$")
    ax.set_ylabel(r"$A^*=A/D$")
    ax.grid(True, which="major")
    ax.legend(handles=legend_handles, loc="best", fontsize=6.5 * font_scale)

    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{out_name}.{suffix}")
    plt.close(fig)

    n_settled = int(is_settled.sum())
    n_total = len(df)

    return dict(pdf_path=output_dir / f"{out_name}.pdf",
               png_path=output_dir / f"{out_name}.png",
               n_settled_lco=n_settled, n_total=n_total)


def plot_learning_curve_thesis(
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    output_dir: Path,
    case_label: str,
    condition_label: str | None = None,
    epoch_start: int = 1,
    out_name: str = "learning_curve",
    font_scale: float = 1.0,
) -> dict:
    """Training and validation loss over the epochs."""
    apply_thesis_style()
    if font_scale != 1.0:
        mpl.rcParams.update({
            "axes.labelsize": mpl.rcParams["axes.labelsize"] * font_scale,
            "xtick.labelsize": mpl.rcParams["xtick.labelsize"] * font_scale,
            "ytick.labelsize": mpl.rcParams["ytick.labelsize"] * font_scale,
            "legend.fontsize": mpl.rcParams["legend.fontsize"] * font_scale,
        })
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_losses = np.asarray(train_losses, dtype=float)
    val_losses = np.asarray(val_losses, dtype=float)
    epochs = np.arange(epoch_start, epoch_start + len(train_losses))
    best_idx = int(np.argmin(val_losses))
    best_epoch = int(epochs[best_idx])

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.2), constrained_layout=True)
    ax.plot(epochs, train_losses, **{**CFD_STYLE, "label": "Training"})
    ax.plot(epochs, val_losses, **{**MODEL_STYLE, "label": "Validation"})
    ax.axvline(best_epoch, color="0.45", linestyle=":", linewidth=0.9)
    ax.plot(best_epoch, val_losses[best_idx], "o", color=MODEL_COLOR, markersize=4,
           zorder=4, label="_nolegend_")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Mean-squared error of standardised $C_L$")
    ax.grid(True, which="major")
    ax.legend(loc="best")

    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{out_name}_{case_label}.{suffix}")
    plt.close(fig)

    caption = (
        f"Training and validation losses for {condition_label or case_label}. "
        f"The marker identifies the checkpoint with the minimum validation loss "
        f"(epoch {best_epoch}), which was retained for subsequent evaluation."
    )
    caption_path = output_dir / f"{out_name}_{case_label}.caption.txt"
    caption_path.write_text(caption + "\n")

    return dict(best_epoch=best_epoch, caption=caption,
               pdf_path=output_dir / f"{out_name}_{case_label}.pdf",
               png_path=output_dir / f"{out_name}_{case_label}.png",
               caption_path=caption_path)


def plot_open_loop_representative(
    cl_pred: np.ndarray,
    cl_true: np.ndarray,
    times: np.ndarray,
    case_label: str,
    output_dir: Path,
    condition_label: str | None = None,
    physical_params_note: str | None = None,
    zoom_duration: float = 20.0,
    out_name: str = "open_loop_representative",
    width_in: float | None = None,
    font_scale: float = 1.0,
    fn: float | None = None,
    U: float | None = None,
    D: float | None = None,
    linewidth_scale: float = 1.0,
    linewidth: float | None = None,
) -> dict:
    """Main-text open-loop figure for one representative test case (full record, zoom, residual)."""
    apply_thesis_style()
    if font_scale != 1.0:
        mpl.rcParams.update({
            "axes.labelsize": mpl.rcParams["axes.labelsize"] * font_scale,
            "xtick.labelsize": mpl.rcParams["xtick.labelsize"] * font_scale,
            "ytick.labelsize": mpl.rcParams["ytick.labelsize"] * font_scale,
            "legend.fontsize": mpl.rcParams["legend.fontsize"] * font_scale,
        })
    if linewidth is not None:
        cfd_lw, model_lw = linewidth, linewidth
    else:
        cfd_lw = CFD_STYLE["linewidth"] * linewidth_scale
        model_lw = MODEL_STYLE["linewidth"] * linewidth_scale
    cfd_kwargs = {**CFD_STYLE, "linestyle": "-", "linewidth": cfd_lw}
    model_kwargs = {**MODEL_STYLE, "linestyle": "-", "linewidth": model_lw}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width_in = width_in or TEXT_WIDTH_IN
    height_in = 6.2 * (width_in / TEXT_WIDTH_IN)

    t = np.asarray(times) - times[0]
    cl_pred = np.asarray(cl_pred)
    cl_true = np.asarray(cl_true)
    residual = cl_pred - cl_true
    zoom_mask = t <= zoom_duration

    if U is not None and D is not None:
        t_disp, xlabel = t * U / D, r"$t^*=(t-t_0)U/D$"
    elif fn is not None:
        Tn = 1.0 / fn
        t_disp, xlabel = t / Tn, r"$(t-t_0)/T_n$"
    else:
        t_disp, xlabel = t, r"Time after evaluation start, $t-t_0$ [s]"

    fig, (ax_full, ax_zoom, ax_res) = plt.subplots(
        3, 1, figsize=(width_in, height_in),
        gridspec_kw={"height_ratios": [1.6, 1.0, 0.7]}, constrained_layout=True,
    )

    t_full, cl_true_full, cl_pred_full = break_at_gaps(t_disp, cl_true, cl_pred)
    ax_full.plot(t_full, cl_true_full, **cfd_kwargs)
    ax_full.plot(t_full, cl_pred_full, **model_kwargs)
    ax_full.set_ylabel(r"$C_L$")
    ax_full.grid(True, which="major")

    ax_full.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
    frameon=False,
    borderaxespad=0.0,
    columnspacing=1.2,
    handlelength=1.8,
    handletextpad=0.5,
    )

    ax_zoom.plot(t_disp[zoom_mask], cl_true[zoom_mask], **{**cfd_kwargs, "label": "_nolegend_"})
    ax_zoom.plot(t_disp[zoom_mask], cl_pred[zoom_mask], **{**model_kwargs, "label": "_nolegend_"})
    ax_zoom.set_ylabel(r"$C_L$")
    ax_zoom.grid(True, which="major")

    ax_res.plot(t_disp[zoom_mask], residual[zoom_mask], **ERROR_STYLE)
    ax_res.axhline(0.0, color=CFD_STYLE["color"], linewidth=0.6)
    ax_res.set_ylabel(r"$e_{C_L}$")
    ax_res.set_xlabel(xlabel)
    ax_res.grid(True, which="major")

    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{out_name}_{case_label}.{suffix}")
    plt.close(fig)

    from sklearn.metrics import r2_score
    r2 = float(r2_score(cl_true, cl_pred))
    caption = (
        f"Open-loop lift-coefficient prediction for {condition_label or case_label}, "
        f"the representative test case for the main text. The upper panel shows the "
        f"complete analysed interval, the middle panel enlarges the first "
        f"{zoom_duration:g} seconds, and the lower panel shows the residual "
        f"$e_{{C_L}}$ over that same interval. The model achieved $R^2={r2:.4f}$."
    )
    if physical_params_note:
        caption += f" Physical parameters: {physical_params_note}."
    caption_path = output_dir / f"{out_name}_{case_label}.caption.txt"
    caption_path.write_text(caption + "\n")

    return dict(r2=r2, caption=caption,
               pdf_path=output_dir / f"{out_name}_{case_label}.pdf",
               png_path=output_dir / f"{out_name}_{case_label}.png",
               caption_path=caption_path)


def build_metrics_table(rows: list[dict], output_dir: Path, out_name: str = "test_case_metrics",
                        caption: str | None = None) -> dict:
    """LaTeX table of open-loop metrics per case."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Test case & $R^2$ & RMSE & MAE \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(f"{r['case']} & {r['r2']:.4f} & {r['rmse']:.4f} & {r['mae']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines) + "\n"

    tex_path = output_dir / f"{out_name}.tex"
    tex_path.write_text(tex)

    import pandas as pd
    csv_path = output_dir / f"{out_name}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    caption_path = None
    if caption:
        caption_path = output_dir / f"{out_name}.caption.txt"
        caption_path.write_text(caption + "\n")

    return dict(tex_path=tex_path, csv_path=csv_path, caption_path=caption_path)
