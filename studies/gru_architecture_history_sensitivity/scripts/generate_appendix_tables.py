"""LaTeX tables for thesis Sec. 5.2.1 and Appendix C (architecture and history-length sensitivity) and the case partitions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import STUDY_ROOT

REPORTS = STUDY_ROOT / "reports"
MANIFESTS = STUDY_ROOT / "manifests"
REQUIRED_SEEDS = [123, 456, 789]


def esc(s) -> str:
    return str(s).replace("_", r"\_")


def fmt(median, iqr, decimals=4) -> str:
    if median is None or (isinstance(median, float) and np.isnan(median)):
        return "n/a"
    if iqr is None or (isinstance(iqr, float) and np.isnan(iqr)):
        return f"{median:.{decimals}f}"
    return f"{median:.{decimals}f} [{iqr:.{decimals}f}]"


def hours(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "n/a"
    return f"{s / 3600.0:.1f}"


def load(dataset: str, stage: int, kind: str) -> pd.DataFrame:
    path = REPORTS / f"{kind}_{dataset}_stage{stage}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "history_label" in df.columns:
        df["history_label"] = df["history_label"].astype(object).where(df["history_label"].notna(), None)
    return df


def architecture_table(dataset: str, label: str) -> str:
    pareto = load(dataset, 1, "pareto")
    pareto = pareto.sort_values("hierarchical_rank")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{label}: architecture sensitivity (Stage 1), ranked by validation-only hierarchical selection. "
        r"Median [IQR] across the 3 seeds " + str(REQUIRED_SEEDS) + r".}",
        rf"\label{{tab:appendix_arch_{dataset}}}",
    ]
    if dataset == "bridge":
        lines += [
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{cccccccc}",
            r"\toprule",
            r"Rank & $H$ & $L$ & Params & Open-loop $R^2$ & $N_{\text{non-LCO}}$ & "
            r"$|f_{\text{osc}}|$ rel.\ err.\ & RMS ratio dev.\ from 1 \\",
            r"\midrule",
        ]
    else:
        lines += [
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{ccccccc}",
            r"\toprule",
            r"Rank & $H$ & $L$ & Params & Open-loop $R^2$ & Stable/$N$ & Amp.\ error \\",
            r"\midrule",
        ]

    for _, r in pareto.iterrows():
        rank = int(r["hierarchical_rank"])
        H, L = int(r["hidden_size"]), int(r["num_layers"])
        params = int(r["trainable_parameter_count_median"])
        r2 = fmt(r["open_loop_val_r2_median"], r["open_loop_val_r2_iqr"])
        marker = r" $\ast$" if rank == 1 else ""
        if dataset == "bridge":
            n_non_lco = r.get("closed_loop_validation_n_non_lco_median")
            f_osc = fmt(r.get("closed_loop_validation_median_abs_f_osc_rel_error_median"),
                        r.get("closed_loop_validation_median_abs_f_osc_rel_error_iqr"), 3)
            rms_dev = r.get("_bridge_non_lco_rms_ratio_deviation_median")
            rms_dev_s = f"{rms_dev:.3f}" if pd.notna(rms_dev) else "n/a"
            lines.append(f"{rank}{marker} & {H} & {L} & {params:,} & {r2} & "
                          f"{int(n_non_lco) if pd.notna(n_non_lco) else 'n/a'} & {f_osc} & {rms_dev_s} \\\\")
        else:
            stable = r.get("closed_loop_validation_stable_count_median")
            n = r.get("closed_loop_validation_n_median")
            amp = fmt(r.get("closed_loop_validation_median_abs_rel_amp_error_median"),
                      r.get("closed_loop_validation_median_abs_rel_amp_error_iqr"), 3)
            stable_s = f"{stable:.0f}/{n:.0f}" if pd.notna(stable) and pd.notna(n) else "n/a"
            lines.append(f"{rank}{marker} & {H} & {L} & {params:,} & {r2} & {stable_s} & {amp} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"}"]
    lines.append(r"\begin{minipage}{0.95\linewidth}\vspace{4pt}\footnotesize")
    lines.append(r"$\ast$ selected configuration. Selection basis: validation partition only "
                  r"(open-loop retention gate, then full-duration closed-loop ranking); "
                  r"see Sec.~\ref{sec:appendix_validation_confirmation}.")
    if dataset == "bridge":
        lines.append(r" Every bridge validation case classifies as non-limit-cycle "
                      r"(statistically stationary or slowly-evolving LES, never settled LCO) under "
                      r"\texttt{reference\_quality.classify\_reference\_status} -- confirmed for all 18 "
                      r"configurations, not assumed -- so the amplitude-gate stable-count metric used for "
                      r"cylinder does not apply here. Ranking instead uses (1) the RMS ratio of predicted to "
                      r"reference response among non-LCO cases, expressed as its deviation from 1 (0 = perfect "
                      r"match), then (2) oscillation-frequency relative error, per the "
                      r"pre-declared priority order in \texttt{select\_configuration.py}.")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def history_table(dataset: str, label: str, complete: bool) -> str:
    agg = load(dataset, 2, "aggregated")
    grid = json.loads((STUDY_ROOT / "configs" / f"{dataset if dataset != 'cylinder200' else 'cylinder'}_history_grid.json").read_text())
    all_labels = [p["label"] for p in grid["history_points"]]

    if dataset == "cylinder200":
        pareto = load(dataset, 2, "pareto")
        order = {row["history_label"]: int(row["hierarchical_rank"]) for _, row in pareto.iterrows()}
        agg = agg.copy()
        agg["_rank"] = agg["history_label"].map(order)
        agg = agg.sort_values("_rank")
        rank_header = "Rank"
    else:
        agg = agg.copy()
        rms_col = "closed_loop_validation_mean_rms_ratio_among_non_lco_median"
        if rms_col in agg.columns:
            agg["_dev"] = (agg[rms_col] - 1.0).abs()
            agg = agg.sort_values("_dev")
        rank_header = "(informal)"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{label}: input-history-length sensitivity (Stage 2) at the Stage 1 selected "
        r"architecture. Median [IQR] across the 3 seeds " + str(REQUIRED_SEEDS) + r".}",
        rf"\label{{tab:appendix_hist_{dataset}}}",
    ]
    if dataset == "bridge":
        lines += [
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{clccccc}",
            r"\toprule",
            rf"{rank_header} & History & Duration [s] & Seeds & Open-loop $R^2$ & "
            r"$|f_{\text{osc}}|$ rel.\ err.\ & RMS ratio dev.\ from 1 \\",
            r"\midrule",
        ]
    else:
        lines += [
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{clccccc}",
            r"\toprule",
            rf"{rank_header} & History & Duration [s] & Seeds & Open-loop $R^2$ & Stable/$N$ & Amp.\ error \\",
            r"\midrule",
        ]

    present_labels = set(agg["history_label"].tolist()) if not agg.empty else set()
    i = 1
    for lbl in (sorted(present_labels, key=lambda l: order.get(l, 999)) if dataset == "cylinder200" and not agg.empty
                else (agg["history_label"].tolist() if not agg.empty else [])):
        r = agg[agg["history_label"] == lbl].iloc[0]
        dur = r.get("sequence_duration_s_median")
        n_seeds = int(r.get("n_seeds", 0))
        seeds_ok = "" if n_seeds == 3 else rf" ({n_seeds}/3 seeds)"
        r2 = fmt(r["open_loop_val_r2_median"], r["open_loop_val_r2_iqr"])
        rank_s = str(i) if dataset == "cylinder200" else "--"
        if dataset == "bridge":
            f_osc = fmt(r.get("closed_loop_validation_median_abs_f_osc_rel_error_median"),
                        r.get("closed_loop_validation_median_abs_f_osc_rel_error_iqr"), 3)
            dev = r.get("_dev")
            dev_s = f"{dev:.3f}" if pd.notna(dev) else "n/a"
            lines.append(f"{rank_s} & {esc(lbl)}{seeds_ok} & {dur:.3f} & {n_seeds}/3 & {r2} & {f_osc} & {dev_s} \\\\")
        else:
            stable = r.get("closed_loop_validation_stable_count_median")
            n = r.get("closed_loop_validation_n_median")
            amp = fmt(r.get("closed_loop_validation_median_abs_rel_amp_error_median"),
                      r.get("closed_loop_validation_median_abs_rel_amp_error_iqr"), 3)
            stable_s = f"{stable:.0f}/{n:.0f}" if pd.notna(stable) and pd.notna(n) else "n/a"
            lines.append(f"{rank_s} & {esc(lbl)}{seeds_ok} & {dur:.3f} & {n_seeds}/3 & {r2} & {stable_s} & {amp} \\\\")
        i += 1

    missing = [l for l in all_labels if l not in present_labels]
    for lbl in missing:
        lines.append(f"-- & {esc(lbl)} & \\multicolumn{{5}}{{c}}{{not yet run}} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"}"]
    lines.append(r"\begin{minipage}{0.95\linewidth}\vspace{4pt}\footnotesize")
    if complete:
        lines.append(r"All " + str(len(all_labels)) + r" history points complete (3/3 seeds each).")
    else:
        lines.append(rf"Incomplete as of this writing: {len(present_labels)}/{len(all_labels)} history points have "
                      r"at least one seed, and only those with 3/3 seeds are fully evidenced. Rows are sorted by "
                      r"the deviation of the non-LCO RMS ratio from 1 among points with complete seed sets; "
                      r"partial-seed and not-yet-run points are listed for completeness, not ranked. "
                      r"Regenerate this table (\texttt{generate\_appendix\_tables.py}) once the remaining "
                      r"configurations finish.")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def per_seed_longtable(dataset: str, stage: int, label: str) -> str:
    pr = load(dataset, stage, "per_run")
    if pr.empty:
        return ""
    group_key = ["hidden_size", "num_layers"] if stage == 1 else ["history_label", "seq_len"]
    if stage == 2:
        grid_name = "cylinder" if dataset == "cylinder200" else "bridge"
        grid = json.loads((STUDY_ROOT / "configs" / f"{grid_name}_history_grid.json").read_text())
        order = {p["label"]: i for i, p in enumerate(grid["history_points"])}
        pr = pr.copy()
        pr["_order"] = pr["history_label"].map(order)
        pr = pr.sort_values(["_order", "seed"])

    lines = [
        r"{\footnotesize",
        r"\begin{longtable}{" + ("cccccc" if dataset == "cylinder200" else "cccccc") + "}",
        rf"\caption{{{label}, Stage {stage}: per-seed evidence.}} "
        rf"\label{{tab:appendix_perseed_{dataset}_stage{stage}}} \\",
        r"\toprule",
    ]
    if dataset == "bridge":
        header = r"Config & Seed & $R^2$ & $f_{\text{osc}}$ err.\ & RMS ratio & $t_{\text{train}}$ [h] \\"
    else:
        header = r"Config & Seed & $R^2$ & Amp.\ error & Stable & $t_{\text{train}}$ [h] \\"
    lines += [header, r"\midrule", r"\endfirsthead", r"\toprule", header, r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]

    for key, grp in pr.groupby(group_key, dropna=False, sort=False):
        if stage == 1:
            cfg = f"H{int(key[0])}\\_L{int(key[1])}"
        else:
            cfg = esc(key[0]) if key[0] else "baseline"
        grp = grp.sort_values("seed")
        for j, (_, r) in enumerate(grp.iterrows()):
            cfg_s = cfg if j == 0 else ""
            r2 = f"{r['open_loop_val_r2']:.4f}" if pd.notna(r["open_loop_val_r2"]) else "n/a"
            th = hours(r.get("elapsed_training_time_s"))
            if dataset == "bridge":
                f_osc = r.get("closed_loop_validation_median_abs_f_osc_rel_error")
                rms = r.get("closed_loop_validation_mean_rms_ratio_among_non_lco")
                f_osc_s = f"{f_osc:.3f}" if pd.notna(f_osc) else "n/a"
                rms_s = f"{rms:.3f}" if pd.notna(rms) else "n/a"
                lines.append(f"{cfg_s} & {int(r['seed'])} & {r2} & {f_osc_s} & {rms_s} & {th} \\\\")
            else:
                amp = r.get("closed_loop_validation_median_abs_rel_amp_error")
                stable = r.get("closed_loop_validation_stable_count")
                n = r.get("closed_loop_validation_n")
                amp_s = f"{amp:.3f}" if pd.notna(amp) else "n/a"
                stable_s = f"{int(stable)}/{int(n)}" if pd.notna(stable) and pd.notna(n) else "n/a"
                lines.append(f"{cfg_s} & {int(r['seed'])} & {r2} & {amp_s} & {stable_s} & {th} \\\\")

    lines.append(r"\end{longtable}")
    lines.append(r"}")
    return "\n".join(lines)


def validation_only_note() -> str:
    sel_c = json.loads((MANIFESTS / "selection_manifest_cylinder200_stage1.json").read_text())
    sel_b = json.loads((MANIFESTS / "selection_manifest_bridge_stage1.json").read_text())
    return "\n".join([
        r"\subsection{Confirmation: selection used validation cases only}",
        r"\label{sec:appendix_validation_confirmation}",
        r"Every ranking in this appendix is computed from the validation partition alone. "
        r"Test cases are structurally excluded, not merely unused: every training run in this "
        r"study passes \texttt{train\_gru.py}'s \texttt{--skip\_test\_eval} flag unconditionally "
        r"(\texttt{train\_sensitivity.py}), which skips constructing the test dataframe/loader "
        r"entirely -- no test metric is computed anywhere in the pipeline, so none could leak "
        r"into a ranking decision even by accident. Each configuration's "
        r"\texttt{study\_receipt.json} records \texttt{skip\_test\_eval: true} and "
        r"\texttt{test\_partition\_evaluated: false} explicitly. Test evaluation is only ever "
        r"unlocked, for a single already-frozen selection, via a separate "
        r"\texttt{unlock\_test\_evaluation.py} script gated on the frozen selection manifest -- "
        r"it was not run for either geometry as of this writing.",
        r"",
        rf"Cylinder200's frozen Stage 1 selection manifest records "
        rf"\texttt{{selection\_basis}} = ``{esc(sel_c['selection_basis'])}''. "
        rf"Bridge's records the same basis: ``{esc(sel_b['selection_basis'])}''.",
    ])


def selection_summary() -> str:
    sel_c = json.loads((MANIFESTS / "selection_manifest_cylinder200_stage1.json").read_text())
    sel_b = json.loads((MANIFESTS / "selection_manifest_bridge_stage1.json").read_text())
    return "\n".join([
        r"\subsection{Selected configuration per geometry}",
        r"\begin{itemize}",
        rf"\item \textbf{{Cylinder200}}: $H={sel_c['selected_configuration']['hidden_size']}$, "
        rf"$L={sel_c['selected_configuration']['num_layers']}$ (hierarchical rank "
        rf"{sel_c['hierarchical_rank']} of 6). Input-history sweep (Stage 2, complete, "
        r"Table~\ref{tab:appendix_hist_cylinder200}) found no history point beats the original "
        r"1$T_n$ baseline duration.",
        rf"\item \textbf{{Bridge}}: $H={sel_b['selected_configuration']['hidden_size']}$, "
        rf"$L={sel_b['selected_configuration']['num_layers']}$ (hierarchical rank "
        rf"{sel_b['hierarchical_rank']} of 6). Input-history sweep (Stage 2, complete, "
        r"Table~\ref{tab:appendix_hist_bridge}) likewise found no history point beats the "
        r"original baseline duration (\texttt{seq\_len}=2500, 5.0\,s, $\approx 1.6\,T_n$), "
        r"which the production model continues to use. The longest point, 4Tn "
        r"(\texttt{seq\_len}=6250, 12.5\,s), needed a disclosed exception to train at all: "
        r"all 3 seeds exhausted GPU memory at the grid's fixed batch\_size=512 (BPTT "
        r"activation memory scaling with batch\_size$\times$seq\_len), confirmed as a real "
        r"capacity limit rather than contention (one failure reported CUDA requesting over "
        r"50GB on a fully exclusive $\sim$40GB GPU); retrained at batch\_size=128 for this "
        r"point only, recorded in each run's \texttt{study\_receipt.json} "
        r"(\texttt{fixed\_hyperparameters\_deviation}) rather than silently changed.",
        r"\end{itemize}",
        r"Both manifests are frozen (\texttt{frozen: true}) at the architecture level; neither "
        r"records a separate frozen Stage 2 (history-length) selection, since Stage 2 is "
        r"exploratory evidence about the frozen architecture rather than a second gate.",
    ])


def case_partitions_tex() -> str:
    sm = json.loads((MANIFESTS / "study_manifest.json").read_text())
    fz = sm["frozen_at_git_commit"]
    lines = [
        r"\section{Train / validation / test case partitions}",
        r"\label{app:case_partitions}",
        r"Exact $U_r$ case labels assigned to each partition, fixed once at Stage 0 and reused "
        r"unchanged for every configuration in the architecture and input-history sensitivity "
        r"study (Appendix~\ref{app:architecture_history_sensitivity}). Scalers "
        r"(\texttt{fit\_scalers}) are fit on training cases only.",
        "",
    ]
    for ds, title in (("cylinder200", "Cylinder200"), ("bridge", "Bridge")):
        d = fz[ds]
        lines += [
            rf"\subsection{{{title} ({d['n_retained_cases']} retained cases: "
            rf"{d['n_train']} train / {d['n_val']} val / {d['n_test']} test)}}",
            r"\begin{description}",
            rf"\item[Train] {', '.join(esc(c) for c in d['train_cases'])}",
            rf"\item[Validation] {', '.join(esc(c) for c in d['val_cases'])}",
            rf"\item[Test] {', '.join(esc(c) for c in d['test_cases'])} "
            r"(never evaluated in this study -- see Sec.~\ref{sec:appendix_validation_confirmation})",
            r"\end{description}",
            "",
        ]
    return "\n".join(lines)


def main():
    parts = [
        r"\section{Architecture and input-history sensitivity}",
        r"\label{app:architecture_history_sensitivity}",
        r"Full Stage 1 (architecture: hidden size $H\in\{32,64,128\}$, layers $L\in\{1,2\}$) and "
        r"Stage 2 (input-history length, at the Stage 1 winning architecture) sensitivity sweeps "
        r"for both geometries, 3 random seeds per configuration throughout. "
        r"Chapter~4 reports only the selected configuration and headline result; this appendix "
        r"gives the full ranking and per-seed spread behind that choice.",
        "",
        r"\subsection{Cylinder200 architecture ranking (Stage 1)}",
        architecture_table("cylinder200", "Cylinder200"),
        "",
        r"\subsection{Bridge architecture ranking (Stage 1)}",
        architecture_table("bridge", "Bridge"),
        "",
        r"\subsection{Cylinder200 input-history sensitivity (Stage 2)}",
        history_table("cylinder200", "Cylinder200", complete=True),
        "",
        r"\subsection{Bridge input-history sensitivity (Stage 2)}",
        history_table("bridge", "Bridge", complete=True),
        "",
        selection_summary(),
        "",
        validation_only_note(),
        "",
        r"\subsection{Per-seed evidence}",
        r"Individual-seed values underlying every median [IQR] above.",
        per_seed_longtable("cylinder200", 1, "Cylinder200"),
        per_seed_longtable("bridge", 1, "Bridge"),
        per_seed_longtable("cylinder200", 2, "Cylinder200"),
        per_seed_longtable("bridge", 2, "Bridge"),
    ]
    out = REPORTS / "appendix_architecture_history_sensitivity.tex"
    out.write_text("\n".join(parts) + "\n")
    print(f"Wrote {out}")

    out2 = REPORTS / "appendix_case_partitions.tex"
    out2.write_text(case_partitions_tex())
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
