"""Shared helpers of the architecture/history study: paths, run folders, training and evaluation wrappers around the production modules."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
SRC_ROOT = REPO_ROOT / "src"
STUDY_VERSION = "1.0.0"

REQUIRED_SEEDS = (123, 456, 789)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def dataset_cache_version(dataset: str) -> str:
    cache_dir = REPO_ROOT / "data" / "cache"
    if not cache_dir.exists():
        return "no_cache_dir"
    matches = sorted(p for p in cache_dir.glob(f"*{dataset}*"))
    if not matches:
        matches = sorted(cache_dir.glob("*bridge*" if dataset == "bridge" else "*"))
    h = hashlib.sha256()
    for p in matches:
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16] if matches else "no_matching_cache_files"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def run_output_dir(dataset: str, stage: int, hidden_size: int, num_layers: int,
                    seq_len: int, seed: int, history_label: str | None = None,
                    smoke: bool = False) -> Path:
    tag = f"H{hidden_size}_L{num_layers}_seq{seq_len}_seed{seed}"
    if history_label:
        tag = f"{history_label}_{tag}"
    root = STUDY_ROOT / "smoke" if smoke else STUDY_ROOT / "results"
    return root / dataset / f"stage{stage}" / tag


def exp_subdir_arg(output_dir: Path) -> str:
    results_root = REPO_ROOT / "results"
    return os.path.relpath(output_dir, results_root)


def model_subdir_alias(run_dir: Path) -> str:
    import uuid
    token = uuid.uuid4().hex[:12]
    link_name = "_gru_arch_hist_study_symlink_" + "_".join(run_dir.parts[-3:]) + f"_{token}"
    link_path = REPO_ROOT / "results" / link_name
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(run_dir.resolve(), target_is_directory=True)
    return link_name


def cleanup_model_subdir_alias(link_name: str) -> None:
    link_path = REPO_ROOT / "results" / link_name
    if link_path.is_symlink():
        link_path.unlink()


FULL_DURATION_S = {"cylinder200": 500.0, "bridge": 300.0}


def run_coupled_sweep(run_dir: Path, cases: list[str], out_dir: Path,
                       handoff_offset: int = 2000,
                       window_frac: float = 0.5, pass_amp_rel_error_threshold: float = 0.20,
                       timeout_s: int = 1800, total_time_override: float | None = None) -> "tuple":
    import sys
    import time

    import numpy as np
    import pandas as pd

    from viv_analysis.evaluate_all import (
        _compute_gate, extract_npz_path, extract_steady_state_amplitude,
        cfd_steady_state_amplitude, load_full_cfd_df, CYLINDER200_D, BRIDGE_D,
    )
    from viv_analysis.closed_loop_metrics import compute_case_metrics
    from viv_analysis.utils import parse_ur_label

    run_config = load_json(run_dir / "run_config.json")
    dataset = run_config["cfd_dataset"]
    total_time = total_time_override if total_time_override is not None else FULL_DURATION_S[dataset]
    D = CYLINDER200_D if dataset == "cylinder200" else BRIDGE_D

    Ur_list = sorted({parse_ur_label(c) for c in cases})
    model_subdir = model_subdir_alias(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_cfd_df = load_full_cfd_df(dataset)
    cfd_amplitudes = {Ur: (cfd_steady_state_amplitude(full_cfd_df, Ur, D)
                            if not full_cfd_df.empty else None) for Ur in Ur_list}

    results, gate_results, freq_results, timings = {}, {}, {}, {}
    try:
        for i, Ur in enumerate(Ur_list, 1):
            print(f"[{i}/{len(Ur_list)}] Ur={Ur} ... ", end="", flush=True)
            cmd = [
                sys.executable, "-m", "src.viv_analysis.coupled_inference",
                "--model_subdir", model_subdir,
                "--checkpoint", "gru_best.pt",
                "--cfd_dataset", dataset,
                "--Ur", str(Ur),
                "--total_time", str(total_time),
                "--handoff_offset", str(handoff_offset),
                "--output_dir", str(out_dir.resolve()),
            ]
            cmd.append("--nd_inputs" if run_config["nd_inputs"] else "--dim_inputs")

            t0 = time.time()
            try:
                out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True,
                                      text=True, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                print("TIMEOUT")
                results[Ur] = None
                gate_results[Ur] = None
                continue
            timings[Ur] = time.time() - t0

            if out.returncode != 0:
                print("FAILED")
                print("\n".join((out.stderr or "").splitlines()[-30:]))
                results[Ur] = None
                gate_results[Ur] = None
                continue

            amp = extract_steady_state_amplitude(out.stdout)
            npz_path = extract_npz_path(out.stdout)
            results[Ur] = amp
            gate_results[Ur] = (
                _compute_gate(npz_path, window_frac, pass_amp_rel_error_threshold)
                if npz_path is not None else None
            )
            try:
                freq_results[Ur] = (
                    compute_case_metrics(npz_path, window_frac=window_frac)
                    if npz_path is not None else None
                )
            except Exception:
                freq_results[Ur] = None
            gate = gate_results[Ur]
            print(f"A/D={amp}  {gate['stability_label'] if gate else 'N/A'}")
    finally:
        cleanup_model_subdir_alias(model_subdir)

    label = f"{dataset}_{run_dir.name}"
    rows = []
    for Ur in Ur_list:
        row = {"Ur": Ur, "CFD": cfd_amplitudes.get(Ur)}
        gate = gate_results.get(Ur)
        freq = freq_results.get(Ur)
        row[label] = results.get(Ur)
        row[f"{label}_stability_label"] = gate["stability_label"] if gate else None
        row[f"{label}_A_star_rel_error"] = gate["A_star_rel_error"] if gate else None
        row[f"{label}_pass"] = gate["pass"] if gate else False
        row[f"{label}_f_osc_rel_error"] = freq.get("f_osc_rel_error") if freq else None
        row["inference_time_s"] = timings.get(Ur)
        rows.append(row)
    sweep_df = pd.DataFrame(rows).sort_values("Ur")
    return sweep_df, timings, label


def assert_frozen_and_get_selected_run_dirs(selection_manifest: Path) -> list[str]:
    if not selection_manifest.exists():
        raise SystemExit(
            f"Test evaluation requires an existing frozen selection manifest; "
            f"{selection_manifest} not found.")
    sel = load_json(selection_manifest)
    if not sel.get("frozen", False):
        raise SystemExit(f"{selection_manifest} is not marked frozen. "
                          f"Test evaluation refused.")

    run_dirs = sel.get("selected_run_dirs", [])
    if len(run_dirs) != 3:
        raise SystemExit(
            f"{selection_manifest} lists {len(run_dirs)} selected_run_dirs; "
            f"expected exactly 3 (one per seed 123/456/789) for a "
            f"configuration-level selection. Test evaluation refused.")
    for rd in run_dirs:
        if not (Path(rd) / "run_config.json").exists():
            raise SystemExit(f"Selected run dir {rd} has no run_config.json.")
    return run_dirs


CANONICAL_VAL_CASE_COUNT = {"cylinder200": 4, "bridge": 5}
CANONICAL_TEST_CASE_COUNT = {"cylinder200": 4, "bridge": 5}


def release_time_for(dataset: str, case: str, cfg: dict) -> float:
    from viv_analysis.config import cylinder200_release_time
    from viv_analysis.utils import parse_ur_label

    if dataset == "bridge":
        ur = parse_ur_label(case)
        U = ur * cfg["bridge_fn_hz"] * cfg["bridge_D_ref"]
        return float(cfg["bridge_t_star_release"] * cfg["bridge_D_ref"] / U)
    return cylinder200_release_time(parse_ur_label(case))


def compute_open_loop_metrics(run_dir: Path, cases: list[str], case_kind: str) -> dict:
    import pickle

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from viv_analysis.train_gru import teacher_forcing_rollout, run_validation, apply_nd_transform
    from viv_analysis.models.gru import VIV_GRU, VIVSequenceDataset, apply_scalers_to_df
    from viv_analysis.preprocess import merge_dataframes, downsample, compute_kinematics
    from viv_analysis.config import (
        config, prepare_gru_config, bridge_structural_params,
        cylinder200_structural_params,
    )
    from viv_analysis.evaluate import evaluate

    run_config = load_json(run_dir / "run_config.json")
    dataset = run_config["cfd_dataset"]
    input_cols = run_config["input_cols"]
    use_ur_context = run_config["use_ur_context"]
    nd_inputs = run_config["nd_inputs"]
    seq_len = run_config["seq_len"]
    D_nd, fn_nd = run_config["D"], run_config["fn"]

    cfg = prepare_gru_config(dataset, config).copy()

    if dataset == "bridge":
        raw_df = merge_dataframes(dataset="bridge", fn_hz=cfg["bridge_fn_hz"],
                                   d_ref=cfg["bridge_D_ref"])
        raw_df = downsample(raw_df, cfg["bridge_downsample"])
        params = bridge_structural_params()
        raw_df = compute_kinematics(raw_df, dataset=dataset, bridge_structural_params=params)
    else:
        raw_df = merge_dataframes(dataset=dataset)
        params = cylinder200_structural_params()
        raw_df = compute_kinematics(raw_df, dataset=dataset, structural_params=params)

    if nd_inputs:
        raw_df = apply_nd_transform(raw_df, nd_inputs=True, D=D_nd, fn=fn_nd,
                                     input_cols=input_cols)

    eval_df = raw_df[raw_df["case"].isin(cases)].copy()

    with open(run_dir / "x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open(run_dir / "y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)
    with open(run_dir / "ur_stats.pkl", "rb") as f:
        ur_stats_dict = pickle.load(f)
    ur_mean, ur_std = ur_stats_dict["mean"], ur_stats_dict["std"]

    eval_df_s = apply_scalers_to_df(eval_df, x_scaler, y_scaler, input_cols, "cl")

    receipt = load_json(run_dir / "study_receipt.json")
    hidden_size = receipt["hidden_size"]
    num_layers = receipt["num_layers"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VIV_GRU(
        input_size=len(input_cols) + (1 if use_ur_context else 0),
        hidden_size=hidden_size, num_layers=num_layers, dropout=config["dropout"],
    ).to(device)
    state = torch.load(run_dir / "gru_best.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    release_time = {c: release_time_for(dataset, c, cfg) for c in cases}
    eval_ds = VIVSequenceDataset(eval_df_s, stride=1, seq_len=seq_len, target_col="cl",
                                  input_cols=input_cols, release_time=release_time,
                                  use_ur_context=use_ur_context, ur_mean=ur_mean, ur_std=ur_std)
    eval_loader = DataLoader(eval_ds, batch_size=512, shuffle=False)
    criterion = torch.nn.MSELoss()
    agg_loss, ep, et, _ = run_validation(model, eval_loader, criterion, device)
    e_pred = y_scaler.inverse_transform(ep.reshape(-1, 1)).ravel()
    e_true = y_scaler.inverse_transform(et.reshape(-1, 1)).ravel()
    agg_metrics = evaluate(e_true, e_pred)
    rng = float(e_true.max() - e_true.min()) if len(e_true) else float("nan")
    agg_metrics["nrmse"] = agg_metrics["rmse"] / rng if rng > 0 else float("nan")
    agg_metrics[f"{case_kind}_loss_scaled_mse"] = agg_loss

    per_case = {}
    for case_name in sorted(cases):
        case_df_s = eval_df_s[eval_df_s["case"] == case_name].copy()
        if case_df_s.empty:
            continue
        rt = release_time[case_name]
        cl_pred, cl_true, _ = teacher_forcing_rollout(
            model, case_df_s, input_cols, seq_len, rt, y_scaler, case_name,
            device, use_ur_context=use_ur_context, ur_stats=(ur_mean, ur_std),
        )
        if len(cl_true) < 20:
            continue
        m = evaluate(cl_true, cl_pred)
        case_rng = float(cl_true.max() - cl_true.min())
        m["nrmse"] = m["rmse"] / case_rng if case_rng > 0 else float("nan")
        per_case[case_name] = m

    return {
        "run_dir": str(run_dir),
        "dataset": dataset,
        f"{case_kind}_cases": sorted(cases),
        f"aggregate_{case_kind}_metrics": agg_metrics,
        f"per_case_{case_kind}_metrics": per_case,
        f"median_{case_kind}_r2": float(np.median([m["r2"] for m in per_case.values()])) if per_case else None,
        f"median_{case_kind}_nrmse": float(np.median([m["nrmse"] for m in per_case.values()])) if per_case else None,
    }
