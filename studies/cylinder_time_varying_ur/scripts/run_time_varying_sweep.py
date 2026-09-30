r"""Continuous closed-loop cylinder simulation under a Ur(t) schedule (thesis Sec. 5.7).

Example:
    PYTHONPATH=src python studies/cylinder_time_varying_ur/scripts/run_time_varying_sweep.py \
        --schedule ascending_cosine --dwell_s 100 --transition_s 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(STUDY_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT / "scripts"))

from viv_analysis.coupled_inference import (  # noqa: E402
    warmup_history, to_model_coords,
    load_artifact_coordinate_mode, check_artifact_dataset_compatibility,
)
from viv_analysis.preprocess import merge_dataframes, compute_kinematics  # noqa: E402
from viv_analysis.utils import format_ur_label  # noqa: E402
from viv_analysis.config import config, cylinder200_structural_params  # noqa: E402
from viv_analysis.models.gru import VIV_GRU  # noqa: E402

from time_varying_coupled import run_coupled_viv_time_varying_ur, CYLINDER200_RE  # noqa: E402
import schedules  # noqa: E402

MODEL_SUBDIR = "gru_cylinder200_nd_context_noacc"
WARMUP_UR = 2.00

OTHER_STUDY_ROOT = REPO_ROOT / "studies" / "gru_architecture_history_sensitivity"
STAGE2_SELECTION_MANIFEST = OTHER_STUDY_ROOT / "manifests" / "selection_manifest_cylinder200_stage2.json"


def assert_stage2_selection_frozen() -> list[str]:
    if not STAGE2_SELECTION_MANIFEST.exists():
        raise SystemExit(
            f"Full time-varying sweeps require a frozen final Stage 2 "
            f"selection manifest with selection_complete=true; "
            f"{STAGE2_SELECTION_MANIFEST} does not exist yet. Use --smoke "
            f"for development runs, which are exempt from this gate.")
    manifest = json.loads(STAGE2_SELECTION_MANIFEST.read_text())
    if not manifest.get("frozen", False):
        raise SystemExit(f"{STAGE2_SELECTION_MANIFEST} exists but is not frozen.")
    if not manifest.get("selection_complete", False):
        raise SystemExit(
            f"{STAGE2_SELECTION_MANIFEST} is frozen but selection_complete "
            f"is not true -- the architecture/history-length choice is not "
            f"final. Full sweeps refused.")
    run_dirs = manifest.get("selected_run_dirs", [])
    if len(run_dirs) != 3:
        raise SystemExit(
            f"{STAGE2_SELECTION_MANIFEST} lists {len(run_dirs)} "
            f"selected_run_dirs; expected exactly 3 (one per seed).")
    return run_dirs


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def load_model_and_artifacts(device: str, artifact_dir: Path | None = None):
    artifact_dir = artifact_dir or (REPO_ROOT / "results" / MODEL_SUBDIR)

    with open(artifact_dir / "x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open(artifact_dir / "y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)
    with open(artifact_dir / "ur_stats.pkl", "rb") as f:
        ur_info = pickle.load(f)

    nd_inputs = load_artifact_coordinate_mode(artifact_dir, True)
    check_artifact_dataset_compatibility(artifact_dir, "cylinder200")

    use_ur_context = bool(ur_info["use_ur_context"])
    ur_mean = float(ur_info["mean"])
    ur_std = float(ur_info["std"]) + 1e-8

    with open(artifact_dir / "run_config.json") as f:
        run_config = json.load(f)
    seq_len = run_config["seq_len"]
    input_cols = run_config["input_cols"]

    receipt_path = artifact_dir / "study_receipt.json"
    metrics_path = artifact_dir / "metrics_gru.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        hidden_size, num_layers = receipt["hidden_size"], receipt["num_layers"]
    elif metrics_path.exists():
        gru_config = json.loads(metrics_path.read_text())["gru_config"]
        hidden_size, num_layers = gru_config["hidden_size"], gru_config["num_layers"]
    else:
        raise FileNotFoundError(
            f"Neither study_receipt.json nor metrics_gru.json found under "
            f"{artifact_dir} -- cannot determine hidden_size/num_layers.")

    if "acc" in input_cols:
        raise ValueError(f"{artifact_dir} includes 'acc' in input_cols -- "
                          f"this study requires a no-acceleration model.")
    input_size = len(input_cols) + (1 if use_ur_context else 0)

    model = VIV_GRU(input_size=input_size, hidden_size=hidden_size,
                     num_layers=num_layers, dropout=0.1).to(device)
    ckpt_path = artifact_dir / "gru_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.eval()

    return dict(
        model=model, x_scaler=x_scaler, y_scaler=y_scaler,
        nd_inputs=nd_inputs, use_ur_context=use_ur_context,
        ur_mean=ur_mean, ur_std=ur_std, seq_len=seq_len, input_cols=input_cols,
        artifact_dir=artifact_dir, checkpoint_path=ckpt_path,
        x_scaler_path=artifact_dir / "x_scaler.pkl",
        y_scaler_path=artifact_dir / "y_scaler.pkl",
    )


def warmup_once(artifacts: dict, dt: float, D: float, fn: float,
                 handoff_offset_steps: int = 2000) -> dict:
    rho = config["cylinder200_rho"]
    t_star_release = config["cylinder200_t_star_release"]
    U_warmup = WARMUP_UR * fn * D
    t_release = t_star_release * D / U_warmup

    raw_df = merge_dataframes(dataset="cylinder200")
    if raw_df.empty:
        raise RuntimeError("Could not load CFD data for warmup.")
    params = cylinder200_structural_params()
    raw_df = compute_kinematics(raw_df, dataset="cylinder200", structural_params=params)

    case_label = format_ur_label(WARMUP_UR)
    case_df = raw_df[raw_df["case"] == case_label].copy()
    if case_df.empty:
        raise ValueError(f"No CFD data for warm-up case '{case_label}'.")

    initial_history, initial_state, t_handoff, handoff_idx = warmup_history(
        cfd_case_df=case_df, release_t=t_release, seq_len=artifacts["seq_len"],
        input_cols=artifacts["input_cols"], x_scaler=artifacts["x_scaler"],
        nd_inputs=artifacts["nd_inputs"], D=D, U=U_warmup,
        use_ur_context=artifacts["use_ur_context"], ur_value=WARMUP_UR,
        ur_stats=(artifacts["ur_mean"], artifacts["ur_std"]),
        handoff_offset_steps=handoff_offset_steps,
    )
    return dict(
        initial_history=initial_history, initial_state=initial_state,
        t_handoff=t_handoff, handoff_idx=handoff_idx, warmup_ur=WARMUP_UR,
        warmup_case=case_label, t_release=t_release, U_warmup=U_warmup,
        source="CFD (cylinder200, Ur=2.00) via warmup_history() -- called once",
    )


def run_sweep(schedule: dict, device: str = None, structural_params: dict = None,
              track_history_provenance: bool = False,
              artifact_dir: Path | None = None) -> tuple[dict, dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    D = config["cylinder200_D_ref"]
    fn = config["cylinder200_fn"]
    dt = config["cylinder200_dt"]
    rho = config["cylinder200_rho"]
    sp = structural_params or cylinder200_structural_params()

    if abs(schedule["dt"] - dt) > 1e-12:
        raise ValueError(f"schedule dt={schedule['dt']} != cylinder200_dt={dt}")

    artifacts = load_model_and_artifacts(device, artifact_dir=artifact_dir)
    warmup = warmup_once(artifacts, dt=dt, D=D, fn=fn)

    result = run_coupled_viv_time_varying_ur(
        model=artifacts["model"], x_scaler=artifacts["x_scaler"],
        y_scaler=artifacts["y_scaler"], seq_len=artifacts["seq_len"],
        input_cols=artifacts["input_cols"],
        initial_history=warmup["initial_history"], initial_state=warmup["initial_state"],
        m=sp["m"], c=sp["c"], k=sp["k"], rho=rho, D=D, fn=fn,
        Ur_schedule=schedule["Ur_schedule"], n_steps=schedule["n_steps"],
        dt=dt, use_ur_context=artifacts["use_ur_context"],
        ur_stats=(artifacts["ur_mean"], artifacts["ur_std"]),
        device=device, nd_inputs=artifacts["nd_inputs"],
        track_history_provenance=track_history_provenance,
    )

    receipt = {
        "study": "cylinder_time_varying_ur",
        "git_commit": _git_commit(),
        "model_subdir": str(artifact_dir) if artifact_dir is not None else MODEL_SUBDIR,
        "checkpoint_path": str(artifacts["checkpoint_path"]),
        "checkpoint_sha256_16": _sha256_file(artifacts["checkpoint_path"]),
        "x_scaler_sha256_16": _sha256_file(artifacts["x_scaler_path"]),
        "y_scaler_sha256_16": _sha256_file(artifacts["y_scaler_path"]),
        "input_cols": artifacts["input_cols"],
        "acceleration_present": "acc" in artifacts["input_cols"],
        "seq_len": artifacts["seq_len"],
        "nd_inputs": artifacts["nd_inputs"],
        "use_ur_context": artifacts["use_ur_context"],
        "schedule": {k: v for k, v in schedule.items() if k != "Ur_schedule"},
        "dt": dt, "D": D, "fn": fn, "rho": rho,
        "structural_params": {"m": sp["m"], "c": sp["c"], "k": sp["k"]},
        "warmup_source": warmup["source"],
        "warmup_ur": warmup["warmup_ur"],
        "warmup_case": warmup["warmup_case"],
        "warmup_handoff_time_s": warmup["t_handoff"],
        "warmup_handoff_idx": warmup["handoff_idx"],
        "device": device,
    }
    return result, receipt


def save_result(result: dict, receipt: dict, tag: str) -> Path:
    out_dir = STUDY_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"time_varying_ur_{tag}.npz"
    save_kwargs = {k: v for k, v in result.items()
                   if isinstance(v, np.ndarray)}
    np.savez(npz_path, **save_kwargs)

    receipts_dir = STUDY_ROOT / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts_dir / f"time_varying_ur_{tag}.receipt.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"Saved {npz_path}")
    print(f"Saved {receipt_path}")
    return npz_path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schedule", choices=["ascending", "ascending_cosine", "triangular_cosine"],
                   default="ascending",
                   help="The 3 approved full-campaign schedules: ascending "
                        "instantaneous, ascending cosine-transition, and "
                        "triangular cosine-transition (2->12->2). "
                        "Instantaneous-transition triangular is available in "
                        "schedules.py for diagnostics but is not part of the "
                        "approved production campaign, so it has no CLI option here.")
    p.add_argument("--dwell_s", type=float, default=100.0)
    p.add_argument("--transition_s", type=float, default=5.0)
    p.add_argument("--tag", default=None)
    p.add_argument("--track_history_provenance", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Bypass the frozen-Stage-2-selection preflight gate "
                        "and use MODEL_SUBDIR directly, for development/"
                        "diagnostic runs. The canonical smoke test (item 7) "
                        "is run_smoke_test.py, which does not go through "
                        "this driver or flag at all; this exists for ad hoc "
                        "checks against the same development checkpoint.")
    p.add_argument("--single_seed_production", action="store_true",
                   help="Bypass the frozen-Stage-2-selection preflight gate "
                        "and run the FULL production schedule (unlike "
                        "--smoke, not shortened) against MODEL_SUBDIR, using "
                        "a single seed. Distinct from --smoke: this is an "
                        "explicit, labeled production decision -- the "
                        "receipt records single_seed_production=true, "
                        "n_seeds=1, and stage2_gate_bypassed=true, and the "
                        "output tag says so too, so it can never be mistaken "
                        "for the originally-planned 3-seed median/IQR result "
                        "once the sensitivity study's Stage 2 does freeze. "
                        "Chosen when this study's figures cannot wait for "
                        "that freeze.")
    return p.parse_args()


def _build_schedule(kind: str, dwell_s: float, transition_s: float, dt: float) -> dict:
    if kind == "ascending":
        return schedules.build_ascending_schedule(schedules.CYLINDER_UR_LIST, dwell_s, dt)
    if kind == "ascending_cosine":
        return schedules.build_ascending_cosine_schedule(
            schedules.CYLINDER_UR_LIST, dwell_s, transition_s, dt)
    if kind == "triangular_cosine":
        return schedules.build_triangular_schedule(
            schedules.CYLINDER_UR_LIST, dwell_s, dt, transition="cosine",
            transition_s=transition_s)
    raise ValueError(f"unknown schedule kind: {kind}")


def main():
    args = parse_args()
    dt = config["cylinder200_dt"]
    schedule = _build_schedule(args.schedule, args.dwell_s, args.transition_s, dt)

    print(f"Running schedule={args.schedule}  n_steps={schedule['n_steps']}  "
          f"duration={schedule['n_steps']*dt:.1f}s  "
          f"transition_duration_convention="
          f"{schedule.get('transition_duration_convention', 'n/a')}")

    if args.smoke and args.single_seed_production:
        raise SystemExit("--smoke and --single_seed_production are mutually "
                          "exclusive (one runs a shortened dev schedule, the "
                          "other the full production schedule).")

    if args.smoke:
        print(f"[--smoke] Bypassing the Stage 2 selection preflight gate; "
              f"using development checkpoint {MODEL_SUBDIR}.")
        tag = args.tag or f"{args.schedule}_smoke"
        result, receipt = run_sweep(schedule, track_history_provenance=args.track_history_provenance)
        save_result(result, receipt, tag)
        return

    if args.single_seed_production:
        print(f"[--single_seed_production] Bypassing the Stage 2 selection "
              f"preflight gate by explicit request; running the FULL "
              f"production schedule against the single existing checkpoint "
              f"{MODEL_SUBDIR}. This is NOT the originally-planned 3-seed "
              f"median/IQR result -- the sensitivity study's Stage 2 "
              f"selected_run_dirs were not available when this was run.")
        tag = args.tag or f"{args.schedule}_single_seed_production"
        result, receipt = run_sweep(schedule, track_history_provenance=args.track_history_provenance)
        receipt["single_seed_production"] = True
        receipt["n_seeds"] = 1
        receipt["stage2_gate_bypassed"] = True
        receipt["stage2_gate_bypass_reason"] = (
            "Stage 2 of the architecture/history-length sensitivity study had "
            "not produced a frozen, complete selection_manifest_cylinder200_stage2.json "
            "at the time this sweep was run; the existing single-seed "
            "production checkpoint was used instead of the originally-"
            "planned 3-seed selected configuration."
        )
        save_result(result, receipt, tag)
        return

    run_dirs = assert_stage2_selection_frozen()
    print(f"Frozen, complete Stage 2 selection confirmed. Running all 3 "
          f"seeds independently: {run_dirs}")
    for run_dir_str in run_dirs:
        run_dir = Path(run_dir_str)
        seed = json.loads((run_dir / "study_receipt.json").read_text())["seed"]
        print(f"--- seed={seed}  ({run_dir}) ---")
        result, receipt = run_sweep(
            schedule, track_history_provenance=args.track_history_provenance,
            artifact_dir=run_dir,
        )
        receipt["seed"] = seed
        tag = f"{args.tag or args.schedule}_seed{seed}"
        save_result(result, receipt, tag)


if __name__ == "__main__":
    main()
