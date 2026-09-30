import json

import pandas as pd
import pytest

CANONICAL_VAL = ["Ur4.25", "Ur6.25", "Ur9", "Ur10"]
CANONICAL_TEST = ["Ur3.5", "Ur5.5", "Ur7", "Ur11"]
CANONICAL_TRAIN = ["Ur2", "Ur3"]


def _write_stage0_audit(study_root):
    (study_root / "manifests").mkdir(parents=True, exist_ok=True)
    (study_root / "manifests" / "stage0_audit.json").write_text(json.dumps({
        "cylinder200": {"val_cases": CANONICAL_VAL, "test_cases": CANONICAL_TEST},
        "bridge": {"val_cases": ["Ur4.8433", "Ur5.6856", "Ur6.4227", "Ur7.1597", "Ur8.002"],
                   "test_cases": ["Ur4.6327", "Ur5.5804", "Ur6.3174", "Ur6.9491", "Ur7.7914"]},
    }))


def _write_clean_run_dir(study_root, tag="H64_L2_seq1000_seed123", dataset="cylinder200",
                          val_cases=None, sweep_ur=None, extra_sweep_ur=None):
    val_cases = val_cases if val_cases is not None else CANONICAL_VAL
    sweep_ur = sweep_ur if sweep_ur is not None else [4.25, 6.25, 9.0, 10.0]
    if extra_sweep_ur:
        sweep_ur = sweep_ur + extra_sweep_ur

    run_dir = study_root / "results" / dataset / "stage1" / tag
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps({
        "cfd_dataset": dataset,
        "train_cases": CANONICAL_TRAIN, "val_cases": val_cases, "test_cases": CANONICAL_TEST,
    }))
    (run_dir / "study_receipt.json").write_text(json.dumps({"hidden_size": 64, "num_layers": 2}))
    (run_dir / "gru_best.pt").write_bytes(b"fake checkpoint bytes")
    (run_dir / "x_scaler.pkl").write_bytes(b"fake x scaler bytes")
    (run_dir / "y_scaler.pkl").write_bytes(b"fake y scaler bytes")
    (run_dir / "open_loop_val_metrics.json").write_text(json.dumps({
        "val_cases": sorted(val_cases),
        "aggregate_val_metrics": {"r2": 0.99, "rmse": 0.01, "nrmse": 0.02, "mae": 0.005},
    }))

    cl_dir = run_dir / "closed_loop_eval"
    cl_dir.mkdir()
    pd.DataFrame({"Ur": sweep_ur}).to_csv(cl_dir / "sweep_results.csv", index=False)
    (cl_dir / "closed_loop_summary.json").write_text(json.dumps({"validation_n": len(sweep_ur)}))
    return run_dir


def test_clean_run_dir_reports_ok(tmp_path, monkeypatch):
    import check_validation_completion as cvc
    monkeypatch.setattr(cvc, "STUDY_ROOT", tmp_path)
    _write_stage0_audit(tmp_path)
    _write_clean_run_dir(tmp_path)

    import sys
    old_argv = sys.argv
    sys.argv = ["check_validation_completion.py", "--dataset", "cylinder200", "--stage", "1"]
    try:
        cvc.main()
    finally:
        sys.argv = old_argv


def test_missing_receipts_fail_closed(tmp_path, monkeypatch):
    import check_validation_completion as cvc
    monkeypatch.setattr(cvc, "STUDY_ROOT", tmp_path)
    _write_stage0_audit(tmp_path)

    run_dir = tmp_path / "results" / "cylinder200" / "stage1" / "H64_L2_seq1000_seed123"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps({
        "cfd_dataset": "cylinder200",
        "train_cases": CANONICAL_TRAIN, "val_cases": CANONICAL_VAL, "test_cases": CANONICAL_TEST,
    }))
    (run_dir / "study_receipt.json").write_text(json.dumps({"hidden_size": 64, "num_layers": 2}))

    import sys
    old_argv = sys.argv
    sys.argv = ["check_validation_completion.py", "--dataset", "cylinder200", "--stage", "1"]
    try:
        with pytest.raises(SystemExit) as exc:
            cvc.main()
    finally:
        sys.argv = old_argv
    assert exc.value.code == 1


def test_wrong_validation_case_identity_is_caught(tmp_path, monkeypatch):
    import check_validation_completion as cvc
    monkeypatch.setattr(cvc, "STUDY_ROOT", tmp_path)
    _write_stage0_audit(tmp_path)
    _write_clean_run_dir(tmp_path, val_cases=["Ur2", "Ur3", "Ur4", "Ur5"],
                          sweep_ur=[2.0, 3.0, 4.0, 5.0])

    import sys
    old_argv = sys.argv
    sys.argv = ["check_validation_completion.py", "--dataset", "cylinder200", "--stage", "1"]
    try:
        with pytest.raises(SystemExit) as exc:
            cvc.main()
    finally:
        sys.argv = old_argv
    assert exc.value.code == 1


def test_test_case_ur_in_sweep_results_is_a_hard_failure(tmp_path, monkeypatch):
    import check_validation_completion as cvc
    monkeypatch.setattr(cvc, "STUDY_ROOT", tmp_path)
    _write_stage0_audit(tmp_path)
    _write_clean_run_dir(tmp_path, extra_sweep_ur=[7.0])

    import sys
    old_argv = sys.argv
    sys.argv = ["check_validation_completion.py", "--dataset", "cylinder200", "--stage", "1"]
    try:
        with pytest.raises(SystemExit) as exc:
            cvc.main()
    finally:
        sys.argv = old_argv
    assert exc.value.code == 1


def test_expect_count_mismatch_fails(tmp_path, monkeypatch):
    import check_validation_completion as cvc
    monkeypatch.setattr(cvc, "STUDY_ROOT", tmp_path)
    _write_stage0_audit(tmp_path)
    _write_clean_run_dir(tmp_path)

    import sys
    old_argv = sys.argv
    sys.argv = ["check_validation_completion.py", "--dataset", "cylinder200", "--stage", "1",
                "--expect", "27"]
    try:
        with pytest.raises(SystemExit) as exc:
            cvc.main()
    finally:
        sys.argv = old_argv
    assert exc.value.code == 1


def test_sha256_is_computed_for_every_hashable_artifact(tmp_path, monkeypatch):
    import check_validation_completion as cvc
    monkeypatch.setattr(cvc, "STUDY_ROOT", tmp_path)
    _write_stage0_audit(tmp_path)
    run_dir = _write_clean_run_dir(tmp_path)

    audit = json.loads((tmp_path / "manifests" / "stage0_audit.json").read_text())
    result = cvc.check_run_dir(run_dir, set(audit["cylinder200"]["val_cases"]),
                                set(audit["cylinder200"]["test_cases"]))
    assert result["ok"], result["problems"]
    for name in ("gru_best.pt", "x_scaler.pkl", "y_scaler.pkl"):
        digest = result["sha256"][name]
        assert digest is not None
        assert len(digest) == 64
