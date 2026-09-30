import json

import pytest

from _common import assert_frozen_and_get_selected_run_dirs


def _make_run_dir(tmp_path, seed, skip_test_eval=True):
    run_dir = tmp_path / "results" / "cylinder200" / "stage1" / f"H64_L2_seq1000_seed{seed}"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps({
        "cfd_dataset": "cylinder200",
        "train_cases": ["Ur2", "Ur3"],
        "val_cases": ["Ur4.25", "Ur6.25", "Ur9", "Ur10"],
        "test_cases": ["Ur3.5", "Ur5.5", "Ur7", "Ur11"],
        "skip_test_eval": skip_test_eval,
    }))
    (run_dir / "study_receipt.json").write_text(json.dumps({"seed": seed}))
    return run_dir


def _make_manifest(tmp_path, run_dirs, frozen=True):
    manifest_path = tmp_path / "selection_manifest_stage1.json"
    manifest_path.write_text(json.dumps({
        "frozen": frozen,
        "dataset": "cylinder200",
        "stage": 1,
        "selected_configuration": {"hidden_size": 64, "num_layers": 2},
        "selected_run_dirs": [str(d) for d in run_dirs],
    }))
    return manifest_path


def test_unlock_fails_without_manifest(tmp_path):
    with pytest.raises(SystemExit):
        assert_frozen_and_get_selected_run_dirs(tmp_path / "no_such_manifest.json")


def test_unlock_fails_if_manifest_not_frozen(tmp_path):
    run_dirs = [_make_run_dir(tmp_path, s) for s in (123, 456, 789)]
    manifest_path = _make_manifest(tmp_path, run_dirs, frozen=False)
    with pytest.raises(SystemExit):
        assert_frozen_and_get_selected_run_dirs(manifest_path)


def test_unlock_fails_with_only_one_run_dir(tmp_path):
    run_dirs = [_make_run_dir(tmp_path, 123)]
    manifest_path = _make_manifest(tmp_path, run_dirs, frozen=True)
    with pytest.raises(SystemExit):
        assert_frozen_and_get_selected_run_dirs(manifest_path)


def test_unlock_fails_with_only_two_run_dirs(tmp_path):
    run_dirs = [_make_run_dir(tmp_path, s) for s in (123, 456)]
    manifest_path = _make_manifest(tmp_path, run_dirs, frozen=True)
    with pytest.raises(SystemExit):
        assert_frozen_and_get_selected_run_dirs(manifest_path)


def test_unlock_succeeds_with_exactly_3_seed_run_dirs(tmp_path):
    run_dirs = [_make_run_dir(tmp_path, s) for s in (123, 456, 789)]
    manifest_path = _make_manifest(tmp_path, run_dirs, frozen=True)
    result = assert_frozen_and_get_selected_run_dirs(manifest_path)
    assert len(result) == 3
    assert set(result) == {str(d) for d in run_dirs}


def test_unlock_test_evaluation_refuses_a_run_not_trained_with_skip_test_eval(tmp_path):
    good = [_make_run_dir(tmp_path, s) for s in (123, 456)]
    bad = _make_run_dir(tmp_path, 789, skip_test_eval=False)
    run_dirs = good + [bad]
    manifest_path = _make_manifest(tmp_path, run_dirs, frozen=True)

    selected = assert_frozen_and_get_selected_run_dirs(manifest_path)
    assert len(selected) == 3
    from pathlib import Path
    run_config = json.loads((Path(bad) / "run_config.json").read_text())
    assert run_config["skip_test_eval"] is False
