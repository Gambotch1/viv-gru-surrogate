import json

from collect_results import aggregate, collect


def _write_receipt(base, dataset, stage, H, L, seq_len, seed, r2=0.99):
    tag = f"H{H}_L{L}_seq{seq_len}_seed{seed}"
    run_dir = base / dataset / f"stage{stage}" / tag
    run_dir.mkdir(parents=True)
    (run_dir / "study_receipt.json").write_text(json.dumps({
        "hidden_size": H, "num_layers": L, "sequence_samples": seq_len,
        "sequence_duration_s": seq_len * 0.005, "history_label": None,
        "seed": seed, "trainable_parameter_count": 1000 + H,
        "elapsed_training_time_s": 60.0, "peak_gpu_memory_mb": 500.0,
    }))
    (run_dir / "open_loop_val_metrics.json").write_text(json.dumps({
        "aggregate_val_metrics": {"r2": r2, "rmse": 0.01, "nrmse": 0.02, "mae": 0.005},
        "median_val_r2": r2,
    }))
    return run_dir


def test_collect_only_returns_requested_dataset_and_stage(tmp_path, monkeypatch):
    import collect_results
    monkeypatch.setattr(collect_results, "STUDY_ROOT", tmp_path)

    _write_receipt(tmp_path / "results", "cylinder200", 1, 64, 2, 1000, 123, r2=0.99)
    _write_receipt(tmp_path / "results", "cylinder200", 1, 64, 2, 1000, 456, r2=0.98)
    _write_receipt(tmp_path / "results", "bridge", 1, 64, 2, 2500, 123, r2=0.90)
    _write_receipt(tmp_path / "results", "cylinder200", 2, 64, 2, 2000, 123, r2=0.95)

    df = collect("cylinder200", 1)
    assert set(df["dataset"]) == {"cylinder200"}
    assert set(df["stage"]) == {1}
    assert len(df) == 2

    df_bridge = collect("bridge", 1)
    assert len(df_bridge) == 1
    assert df_bridge["dataset"].iloc[0] == "bridge"

    df_stage2 = collect("cylinder200", 2)
    assert len(df_stage2) == 1
    assert df_stage2["stage"].iloc[0] == 2


def test_aggregate_keeps_seeds_as_a_list_not_averaged_away(tmp_path, monkeypatch):
    import collect_results
    monkeypatch.setattr(collect_results, "STUDY_ROOT", tmp_path)

    _write_receipt(tmp_path / "results", "cylinder200", 1, 64, 2, 1000, 123, r2=0.99)
    _write_receipt(tmp_path / "results", "cylinder200", 1, 64, 2, 1000, 456, r2=0.97)
    _write_receipt(tmp_path / "results", "cylinder200", 1, 64, 2, 1000, 789, r2=0.95)

    df = collect("cylinder200", 1)
    agg = aggregate(df)
    assert len(agg) == 1
    row = agg.iloc[0]
    assert sorted(row["seeds"]) == [123, 456, 789]
    assert row["n_seeds"] == 3
    assert abs(row["open_loop_val_r2_median"] - 0.97) < 1e-9


def test_aggregate_rejects_configuration_missing_a_required_seed(tmp_path, monkeypatch):
    import collect_results
    monkeypatch.setattr(collect_results, "STUDY_ROOT", tmp_path)

    _write_receipt(tmp_path / "results", "cylinder200", 1, 64, 2, 1000, 123, r2=0.99)
    _write_receipt(tmp_path / "results", "cylinder200", 1, 64, 2, 1000, 456, r2=0.98)
    _write_receipt(tmp_path / "results", "cylinder200", 1, 32, 1, 1000, 123, r2=0.90)
    _write_receipt(tmp_path / "results", "cylinder200", 1, 32, 1, 1000, 456, r2=0.91)
    _write_receipt(tmp_path / "results", "cylinder200", 1, 32, 1, 1000, 789, r2=0.92)

    df = collect("cylinder200", 1)
    agg = aggregate(df)
    assert len(agg) == 2

    partial = agg[(agg["hidden_size"] == 64) & (agg["num_layers"] == 2)].iloc[0]
    assert partial["n_seeds"] == 2
    assert not bool(partial["all_valid"]), (
        "a 2-of-3-seed configuration must never be treated as valid, even "
        "though every row it does have is individually finite/non-NaN")

    complete = agg[(agg["hidden_size"] == 32) & (agg["num_layers"] == 1)].iloc[0]
    assert complete["n_seeds"] == 3
    assert bool(complete["all_valid"])
