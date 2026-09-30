from pathlib import Path

from _common import REPO_ROOT, cleanup_model_subdir_alias, model_subdir_alias


def test_model_subdir_alias_is_flat_and_resolves_correctly(tmp_path):
    run_dir = tmp_path / "cylinder200" / "stage1" / "H64_L2_seq1000_seed123"
    run_dir.mkdir(parents=True)
    (run_dir / "gru_best.pt").write_text("fake checkpoint")

    alias = model_subdir_alias(run_dir)
    try:
        assert "/" not in alias, f"alias {alias!r} must be a flat name (no '/')"
        link_path = REPO_ROOT / "results" / alias
        assert link_path.is_symlink()
        resolved = (REPO_ROOT / "results" / alias).resolve()
        assert resolved == run_dir.resolve()
        assert (REPO_ROOT / "results" / alias / "gru_best.pt").read_text() == "fake checkpoint"
    finally:
        cleanup_model_subdir_alias(alias)

    assert not (REPO_ROOT / "results" / alias).exists()
    assert not (REPO_ROOT / "results" / alias).is_symlink()


def test_model_subdir_alias_does_not_collide_across_configs(tmp_path):
    run_dir_a = tmp_path / "cylinder200" / "stage1" / "H32_L1_seq1000_seed123"
    run_dir_b = tmp_path / "cylinder200" / "stage1" / "H64_L2_seq1000_seed123"
    run_dir_a.mkdir(parents=True)
    run_dir_b.mkdir(parents=True)

    alias_a = model_subdir_alias(run_dir_a)
    alias_b = model_subdir_alias(run_dir_b)
    try:
        assert alias_a != alias_b
    finally:
        cleanup_model_subdir_alias(alias_a)
        cleanup_model_subdir_alias(alias_b)


def test_model_subdir_alias_is_safe_under_concurrent_calls_on_the_SAME_run_dir(tmp_path):
    import threading
    import time

    run_dir = tmp_path / "bridge" / "stage1" / "H64_L2_seq2500_seed123"
    run_dir.mkdir(parents=True)
    (run_dir / "gru_best.pt").write_text("fake checkpoint")

    n_threads = 40
    aliases: list[str] = []
    errors: list[Exception] = []
    lock = threading.Lock()
    barrier = threading.Barrier(n_threads)

    def worker():
        alias = None
        try:
            barrier.wait()
            alias = model_subdir_alias(run_dir)
            link_path = REPO_ROOT / "results" / alias
            time.sleep(0.01)
            assert link_path.is_symlink(), f"{alias} vanished while still in use"
            assert link_path.resolve() == run_dir.resolve()
            assert (link_path / "gru_best.pt").read_text() == "fake checkpoint"
            with lock:
                aliases.append(alias)
        except Exception as e:  # pragma: no cover - surfaced via errors list
            with lock:
                errors.append(e)
        finally:
            if alias is not None:
                cleanup_model_subdir_alias(alias)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent alias use failed: {errors}"
    assert len(aliases) == n_threads
    assert len(set(aliases)) == n_threads, "two concurrent calls returned the same alias"
    for alias in aliases:
        assert not (REPO_ROOT / "results" / alias).exists(), f"{alias} was not cleaned up"
