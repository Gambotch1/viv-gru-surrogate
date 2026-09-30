"""Time-gap sequence segmentation: no GRU training window may span a
discontinuous CFD restart (see Ur=6.9491's ~77.5s report-file gap)."""
import numpy as np
import pandas as pd
import pytest

from viv_analysis.utils import segment_by_time_gaps
from viv_analysis.models.gru import VIVSequenceDataset


def test_no_gap_returns_single_segment():
    t = np.arange(0, 10, 0.002)
    assert segment_by_time_gaps(t) == [(0, len(t))]


def test_single_gap_splits_into_two_segments():
    t1 = np.arange(0, 5, 0.002)
    t2 = np.arange(0, 5, 0.002) + 100.0  # huge gap
    t = np.concatenate([t1, t2])
    segs = segment_by_time_gaps(t)
    assert len(segs) == 2
    assert segs[0] == (0, len(t1))
    assert segs[1] == (len(t1), len(t))


def test_two_gaps_splits_into_three_segments():
    t1 = np.arange(0, 3, 0.002)
    t2 = np.arange(0, 3, 0.002) + 50.0
    t3 = np.arange(0, 3, 0.002) + 100.0
    t = np.concatenate([t1, t2, t3])
    segs = segment_by_time_gaps(t)
    assert len(segs) == 3


def test_short_arrays_dont_crash():
    assert segment_by_time_gaps(np.array([])) == [(0, 0)]
    assert segment_by_time_gaps(np.array([1.0])) == [(0, 1)]


def _make_case_df(case_name, t, disp_freq=0.3):
    return pd.DataFrame({
        "case": case_name, "step": np.arange(len(t)), "time": t,
        "disp": np.sin(2 * np.pi * disp_freq * t) * 0.1,
        "vel": np.zeros(len(t)), "acc": np.zeros(len(t)),
        "cl": np.sin(2 * np.pi * disp_freq * t) * 0.2,
    })


def test_dataset_no_window_crosses_a_gap():
    seq_len = 20
    t1 = np.arange(0, 2, 0.002)          # 1000 samples
    t2 = np.arange(0, 2, 0.002) + 500.0  # another 1000 samples, huge gap after t1
    t = np.concatenate([t1, t2])
    df = _make_case_df("Ur5.0", t)

    ds = VIVSequenceDataset(
        df, seq_len=seq_len, target_col="cl", input_cols=["disp", "vel"],
        release_time={"Ur5.0": -np.inf}, stride=1,
    )
    assert len(ds) > 0

    n_seg1 = len(t1)
    n_seg2 = len(t2)
    expected_n = max(0, n_seg1 - seq_len) + max(0, n_seg2 - seq_len)
    assert len(ds) == expected_n, (
        "dataset size should match windowing each segment independently, "
        "not the whole (gapped) array as one block")


def test_dataset_matches_ungapped_case_when_no_gap_present():
    seq_len = 20
    t = np.arange(0, 4, 0.002)
    df = _make_case_df("Ur5.0", t)
    ds = VIVSequenceDataset(
        df, seq_len=seq_len, target_col="cl", input_cols=["disp", "vel"],
        release_time={"Ur5.0": -np.inf}, stride=1,
    )
    assert len(ds) == len(t) - seq_len


def test_dataset_with_gap_produces_fewer_windows_than_naive_ungapped_count():
    """Sanity check on the bug this fixes: treating the gapped array as one
    contiguous block would overcount by seq_len-1 windows that illegally
    span the gap. The segmented count must be strictly smaller."""
    seq_len = 20
    t1 = np.arange(0, 2, 0.002)
    t2 = np.arange(0, 2, 0.002) + 500.0
    t = np.concatenate([t1, t2])
    df = _make_case_df("Ur5.0", t)
    ds = VIVSequenceDataset(
        df, seq_len=seq_len, target_col="cl", input_cols=["disp", "vel"],
        release_time={"Ur5.0": -np.inf}, stride=1,
    )
    naive_ungapped_count = len(t) - seq_len
    assert len(ds) < naive_ungapped_count
