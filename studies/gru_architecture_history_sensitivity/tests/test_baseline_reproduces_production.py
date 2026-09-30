from pathlib import Path

import torch

from viv_analysis.models.gru import VIV_GRU
from _common import REPO_ROOT


def test_fresh_h64_l2_matches_existing_production_checkpoint_shapes():
    candidates = [
        REPO_ROOT / "results" / "gru_cylinder200_nd_context_noacc" / "gru_best.pt",
        REPO_ROOT / "results" / "gru_cylinder200_dim_context_noacc" / "gru_best.pt",
    ]
    ckpt_path = next((c for c in candidates if c.exists()), None)
    if ckpt_path is None:
        import pytest
        pytest.skip("no existing production checkpoint found on disk to compare against")

    state = torch.load(ckpt_path, map_location="cpu")
    fresh = VIV_GRU(input_size=3, hidden_size=64, num_layers=2, dropout=0.1)
    fresh_state = fresh.state_dict()

    assert set(state.keys()) == set(fresh_state.keys())
    for k in state:
        assert state[k].shape == fresh_state[k].shape, f"shape mismatch at {k}"

    fresh_param_count = sum(p.numel() for p in fresh.parameters() if p.requires_grad)
    ckpt_param_count = sum(v.numel() for v in state.values())
    assert fresh_param_count == ckpt_param_count
