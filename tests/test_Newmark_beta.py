import numpy as np
import torch
from viv_analysis.coupled_inference import Newmark_beta

def newmark_beta_np(F, h, h_dot, h_ddot, dt, m, c, k, beta=0.25, gamma=0.5):
    a1 = m / (beta * dt**2) + gamma * c / (beta * dt)
    a2 = m / (beta * dt) + (gamma / beta - 1.0) * c
    a3 = (0.5 / beta - 1.0) * m + dt * (gamma / (2.0 * beta) - 1.0) * c
    kbar = k + a1

    pbar = F + a1 * h + a2 * h_dot + a3 * h_ddot
    h_new = pbar / kbar
    h_dot_new = (gamma / (beta * dt)) * (h_new - h) + (1.0 - gamma / beta) * h_dot \
                + dt * (1.0 - gamma / (2.0 * beta)) * h_ddot
    h_ddot_new = (1.0 / (beta * dt**2)) * (h_new - h) - (1.0 / (beta * dt)) * h_dot \
                 - (0.5 / beta - 1.0) * h_ddot
    return h_new, h_dot_new, h_ddot_new

def test_newmark_beta_matches_numpy_and_has_grad():
    F = torch.tensor([0.001], dtype=torch.float32, requires_grad=True)
    h = torch.tensor([0.05], dtype=torch.float32)
    h_dot = torch.tensor([0.001], dtype=torch.float32)
    h_ddot = torch.tensor([-0.01], dtype=torch.float32)

    dt = 0.005
    m = 0.2513
    c = 0.004421
    k = 0.3969

    h_new, v_new, a_new = Newmark_beta(F, h, h_dot, h_ddot, dt, m, c, k)

    expected = newmark_beta_np(
        F.detach().cpu().numpy(),
        h.detach().cpu().numpy(),
        h_dot.detach().cpu().numpy(),
        h_ddot.detach().cpu().numpy(),
        dt, m, c, k,
    )

    assert torch.allclose(h_new, torch.tensor(expected[0], dtype=h_new.dtype), atol=1e-6)
    assert torch.allclose(v_new, torch.tensor(expected[1], dtype=v_new.dtype), atol=1e-6)
    assert torch.allclose(a_new, torch.tensor(expected[2], dtype=a_new.dtype), atol=1e-6)

    loss = h_new.sum()
    loss.backward()
    assert F.grad is not None
    assert torch.isfinite(F.grad).all()