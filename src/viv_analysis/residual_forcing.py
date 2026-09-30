"""Random lift forcing for the residual forcing test (thesis Sec. 6.6.2).

The residual is the part of the CFD lift the teacher-forced GRU does not
reproduce (cl_true - cl_tf). make_forcing() turns it into an additive
forcing e(t) that coupled_inference adds to the GRU lift at every step.
"""

import numpy as np


def make_forcing(resid, n_steps, mode="surrogate", scale=1.0, seed=0):
    """Build a forcing vector of length n_steps from a measured residual.

    mode "surrogate": same amplitude spectrum as the residual, random phases
                      (the "frequency-matched residual" of Fig. 6.13).
    mode "white":     white noise with the residual's standard deviation (control).
    The result is multiplied by `scale`; `seed` makes it reproducible.
    """
    resid = np.asarray(resid, dtype=np.float64)
    n_resid = len(resid)

    rng = np.random.default_rng(seed)

    if mode == "surrogate":
        tiles_needed = int(np.ceil(n_steps / n_resid))
        pieces = []
        for _ in range(tiles_needed):
            fft = np.fft.rfft(resid)
            n_rfft = len(fft)
            phases = rng.uniform(0.0, 2.0 * np.pi, n_rfft)
            phases[0] = 0.0
            if n_resid % 2 == 0:
                phases[-1] = 0.0
            fft_surr = np.abs(fft) * np.exp(1j * phases)
            pieces.append(np.fft.irfft(fft_surr, n=n_resid))
        e = np.concatenate(pieces)[:n_steps]

    elif mode == "white":
        e = rng.normal(0.0, float(resid.std()), n_steps)

    else:
        raise ValueError(f"make_forcing: unknown mode '{mode}'. "
                         "Use 'surrogate' or 'white'.")

    return (scale * e).astype(np.float32)
