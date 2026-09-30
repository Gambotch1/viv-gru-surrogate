import inspect

import numpy as np
import pandas as pd

from viv_analysis.train_gru import apply_nd_transform


def _toy_df():
    n = 50
    return pd.DataFrame({
        "case": ["Ur5.0"] * n,
        "time": np.arange(n) * 0.005,
        "step": np.arange(n),
        "disp": np.linspace(0.1, 1.0, n),
        "vel": np.linspace(0.01, 0.1, n),
        "cl": np.sin(np.linspace(0, 3, n)),
    })


def test_applying_transform_twice_differs_from_once():
    df = _toy_df()
    D, fn = 0.2, 0.2
    once = apply_nd_transform(df.copy(), nd_inputs=True, D=D, fn=fn,
                               input_cols=["disp", "vel"])
    twice = apply_nd_transform(once.copy(), nd_inputs=True, D=D, fn=fn,
                                input_cols=["disp", "vel"])
    assert not np.allclose(once["disp"].to_numpy(), twice["disp"].to_numpy())
    assert not np.allclose(once["vel"].to_numpy(), twice["vel"].to_numpy())
    ratio_disp = twice["disp"].to_numpy() / once["disp"].to_numpy()
    np.testing.assert_allclose(ratio_disp, 1.0 / D, rtol=1e-5)


def test_open_loop_metrics_calls_apply_nd_transform_exactly_once_in_source():
    import _common
    src = inspect.getsource(_common.compute_open_loop_metrics)
    assert src.count("apply_nd_transform(") == 1
