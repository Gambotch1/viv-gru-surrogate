import numpy as np
import pandas as pd

from viv_analysis.models.gru import fit_scalers


def test_fit_scalers_ignores_rows_not_in_train_df():
    train_df = pd.DataFrame({
        "case": ["Ur5.0"] * 20,
        "disp": np.full(20, 1.0),
        "vel": np.full(20, 2.0),
        "cl": np.full(20, 0.5),
    })
    val_df = pd.DataFrame({
        "case": ["Ur9.0"] * 20,
        "disp": np.full(20, 1000.0),
        "vel": np.full(20, 2000.0),
        "cl": np.full(20, 500.0),
    })

    x_scaler, y_scaler = fit_scalers(train_df, ["disp", "vel"], "cl")

    np.testing.assert_allclose(x_scaler.mean_, [1.0, 2.0])
    np.testing.assert_allclose(y_scaler.mean_, [0.5])

    combined = pd.concat([train_df, val_df], ignore_index=True)
    x_scaler_leaked, _ = fit_scalers(combined, ["disp", "vel"], "cl")
    assert not np.allclose(x_scaler_leaked.mean_, x_scaler.mean_)
