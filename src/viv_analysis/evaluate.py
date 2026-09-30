"""Open-loop error metrics (MAE, RMSE, R^2) used after training."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE and R^2 between true and predicted lift coefficients."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2":   float(r2_score(y_true, y_pred)),
    }
