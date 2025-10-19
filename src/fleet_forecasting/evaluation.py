"""Evaluation utilities for fleet utilization forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class ForecastMetrics:
    rmse: float
    mape: float
    mae: float

    def to_dict(self) -> Dict[str, float]:
        return {"rmse": self.rmse, "mape": self.mape, "mae": self.mae}


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> ForecastMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.abs((y_true - y_pred) / y_true.replace(0, np.nan)).mean() * 100
    mape = float(np.nan_to_num(mape))
    return ForecastMetrics(rmse=rmse, mape=mape, mae=mae)
