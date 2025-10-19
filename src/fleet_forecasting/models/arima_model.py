"""ARIMA implementation for fleet utilization forecasting."""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from .base import ForecastModel


class ARIMAForecastModel(ForecastModel):
    def __init__(self, order: Tuple[int, int, int] = (2, 1, 2)):
        self.order = order
        self._model_fit: Optional[ARIMAResults] = None

    def fit(self, history: pd.DataFrame) -> None:
        series = history[self.target_column]
        model = ARIMA(series, order=self.order)
        self._model_fit = model.fit()

    def predict(self, future: pd.DataFrame) -> pd.Series:
        if self._model_fit is None:
            raise RuntimeError("Model must be fitted before calling predict")

        steps = len(future)
        forecast = self._model_fit.forecast(steps=steps)
        forecast.index = future.index
        forecast.name = self.target_column
        return forecast
