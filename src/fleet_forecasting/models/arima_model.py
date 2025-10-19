"""ARIMA implementation for fleet utilization forecasting."""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from .base import ForecastModel


class ARIMAForecastModel(ForecastModel):
    def __init__(self, order: Tuple[int, int, int] = (2, 1, 2)):
        self.order = order
        self._model_fit: Optional[ARIMAResults] = None

    def fit(self, history: pd.DataFrame) -> None:
        df = self._coerce_datetime_index(history)
        if self.target_column not in df.columns:
            raise ValueError(f"History must contain '{self.target_column}'")

        series = df[self.target_column].astype(float)
        freq = df.index.freq or pd.infer_freq(df.index)
        if freq is not None:
            series.index.freq = freq  # type: ignore[attr-defined]

        model = ARIMA(
            series,
            order=self.order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self._model_fit = model.fit(method_kwargs={"warn_convergence": False})

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        if self._model_fit is None:
            raise RuntimeError("Model must be fitted before calling predict")

        future_df = self._coerce_datetime_index(future)
        steps = len(future_df)
        if steps == 0:
            empty_series = pd.Series(dtype=float, index=future_df.index)
            return self._format_forecast(empty_series, future_df.index)

        forecast_res = self._model_fit.get_forecast(steps=steps)
        mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=0.05)
        lower = conf_int.iloc[:, 0]
        upper = conf_int.iloc[:, 1]
        mean.index = future_df.index
        lower.index = future_df.index
        upper.index = future_df.index
        return self._format_forecast(mean, future_df.index, lower=lower, upper=upper)
