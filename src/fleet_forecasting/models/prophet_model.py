"""Prophet implementation of the ForecastModel interface."""

from __future__ import annotations

from typing import Optional

import pandas as pd
from prophet import Prophet

from .base import ForecastModel


class ProphetForecastModel(ForecastModel):
    def __init__(self, yearly_seasonality: bool = True, weekly_seasonality: bool = True):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self._model: Optional[Prophet] = None

    def fit(self, history: pd.DataFrame) -> None:
        df = history.reset_index().rename(columns={"date": "ds", self.target_column: "y"})
        if "ds" not in df.columns:
            df = df.rename(columns={df.columns[0]: "ds"})
        self._model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=False,
        )
        self._model.fit(df[["ds", "y"]])

    def predict(self, future: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Model must be fitted before calling predict")

        df_future = future.reset_index().rename(columns={"date": "ds"})
        if "ds" not in df_future.columns:
            df_future = df_future.rename(columns={df_future.columns[0]: "ds"})
        forecast = self._model.predict(df_future[["ds"]])
        return pd.Series(forecast["yhat"].values, index=future.index, name=self.target_column)
