from __future__ import annotations

"""Prophet implementation of the ForecastModel interface."""

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
        df = history.copy()
        if "ds" in df.columns:
            ds_series = df["ds"]
        else:
            ds_series = df.index.to_series(index=df.index, name="ds")

        if self.target_column not in df.columns:
            raise ValueError(f"History must contain target column '{self.target_column}'")

        y_series = df[self.target_column]
        df = pd.DataFrame({"ds": ds_series, "y": y_series})
        df["ds"] = pd.to_datetime(df["ds"], utc=True).dt.tz_localize(None)
        self._model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=False,
        )
        self._model.fit(df[["ds", "y"]])

    def predict(self, future: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Model must be fitted before calling predict")

        df_future = future.copy()
        if "ds" in df_future.columns:
            ds_series = df_future["ds"]
        else:
            ds_series = df_future.index.to_series(index=df_future.index, name="ds")

        df_future = pd.DataFrame({"ds": ds_series})
        df_future["ds"] = pd.to_datetime(df_future["ds"], utc=True).dt.tz_localize(None)
        forecast = self._model.predict(df_future[["ds"]])
        return pd.Series(forecast["yhat"].values, index=future.index, name=self.target_column)
