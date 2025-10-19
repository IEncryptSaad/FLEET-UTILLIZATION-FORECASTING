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
        self._history: Optional[pd.DataFrame] = None

    def fit(self, history: pd.DataFrame) -> None:
        df = self._coerce_datetime_index(history)
        if self.target_column not in df.columns:
            raise ValueError(f"History must contain target column '{self.target_column}'")

        training_df = df[["ds", self.target_column]].rename(columns={self.target_column: "y"})
        training_df = training_df.dropna(subset=["y"])

        if len(training_df) < 15:
            raise ValueError("Prophet requires at least 15 observations for stable fitting")

        self._model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=False,
        )
        self._model.fit(training_df[["ds", "y"]])
        self._history = training_df

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Model must be fitted before calling predict")

        df_future = self._coerce_datetime_index(future)
        ds_values = df_future.index
        forecast = self._model.predict(pd.DataFrame({"ds": ds_values}))
        forecast["ds"] = pd.to_datetime(forecast["ds"], utc=True).dt.tz_localize(None)
        forecast = forecast.set_index("ds").loc[ds_values]
        return self._format_forecast(
            predictions=forecast["yhat"],
            index=ds_values,
            lower=forecast.get("yhat_lower"),
            upper=forecast.get("yhat_upper"),
        )

    def component_plot(self) -> Optional["go.Figure"]:
        if self._model is None or self._history is None:
            return None
        try:  # pragma: no cover - optional visualization path
            from prophet.plot import plot_components_plotly
        except ImportError:  # pragma: no cover
            return None
        forecast = self._model.predict(self._history[["ds"]])
        figure = plot_components_plotly(self._model, forecast)
        return figure
