"""Model interface for fleet utilization forecasting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class ForecastModel(ABC):
    target_column: str = "utilization_rate"

    @abstractmethod
    def fit(self, history: pd.DataFrame) -> None:
        """Train the model using a dataframe with a DateTimeIndex."""

    @abstractmethod
    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        """Forecast the target values for provided timestamps."""

    def save(self, path: Path | str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path | str) -> "ForecastModel":
        model = joblib.load(path)
        if not isinstance(model, ForecastModel):
            raise TypeError("Loaded object is not a ForecastModel")
        return model

    @staticmethod
    def _coerce_datetime_index(data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if "ds" in df.columns:
            df["ds"] = pd.to_datetime(df["ds"], errors="coerce", utc=True).dt.tz_localize(None)
            df = df.dropna(subset=["ds"])
            df = df.set_index("ds", drop=False)
        else:
            df.index = pd.to_datetime(df.index, errors="coerce", utc=True).tz_localize(None)
        return df.sort_index()

    @staticmethod
    def _format_forecast(
        predictions: pd.Series,
        index: pd.Index,
        lower: pd.Series | None = None,
        upper: pd.Series | None = None,
    ) -> pd.DataFrame:
        df = pd.DataFrame({"yhat": predictions.astype(float)}, index=index)
        if lower is None:
            df["yhat_lower"] = df["yhat"]
        else:
            df["yhat_lower"] = lower.astype(float)
        if upper is None:
            df["yhat_upper"] = df["yhat"]
        else:
            df["yhat_upper"] = upper.astype(float)
        df.index.name = "ds"
        df["ds"] = df.index
        return df
