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
    def predict(self, future: pd.DataFrame) -> pd.Series:
        """Forecast the target values for provided timestamps."""

    def save(self, path: Path | str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path | str) -> "ForecastModel":
        model = joblib.load(path)
        if not isinstance(model, ForecastModel):
            raise TypeError("Loaded object is not a ForecastModel")
        return model
