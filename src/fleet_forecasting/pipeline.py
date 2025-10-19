"""Training pipeline for fleet utilization forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Type

import pandas as pd

from .data import load_dataset, train_test_split_time_series
from .evaluation import evaluate_forecast, ForecastMetrics
from .models.arima_model import ARIMAForecastModel
from .models.base import ForecastModel
from .models.prophet_model import ProphetForecastModel


MODEL_REGISTRY: Dict[str, Type[ForecastModel]] = {
    "prophet": ProphetForecastModel,
    "arima": ARIMAForecastModel,
}


@dataclass
class PipelineResult:
    model_name: str
    metrics: ForecastMetrics
    forecast: pd.Series


def run_training_pipeline(
    model_name: str = "prophet",
    dataset_path: Path | str | None = None,
    test_days: int = 30,
) -> PipelineResult:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'. Options: {list(MODEL_REGISTRY)}")

    data = load_dataset(dataset_path) if dataset_path else load_dataset()
    train, test = train_test_split_time_series(data, test_days=test_days)

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls()
    model.fit(train)

    predictions = model.predict(test)
    metrics = evaluate_forecast(test[model.target_column], predictions)

    return PipelineResult(model_name=model_name, metrics=metrics, forecast=predictions)


def forecast_future(
    model: ForecastModel,
    history: pd.DataFrame,
    periods: int = 30,
) -> pd.Series:
    """Generate future forecasts by extending the index."""

    last_timestamp = history.index[-1]
    future_index = pd.date_range(last_timestamp + pd.Timedelta(days=1), periods=periods, freq="D")
    future_df = pd.DataFrame(index=future_index)
    return model.predict(future_df)
