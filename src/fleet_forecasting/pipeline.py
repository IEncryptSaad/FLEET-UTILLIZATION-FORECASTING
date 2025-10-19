"""Training pipeline for fleet utilization forecasting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Type

import pandas as pd

from .data import load_dataset, train_test_split_time_series
from .evaluation import ForecastMetrics, evaluate_forecast
from .models.arima_model import ARIMAForecastModel
from .models.base import ForecastModel
from .models.prophet_model import ProphetForecastModel


LOGGER = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Type[ForecastModel]] = {
    "prophet": ProphetForecastModel,
    "arima": ARIMAForecastModel,
}


@dataclass
class PipelineResult:
    model_name: str
    metrics: ForecastMetrics
    forecast: pd.DataFrame
    model: ForecastModel
    train: pd.DataFrame
    test: pd.DataFrame


def _resolve_model_order(model_name: str, fallback_order: Sequence[str] | None) -> List[str]:
    order: List[str] = [model_name]
    if fallback_order:
        for candidate in fallback_order:
            if candidate not in order:
                order.append(candidate)
    return order


def run_training_pipeline(
    model_name: str = "prophet",
    dataset_path: Path | str | None = None,
    dataset: Optional[pd.DataFrame] = None,
    test_days: int = 30,
    fallback_models: Sequence[str] | None = ("arima",),
) -> PipelineResult:
    """Train a forecasting model with optional fallbacks for robustness."""

    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'. Options: {list(MODEL_REGISTRY)}")

    data = dataset.copy() if dataset is not None else load_dataset(dataset_path) if dataset_path else load_dataset()
    train, test = train_test_split_time_series(data, test_days=test_days)

    errors: List[Tuple[str, Exception]] = []
    for candidate_name in _resolve_model_order(model_name, fallback_models):
        model_cls = MODEL_REGISTRY.get(candidate_name)
        if model_cls is None:
            LOGGER.warning("Skipping unknown model '%s' in fallback list", candidate_name)
            continue
        model = model_cls()
        try:
            model.fit(train)
            predictions = model.predict(test)
            if "yhat" not in predictions.columns:
                raise ValueError("Model predictions must include a 'yhat' column")
            metrics = evaluate_forecast(test[model.target_column], predictions["yhat"])
            return PipelineResult(
                model_name=candidate_name,
                metrics=metrics,
                forecast=predictions,
                model=model,
                train=train,
                test=test,
            )
        except Exception as exc:  # pragma: no cover - error path
            LOGGER.exception("Model '%s' failed during training", candidate_name)
            errors.append((candidate_name, exc))

    error_messages = ", ".join(f"{name}: {error}" for name, error in errors) or "no models tried"
    raise RuntimeError(f"All models failed to train successfully ({error_messages})")


def forecast_future(
    model: ForecastModel,
    history: pd.DataFrame,
    periods: int = 30,
) -> pd.DataFrame:
    """Generate future forecasts by extending the index."""

    if periods <= 0:
        raise ValueError("periods must be positive")
    if history.empty:
        raise ValueError("history must contain data")

    history = history.sort_index()
    last_timestamp = history.index[-1]
    inferred_freq = history.index.freq or pd.infer_freq(history.index)
    if inferred_freq is None:
        inferred_freq = pd.Timedelta(history.index.to_series().diff().dropna().median() or pd.Timedelta(days=1))
    if isinstance(inferred_freq, pd.Timedelta):
        future_index = pd.date_range(last_timestamp + inferred_freq, periods=periods, freq=inferred_freq)
    else:
        future_index = pd.date_range(last_timestamp, periods=periods + 1, freq=inferred_freq)[1:]

    future_df = pd.DataFrame({"ds": future_index})
    future_df = future_df.set_index("ds", drop=False)
    forecast = model.predict(future_df)
    forecast = forecast.copy()
    forecast["ds"] = forecast.index
    return forecast
