"""Command line interface for the fleet forecasting MVP."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from . import data as data_utils
from .models.base import ForecastModel
from .pipeline import MODEL_REGISTRY, forecast_future, run_training_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fleet utilization forecasting CLI")
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="prophet",
        help="Model to train",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional path to a custom dataset",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=30,
        help="Number of days to hold out for evaluation",
    )
    parser.add_argument(
        "--future-periods",
        type=int,
        default=30,
        help="Days to forecast beyond the dataset",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Path to save future forecasts as CSV",
    )
    return parser


def train_and_forecast(
    model_name: str,
    dataset_path: Optional[Path],
    test_days: int,
    future_periods: int,
    export_path: Optional[Path],
) -> None:
    result = run_training_pipeline(model_name=model_name, dataset_path=dataset_path, test_days=test_days)
    print(f"Model: {result.model_name}")
    print("Evaluation metrics (test set):")
    for key, value in result.metrics.to_dict().items():
        print(f"  {key}: {value:.4f}")

    data = data_utils.load_dataset(dataset_path) if dataset_path else data_utils.load_dataset()
    model_cls = MODEL_REGISTRY[model_name]
    model: ForecastModel = model_cls()
    model.fit(data)

    future_forecast = forecast_future(model, data, periods=future_periods)
    print(f"\nForecast for the next {future_periods} days:")
    print(future_forecast.head())

    if export_path:
        export_df = future_forecast.reset_index()
        export_df.columns = ["date", "utilization_rate"]
        export_df.to_csv(export_path, index=False)
        print(f"\nSaved future forecast to {export_path}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    train_and_forecast(
        model_name=args.model,
        dataset_path=args.dataset,
        test_days=args.test_days,
        future_periods=args.future_periods,
        export_path=args.export,
    )


if __name__ == "__main__":
    main()
