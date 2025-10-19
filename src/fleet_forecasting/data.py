"""Utility functions for loading and preparing fleet utilization datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "fleet_utilization_sample.csv"


def load_dataset(csv_path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the fleet utilization dataset.

    Parameters
    ----------
    csv_path:
        Path to the CSV file. Defaults to the packaged demo dataset.

    Returns
    -------
    pandas.DataFrame
        Dataset sorted by date with a DateTimeIndex.
    """

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    if "date" in df.columns:
        df.rename(columns={"date": "ds"}, inplace=True)

    if "ds" not in df.columns:
        raise ValueError("Dataset must contain a 'ds' column")

    df["ds"] = pd.to_datetime(df["ds"], utc=True).dt.tz_localize(None)
    df = df.sort_values("ds").set_index("ds", drop=False)

    numeric_cols = [
        "total_fleet",
        "active_vehicles",
        "idle_vehicles",
        "maintenance_events",
        "miles_driven",
        "fuel_consumed_gallons",
        "utilization_rate",
    ]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    if df[numeric_cols].isnull().any().any():
        raise ValueError("Dataset contains non-numeric values")

    return df


def train_test_split_time_series(
    data: pd.DataFrame, test_days: int = 30
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train/test segments keeping temporal order."""

    if test_days <= 0:
        raise ValueError("test_days must be positive")
    if len(data) <= test_days:
        raise ValueError("Dataset must be longer than test_days")

    train = data.iloc[:-test_days]
    test = data.iloc[-test_days:]
    return train, test
