"""Utility functions for loading and preparing fleet utilization datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import IO, Iterable, Tuple

import pandas as pd


LOGGER = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "fleet_utilization_sample.csv"
DATE_COLUMN_CANDIDATES = ("ds", "date", "datetime", "timestamp", "day")
TARGET_COLUMN_CANDIDATES = ("utilization_rate", "utilization", "utilisation_rate")


def _read_csv(csv_source: Path | str | IO[str] | IO[bytes]) -> pd.DataFrame:
    if hasattr(csv_source, "read"):
        return pd.read_csv(csv_source)  # type: ignore[arg-type]
    path = Path(csv_source)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def _detect_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def load_dataset(
    csv_path: Path | str | IO[str] | IO[bytes] = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    """Load the fleet utilization dataset with robust cleaning and validation."""

    raw_df = _read_csv(csv_path)
    if raw_df.empty:
        raise ValueError("Dataset is empty")

    raw_df.columns = raw_df.columns.str.strip().str.lower()

    date_column = _detect_column(raw_df.columns, DATE_COLUMN_CANDIDATES)
    if date_column is None:
        raise ValueError("Dataset must include a date or 'ds' column")

    target_column = _detect_column(raw_df.columns, TARGET_COLUMN_CANDIDATES)
    if target_column is None:
        raise ValueError("Dataset must include a utilization rate column")

    df = raw_df.copy()
    df.rename(columns={date_column: "ds", target_column: "utilization_rate"}, inplace=True)

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce", utc=True).dt.tz_localize(None)
    invalid_dates = df["ds"].isna().sum()
    if invalid_dates:
        LOGGER.warning("Dropping %s rows with invalid dates", invalid_dates)
    df = df.dropna(subset=["ds"]).drop_duplicates(subset=["ds"], keep="last")

    numeric_columns = [col for col in df.columns if col != "ds"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_numeric = df[numeric_columns].isna().any(axis=1)
    if invalid_numeric.any():
        LOGGER.warning("Dropping %s rows with invalid numeric values", int(invalid_numeric.sum()))
        df = df.loc[~invalid_numeric]

    df = df.sort_values("ds")
    df = df.set_index("ds", drop=False)

    try:
        inferred_freq = pd.infer_freq(df.index)
    except ValueError:
        inferred_freq = None
    if inferred_freq:
        df.index.freq = inferred_freq

    if df.empty:
        raise ValueError("Dataset did not contain any valid rows after cleaning")

    return df


def train_test_split_time_series(
    data: pd.DataFrame, test_days: int = 30
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train/test segments keeping temporal order."""

    if test_days <= 0:
        raise ValueError("test_days must be positive")
    if len(data) <= test_days:
        raise ValueError("Dataset must be longer than test_days")

    data = data.sort_index()
    train = data.iloc[:-test_days]
    test = data.iloc[-test_days:]
    return train, test
