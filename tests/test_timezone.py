from fleet_forecasting.data import load_dataset
from fleet_forecasting.models.prophet_model import ProphetForecastModel


def test_load_dataset_returns_tz_naive_ds_column():
    df = load_dataset()
    assert "ds" in df.columns, "load_dataset should expose a 'ds' column"
    assert df["ds"].dt.tz is None, "'ds' column must be timezone naive"


def test_prophet_model_handles_tz_naive_and_aware_inputs():
    full_df = load_dataset()
    history = full_df.iloc[:-7]
    future = full_df.iloc[-7:]

    # Naive inputs should work without issue
    model_naive = ProphetForecastModel()
    model_naive.fit(history)
    naive_forecast = model_naive.predict(future)
    assert len(naive_forecast) == len(future)

    # Prepare timezone-aware copies
    tz_history = history.copy()
    tz_history.index = tz_history.index.tz_localize("UTC")
    tz_history["ds"] = tz_history["ds"].dt.tz_localize("UTC")

    tz_future = future.copy()
    tz_future.index = tz_future.index.tz_localize("UTC")
    tz_future["ds"] = tz_future["ds"].dt.tz_localize("UTC")

    model_tz = ProphetForecastModel()
    model_tz.fit(tz_history)
    tz_forecast = model_tz.predict(tz_future)
    assert len(tz_forecast) == len(tz_future)
