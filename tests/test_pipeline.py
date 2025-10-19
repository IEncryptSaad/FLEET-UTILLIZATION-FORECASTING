from fleet_forecasting.data import load_dataset
from fleet_forecasting.pipeline import forecast_future, run_training_pipeline


def test_run_training_pipeline_returns_forecast_dataframe() -> None:
    dataset = load_dataset()
    result = run_training_pipeline(model_name="prophet", dataset=dataset, test_days=14)
    assert result.metrics.rmse >= 0
    assert set(["yhat", "yhat_lower", "yhat_upper", "ds"]).issubset(result.forecast.columns)
    assert len(result.forecast) == 14


def test_forecast_future_extends_timeline() -> None:
    dataset = load_dataset()
    result = run_training_pipeline(model_name="prophet", dataset=dataset, test_days=30)
    future = forecast_future(result.model, dataset, periods=10)
    assert len(future) == 10
    assert future.index[0] > dataset.index[-1]
