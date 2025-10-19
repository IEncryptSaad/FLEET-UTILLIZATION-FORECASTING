from pathlib import Path

import pandas as pd

from fleet_forecasting.pipeline import run_training_pipeline


def test_run_training_pipeline_arima(tmp_path: Path) -> None:
    result = run_training_pipeline(model_name="arima", test_days=14)
    assert result.metrics.rmse >= 0
    assert len(result.forecast) == 14

    # ensure CLI compatible export logic
    export_path = tmp_path / "forecast.csv"
    df = result.forecast.reset_index()
    df.columns = ["date", "utilization_rate"]
    df.to_csv(export_path, index=False)
    loaded = pd.read_csv(export_path)
    assert {"date", "utilization_rate"} == set(loaded.columns)
