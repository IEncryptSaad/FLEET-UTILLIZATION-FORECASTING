# Fleet Utilization Forecasting MVP

This repository contains a small but complete fleet-utilization forecasting MVP. It ships with a
synthetic dataset, reproducible training pipeline, evaluation utilities, automated tests, and a
Streamlit dashboard that can be deployed for free.

## Project structure

```
.
├── data/
│   └── fleet_utilization_sample.csv      # synthetic dataset with two years of daily history
├── src/fleet_forecasting/
│   ├── cli.py                            # command line interface for training and forecasting
│   ├── data.py                           # dataset loading helpers
│   ├── evaluation.py                     # RMSE / MAPE / MAE metrics
│   ├── models/                           # Prophet and ARIMA model wrappers
│   └── pipeline.py                       # training + forecasting orchestration
├── streamlit_app.py                      # interactive dashboard for deployment
├── tests/                                # pytest-based regression test
├── requirements.txt
└── pyproject.toml                        # enables `pip install -e .`
```

The MVP uses two complementary models:

- **Prophet** for flexible seasonality modelling
- **ARIMA** as a lightweight baseline suitable for automated testing

Both models learn the historical utilization rate and produce short-range forecasts suitable for
capacity planning and maintenance scheduling.

## Getting started locally

1. Create a virtual environment (Python 3.10+ recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

2. Run the training CLI (Prophet by default):

   ```bash
   python -m fleet_forecasting.cli --test-days 30 --future-periods 45
   ```

   Output includes evaluation metrics on the hold-out window and a preview of the future forecast.
   Pass `--export forecasts.csv` to persist the results. Use `--model arima` for a lighter-weight run.

3. Execute the automated test suite (uses the ARIMA model for speed):

   ```bash
   pytest
   ```

4. Launch the Streamlit dashboard locally:

   ```bash
   streamlit run streamlit_app.py
   ```

   The dashboard lets you explore the dataset, compare model forecasts, and download predictions as a
   CSV file for downstream planning.

## Deploying on free infrastructure

The project is designed around free-to-use services:

- **Streamlit Community Cloud** – deploy the dashboard directly from this repository without any
  infrastructure cost. After signing in at [share.streamlit.io](https://share.streamlit.io), click
  *New app*, point it at your fork, set the entrypoint to `streamlit_app.py`, and add the contents of
  `requirements.txt`.
- **GitHub Actions** – add a workflow (example below) to automatically run `pytest` on pushes/pull
  requests for continuous validation using GitHub’s free minutes.

Example `.github/workflows/tests.yml` snippet:

```yaml
name: tests
on: [push, pull_request]
jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pip install -e .
      - run: pytest
```

## Customising the MVP

- Replace `data/fleet_utilization_sample.csv` with your actual fleet telemetry data. Ensure it keeps a
  `date` column plus the `utilization_rate` target.
- Extend `MODEL_REGISTRY` in `src/fleet_forecasting/pipeline.py` with additional models (e.g. XGBoost
  regressors on engineered features) to create ensembles.
- Modify `streamlit_app.py` to surface extra KPIs such as maintenance events or fuel consumption.

## License

This project is provided as-is for demonstration purposes. Adapt and extend it to meet your fleet
operations requirements.
