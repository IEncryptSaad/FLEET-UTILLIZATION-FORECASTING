# Fleet Utilization Forecasting MVP

This repository delivers a production-ready minimum viable product for forecasting fleet utilization.
It includes a hardened data cleaning layer, Prophet and ARIMA forecasting models, an end-to-end
training pipeline, automated regression tests, and a Streamlit dashboard that is ready for one-click
deployment on Streamlit Community Cloud.

## Key features

- **Robust ingestion** – automatic column normalisation, timezone handling, and schema validation for
  uploaded CSV files.
- **Battle-tested models** – Prophet with graceful fallbacks to ARIMA when data volume or quality is
  insufficient.
- **Rich dashboard** – cached data/model pipelines, interactive Plotly visuals, download buttons, and
  optional Prophet component plots.
- **Deployment ready** – minimal requirements, Streamlit theme configuration, and clean pytest suite.

## Project structure

```
.
├── data/
│   └── fleet_utilization_sample.csv
├── src/fleet_forecasting/
│   ├── data.py
│   ├── evaluation.py
│   ├── models/
│   ├── pipeline.py
│   └── …
├── streamlit_app.py
├── tests/
├── requirements.txt
├── pyproject.toml
└── .streamlit/config.toml
```

## Local development workflow

1. **Create and activate a virtual environment (Python 3.10+ recommended):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   ```

2. **Install dependencies and register the package in editable mode:**

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Run the automated test suite:**

   ```bash
   pytest -q
   ```

4. **Launch the Streamlit dashboard:**

   ```bash
   streamlit run streamlit_app.py
   ```

   The app loads the bundled dataset by default, supports CSV uploads, displays validation metrics,
   and offers forecast downloads.

## Deployment on Streamlit Community Cloud

1. Push your fork of this repository to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) and choose **New app**.
3. Select your repository, set the branch, and use `streamlit_app.py` as the entry point.
4. The default `requirements.txt` and `.streamlit/config.toml` ensure the app launches with the same
   theme and dependencies used locally.

## Working with your own data

- Replace or augment `data/fleet_utilization_sample.csv` with your telemetry export. The loader will
  automatically detect and rename the date column to `ds`, coerce numeric values, and drop invalid
  rows.
- Extend `MODEL_REGISTRY` in `src/fleet_forecasting/pipeline.py` to experiment with additional
  forecasting models or ensembles.
- Update `streamlit_app.py` to surface extra KPIs (maintenance events, miles driven, etc.) for your
  stakeholders.

## License

This project is provided as-is for demonstration purposes. Adapt and extend it to meet your fleet
operations requirements.
