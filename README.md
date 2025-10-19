# Fleet Utilization Forecasting MVP

This repository provides a production-ready minimum viable product for forecasting fleet utilization. The system ingests historical vehicle data, cleans and validates it, and trains time-series forecasting models to predict future utilization trends. It includes automated model evaluation and visualization through a fully interactive Streamlit dashboard.

## Overview

The Fleet Utilization Forecasting MVP demonstrates how organizations can use machine learning to monitor and forecast operational efficiency in transportation or logistics environments. The project covers the full workflow, from data ingestion to forecast interpretation, and follows industry-standard engineering and data science practices.

## Key Features

**1. Robust Data Ingestion**
Automated reading of uploaded CSV datasets or use of the included synthetic dataset.
Validation of columns, schema consistency, and automatic normalization of datetime fields.
Timezone and missing value handling with explicit logging.

**2. Forecasting Models**
Primary forecasting models implemented with Prophet and ARIMA.
Configurable training parameters for validation window and forecast horizon.
Automatic metric computation including RMSE, MAE, and MAPE for model comparison.

**3. Interactive Streamlit Dashboard**
Clean and modular interface for uploading datasets, configuring model parameters, and visualizing results.
Real-time rendering of forecast curves with confidence intervals.
Tabular output of future predictions for inspection or export.
Validation and backtesting to assess model accuracy on historical data.

**4. End-to-End Architecture**
Modular Python package layout under `src/fleet_forecasting`.
Automated tests under `tests/` for pipeline and timezone validation.
Compatible with cloud-based deployment on Streamlit Community Cloud or containerized environments.

## Repository Structure

```
FLEET-UTILILIZATION-FORECASTING/
│
├── data/
│   └── fleet_utilization_sample.csv
│
├── src/
│   └── fleet_forecasting/
│       ├── data.py                 # Data loading and validation
│       ├── pipeline.py             # Main training and forecasting logic
│       ├── models/
│       │   ├── prophet_model.py    # Prophet model wrapper
│       │   └── arima_model.py      # ARIMA model wrapper
│       └── evaluation.py           # Metric computation
│
├── tests/
│   ├── test_pipeline.py
│   └── test_timezone.py
│
├── streamlit_app.py                # Streamlit dashboard entry point
├── pyproject.toml                  # Package configuration
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .gitignore
```

## Setup Instructions

### 1. Create and activate a virtual environment

```
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```
pip install -e .
pip install -r requirements.txt
```

### 3. Run the test suite

```
python -m pytest -q
```

All tests should pass successfully before running the Streamlit app.

### 4. Launch the Streamlit dashboard

```
streamlit run streamlit_app.py --server.port 3000 --server.address 0.0.0.0
```

Then open the displayed URL in a web browser. The dashboard will load automatically.

## Usage Workflow

1. Upload a dataset in CSV format, or use the included sample file.
2. Select the desired forecasting model (Prophet or ARIMA).
3. Configure the validation window and forecast horizon.
4. Train the model and view the RMSE, MAPE, and MAE metrics.
5. Examine the generated forecast chart and tabular output.
6. Optionally export predictions or retrain with adjusted parameters.

## Example Metrics

| Metric | Description                    | Example Value |
| ------ | ------------------------------ | ------------- |
| RMSE   | Root Mean Squared Error        | 0.0464        |
| MAPE   | Mean Absolute Percentage Error | 5.01%         |
| MAE    | Mean Absolute Error            | 0.0366        |

## Technology Stack

Language: Python 3.12
Libraries: Prophet, scikit-learn, pandas, numpy, statsmodels
Interface: Streamlit
Testing: Pytest
Environment: Cloud-ready, virtual environment compatible

## Project Highlights

Fully modular codebase ready for extension and scaling.
Implements industry-standard best practices for reproducibility.
Proven performance on the provided dataset with low forecast error.
Clean UI for operational monitoring and executive visualization.

## License

This project is released under the MIT License.

## Acknowledgment

Developed as part of an applied AI engineering workflow for production-grade forecasting systems. The architecture emphasizes clarity, reproducibility, and deployment readiness across cloud environments.
