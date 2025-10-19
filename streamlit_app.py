"""Streamlit dashboard for the fleet utilization forecasting MVP."""

from __future__ import annotations

import hashlib
import io
import logging
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fleet_forecasting.data import load_dataset
from fleet_forecasting.pipeline import MODEL_REGISTRY, PipelineResult, forecast_future, run_training_pipeline


LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)


def _hash_dataframe(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.md5(hashed).hexdigest()


@st.cache_data(show_spinner=False)
def _load_data(file_bytes: Optional[bytes]) -> pd.DataFrame:
    if file_bytes:
        return load_dataset(io.BytesIO(file_bytes))
    return load_dataset()


@st.cache_resource(show_spinner=False, hash_funcs={pd.DataFrame: _hash_dataframe})
def _train_model_cached(model_name: str, dataset: pd.DataFrame, test_days: int) -> PipelineResult:
    return run_training_pipeline(model_name=model_name, dataset=dataset, test_days=test_days)


def _render_metrics(result: PipelineResult) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{result.metrics.rmse:.4f}")
    col2.metric("MAPE", f"{result.metrics.mape:.2f}%")
    col3.metric("MAE", f"{result.metrics.mae:.4f}")


def _build_forecast_chart(history: pd.DataFrame, backtest: pd.DataFrame, future: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["utilization_rate"],
            mode="lines",
            name="Actual utilization",
            line=dict(color="#1f77b4"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=backtest.index,
            y=backtest["yhat"],
            mode="lines",
            name="Backtest forecast",
            line=dict(color="#ff7f0e"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future.index,
            y=future["yhat"],
            mode="lines",
            name="Future forecast",
            line=dict(color="#2ca02c"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(future.index) + list(future.index[::-1]),
            y=list(future["yhat_upper"]) + list(future["yhat_lower"][::-1]),
            fill="toself",
            fillcolor="rgba(44, 160, 44, 0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Forecast interval",
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Utilization rate",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.2, x=0),
        margin=dict(l=40, r=20, t=40, b=60),
    )
    return fig


def main() -> None:
    _configure_logging()
    st.set_page_config(page_title="Fleet Utilization Forecasting", layout="wide")
    st.title("Fleet Utilization Forecasting Dashboard")
    st.write(
        "Upload fleet utilization data or use the sample dataset to train forecasting models. "
        "The app validates the data, trains a Prophet or ARIMA model, and visualises forecasts with metrics."
    )

    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader("Upload CSV dataset", type="csv")
        selected_model = st.selectbox("Primary model", options=list(MODEL_REGISTRY.keys()), index=0)
        test_days = st.slider("Validation window (days)", min_value=7, max_value=90, value=30, step=7)
        forecast_horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=120, value=30, step=7)
        show_components = st.checkbox("Show Prophet components (if available)", value=False)

    file_bytes = uploaded_file.getvalue() if uploaded_file else None

    try:
        with st.spinner("Loading dataset..."):
            dataset = _load_data(file_bytes)
        st.success("Dataset loaded successfully")
    except Exception as exc:  # pragma: no cover - UI feedback path
        LOGGER.exception("Failed to load dataset")
        st.error(f"Unable to load dataset: {exc}")
        st.stop()

    st.subheader("Dataset preview")
    st.dataframe(dataset.tail(20), use_container_width=True)

    st.caption(
        f"Rows: {len(dataset):,} · Columns: {', '.join(dataset.columns)}"
    )

    try:
        with st.spinner("Training model and generating validation forecast..."):
            pipeline_result = _train_model_cached(selected_model, dataset, test_days)
        st.success(f"Model '{pipeline_result.model_name.title()}' trained successfully")
        if pipeline_result.model_name != selected_model:
            st.info(
                f"Primary model '{selected_model}' was unavailable; fallback model '{pipeline_result.model_name}' was used instead."
            )
    except Exception as exc:  # pragma: no cover - UI feedback path
        LOGGER.exception("Training pipeline failed")
        st.error(f"Model training failed: {exc}")
        st.stop()

    _render_metrics(pipeline_result)

    try:
        with st.spinner("Generating future forecast..."):
            future_forecast = forecast_future(pipeline_result.model, dataset, periods=forecast_horizon)
        st.success("Future forecast generated")
    except Exception as exc:  # pragma: no cover - UI feedback path
        LOGGER.exception("Future forecasting failed")
        st.error(f"Unable to generate future forecast: {exc}")
        st.stop()

    forecast_chart = _build_forecast_chart(dataset, pipeline_result.forecast, future_forecast)
    st.subheader("Forecast overview")
    st.plotly_chart(forecast_chart, use_container_width=True)

    forecast_output = future_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    st.subheader("Forecast table")
    st.dataframe(forecast_output, use_container_width=True)
    st.download_button(
        label="Download forecast CSV",
        data=forecast_output.to_csv(index=False).encode("utf-8"),
        file_name="fleet_utilization_forecast.csv",
        mime="text/csv",
    )

    if show_components and hasattr(pipeline_result.model, "component_plot"):
        component_fig = pipeline_result.model.component_plot()
        if component_fig is not None:
            st.subheader("Prophet components")
            st.plotly_chart(component_fig, use_container_width=True)
        else:
            st.info("Component plots are not available for the selected model.")


if __name__ == "__main__":
    main()
