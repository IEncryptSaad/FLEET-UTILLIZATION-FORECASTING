"""Streamlit dashboard for the fleet utilization forecasting MVP."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from fleet_forecasting.data import load_dataset
from fleet_forecasting.pipeline import MODEL_REGISTRY, forecast_future


def main() -> None:
    st.set_page_config(page_title="Fleet Utilization Forecasting", layout="wide")
    st.title("Fleet Utilization Forecasting")
    st.write(
        "This demo trains a time-series model on historical fleet utilization data and"
        " projects utilization rates into the future."
    )

    dataset = load_dataset()
    st.subheader("Dataset preview")
    st.dataframe(dataset.tail(10))

    st.subheader("Model configuration")
    model_name = st.selectbox("Model", options=list(MODEL_REGISTRY.keys()), index=0)
    forecast_days = st.slider("Days to forecast", min_value=7, max_value=90, value=30, step=7)

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls()
    model.fit(dataset)
    future_forecast = forecast_future(model, dataset, periods=forecast_days)

    forecast_df = future_forecast.reset_index()
    forecast_df.columns = ["date", "forecast_utilization"]
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    history_df = dataset.reset_index()[["date", "utilization_rate"]]
    history_df["date"] = pd.to_datetime(history_df["date"])

    chart = (
        alt.Chart(history_df)
        .mark_line(color="#1f77b4")
        .encode(x="date:T", y=alt.Y("utilization_rate:Q", title="Utilization rate"), tooltip=["date", "utilization_rate"])
    )

    forecast_chart = (
        alt.Chart(forecast_df)
        .mark_line(color="#ff7f0e")
        .encode(x="date:T", y="forecast_utilization:Q", tooltip=["date", "forecast_utilization"])
    )

    st.subheader("Historical utilization vs forecast")
    st.altair_chart(chart + forecast_chart, use_container_width=True)

    st.subheader("Download forecast")
    st.download_button(
        label="Download CSV",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name="fleet_utilization_forecast.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
