from __future__ import annotations

from datetime import datetime
from io import BytesIO
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import (  # noqa: E402
    classify_glucose_risk,
    classify_trend_direction,
    load_saved_models,
    predict_future_glucose,
)
from src.preprocessing import clean_glucose_dataframe, load_libreview_csv  # noqa: E402
from src.train_utils import load_feature_columns  # noqa: E402


MODELS_DIR = ROOT / "models"
DISCLAIMER = (
    "This is an educational machine learning project and is not medical advice "
    "or insulin dosing guidance."
)


@st.cache_resource
def load_artifacts() -> tuple[dict[str, object], list[str]]:
    models = load_saved_models(MODELS_DIR)
    feature_columns = load_feature_columns(MODELS_DIR / "feature_columns.json")
    return models, feature_columns


@st.cache_data
def load_clean_libreview_upload(file_bytes: bytes) -> pd.DataFrame:
    raw_df = load_libreview_csv(BytesIO(file_bytes))
    return clean_glucose_dataframe(raw_df)


def append_glucose_reading(
    clean_df: pd.DataFrame,
    timestamp: datetime,
    glucose_mmol: float,
) -> pd.DataFrame:
    reading = pd.DataFrame(
        {
            "patient_id": ["libreview_user"],
            "timestamp": [pd.Timestamp(timestamp).floor("5min")],
            "glucose_mgdl": [glucose_mmol * 18],
            "glucose_mmol": [glucose_mmol],
            "weight": [np.nan],
            "insulin_type": [np.nan],
        }
    )
    combined = pd.concat([clean_df, reading], ignore_index=True)
    combined = combined.sort_values(["patient_id", "timestamp"])
    combined = combined.drop_duplicates(subset=["patient_id", "timestamp"], keep="last")
    return combined.reset_index(drop=True)


def prediction_table(predictions: pd.DataFrame) -> pd.DataFrame:
    display_predictions = predictions.copy()
    display_predictions["glucose_mmol"] = display_predictions["glucose_mmol"].map(lambda x: f"{x:.1f}")
    display_predictions["glucose_mgdl"] = display_predictions["glucose_mgdl"].map(lambda x: f"{x:.0f}")
    return display_predictions.rename(
        columns={
            "horizon": "Horizon",
            "glucose_mmol": "Prediction mmol/L",
            "glucose_mgdl": "Prediction mg/dL",
            "risk_zone": "Risk zone",
        }
    )


def render_results(clean_df: pd.DataFrame, predictions: pd.DataFrame) -> None:
    latest = clean_df.sort_values("timestamp").iloc[-1]
    latest_mmol = float(latest["glucose_mmol"])
    latest_mgdl = float(latest["glucose_mgdl"])
    trend = classify_trend_direction(clean_df)
    risk = classify_glucose_risk(latest_mmol)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Latest glucose", f"{latest_mmol:.1f} mmol/L", f"{latest_mgdl:.0f} mg/dL")
    metric_cols[1].metric("Current risk zone", risk)
    metric_cols[2].metric("Trend", trend)
    metric_cols[3].metric("Readings used", f"{len(clean_df):,}")

    st.dataframe(prediction_table(predictions), use_container_width=True, hide_index=True)

    timeline_df = clean_df.sort_values("timestamp").tail(1200)
    fig = px.line(
        timeline_df,
        x="timestamp",
        y="glucose_mmol",
        color="patient_id",
        labels={"timestamp": "Time", "glucose_mmol": "Glucose mmol/L", "patient_id": "Patient"},
    )
    fig.add_hrect(y0=0, y1=3.9, fillcolor="red", opacity=0.08, line_width=0)
    fig.add_hrect(y0=3.9, y1=10, fillcolor="green", opacity=0.08, line_width=0)
    fig.add_hrect(y0=10, y1=25, fillcolor="orange", opacity=0.08, line_width=0)
    st.plotly_chart(fig, use_container_width=True)


def create_predictions(clean_df: pd.DataFrame) -> pd.DataFrame:
    models, feature_columns = load_artifacts()
    return predict_future_glucose(clean_df, models, feature_columns)


st.set_page_config(page_title="Glucose Forecasting", page_icon="G", layout="wide")
st.title("Glucose Forecasting")
st.caption(DISCLAIMER)

uploaded_file = st.file_uploader("LibreView CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a LibreView CSV after training notebooks have saved model files.")
    st.stop()

try:
    base_df = load_clean_libreview_upload(uploaded_file.getvalue())
    if base_df.empty:
        raise ValueError("No valid glucose readings were found in the uploaded CSV.")

    latest = base_df.sort_values("timestamp").iloc[-1]
    default_timestamp = pd.Timestamp(latest["timestamp"]) + pd.Timedelta(minutes=5)
    default_glucose = float(latest["glucose_mmol"])

    input_cols = st.columns([1, 1, 1])
    glucose_mmol = input_cols[0].number_input(
        "Current glucose (mmol/L)",
        min_value=2.0,
        max_value=25.0,
        value=round(default_glucose, 1),
        step=0.1,
    )
    reading_date = input_cols[1].date_input("Reading date", value=default_timestamp.date())
    reading_time = input_cols[2].time_input(
        "Reading time",
        value=default_timestamp.time().replace(second=0, microsecond=0),
    )

    reading_timestamp = datetime.combine(reading_date, reading_time)
    clean_df = append_glucose_reading(base_df, reading_timestamp, glucose_mmol)
    latest_gap = pd.Timestamp(reading_timestamp) - pd.Timestamp(latest["timestamp"])
    if latest_gap > pd.Timedelta(minutes=30):
        st.warning(
            "The entered reading is more than 30 minutes after the uploaded CSV history. "
            "Predictions work best with recent CGM history."
        )

    predictions = create_predictions(clean_df)
    render_results(clean_df, predictions)
except FileNotFoundError as exc:
    st.error(f"Saved model artifact not found: {exc}")
except Exception as exc:
    st.error(f"Could not create predictions: {exc}")
