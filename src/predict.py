from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features import create_model_features
from src.train_utils import load_model


MODEL_FILES = {
    "30min": "xgboost_30min.pkl",
    "60min": "xgboost_60min.pkl",
    "120min": "xgboost_120min.pkl",
}


def load_saved_models(models_dir: str | Path) -> dict[str, object]:
    models_path = Path(models_dir)
    models: dict[str, object] = {}
    missing: list[str] = []
    for horizon, filename in MODEL_FILES.items():
        path = models_path / filename
        if path.exists():
            models[horizon] = load_model(path)
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing saved model files: {missing}")
    return models


def prepare_latest_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    featured = create_model_features(df, include_targets=False)
    if featured.empty:
        raise ValueError("Not enough glucose history to create prediction features.")

    missing = [column for column in feature_columns if column not in featured.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    return featured.sort_values(["patient_id", "timestamp"]).tail(1)[feature_columns]


def predict_future_glucose(
    df: pd.DataFrame,
    models: dict[str, object],
    feature_columns: list[str],
) -> pd.DataFrame:
    latest_features = prepare_latest_features(df, feature_columns)
    rows: list[dict[str, float | str]] = []
    for horizon, model in models.items():
        glucose_mgdl = float(model.predict(latest_features)[0])
        glucose_mmol = glucose_mgdl / 18
        rows.append(
            {
                "horizon": horizon,
                "glucose_mgdl": glucose_mgdl,
                "glucose_mmol": glucose_mmol,
                "risk_zone": classify_glucose_risk(glucose_mmol),
            }
        )
    return pd.DataFrame(rows)


def classify_glucose_risk(glucose_mmol: float) -> str:
    if glucose_mmol < 3.9:
        return "low"
    if glucose_mmol > 10:
        return "high"
    return "normal"


def classify_trend_direction(df: pd.DataFrame) -> str:
    ordered = df.sort_values(["patient_id", "timestamp"])
    latest_patient = ordered["patient_id"].iloc[-1]
    patient_rows = ordered[ordered["patient_id"] == latest_patient].tail(4)
    if len(patient_rows) < 2:
        return "stable"

    change_mmol = patient_rows["glucose_mmol"].iloc[-1] - patient_rows["glucose_mmol"].iloc[0]
    if change_mmol > 0.5:
        return "rising"
    if change_mmol < -0.5:
        return "falling"
    return "stable"

