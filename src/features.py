from __future__ import annotations

import pandas as pd


LAG_STEPS = {
    "lag_5min": 1,
    "lag_10min": 2,
    "lag_15min": 3,
    "lag_30min": 6,
    "lag_60min": 12,
    "lag_120min": 24,
}

TARGET_STEPS = {
    "target_30min": 6,
    "target_60min": 12,
    "target_120min": 24,
}


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    return result.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    result = _sorted(df)
    grouped = result.groupby("patient_id")["glucose_mgdl"]
    for column, periods in LAG_STEPS.items():
        result[column] = grouped.shift(periods)
    return result


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    result = _sorted(df)
    grouped = result.groupby("patient_id")["glucose_mgdl"]
    result["rolling_mean_30min"] = grouped.transform(lambda s: s.rolling(6).mean())
    result["rolling_mean_60min"] = grouped.transform(lambda s: s.rolling(12).mean())
    result["rolling_std_30min"] = grouped.transform(lambda s: s.rolling(6).std())
    result["rolling_min_60min"] = grouped.transform(lambda s: s.rolling(12).min())
    result["rolling_max_60min"] = grouped.transform(lambda s: s.rolling(12).max())
    return result


def create_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    result = _sorted(df)
    grouped = result.groupby("patient_id")["glucose_mgdl"]
    result["glucose_rate_of_change"] = grouped.diff() / 5
    result["glucose_slope_15min"] = (result["glucose_mgdl"] - grouped.shift(3)) / 15
    result["glucose_slope_30min"] = (result["glucose_mgdl"] - grouped.shift(6)) / 30
    return result


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result = _sorted(df)
    result["hour"] = result["timestamp"].dt.hour
    result["day_of_week"] = result["timestamp"].dt.dayofweek
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
    return result


def create_future_targets(df: pd.DataFrame) -> pd.DataFrame:
    result = _sorted(df)
    grouped = result.groupby("patient_id")["glucose_mgdl"]
    for column, periods in TARGET_STEPS.items():
        result[column] = grouped.shift(-periods)
    return result


def create_model_features(df: pd.DataFrame, include_targets: bool = True) -> pd.DataFrame:
    result = create_lag_features(df)
    result = create_rolling_features(result)
    result = create_trend_features(result)
    result = create_time_features(result)
    if include_targets:
        result = create_future_targets(result)

    feature_columns = get_feature_columns(result)
    required_columns = feature_columns + (list(TARGET_STEPS) if include_targets else [])
    return result.dropna(subset=required_columns).reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame | None = None) -> list[str]:
    base_columns = [
        "glucose_mgdl",
        "glucose_mmol",
        *LAG_STEPS.keys(),
        "rolling_mean_30min",
        "rolling_mean_60min",
        "rolling_std_30min",
        "rolling_min_60min",
        "rolling_max_60min",
        "glucose_rate_of_change",
        "glucose_slope_15min",
        "glucose_slope_30min",
        "hour",
        "day_of_week",
        "is_weekend",
    ]
    if df is None:
        return base_columns
    return [column for column in base_columns if column in df.columns]

