from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return common regression metrics for glucose forecasts."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_model(
    model,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    horizon: str | None = None,
) -> dict[str, float | str]:
    predictions = model.predict(test_df[feature_columns])
    metrics = regression_metrics(test_df[target_column], predictions)
    if horizon is not None:
        metrics = {"horizon": horizon, **metrics}
    return metrics


def compare_model_results(results: list[dict[str, object]]) -> pd.DataFrame:
    """Convert a list of metric dictionaries into a sorted comparison table."""
    comparison = pd.DataFrame(results)
    sort_columns = [column for column in ["horizon", "model"] if column in comparison.columns]
    if sort_columns:
        comparison = comparison.sort_values(sort_columns)
    return comparison.reset_index(drop=True)

