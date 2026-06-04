from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def time_series_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each patient chronologically into train and test sets."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    ordered = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    for _, group in ordered.groupby("patient_id", sort=False):
        split_index = max(1, int(len(group) * (1 - test_size)))
        if split_index >= len(group):
            split_index = len(group) - 1
        if split_index <= 0:
            continue
        train_parts.append(group.iloc[:split_index])
        test_parts.append(group.iloc[split_index:])

    train = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    test = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
    return train, test


def save_model(model: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> Any:
    return joblib.load(path)


def save_feature_columns(feature_columns: list[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")


def load_feature_columns(path: str | Path) -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

