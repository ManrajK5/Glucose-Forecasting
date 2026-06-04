from __future__ import annotations

from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


OHIO_COLUMNS = [
    "patient_id",
    "timestamp",
    "glucose_mgdl",
    "glucose_mmol",
    "weight",
    "insulin_type",
]


def parse_ohio_xml_file(file_path: str | Path) -> pd.DataFrame:
    """Parse one OhioT1DM-style XML file into a glucose dataframe."""
    path = Path(file_path)
    tree = ET.parse(path)
    root = tree.getroot()

    patient_id = root.attrib.get("id", path.stem)
    weight = pd.to_numeric(root.attrib.get("weight"), errors="coerce")
    insulin_type = root.attrib.get("insulin_type")

    rows: list[dict[str, object]] = []
    glucose_node = root.find("glucose_level")
    if glucose_node is not None:
        for event in glucose_node.findall("event"):
            timestamp = pd.to_datetime(
                event.attrib.get("ts"),
                dayfirst=True,
                errors="coerce",
            )
            glucose_mgdl = pd.to_numeric(event.attrib.get("value"), errors="coerce")
            rows.append(
                {
                    "patient_id": str(patient_id),
                    "timestamp": timestamp,
                    "glucose_mgdl": glucose_mgdl,
                    "glucose_mmol": glucose_mgdl / 18 if pd.notna(glucose_mgdl) else np.nan,
                    "weight": weight,
                    "insulin_type": insulin_type,
                }
            )

    return pd.DataFrame(rows, columns=OHIO_COLUMNS)


def parse_ohio_xml_folder(folder_path: str | Path) -> pd.DataFrame:
    """Recursively parse all XML files in an OhioT1DM data folder."""
    folder = Path(folder_path)
    frames = [parse_ohio_xml_file(path) for path in sorted(folder.rglob("*.xml"))]
    if not frames:
        return pd.DataFrame(columns=OHIO_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["timestamp", "glucose_mgdl"])
    return combined.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)


def _reset_file(file_path: str | Path) -> None:
    if hasattr(file_path, "seek"):
        file_path.seek(0)


def _normalize_column_name(column: object) -> str:
    column_text = str(column).replace("\ufeff", "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "", column_text)


def _read_libreview_csv(file_path: str | Path) -> pd.DataFrame:
    """Read LibreView exports that may include metadata rows before the header."""
    _reset_file(file_path)
    preview = pd.read_csv(
        file_path,
        header=None,
        dtype=str,
        nrows=80,
        keep_default_na=False,
        engine="python",
        on_bad_lines="skip",
    )

    header_row = 0
    for index, row in preview.iterrows():
        normalized = {_normalize_column_name(value) for value in row.tolist()}
        has_timestamp = bool({"timestamp", "devicetimestamp"} & normalized)
        has_glucose = any("glucose" in value for value in normalized)
        if has_timestamp and has_glucose:
            header_row = int(index)
            break

    _reset_file(file_path)
    df = pd.read_csv(
        file_path,
        header=header_row,
        dtype=str,
        keep_default_na=False,
        engine="python",
        on_bad_lines="skip",
    )
    df.columns = [str(column).replace("\ufeff", "").strip() for column in df.columns]
    df = df.map(lambda value: np.nan if isinstance(value, str) and not value.strip() else value)
    return df.dropna(how="all")


def _find_column(columns: list[str], *patterns: str) -> str | None:
    normalized = {column: _normalize_column_name(column) for column in columns}
    for column, value in normalized.items():
        if all(pattern in value for pattern in patterns):
            return column
    return None


def _find_timestamp_column(columns: list[str]) -> str | None:
    normalized = {column: _normalize_column_name(column) for column in columns}
    for column, value in normalized.items():
        if value in {"timestamp", "devicetimestamp"}:
            return column
    for column, value in normalized.items():
        if "timestamp" in value or value in {"time", "date"}:
            return column
    return None


def _parse_glucose_column(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    glucose = pd.to_numeric(df[column], errors="coerce")
    normalized = _normalize_column_name(column)
    if "mgdl" in normalized:
        glucose = glucose / 18
    return glucose


def _parse_timestamps(values: pd.Series) -> pd.Series:
    month_first = pd.to_datetime(values, errors="coerce", dayfirst=False, format="mixed")
    day_first = pd.to_datetime(values, errors="coerce", dayfirst=True, format="mixed")

    month_parse_rate = month_first.notna().mean()
    day_parse_rate = day_first.notna().mean()
    if day_parse_rate > month_parse_rate + 0.05:
        return day_first
    if month_parse_rate > day_parse_rate + 0.05:
        return month_first

    tomorrow = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    month_future_count = (month_first > tomorrow).sum()
    day_future_count = (day_first > tomorrow).sum()
    if day_future_count < month_future_count:
        return day_first
    return month_first


def load_libreview_csv(file_path: str | Path) -> pd.DataFrame:
    """Load LibreView CSV data and normalize it to the project schema."""
    df = _read_libreview_csv(file_path)

    timestamp_column = _find_timestamp_column(list(df.columns))
    historic_column = _find_column(list(df.columns), "historic", "glucose")
    scan_column = _find_column(list(df.columns), "scan", "glucose")

    generic_glucose_column = None
    if historic_column is None and scan_column is None:
        glucose_columns = [
            column
            for column in df.columns
            if "glucose" in _normalize_column_name(column)
            and "insulin" not in _normalize_column_name(column)
            and "ketone" not in _normalize_column_name(column)
        ]
        generic_glucose_column = glucose_columns[0] if glucose_columns else None

    missing = []
    if timestamp_column is None:
        missing.append("timestamp column, for example Timestamp or Device Timestamp")
    if historic_column is None and scan_column is None and generic_glucose_column is None:
        missing.append("glucose column, for example Historic Glucose mmol/L")
    if missing:
        available = ", ".join(str(column) for column in df.columns[:20])
        raise ValueError(
            "Missing LibreView data: "
            + "; ".join(missing)
            + f". Available columns include: {available}"
        )

    historic = _parse_glucose_column(df, historic_column)
    scan = _parse_glucose_column(df, scan_column)
    generic = _parse_glucose_column(df, generic_glucose_column)
    glucose_mmol = historic.combine_first(scan)
    glucose_mmol = glucose_mmol.combine_first(generic)

    normalized = pd.DataFrame(
        {
            "patient_id": "libreview_user",
            "timestamp": _parse_timestamps(df[timestamp_column]),
            "glucose_mmol": glucose_mmol,
        }
    )
    normalized["glucose_mgdl"] = normalized["glucose_mmol"] * 18
    normalized["weight"] = np.nan
    normalized["insulin_type"] = np.nan

    return normalized[OHIO_COLUMNS].dropna(subset=["timestamp", "glucose_mmol"])


def clean_glucose_dataframe(
    df: pd.DataFrame,
    freq: str = "5min",
    interpolation_limit: int = 3,
) -> pd.DataFrame:
    """Clean glucose rows, resample to five minutes, and drop long gaps."""
    if df.empty:
        return pd.DataFrame(columns=OHIO_COLUMNS)

    cleaned = df.copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
    cleaned["glucose_mgdl"] = pd.to_numeric(cleaned["glucose_mgdl"], errors="coerce")
    cleaned = cleaned.dropna(subset=["patient_id", "timestamp", "glucose_mgdl"])
    cleaned = cleaned.drop_duplicates(subset=["patient_id", "timestamp"], keep="last")
    cleaned = cleaned[
        cleaned["glucose_mgdl"].between(40, 400, inclusive="both")
    ].sort_values(["patient_id", "timestamp"])

    patient_frames: list[pd.DataFrame] = []
    for patient_id, group in cleaned.groupby("patient_id", sort=False):
        group = group.sort_values("timestamp").set_index("timestamp")
        numeric = group[["glucose_mgdl", "glucose_mmol", "weight"]].resample(freq).mean()
        numeric[["glucose_mgdl", "glucose_mmol"]] = numeric[
            ["glucose_mgdl", "glucose_mmol"]
        ].interpolate(method="time", limit=interpolation_limit, limit_area="inside")

        metadata = group[["insulin_type"]].resample(freq).first().ffill().bfill()
        resampled = pd.concat([numeric, metadata], axis=1)
        resampled["patient_id"] = str(patient_id)
        resampled = resampled.dropna(subset=["glucose_mgdl", "glucose_mmol"])
        patient_frames.append(resampled.reset_index())

    if not patient_frames:
        return pd.DataFrame(columns=OHIO_COLUMNS)

    result = pd.concat(patient_frames, ignore_index=True)
    return result[OHIO_COLUMNS].sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
