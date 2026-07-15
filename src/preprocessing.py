"""Shared, leakage-safe Oura preprocessing for training and inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

TARGET_NAME = "efficiency"
PARTICIPANT_COLUMN = "participant_id"
MODEL_FEATURES = [
    "onset_latency",
    "midpoint_time",
    "restless",
    "hr_average",
    "hr_lowest",
    "rmssd",
    "breath_average",
    "temperature_deviation",
    "temperature_trend_deviation",
    "rem_sleep_percent",
    "deep_sleep_percent",
    "bedtime_sin",
    "bedtime_cos",
    "day_sin",
    "day_cos",
]

LEAKAGE_COLUMNS = {
    "efficiency", "duration", "total", "awake", "light", "rem", "deep",
    "score_efficiency", "score", "score_alignment", "score_deep",
    "score_disturbances", "score_latency", "score_rem", "score_total",
    "score_personal_adjusted", "sleep_variance", "sleep_sd", "bad_recovery",
}
EXCLUDED_COLUMNS = {
    "Unnamed: 0", "participant_uid", "participant_email", "email", "quality",
    "sleeper_category", "temperature_delta", "bedtime_end_delta", "wakeup_hour",
}
MISSING_TEXT = {"", "na", "n/a", "null", "none"}
RAW_NUMERIC_COLUMNS = sorted(
    {
        *MODEL_FEATURES,
        "efficiency", "total", "rem", "deep", "bedtime_start_delta",
    }
    - {"rem_sleep_percent", "deep_sleep_percent", "bedtime_sin", "bedtime_cos", "day_sin", "day_cos"}
)


class PreprocessingError(ValueError):
    """Raised when model-ready features cannot be created safely."""


def load_oura_workbook(path: str | Path, sheet_name: str = "in") -> pd.DataFrame:
    """Load the configured Oura worksheet without applying feature engineering."""
    workbook = Path(path)
    if not workbook.exists():
        raise FileNotFoundError(
            "Training dataset not found. Place sleep_parsed_anonymised.xlsx inside the data directory."
        )
    LOGGER.info("Loading Oura workbook %s (sheet=%s)", workbook, sheet_name)
    try:
        return pd.read_excel(workbook, sheet_name=sheet_name)
    except ValueError as exc:
        raise PreprocessingError(f"Unable to read worksheet {sheet_name!r}: {exc}") from exc


def convert_excel_date(value: Any) -> pd.Timestamp:
    """Convert an Excel serial number or parseable date into a pandas Timestamp."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return pd.Timestamp("1899-12-30") + pd.to_timedelta(float(value), unit="D")
    return pd.to_datetime(value, errors="coerce")


def _replace_text_missing(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].map(
            lambda value: np.nan
            if isinstance(value, str) and value.strip().lower() in MISSING_TEXT
            else value
        )
    return result


def clean_numeric_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Coerce selected columns to finite numeric values, leaving invalid data missing."""
    result = _replace_text_missing(frame)
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


def engineer_dataframe_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the exact shared features used by training and runtime inference."""
    engineered = clean_numeric_columns(frame, RAW_NUMERIC_COLUMNS)
    if "date" not in engineered:
        engineered["date"] = pd.NaT
    engineered["date"] = engineered["date"].map(convert_excel_date)

    total = engineered.get("total", pd.Series(np.nan, index=engineered.index))
    valid_total = total.where(total > 0)
    for raw, output in (("rem", "rem_sleep_percent"), ("deep", "deep_sleep_percent")):
        values = engineered.get(raw, pd.Series(np.nan, index=engineered.index))
        engineered[output] = values.div(valid_total).mul(100)

    bedtime = engineered.get(
        "bedtime_start_delta", pd.Series(np.nan, index=engineered.index)
    )
    bedtime_hour = bedtime.div(3600).mod(24)
    engineered["bedtime_sin"] = np.sin(2 * np.pi * bedtime_hour / 24)
    engineered["bedtime_cos"] = np.cos(2 * np.pi * bedtime_hour / 24)

    day = engineered["date"].dt.dayofweek
    engineered["day_sin"] = np.sin(2 * np.pi * day / 7)
    engineered["day_cos"] = np.cos(2 * np.pi * day / 7)
    for feature in MODEL_FEATURES:
        if feature not in engineered.columns:
            engineered[feature] = np.nan
    return engineered.replace([np.inf, -np.inf], np.nan)


def engineer_single_record_features(raw_record: dict[str, object]) -> dict[str, float]:
    """Engineer one raw Oura record and return features in canonical order."""
    if not isinstance(raw_record, dict):
        raise PreprocessingError("Raw sleep record must be a dictionary.")
    frame = engineer_dataframe_features(pd.DataFrame([raw_record]))
    missing_columns = [name for name in MODEL_FEATURES if name not in frame.columns]
    if missing_columns:
        raise PreprocessingError(
            "Required model features could not be generated: " + ", ".join(missing_columns)
        )
    return {
        name: float(frame.iloc[0][name]) if pd.notna(frame.iloc[0][name]) else np.nan
        for name in MODEL_FEATURES
    }


def prepare_training_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Return ordered features, target, participant groups, and row metadata."""
    data = frame.copy()
    if "email" in data.columns:
        data = data.rename(columns={"email": PARTICIPANT_COLUMN})
    required = {TARGET_NAME, PARTICIPANT_COLUMN, "date"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise PreprocessingError("Training data is missing: " + ", ".join(missing))

    engineered = engineer_dataframe_features(data)
    target = pd.to_numeric(engineered[TARGET_NAME], errors="coerce")
    groups = engineered[PARTICIPANT_COLUMN].astype("string")
    valid = target.notna() & groups.notna() & engineered["date"].notna()
    if not valid.any():
        raise PreprocessingError("No valid training rows remain after preprocessing.")

    features = engineered.loc[valid, MODEL_FEATURES].astype(float)
    metadata = engineered.loc[valid, [PARTICIPANT_COLUMN, "date"]].copy()
    return features, target.loc[valid].astype(float), groups.loc[valid], metadata


def get_feature_names() -> list[str]:
    """Return a copy of the canonical ordered model feature list."""
    return MODEL_FEATURES.copy()
