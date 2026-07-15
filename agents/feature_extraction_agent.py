"""Agent 2: shared leakage-safe feature extraction."""

from __future__ import annotations

from math import isnan

from src.preprocessing import engineer_single_record_features


def feature_extraction_agent(data: dict[str, object]) -> dict[str, object]:
    """Create ordered model features and safe display measurements."""
    model_features = engineer_single_record_features(data)
    warnings = [
        f"{name} is missing and will be imputed by the fitted model pipeline"
        for name, value in model_features.items()
        if isnan(value)
    ]
    display = {
        "date": str(data["date"]),
        "rem_sleep_percent": model_features["rem_sleep_percent"],
        "deep_sleep_percent": model_features["deep_sleep_percent"],
        "average_sleeping_heart_rate": model_features["hr_average"],
        "lowest_sleeping_heart_rate": model_features["hr_lowest"],
        "heart_rate_variability_rmssd": model_features["rmssd"],
        "average_breathing_rate": model_features["breath_average"],
    }
    return {
        "model_features": model_features,
        "display_measurements": display,
        "warnings": warnings,
    }
