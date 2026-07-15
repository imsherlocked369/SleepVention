"""SHAP explanations for individual XGBoost efficiency predictions."""

from __future__ import annotations

from typing import Any

import numpy as np


FEATURE_LABELS = {
    "onset_latency": "sleep-onset latency",
    "midpoint_time": "sleep midpoint",
    "restless": "restlessness",
    "hr_average": "average sleeping heart rate",
    "hr_lowest": "lowest sleeping heart rate",
    "rmssd": "heart-rate variability",
    "breath_average": "average breathing rate",
    "temperature_deviation": "temperature deviation",
    "temperature_trend_deviation": "temperature trend deviation",
    "rem_sleep_percent": "REM-sleep percentage",
    "deep_sleep_percent": "deep-sleep percentage",
    "bedtime_sin": "bedtime timing",
    "bedtime_cos": "bedtime timing",
    "day_sin": "day of week",
    "day_cos": "day of week",
}


class ModelExplanationError(RuntimeError):
    """Raised when a SHAP explanation cannot be generated."""


def explain_xgboost_prediction(
    model: object,
    transformed_features: np.ndarray,
    original_features: dict[str, float],
    feature_names: list[str],
    predicted_efficiency: float,
    max_factors: int = 5,
) -> dict[str, object]:
    """Return the strongest positive and negative SHAP model contributions."""
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        explanation = explainer(transformed_features)
        values = np.asarray(explanation.values)[0]
        base = np.asarray(explanation.base_values).reshape(-1)[0]
    except Exception as exc:
        raise ModelExplanationError(f"Unable to generate SHAP explanation: {exc}") from exc

    factors: list[dict[str, Any]] = []
    for name, shap_value in zip(feature_names, values):
        factors.append(
            {
                "feature": name,
                "display_name": FEATURE_LABELS.get(name, name.replace("_", " ")),
                "feature_value": float(original_features[name]),
                "shap_value": float(shap_value),
            }
        )
    strongest = sorted(
        factors, key=lambda item: abs(item["shap_value"]), reverse=True
    )[:max_factors]
    return {
        "predicted_efficiency": float(predicted_efficiency),
        "base_value": float(base),
        "factors_lowering_prediction": [
            item for item in strongest if item["shap_value"] < 0
        ],
        "factors_increasing_prediction": [
            item for item in strongest if item["shap_value"] > 0
        ],
    }
