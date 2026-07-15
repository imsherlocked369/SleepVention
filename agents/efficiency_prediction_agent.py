"""Agent 3: efficiency prediction and model-supported explanation."""

from __future__ import annotations

from math import isfinite


class EfficiencyModelNotConfiguredError(RuntimeError):
    """Raised when no trained efficiency-model adapter is available."""


def _efficiency_band(value: float) -> str:
    """Map estimates to prototype communication categories, not diagnoses."""
    if value >= 90:
        return "high estimated efficiency"
    if value >= 85:
        return "generally efficient"
    if value >= 80:
        return "mildly reduced efficiency"
    return "reduced efficiency"


def efficiency_prediction_agent(
    raw_record: dict[str, object], model: object | None
) -> dict[str, object]:
    """Request a prediction and SHAP evidence from the configured adapter."""
    if model is None:
        raise EfficiencyModelNotConfiguredError(
            "No trained efficiency model is configured."
        )
    predictor = getattr(model, "predict_efficiency", None)
    explainer = getattr(model, "explain_prediction", None)
    if not callable(predictor) or not callable(explainer):
        raise TypeError(
            "Efficiency model must implement predict_efficiency(raw_record) and "
            "explain_prediction(raw_record)."
        )
    predicted = float(predictor(raw_record))
    if not isfinite(predicted):
        raise ValueError("Efficiency model returned a non-finite prediction.")
    explanation = explainer(raw_record)
    if not isinstance(explanation, dict):
        raise ValueError("Model explanation must be a dictionary.")
    warnings = list(explanation.get("warnings", []))
    return {
        "predicted_efficiency_percent": predicted,
        "efficiency_band": _efficiency_band(predicted),
        "model_mae_percent": getattr(model, "model_mae", None),
        "base_value": explanation.get("base_value"),
        "factors_lowering_prediction": list(
            explanation.get("factors_lowering_prediction", [])
        ),
        "factors_increasing_prediction": list(
            explanation.get("factors_increasing_prediction", [])
        ),
        "warnings": warnings,
    }
