"""End-to-end Oura efficiency prediction and reporting pipeline."""

from __future__ import annotations

from agents.data_quality_agent import data_quality_agent
from agents.efficiency_prediction_agent import efficiency_prediction_agent
from agents.feature_extraction_agent import feature_extraction_agent
from agents.report_generation_agent import report_generation_agent


class DataQualityError(ValueError):
    """Raised when poor input must stop before model inference."""

    def __init__(self, result: dict[str, object]):
        self.result = result
        super().__init__("; ".join(result["errors"]))


def run_sleep_pipeline(
    sleep_input: dict[str, object],
    efficiency_model: object,
    report_client: object | None = None,
    generate_report: bool = True,
    fallback_only: bool = False,
) -> dict[str, object]:
    """Run quality, shared features, XGBoost/SHAP inference, and reporting."""
    quality = data_quality_agent(sleep_input)
    if quality["quality"] == "Poor":
        raise DataQualityError(quality)
    extracted = feature_extraction_agent(sleep_input)
    prediction = efficiency_prediction_agent(sleep_input, efficiency_model)
    all_warnings = [
        *quality["warnings"], *extracted["warnings"], *prediction["warnings"]
    ]
    report_quality = {**quality, "warnings": all_warnings}
    report = None
    if generate_report:
        report = report_generation_agent(
            data_quality=report_quality,
            display_measurements=extracted["display_measurements"],
            efficiency_result=prediction,
            client=False if fallback_only else report_client,
        )
    return {
        "data_quality": report_quality,
        "display_measurements": extracted["display_measurements"],
        "model_features": extracted["model_features"],
        "predicted_efficiency_percent": prediction["predicted_efficiency_percent"],
        "efficiency_band": prediction["efficiency_band"],
        "model_mae_percent": prediction["model_mae_percent"],
        "base_value": prediction["base_value"],
        "factors_lowering_prediction": prediction["factors_lowering_prediction"],
        "factors_increasing_prediction": prediction["factors_increasing_prediction"],
        "report": report,
    }
