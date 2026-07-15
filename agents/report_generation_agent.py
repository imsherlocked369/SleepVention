"""Safe LLM reporting with a deterministic offline fallback."""

from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """You write a clear, non-diagnostic wearable sleep report.
Use only the verified information supplied. State the predicted sleep efficiency
and explain the strongest SHAP-supported model contributions. Describe them as
associations with the model estimate, never as causes or clinical findings.
Mention prediction uncertainty using model MAE when supplied. Give only simple,
low-risk sleep-hygiene suggestions. Do not diagnose insomnia, sleep apnoea, or
any other condition. Do not invent caffeine use, stress, illness, alcohol,
medication, symptoms, or lifestyle details. State that wearable measurements
are estimates and recommend professional advice only for persistent concerning
symptoms. Use clear non-technical language and exactly these headings:
Estimated sleep efficiency; What influenced the estimate; Practical suggestions;
Limitations. Return only the report.
"""


def _factor_lines(factors: list[dict[str, object]], direction: str) -> list[str]:
    lines = []
    for factor in factors[:5]:
        label = factor.get("display_name", factor.get("feature", "A model feature"))
        contribution = abs(float(factor.get("shap_value", 0.0)))
        lines.append(
            f"- {label} {direction}; its SHAP contribution was approximately "
            f"{contribution:.2f} percentage points."
        )
    return lines


def deterministic_report(
    efficiency_result: dict[str, object],
    data_quality: dict[str, object],
) -> str:
    """Create a safe report without an API key or external request."""
    predicted = float(efficiency_result["predicted_efficiency_percent"])
    band = efficiency_result["efficiency_band"]
    mae = efficiency_result.get("model_mae_percent")
    uncertainty = (
        f"Typical validation MAE was {float(mae):.1f} percentage points, so the "
        "estimate should not be treated as exact."
        if mae is not None
        else "Validated model error is not available, so uncertainty cannot yet be quantified."
    )
    influence = _factor_lines(
        list(efficiency_result.get("factors_lowering_prediction", [])),
        "was associated with a lower model estimate",
    ) + _factor_lines(
        list(efficiency_result.get("factors_increasing_prediction", [])),
        "contributed to a higher predicted efficiency",
    )
    if not influence:
        influence = ["- No individual model contributions were available."]
    warnings = list(data_quality.get("warnings", []))
    warning_text = (
        " Data-quality notes: " + "; ".join(str(item) for item in warnings)
        if warnings else ""
    )
    return "\n".join(
        [
            "Estimated sleep efficiency",
            f"The model estimated sleep efficiency at {predicted:.1f}% ({band}).",
            "",
            "What influenced the estimate",
            *influence,
            "These values influenced the model prediction; they do not prove causation.",
            "",
            "Practical suggestions",
            "Keep sleep and wake timing regular and use a calm, consistent wind-down routine.",
            "",
            "Limitations",
            uncertainty,
            "Wearable measurements are estimates, and this model is non-diagnostic."
            + warning_text,
            "Seek professional advice for persistent or concerning symptoms.",
        ]
    )


def report_generation_agent(
    data_quality: dict[str, object],
    display_measurements: dict[str, object],
    efficiency_result: dict[str, object],
    client: object | None = None,
) -> str:
    """Generate an LLM report, falling back deterministically on any API issue."""
    fallback = deterministic_report(efficiency_result, data_quality)
    if client is False:
        return fallback
    load_dotenv()
    if client is None:
        if not os.getenv("OPENAI_API_KEY"):
            return fallback
        client = OpenAI()

    payload = {
        "predicted_efficiency_percent": efficiency_result["predicted_efficiency_percent"],
        "efficiency_band": efficiency_result["efficiency_band"],
        "model_mae_percent": efficiency_result.get("model_mae_percent"),
        "verified_measurements": display_measurements,
        "factors_lowering_prediction": efficiency_result.get(
            "factors_lowering_prediction", []
        ),
        "factors_increasing_prediction": efficiency_result.get(
            "factors_increasing_prediction", []
        ),
        "data_quality_warnings": data_quality.get("warnings", []),
    }
    try:
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL") or "gpt-5-mini",
            instructions=SYSTEM_PROMPT,
            input="Write the report from this verified JSON:\n" + json.dumps(payload, default=str),
        )
        text = response.output_text.strip()
        return text or fallback
    except Exception as exc:
        LOGGER.warning("OpenAI report generation failed; using fallback: %s", exc)
        return fallback
