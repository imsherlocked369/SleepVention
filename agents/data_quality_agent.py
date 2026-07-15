"""Technical quality checks for one Oura-compatible sleep record."""

from __future__ import annotations

from math import isfinite

import pandas as pd


REQUIRED_FIELDS = [
    "date", "onset_latency", "midpoint_time", "restless", "hr_average",
    "hr_lowest", "rmssd", "breath_average", "temperature_deviation",
    "rem", "deep", "total", "bedtime_start_delta",
]
OPTIONAL_FIELDS = ["temperature_trend_deviation"]
NON_NEGATIVE_FIELDS = [
    "onset_latency", "midpoint_time", "restless", "hr_average", "hr_lowest",
    "rmssd", "breath_average", "rem", "deep",
]


def data_quality_agent(data: dict[str, object]) -> dict[str, object]:
    """Validate model input without making medical or diagnostic judgments."""
    errors: list[str] = []
    warnings: list[str] = []
    missing_optional = [field for field in OPTIONAL_FIELDS if data.get(field) is None]

    missing = [field for field in REQUIRED_FIELDS if data.get(field) is None]
    errors.extend(f"{field} is missing" for field in missing)

    if "date" not in missing and pd.isna(pd.to_datetime(data.get("date"), errors="coerce")):
        errors.append("date is invalid or cannot be parsed")

    numeric: dict[str, float] = {}
    for field in [*REQUIRED_FIELDS[1:], *OPTIONAL_FIELDS]:
        value = data.get(field)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors.append(f"{field} must be numeric")
            continue
        if not isfinite(number):
            errors.append(f"{field} must be finite")
            continue
        numeric[field] = number

    for field in NON_NEGATIVE_FIELDS:
        if field in numeric and numeric[field] < 0:
            errors.append(f"{field} cannot be negative")
    if "total" in numeric and numeric["total"] <= 0:
        errors.append("total must be greater than zero")
    if all(field in numeric for field in ("rem", "deep", "total")):
        if numeric["rem"] + numeric["deep"] > numeric["total"]:
            errors.append("rem + deep cannot be greater than total")

    if missing_optional:
        warnings.append(
            "Optional temperature trend data is missing; the model imputer will supply it."
        )
    if "hr_lowest" in numeric and "hr_average" in numeric:
        if numeric["hr_lowest"] > numeric["hr_average"]:
            warnings.append("hr_lowest is greater than hr_average; verify the source record")
    if "breath_average" in numeric and not 3 <= numeric["breath_average"] <= 60:
        warnings.append("breath_average is outside the configured technical plausibility range")
    if "hr_average" in numeric and not 20 <= numeric["hr_average"] <= 220:
        warnings.append("hr_average is outside the configured technical plausibility range")

    quality = "Poor" if errors else ("Limited" if warnings else "Good")
    return {
        "quality": quality,
        "warnings": warnings,
        "errors": errors,
        "missing_optional_fields": missing_optional,
    }
