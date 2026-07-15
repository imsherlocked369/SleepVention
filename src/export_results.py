"""In-memory JSON, CSV, and Excel download generation."""

from __future__ import annotations

import json
from io import BytesIO

import pandas as pd

SENSITIVE_KEYS = {"api_key", "openai_api_key", "participant_uid", "participant_email"}


def _safe(value):
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items() if key.lower() not in SENSITIVE_KEYS}
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def single_result_json_bytes(payload: dict[str, object]) -> bytes:
    """Serialize a privacy-filtered single result for download."""
    return json.dumps(_safe(payload), indent=2, default=str).encode("utf-8")


def batch_csv_bytes(results: pd.DataFrame) -> bytes:
    """Serialize batch results as UTF-8 CSV bytes."""
    return results.to_csv(index=False).encode("utf-8")


def batch_excel_bytes(
    results: pd.DataFrame,
    errors: pd.DataFrame,
    input_preview: pd.DataFrame | None = None,
) -> bytes:
    """Build an in-memory workbook without modifying the uploaded source."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="Analysis_Results", index=False)
        errors.to_excel(writer, sheet_name="Processing_Errors", index=False)
        if input_preview is not None:
            input_preview.head(100).to_excel(writer, sheet_name="Input_Preview", index=False)
    return buffer.getvalue()
