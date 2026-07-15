"""In-memory Excel loading, schema validation, and row preparation for the UI."""

from __future__ import annotations

import re
from io import BytesIO
from typing import BinaryIO

import pandas as pd

from agents.data_quality_agent import OPTIONAL_FIELDS, REQUIRED_FIELDS, data_quality_agent

COLUMN_ALIASES = {
    "hrv": "rmssd",
    "average_hr": "hr_average",
    "lowest_hr": "hr_lowest",
    "respiratory_rate": "breath_average",
    "breathing_rate": "breath_average",
    "rem_seconds": "rem",
    "deep_seconds": "deep",
    "total_sleep_seconds": "total",
}
METADATA_COLUMNS = {
    "participant_id", "email", "participant_uid", "participant_email", "efficiency"
}
NUMERIC_INPUT_FIELDS = set(REQUIRED_FIELDS[1:]) | set(OPTIONAL_FIELDS)


class ExcelInputError(ValueError):
    """Raised when an uploaded workbook or worksheet cannot be used."""


class DuplicateColumnError(ExcelInputError):
    """Raised when different source columns normalize to the same name."""


def _rewind(source: BinaryIO | BytesIO) -> None:
    if hasattr(source, "seek"):
        source.seek(0)


def list_excel_sheets(source: BinaryIO | BytesIO) -> list[str]:
    """List workbook worksheet names without saving the upload to disk."""
    try:
        _rewind(source)
        names = pd.ExcelFile(source).sheet_names
        _rewind(source)
        return list(names)
    except Exception as exc:
        raise ExcelInputError(
            "The uploaded workbook could not be read. Confirm that it is a valid "
            "Excel file and is not password protected."
        ) from exc


def load_excel_sheet(source: BinaryIO | BytesIO, sheet_name: str) -> pd.DataFrame:
    """Load only the selected worksheet from an in-memory workbook."""
    try:
        _rewind(source)
        frame = pd.read_excel(source, sheet_name=sheet_name)
        _rewind(source)
        return frame
    except Exception as exc:
        raise ExcelInputError(
            "The selected worksheet could not be read. Confirm that the workbook "
            "is valid and is not password protected."
        ) from exc


def normalise_column_name(name: object) -> str:
    """Normalize one obvious technical column variant."""
    normalized = str(name).strip().lower().replace("-", " ")
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return COLUMN_ALIASES.get(normalized, normalized)


def normalise_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns and reject collisions rather than silently overwriting."""
    names = [normalise_column_name(column) for column in frame.columns]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise DuplicateColumnError(
            "Duplicate columns after normalisation: " + ", ".join(duplicates)
        )
    result = frame.copy()
    result.columns = names
    return result


def prepare_row_for_pipeline(row: pd.Series) -> dict[str, object]:
    """Return only supported raw fields, excluding target and identifiers."""
    record: dict[str, object] = {}
    for field in [*REQUIRED_FIELDS, *OPTIONAL_FIELDS]:
        if field not in row or pd.isna(row[field]):
            continue
        value = row[field]
        if field == "date":
            parsed = pd.to_datetime(value, errors="coerce")
            record[field] = value if pd.isna(parsed) else parsed.date().isoformat()
        elif field in NUMERIC_INPUT_FIELDS:
            converted = pd.to_numeric(value, errors="coerce")
            record[field] = value if pd.isna(converted) else float(converted)
        else:
            record[field] = value
    return record


def _display_identifier(row: pd.Series) -> str | None:
    participant = row.get("participant_id")
    if pd.notna(participant) and str(participant).strip():
        return str(participant).strip()
    email = row.get("email")
    if pd.notna(email) and str(email).strip():
        value = str(email).strip()
        if "@" in value:
            local, domain = value.split("@", 1)
            return f"{local[:2]}***@{domain}"
        return value[:2] + "***"
    return None


def create_record_label(row: pd.Series, dataframe_index: object) -> str:
    """Create a stable local-only label using identifier, date, then source row."""
    row_number = int(dataframe_index) + 2 if isinstance(dataframe_index, int) else dataframe_index
    parts: list[str] = []
    identifier = _display_identifier(row)
    if identifier:
        parts.append(identifier)
    parsed_date = pd.to_datetime(row.get("date"), errors="coerce")
    if pd.notna(parsed_date):
        parts.append(parsed_date.date().isoformat())
    parts.append(f"Row {row_number}")
    return " — ".join(parts)


def validate_uploaded_schema(frame: pd.DataFrame) -> dict[str, object]:
    """Validate worksheet-level schema and determine processable row indices."""
    errors: list[str] = []
    warnings: list[str] = []
    missing = [field for field in REQUIRED_FIELDS if field not in frame.columns]
    optional_present = [field for field in OPTIONAL_FIELDS if field in frame.columns]
    supported = set(REQUIRED_FIELDS) | set(OPTIONAL_FIELDS) | METADATA_COLUMNS
    ignored = [column for column in frame.columns if column not in supported]

    if frame.empty:
        errors.append("The selected worksheet is empty.")
    if missing:
        errors.append("Missing required columns: " + ", ".join(missing))
    if "temperature_trend_deviation" not in frame.columns:
        warnings.append(
            "temperature_trend_deviation is missing and will be imputed by the trained pipeline."
        )
    if "efficiency" in frame.columns:
        warnings.append(
            "The workbook contains actual efficiency values; these are used only for comparison."
        )

    valid_indices: list[object] = []
    row_errors: dict[object, list[str]] = {}
    if not missing:
        for index, row in frame.iterrows():
            quality = data_quality_agent(prepare_row_for_pipeline(row))
            if quality["errors"]:
                row_errors[index] = list(quality["errors"])
            else:
                valid_indices.append(index)
        if row_errors:
            warnings.append(
                f"{len(row_errors)} row(s) contain incomplete or invalid values and cannot be processed."
            )
        if not valid_indices and not frame.empty:
            errors.append("No rows contain all required valid values.")

    state = "Cannot be analysed" if errors else (
        "Ready with warnings" if warnings else "Ready for analysis"
    )
    return {
        "state": state,
        "errors": errors,
        "warnings": warnings,
        "detected_columns": list(frame.columns),
        "missing_required_columns": missing,
        "optional_columns_present": optional_present,
        "ignored_columns": ignored,
        "valid_row_indices": valid_indices,
        "row_errors": row_errors,
    }
