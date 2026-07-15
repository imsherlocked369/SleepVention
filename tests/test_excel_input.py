from io import BytesIO

import pandas as pd
import pytest

from src.excel_input import (
    DuplicateColumnError,
    create_record_label,
    list_excel_sheets,
    load_excel_sheet,
    normalise_column_names,
    prepare_row_for_pipeline,
    validate_uploaded_schema,
)


def _workbook_bytes(frame):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Nights", index=False)
        frame.head(1).to_excel(writer, sheet_name="Other", index=False)
    buffer.seek(0)
    return buffer


def test_in_memory_workbook_sheet_listing_and_loading(valid_record):
    source = _workbook_bytes(pd.DataFrame([valid_record]))
    assert list_excel_sheets(source) == ["Nights", "Other"]
    loaded = load_excel_sheet(source, "Nights")
    assert len(loaded) == 1


def test_column_normalisation_aliases_and_duplicates():
    frame = pd.DataFrame(columns=[" HR Average ", "HRV", "REM-Seconds"])
    assert list(normalise_column_names(frame).columns) == ["hr_average", "rmssd", "rem"]
    with pytest.raises(DuplicateColumnError):
        normalise_column_names(pd.DataFrame(columns=["HRV", "rmssd"]))


def test_schema_validation_and_stable_labels(valid_record):
    row = {**valid_record, "participant_id": "P39", "efficiency": 84.2}
    frame = pd.DataFrame([row])
    result = validate_uploaded_schema(frame)
    assert result["state"] == "Ready with warnings"
    assert result["valid_row_indices"] == [0]
    assert create_record_label(frame.loc[0], 0) == "P39 — 2020-06-18 — Row 2"
    missing = validate_uploaded_schema(frame.drop(columns=["rmssd"]))
    assert "rmssd" in missing["missing_required_columns"]


def test_row_preparation_excludes_target_and_identifiers(valid_record):
    row = pd.Series(
        {
            **valid_record,
            "efficiency": 84.2,
            "participant_id": "P39",
            "email": "person@example.test",
        }
    )
    record = prepare_row_for_pipeline(row)
    assert "efficiency" not in record
    assert "participant_id" not in record
    assert "email" not in record
    assert record["rmssd"] == 34.0
