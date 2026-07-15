"""Reusable row-by-row batch orchestration for the Streamlit frontend."""

from __future__ import annotations

from typing import Callable, Iterable

import pandas as pd

from pipeline import run_sleep_pipeline
from src.excel_input import create_record_label, prepare_row_for_pipeline

BATCH_RESULT_COLUMNS = [
    "source_row", "record_label", "participant_id", "date", "data_quality",
    "predicted_efficiency", "actual_efficiency", "absolute_error",
    "efficiency_band", "model_mae", "main_lowering_factor",
    "main_increasing_factor", "report_status", "error",
]


def _local_participant(row: pd.Series) -> object:
    for field in ("participant_id", "email"):
        value = row.get(field)
        if pd.notna(value):
            return value
    return None


def process_batch_records(
    frame: pd.DataFrame,
    row_indices: Iterable[object],
    efficiency_model: object,
    report_client: object | None = None,
    generate_reports: bool = False,
    fallback_only: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Process rows independently, recording errors without aborting the batch."""
    indices = list(row_indices)
    output: list[dict[str, object]] = []
    for position, index in enumerate(indices, start=1):
        row = frame.loc[index]
        label = create_record_label(row, index)
        source_row = int(index) + 2 if isinstance(index, int) else index
        base: dict[str, object] = {
            "source_row": source_row,
            "record_label": label,
            "participant_id": _local_participant(row),
            "date": row.get("date"),
            "actual_efficiency": row.get("efficiency") if pd.notna(row.get("efficiency")) else None,
        }
        try:
            record = prepare_row_for_pipeline(row)
            result = run_sleep_pipeline(
                record,
                efficiency_model,
                report_client=report_client,
                generate_report=generate_reports,
                fallback_only=fallback_only,
            )
            predicted = float(result["predicted_efficiency_percent"])
            actual = base["actual_efficiency"]
            lowering = result["factors_lowering_prediction"]
            increasing = result["factors_increasing_prediction"]
            base.update(
                {
                    "data_quality": result["data_quality"]["quality"],
                    "predicted_efficiency": predicted,
                    "absolute_error": abs(predicted - float(actual)) if actual is not None else None,
                    "efficiency_band": result["efficiency_band"],
                    "model_mae": result["model_mae_percent"],
                    "main_lowering_factor": lowering[0].get("display_name") if lowering else None,
                    "main_increasing_factor": increasing[0].get("display_name") if increasing else None,
                    "report_status": (
                        "deterministic fallback" if generate_reports and fallback_only
                        else "generated" if generate_reports else "not requested"
                    ),
                    "error": None,
                }
            )
        except Exception as exc:
            base.update(
                {
                    "data_quality": "Poor",
                    "predicted_efficiency": None,
                    "absolute_error": None,
                    "efficiency_band": None,
                    "model_mae": None,
                    "main_lowering_factor": None,
                    "main_increasing_factor": None,
                    "report_status": "failed",
                    "error": str(exc),
                }
            )
        output.append(base)
        if progress_callback:
            progress_callback(position, len(indices), label)
    return pd.DataFrame(output, columns=BATCH_RESULT_COLUMNS)
