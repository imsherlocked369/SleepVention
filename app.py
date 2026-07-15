"""Skeletal Streamlit frontend for Excel-based SleepVention analysis."""

from __future__ import annotations

import hashlib
import logging
import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agents.feature_extraction_agent import feature_extraction_agent
from pipeline import DataQualityError, run_sleep_pipeline
from src.batch_processing import process_batch_records
from src.excel_input import (
    DuplicateColumnError,
    ExcelInputError,
    create_record_label,
    list_excel_sheets,
    load_excel_sheet,
    normalise_column_names,
    prepare_row_for_pipeline,
    validate_uploaded_schema,
)
from src.export_results import batch_csv_bytes, batch_excel_bytes, single_result_json_bytes
from src.model_adapter import XGBoostEfficiencyAdapter
from src.ui_visuals import create_contribution_chart
from src.ui_state import model_is_available
from src.ui_theme import (
    apply_sleepvention_theme,
    console_log_html,
    empty_state_html,
    section_heading_html,
    terminal_header_html,
)

LOGGER = logging.getLogger(__name__)
MODEL_PATH = Path("models/xgboost_efficiency_model.joblib")
MAX_BATCH_RECORDS = 100
TRAINING_COMMAND = (
    "python -m training.train_efficiency_model --data "
    "data/sleep_parsed_anonymised.xlsx --sheet in"
)
NO_MODEL_MESSAGE = (
    "No trained XGBoost model was found. The workbook can still be uploaded, "
    "previewed and validated, but efficiency prediction and final report "
    "generation require the model to be trained."
)

st.set_page_config(page_title="SleepVention", page_icon="🌙", layout="wide")
apply_sleepvention_theme()


@st.cache_resource
def load_efficiency_model(model_path: str) -> XGBoostEfficiencyAdapter:
    """Load the trained bundle once; this is never called when it is absent."""
    return XGBoostEfficiencyAdapter(Path(model_path))


def clear_results() -> None:
    """Clear analysis outputs while retaining the currently loaded worksheet."""
    for key in ("single_result", "single_download", "batch_results"):
        st.session_state.pop(key, None)


def _factor_panel(title: str, factors: list[dict[str, object]]) -> None:
    st.subheader(title)
    if not factors:
        st.write("No contribution was available in this direction.")
    for factor in factors:
        st.markdown(f"**{factor['display_name']}**")
        st.write(f"Observed value: {factor['feature_value']}")
        st.write(f"Model contribution: {float(factor['shap_value']):+.2f} percentage points")


load_dotenv()
model_ready = model_is_available(MODEL_PATH)
llm_ready = bool(os.getenv("OPENAI_API_KEY"))

with st.sidebar:
    st.markdown("### :: configuration")
    st.caption("MODEL")
    st.code("model_path\nmodels/xgboost_efficiency_model.joblib")
    st.write(f"status: {'ready' if model_ready else 'not_trained'}")
    if not model_ready:
        st.warning(NO_MODEL_MESSAGE)
        st.code(TRAINING_COMMAND)
    st.caption("REPORT MODE")
    st.write(f"provider: {'openai' if llm_ready else 'deterministic_fallback'}")
    st.caption("PROCESSING MODE")
    processing_mode = st.radio(
        "Processing mode", ["Single-record analysis", "Batch analysis"], index=0
    )
    report_mode = st.radio(
        "Report mode",
        ["Generate report with configured LLM", "Use deterministic fallback only"],
        index=0,
    )
    fallback_only = report_mode == "Use deterministic fallback only"
    if st.button("Clear analysis results"):
        clear_results()
    st.divider()
    st.caption("PRIVACY")
    st.caption("identifiers remain local and are not sent to the language model")

header_kind = "ready" if model_ready else "unavailable"
header_status = "system ready" if model_ready else "model unavailable"
st.markdown(
    terminal_header_html(
        "SleepVention", "wearable sleep interpretation workspace", header_status, header_kind
    ),
    unsafe_allow_html=True,
)
st.markdown(section_heading_html("01", "INPUT"), unsafe_allow_html=True)
st.caption("accepted: .xlsx, .xls")
uploaded_file = st.file_uploader("Upload wearable sleep data", type=["xlsx", "xls"])

if uploaded_file is not None:
    workbook_bytes = uploaded_file.getvalue()
    workbook_key = hashlib.sha256(workbook_bytes).hexdigest()
    if st.session_state.get("workbook_key") != workbook_key:
        clear_results()
        st.session_state["workbook_key"] = workbook_key
        st.session_state.pop("worksheet", None)

    try:
        sheets = list_excel_sheets(BytesIO(workbook_bytes))
        selected_sheet = st.selectbox("Select worksheet", options=sheets)
        if st.session_state.get("worksheet") != selected_sheet:
            clear_results()
            st.session_state["worksheet"] = selected_sheet
        raw_frame = load_excel_sheet(BytesIO(workbook_bytes), selected_sheet)
        frame = normalise_column_names(raw_frame)
        validation = validate_uploaded_schema(frame)
    except (ExcelInputError, DuplicateColumnError) as exc:
        LOGGER.warning("Workbook validation failed: %s", exc)
        st.error(str(exc))
        with st.expander("Technical detail"):
            st.write(type(exc).__name__)
    except Exception as exc:
        LOGGER.exception("Unexpected workbook error")
        st.error(
            "The uploaded workbook could not be read. Confirm that it is a valid "
            "Excel file and is not password protected."
        )
        with st.expander("Technical detail"):
            st.write(type(exc).__name__)
    else:
        st.markdown(
            console_log_html(
                [
                    ("ok", "workbook opened successfully"),
                    ("info", f"file: {uploaded_file.name}"),
                    ("info", f"sheet: {selected_sheet}"),
                    ("info", f"size: {len(workbook_bytes) / 1024:.1f} KB"),
                ]
            ),
            unsafe_allow_html=True,
        )
        st.markdown(section_heading_html("02", "VALIDATE"), unsafe_allow_html=True)
        col_rows, col_columns = st.columns(2)
        col_rows.metric("Rows", len(frame))
        col_columns.metric("Columns", len(frame.columns))
        validation_entries = [
            ("ok", f'worksheet "{selected_sheet}" contains {len(frame)} records'),
            (
                "ok" if not validation["missing_required_columns"] else "error",
                "all required columns detected"
                if not validation["missing_required_columns"]
                else "required columns are missing",
            ),
        ]
        if "efficiency" in frame.columns:
            validation_entries.append(("ok", 'target column "efficiency" detected for comparison only'))
        for warning in validation["warnings"]:
            validation_entries.append(("warn", warning))
        for error in validation["errors"]:
            validation_entries.append(("error", error))
        if validation["ignored_columns"]:
            validation_entries.append(
                ("info", f"{len(validation['ignored_columns'])} unrelated columns will be ignored")
            )
        st.markdown(console_log_html(validation_entries), unsafe_allow_html=True)

        st.markdown(section_heading_html("dataset", "preview()"), unsafe_allow_html=True)
        st.caption(f"shape: {frame.shape}   ·   sheet: {selected_sheet}")
        st.dataframe(frame.head(15), use_container_width=True)

        status = validation["state"]
        if status == "Cannot be analysed":
            st.error(status)
        elif status == "Ready with warnings":
            st.warning(status)
        else:
            st.success(status)
        with st.expander("Schema details", expanded=status == "Cannot be analysed"):
            st.write("Detected columns:", validation["detected_columns"])
            st.write("Missing required columns:", validation["missing_required_columns"] or "None")
            st.write("Optional columns present:", validation["optional_columns_present"] or "None")
            st.write("Ignored columns:", validation["ignored_columns"] or "None")
            for error in validation["errors"]:
                st.error(error)
            for warning in validation["warnings"]:
                st.warning(warning)

        valid_indices = validation["valid_row_indices"]
        labels = {create_record_label(frame.loc[index], index): index for index in valid_indices}
        can_predict = model_ready and bool(valid_indices) and not validation["errors"]

        if processing_mode == "Single-record analysis" and valid_indices:
            st.markdown(section_heading_html("03", "SELECT"), unsafe_allow_html=True)
            selected_label = st.selectbox("Valid record", options=list(labels))
            st.session_state["selected_record"] = selected_label
            selected_index = labels[selected_label]
            selected_row = frame.loc[selected_index]
            record = prepare_row_for_pipeline(selected_row)
            st.dataframe(pd.DataFrame([record]), use_container_width=True)
            with st.expander("Prepared model features"):
                st.json(feature_extraction_agent(record)["model_features"])
            if not model_ready:
                st.info(NO_MODEL_MESSAGE)
                st.code(TRAINING_COMMAND)
            st.markdown(section_heading_html("04", "ANALYSE"), unsafe_allow_html=True)
            analyse = st.button(
                "▶ run sleep analysis", type="primary", disabled=not can_predict
            )
            if analyse:
                try:
                    model = load_efficiency_model(str(MODEL_PATH))
                    with st.spinner("Analysing sleep record..."):
                        result = run_sleep_pipeline(
                            record,
                            model,
                            generate_report=True,
                            fallback_only=fallback_only,
                        )
                    actual = selected_row.get("efficiency")
                    actual = float(actual) if pd.notna(actual) else None
                    result["actual_efficiency"] = actual
                    result["prediction_error"] = (
                        result["predicted_efficiency_percent"] - actual
                        if actual is not None else None
                    )
                    result["absolute_error"] = (
                        abs(result["prediction_error"]) if actual is not None else None
                    )
                    st.session_state["single_result"] = result
                    st.session_state["single_download"] = {
                        "input_summary": record,
                        **result,
                        "warnings": result["data_quality"]["warnings"],
                    }
                except DataQualityError as exc:
                    st.error("The selected record failed data-quality validation.")
                    for error in exc.result["errors"]:
                        st.write(f"- {error}")
                except Exception as exc:
                    LOGGER.exception("Single-record analysis failed")
                    st.error(f"Analysis could not be completed: {exc}")

        elif processing_mode == "Batch analysis" and valid_indices:
            st.markdown(section_heading_html("03", "SELECT"), unsafe_allow_html=True)
            batch_choice = st.radio(
                "Batch selection", ["Analyse all valid rows", "Analyse selected rows"]
            )
            if batch_choice == "Analyse selected rows":
                chosen_labels = st.multiselect("Valid records", options=list(labels))
                batch_indices = [labels[label] for label in chosen_labels]
            else:
                batch_indices = valid_indices[:MAX_BATCH_RECORDS]
                if len(valid_indices) > MAX_BATCH_RECORDS:
                    st.warning(
                        f"This run is limited to {MAX_BATCH_RECORDS} records. "
                        "Process the remaining records in another batch."
                    )
            generate_batch_reports = st.checkbox(
                "Generate narrative report for every row", value=False
            )
            if generate_batch_reports:
                st.warning("One report request may be made for every selected row.")
            if not model_ready:
                st.info(NO_MODEL_MESSAGE)
                st.code(TRAINING_COMMAND)
            st.markdown(section_heading_html("04", "ANALYSE"), unsafe_allow_html=True)
            run_batch = st.button(
                "▶ run batch analysis",
                type="primary",
                disabled=not can_predict or not batch_indices,
            )
            if run_batch:
                progress = st.progress(0.0)
                progress_text = st.empty()

                def update_progress(done: int, total: int, label: str) -> None:
                    progress.progress(done / total)
                    progress_text.write(f"Processing {done} of {total}: {label}")

                model = load_efficiency_model(str(MODEL_PATH))
                results = process_batch_records(
                    frame,
                    batch_indices,
                    model,
                    generate_reports=generate_batch_reports,
                    fallback_only=fallback_only,
                    progress_callback=update_progress,
                )
                st.session_state["batch_results"] = results
                progress_text.write("Batch processing complete.")
                completed = int(results["error"].isna().sum())
                failed = int(results["error"].notna().sum())
                average = results["predicted_efficiency"].mean()
                st.markdown(
                    console_log_html(
                        [
                            ("ok", f"processed: {completed}"),
                            ("error" if failed else "info", f"failed: {failed}"),
                            ("info", f"reports: {completed if generate_batch_reports else 0}"),
                            ("info", f"average estimate: {average:.1f}%" if pd.notna(average) else "average estimate: n/a"),
                        ]
                    ),
                    unsafe_allow_html=True,
                )

        result = st.session_state.get("single_result")
        if result and processing_mode == "Single-record analysis":
            st.markdown(section_heading_html("05", "INTERPRET"), unsafe_allow_html=True)
            metrics = st.columns(4)
            metrics[0].metric("Predicted efficiency", f"{result['predicted_efficiency_percent']:.1f}%")
            metrics[1].metric("Efficiency band", result["efficiency_band"])
            mae = result["model_mae_percent"]
            metrics[2].metric("Typical model MAE", f"{mae:.1f}%" if mae is not None else "N/A")
            actual = result.get("actual_efficiency")
            metrics[3].metric("Actual efficiency", f"{actual:.1f}%" if actual is not None else "N/A")
            if actual is not None:
                st.caption(
                    f"Prediction error: {result['prediction_error']:+.1f} points · "
                    f"Absolute error: {result['absolute_error']:.1f} points"
                )
            summary_tab, explanation_tab, report_tab, raw_tab = st.tabs(
                ["SUMMARY", "MODEL TRACE", "REPORT", "RAW OUTPUT"]
            )
            with summary_tab:
                st.write("Data quality:", result["data_quality"]["quality"])
                with st.expander("Data-quality warnings"):
                    for warning in result["data_quality"]["warnings"]:
                        st.warning(warning)
            with explanation_tab:
                base_value = result.get("base_value")
                st.code(
                    f"base estimate       {base_value:.1f}%\n"
                    f"final estimate      {result['predicted_efficiency_percent']:.1f}%"
                    if base_value is not None
                    else f"final estimate      {result['predicted_efficiency_percent']:.1f}%"
                )
                lower, higher = st.columns(2)
                with lower:
                    _factor_panel("Factors lowering the estimate", result["factors_lowering_prediction"])
                with higher:
                    _factor_panel("Factors increasing the estimate", result["factors_increasing_prediction"])
                figure = create_contribution_chart(
                    result["factors_lowering_prediction"],
                    result["factors_increasing_prediction"],
                )
                st.pyplot(figure)
                st.caption(
                    "SHAP values describe how each feature influenced the model estimate. "
                    "They do not establish medical causation."
                )
            with report_tab:
                st.caption(
                    "generated with OpenAI"
                    if llm_ready and not fallback_only
                    else "deterministic fallback"
                )
                st.markdown("#### sleepvention.report")
                st.markdown(result["report"])
            with raw_tab:
                st.json(st.session_state["single_download"])
            st.markdown(section_heading_html("06", "EXPORT"), unsafe_allow_html=True)
            st.download_button(
                "↓ download result.json",
                data=single_result_json_bytes(st.session_state["single_download"]),
                file_name="sleepvention_result.json",
                mime="application/json",
            )

        batch_results = st.session_state.get("batch_results")
        if batch_results is not None and processing_mode == "Batch analysis":
            st.markdown(section_heading_html("05", "INTERPRET"), unsafe_allow_html=True)
            success = batch_results[batch_results["error"].isna()]
            errors = batch_results[batch_results["error"].notna()][
                ["source_row", "record_label", "error"]
            ]
            results_tab, errors_tab, downloads_tab = st.tabs(["RESULTS", "ERRORS", "EXPORT"])
            with results_tab:
                st.dataframe(batch_results, use_container_width=True)
            with errors_tab:
                st.dataframe(errors, use_container_width=True)
            with downloads_tab:
                st.download_button(
                    "↓ download analysis_results.csv",
                    data=batch_csv_bytes(batch_results),
                    file_name="sleepvention_results.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "↓ download analysis_results.xlsx",
                    data=batch_excel_bytes(success, errors, frame.head(100)),
                    file_name="sleepvention_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
else:
    st.markdown(empty_state_html("waiting for workbook input"), unsafe_allow_html=True)

st.divider()
st.caption(
    "SHAP contributions describe model behaviour and do not prove causation. "
    "Uploaded workbooks are not automatically saved or overwritten."
)
