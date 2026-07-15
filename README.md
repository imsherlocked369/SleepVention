# Sleep Efficiency Insight

This repository provides a non-diagnostic prototype that estimates sleep
efficiency from one Oura-compatible sleep record. The deployed model design is
an XGBoost regressor, and SHAP is used to explain individual model predictions.

```text
Raw Oura sleep data
        |
        v
Data Quality Agent
        |
        v
Feature Extraction Agent
        |
        v
XGBoost Efficiency Prediction
        |
        v
SHAP Model Explanation
        |
        v
LLM or deterministic Report Generation
        |
        v
Personalised, non-diagnostic report
```

The repository does not contain a real trained model until the training command
is run manually.

## Method

Sleep efficiency can normally be calculated directly from total sleep time and
time in bed. To avoid target leakage, the direct components of that formula were
excluded from the model. XGBoost therefore estimates efficiency from independent
physiological, temporal and sleep-stage-composition features.

The raw `total`, `rem`, and `deep` values are accepted temporarily only to
calculate REM- and deep-sleep percentages. They are never passed directly into
the model. Participant identity is used only to create participant-disjoint
validation folds and is never a model feature. Date is used for sorting and
day-of-week circular features; the raw date is not a model feature.

## Installation

Python 3.10 or newer is recommended.

```powershell
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` if OpenAI reporting is wanted:

```dotenv
OPENAI_API_KEY=
OPENAI_MODEL=
```

The API key is optional. Without it, or if the API request fails, the application
uses a deterministic safe report template.

## Frontend workflow

```text
Upload Excel
    -> Select worksheet
    -> Preview and validate records
    -> Select one or multiple nights
    -> Run XGBoost and SHAP
    -> Generate an optional report
    -> Download results
```

The Streamlit interface supports single-record and batch analysis. Batch mode
does not generate narrative reports by default, avoiding unexpected API usage.
When reports are enabled, the interface clearly warns that one request may be
made per selected row. The deterministic fallback remains available in both
modes.

The UI can upload, preview, normalize, and validate an Excel workbook even when
the model is absent. Prediction controls remain disabled until this file exists:

```text
models/xgboost_efficiency_model.joblib
```

Uploaded workbooks are processed in memory and are not automatically written
back to disk.

## Interface design

The Streamlit interface uses a Claude Code-inspired terminal aesthetic: warm
charcoal surfaces, monospace typography, terracotta accents, subtle borders,
and console-style workflow panels. This is an independent visual interpretation
and does not use proprietary branding, logos, illustrations, remote assets, or
imply an association with Anthropic.

The layout uses responsive status metadata and Streamlit columns that naturally
stack on narrower screens. Visible keyboard focus outlines, restrained type
sizes, warm high-contrast text, textual status labels, and non-colour-only
validation messages support accessibility.

Screenshot placeholder (no image is currently committed):

```text
docs/screenshots/sleepvention-ui.png
```

## Supported workbook schema

Required columns:

```text
date
onset_latency
midpoint_time
restless
hr_average
hr_lowest
rmssd
breath_average
temperature_deviation
rem
deep
total
bedtime_start_delta
```

Optional model input:

```text
temperature_trend_deviation
```

Optional local metadata and comparison fields:

```text
participant_id
email
participant_uid
participant_email
efficiency
```

Column names are normalized for capitalization, whitespace, and hyphens. A
small alias set supports obvious variants such as `hrv` to `rmssd` and
`average_hr` to `hr_average`. Ambiguous physiological concepts are not mapped.

`efficiency` is optional during inference. When present, it is used only to
compare predicted and actual efficiency and is never sent to XGBoost. Likewise,
`total`, `rem`, and `deep` are accepted only to calculate stage percentages.
Identifiers are never model features and are not sent to the LLM.

One supported row has this raw shape:

```json
{
  "date": "2020-06-18",
  "onset_latency": 1500,
  "midpoint_time": 14500,
  "restless": 42,
  "hr_average": 67.2,
  "hr_lowest": 58,
  "rmssd": 34,
  "breath_average": 15.8,
  "temperature_deviation": 0.12,
  "temperature_trend_deviation": 0.08,
  "rem": 5400,
  "deep": 6300,
  "total": 27000,
  "bedtime_start_delta": 4200
}
```

`temperature_trend_deviation` is optional. Missing model values are handled by
the median imputer fitted inside the persisted sklearn pipeline.

## Frontend results and downloads

Single-record analysis displays data quality, predicted and actual efficiency
when available, prototype efficiency band, model MAE, signed SHAP factors, a
simple contribution chart, and the generated or fallback report. The result can
be downloaded as privacy-filtered JSON.

Batch analysis continues when an individual row fails. It records row-specific
errors and provides CSV and Excel downloads. The Excel export contains
`Analysis_Results` and `Processing_Errors`, with an optional limited input
preview. Participant identifiers remain only in local selections and local
batch downloads.

## Training the model later

The training workbook is expected at `data/sleep_parsed_anonymised.xlsx`, with
the default worksheet named `in`. The continuous target must be `efficiency`,
and `email` is renamed internally to `participant_id` for grouped validation.

Run the real training manually when ready:

```powershell
python -m training.train_efficiency_model --data data/sleep_parsed_anonymised.xlsx --sheet in
```

The command performs five-fold `GroupKFold` validation, verifies participant
disjointness, records MAE, RMSE, R², Pearson and Spearman correlations, writes
evaluation tables and plots under `outputs/`, fits the final pipeline, and saves:

```text
models/xgboost_efficiency_model.joblib
```

The saved bundle contains the pipeline, ordered feature names, target name,
validation metrics, model version, timezone-aware training timestamp, and
training row and participant counts. The application does not need the workbook
after this bundle has been created.

If the workbook is absent, training stops with a clear dataset-not-found error.

## Start the UI

Install dependencies and start the UI:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

When no trained bundle exists, the UI imports normally and explains that manual
training is required before predictions can be generated.

## Testing

```powershell
python -m pytest -q
```

Tests use fake adapters, lightweight estimators, or synthetic data only. They do
not train on the Oura workbook and do not create a production model artifact.

## Project structure

```text
sleep_llm_terminal_framework/
|-- app.py
|-- pipeline.py
|-- sleep_data.py
|-- README.md
|-- requirements.txt
|-- .env.example
|-- agents/
|   |-- __init__.py
|   |-- data_quality_agent.py
|   |-- feature_extraction_agent.py
|   |-- efficiency_prediction_agent.py
|   `-- report_generation_agent.py
|-- src/
|   |-- __init__.py
|   |-- excel_input.py
|   |-- batch_processing.py
|   |-- export_results.py
|   |-- preprocessing.py
|   |-- model_adapter.py
|   |-- model_explainer.py
|   |-- ui_state.py
|   `-- ui_visuals.py
|-- training/
|   |-- __init__.py
|   `-- train_efficiency_model.py
|-- models/
|   `-- .gitkeep
|-- data/
|   |-- sample_night.json
|   `-- sleep_parsed_anonymised.xlsx
|-- outputs/
`-- tests/
```

`main.py` is intentionally absent; Streamlit is the application entry point.

## Limitations and safety

- Wearable measurements are estimates.
- Sleep efficiency can normally be calculated directly by the device.
- This model estimates efficiency only from leakage-safe independent features.
- SHAP explains model behaviour and does not prove causation.
- Performance may vary for participants not represented in training.
- Participant-grouped validation reduces, but does not eliminate, generalisation risk.
- The communication bands are prototype descriptions, not validated clinical thresholds.
- The system is non-diagnostic and must not be used to diagnose a sleep condition.
- External validation is required before any health-related deployment.

## Privacy

Uploaded files are processed locally by the running Streamlit application and
are not automatically saved. Participant identifiers are used only to label
records locally and may appear in user-requested local batch downloads. The LLM
receives only verified display measurements, the de-identified prediction,
model uncertainty, SHAP contributions, and data-quality warnings. It never
receives the full worksheet, participant identifiers, actual efficiency target,
or environment secrets.
