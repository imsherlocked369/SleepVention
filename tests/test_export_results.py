import json
from io import BytesIO

import pandas as pd

from src.export_results import batch_csv_bytes, batch_excel_bytes, single_result_json_bytes


def test_single_json_filters_sensitive_keys():
    content = single_result_json_bytes(
        {"prediction": 82.4, "OPENAI_API_KEY": "secret", "participant_uid": "uuid"}
    )
    decoded = json.loads(content)
    assert decoded == {"prediction": 82.4}
    assert b"secret" not in content


def test_csv_and_excel_downloads_are_valid():
    results = pd.DataFrame([{"source_row": 2, "predicted_efficiency": 82.4}])
    errors = pd.DataFrame([{"source_row": 3, "error": "invalid"}])
    assert b"predicted_efficiency" in batch_csv_bytes(results)
    workbook = pd.ExcelFile(BytesIO(batch_excel_bytes(results, errors)))
    assert workbook.sheet_names == ["Analysis_Results", "Processing_Errors"]
