from types import SimpleNamespace

import pandas as pd

from src.batch_processing import process_batch_records


class FakeModel:
    model_mae = 3.1

    def predict_efficiency(self, record):
        return 82.4

    def explain_prediction(self, record):
        return {
            "base_value": 85.0,
            "factors_lowering_prediction": [
                {
                    "feature": "restless", "display_name": "restlessness",
                    "feature_value": record["restless"], "shap_value": -2.6,
                }
            ],
            "factors_increasing_prediction": [],
        }


class FakeResponses:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(output_text="Generated report")


def test_batch_continues_after_invalid_row_and_skips_reports(valid_record):
    invalid = {**valid_record, "total": 0}
    frame = pd.DataFrame([valid_record, invalid])
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    results = process_batch_records(
        frame, frame.index, FakeModel(), report_client=client, generate_reports=False
    )
    assert results.loc[0, "predicted_efficiency"] == 82.4
    assert pd.notna(results.loc[1, "error"])
    assert responses.calls == 0


def test_batch_calls_report_client_only_when_enabled(valid_record):
    frame = pd.DataFrame([valid_record, valid_record])
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    results = process_batch_records(
        frame, frame.index, FakeModel(), report_client=client, generate_reports=True
    )
    assert responses.calls == 2
    assert (results["report_status"] == "generated").all()
