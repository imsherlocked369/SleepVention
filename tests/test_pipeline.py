import pytest

from pipeline import DataQualityError, run_sleep_pipeline


class FakeEfficiencyAdapter:
    model_mae = 3.1
    calls = 0

    def predict_efficiency(self, raw_record):
        self.calls += 1
        return 82.4

    def explain_prediction(self, raw_record):
        return {
            "base_value": 85.7,
            "factors_lowering_prediction": [
                {
                    "feature": "restless", "display_name": "restlessness",
                    "feature_value": 42.0, "shap_value": -3.2,
                }
            ],
            "factors_increasing_prediction": [],
        }


def test_full_pipeline_works_without_openai(valid_record):
    model = FakeEfficiencyAdapter()
    result = run_sleep_pipeline(valid_record, model, report_client=False)
    assert result["data_quality"]["quality"] == "Good"
    assert result["predicted_efficiency_percent"] == 82.4
    assert result["efficiency_band"] == "mildly reduced efficiency"
    assert "Wearable measurements are estimates" in result["report"]
    assert "severity" not in result
    assert "patterns" not in result


def test_poor_input_stops_before_prediction(valid_record):
    model = FakeEfficiencyAdapter()
    valid_record["total"] = 0
    with pytest.raises(DataQualityError):
        run_sleep_pipeline(valid_record, model, report_client=False)
    assert model.calls == 0


def test_optional_temperature_trend_can_be_imputed(valid_record):
    valid_record.pop("temperature_trend_deviation")
    result = run_sleep_pipeline(valid_record, FakeEfficiencyAdapter(), report_client=False)
    assert result["data_quality"]["quality"] == "Limited"
    assert any("imputed" in warning for warning in result["data_quality"]["warnings"])
