import pytest

from agents.efficiency_prediction_agent import (
    EfficiencyModelNotConfiguredError,
    efficiency_prediction_agent,
)


class FakeAdapter:
    model_mae = 3.1

    def predict_efficiency(self, raw_record):
        return 82.4

    def explain_prediction(self, raw_record):
        return {
            "base_value": 85.7,
            "factors_lowering_prediction": [{"feature": "restless"}],
            "factors_increasing_prediction": [],
        }


def test_agent_returns_regression_result(valid_record):
    result = efficiency_prediction_agent(valid_record, FakeAdapter())
    assert result["predicted_efficiency_percent"] == 82.4
    assert result["efficiency_band"] == "mildly reduced efficiency"
    assert result["model_mae_percent"] == 3.1
    assert "severity" not in result
    assert "patterns" not in result


def test_agent_requires_model(valid_record):
    with pytest.raises(EfficiencyModelNotConfiguredError):
        efficiency_prediction_agent(valid_record, None)
