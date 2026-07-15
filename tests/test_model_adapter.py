import joblib
import numpy as np
import pytest

from src.model_adapter import EfficiencyModelLoadError, XGBoostEfficiencyAdapter
from src.preprocessing import MODEL_FEATURES


class FakeImputer:
    def transform(self, frame):
        return np.nan_to_num(frame.to_numpy(dtype=float), nan=0.0)


class FakeModel:
    def predict(self, values):
        return np.array([82.4])


class FakePipeline:
    def __init__(self):
        self.named_steps = {"imputer": FakeImputer(), "model": FakeModel()}
        self.columns_seen = None

    def predict(self, frame):
        self.columns_seen = list(frame.columns)
        return np.array([82.4])


def _write_bundle(path, feature_names=None):
    joblib.dump(
        {
            "pipeline": FakePipeline(),
            "feature_names": feature_names or MODEL_FEATURES,
            "target_name": "efficiency",
            "metrics": {"mae_mean": 3.1},
        },
        path,
    )


def test_missing_and_malformed_bundles_raise_clear_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="Trained efficiency model not found"):
        XGBoostEfficiencyAdapter(tmp_path / "missing.joblib")
    malformed = tmp_path / "malformed.joblib"
    joblib.dump({"pipeline": object()}, malformed)
    with pytest.raises(EfficiencyModelLoadError, match="malformed"):
        XGBoostEfficiencyAdapter(malformed)


def test_prediction_is_float_and_uses_saved_order(tmp_path, valid_record):
    path = tmp_path / "model.joblib"
    _write_bundle(path)
    adapter = XGBoostEfficiencyAdapter(path)
    prediction = adapter.predict_efficiency(valid_record)
    assert isinstance(prediction, float)
    assert adapter.pipeline.columns_seen == MODEL_FEATURES
    assert not {"total", "rem", "deep", "participant_id"} & set(
        adapter.pipeline.columns_seen
    )


def test_saved_feature_mismatch_is_rejected(tmp_path):
    path = tmp_path / "wrong.joblib"
    _write_bundle(path, list(reversed(MODEL_FEATURES)))
    with pytest.raises(EfficiencyModelLoadError, match="feature order"):
        XGBoostEfficiencyAdapter(path)


def test_explanation_has_required_factor_fields(tmp_path, valid_record, monkeypatch):
    path = tmp_path / "model.joblib"
    _write_bundle(path)

    def fake_explain(model, transformed, original, names, predicted):
        return {
            "predicted_efficiency": predicted,
            "base_value": 85.0,
            "factors_lowering_prediction": [
                {
                    "feature": "restless",
                    "display_name": "restlessness",
                    "feature_value": original["restless"],
                    "shap_value": -2.6,
                }
            ],
            "factors_increasing_prediction": [],
        }

    monkeypatch.setattr("src.model_adapter.explain_xgboost_prediction", fake_explain)
    result = XGBoostEfficiencyAdapter(path).explain_prediction(valid_record)
    factor = result["factors_lowering_prediction"][0]
    assert set(factor) == {"feature", "display_name", "feature_value", "shap_value"}
