from src.ui_state import model_is_available


def test_missing_model_remains_unavailable(tmp_path):
    assert model_is_available(tmp_path / "missing.joblib") is False
