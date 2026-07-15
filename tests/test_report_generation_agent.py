from agents.report_generation_agent import deterministic_report, report_generation_agent


def _result():
    return {
        "predicted_efficiency_percent": 82.4,
        "efficiency_band": "mildly reduced efficiency",
        "model_mae_percent": 3.1,
        "factors_lowering_prediction": [
            {"display_name": "restlessness", "shap_value": -3.2}
        ],
        "factors_increasing_prediction": [],
    }


def test_fallback_is_safe_and_includes_uncertainty():
    report = deterministic_report(_result(), {"warnings": []})
    lower = report.lower()
    assert "3.1 percentage points" in report
    assert "wearable measurements are estimates" in lower
    assert "diagnose insomnia" not in lower
    assert "diagnose sleep apnoea" not in lower
    assert "caffeine" not in lower
    assert "stress" not in lower


def test_explicit_offline_mode_uses_fallback():
    report = report_generation_agent(
        {"warnings": []}, {"rmssd": 34}, _result(), client=False
    )
    assert report.startswith("Estimated sleep efficiency")
