from agents.data_quality_agent import data_quality_agent


def test_valid_record_is_good(valid_record):
    assert data_quality_agent(valid_record)["quality"] == "Good"


def test_missing_required_field_is_error(valid_record):
    valid_record.pop("rmssd")
    result = data_quality_agent(valid_record)
    assert result["quality"] == "Poor"
    assert "rmssd is missing" in result["errors"]


def test_invalid_date_is_error(valid_record):
    valid_record["date"] = "not-a-date"
    assert data_quality_agent(valid_record)["quality"] == "Poor"


def test_nonpositive_total_is_error(valid_record):
    valid_record["total"] = 0
    assert "total must be greater than zero" in data_quality_agent(valid_record)["errors"]


def test_stage_sum_above_total_is_error(valid_record):
    valid_record["rem"] = valid_record["total"]
    assert "rem + deep cannot be greater than total" in data_quality_agent(valid_record)["errors"]


def test_missing_optional_field_is_limited_not_failure(valid_record):
    valid_record.pop("temperature_trend_deviation")
    result = data_quality_agent(valid_record)
    assert result["quality"] == "Limited"
    assert result["errors"] == []
    assert result["missing_optional_fields"] == ["temperature_trend_deviation"]
