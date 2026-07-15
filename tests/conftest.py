import pytest


@pytest.fixture
def valid_record():
    return {
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
        "bedtime_start_delta": 4200,
    }
