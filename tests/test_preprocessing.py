import math

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    LEAKAGE_COLUMNS,
    MODEL_FEATURES,
    convert_excel_date,
    engineer_dataframe_features,
    engineer_single_record_features,
    prepare_training_data,
)


def test_date_conversion_supports_excel_and_iso():
    assert convert_excel_date(43831) == pd.Timestamp("2020-01-01")
    assert convert_excel_date("2020-06-18") == pd.Timestamp("2020-06-18")


def test_feature_engineering_percentages_circular_values_and_missing(valid_record):
    record = {**valid_record, "temperature_trend_deviation": "NA"}
    features = engineer_single_record_features(record)
    assert list(features) == MODEL_FEATURES
    assert features["rem_sleep_percent"] == 20.0
    assert features["deep_sleep_percent"] == pytest.approx(70 / 3)
    assert math.isnan(features["temperature_trend_deviation"])
    assert features["bedtime_sin"] == pytest.approx(
        np.sin(2 * np.pi * ((4200 / 3600) % 24) / 24)
    )
    assert features["bedtime_cos"] == pytest.approx(
        np.cos(2 * np.pi * ((4200 / 3600) % 24) / 24)
    )
    day = pd.Timestamp("2020-06-18").dayofweek
    assert features["day_sin"] == pytest.approx(np.sin(2 * np.pi * day / 7))
    assert features["day_cos"] == pytest.approx(np.cos(2 * np.pi * day / 7))


def test_zero_total_produces_missing_percentages(valid_record):
    engineered = engineer_dataframe_features(pd.DataFrame([{**valid_record, "total": 0}]))
    assert pd.isna(engineered.loc[0, "rem_sleep_percent"])
    assert pd.isna(engineered.loc[0, "deep_sleep_percent"])


def test_training_matrix_excludes_leakage_and_identity(valid_record):
    rows = []
    for index in range(2):
        rows.append(
            {
                **valid_record,
                "date": f"2020-06-{18 + index}",
                "email": f"person{index}@example.test",
                "efficiency": 80 + index,
                "duration": 30000,
                "score": 90,
            }
        )
    features, _, groups, metadata = prepare_training_data(pd.DataFrame(rows))
    assert list(features.columns) == MODEL_FEATURES
    assert not (set(features.columns) & LEAKAGE_COLUMNS)
    assert "participant_id" not in features
    assert list(groups) == ["person0@example.test", "person1@example.test"]
    assert "participant_id" in metadata
