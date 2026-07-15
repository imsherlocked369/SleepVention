import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import MODEL_FEATURES
from training.train_efficiency_model import evaluate_grouped_model


def test_grouped_validation_has_no_participant_overlap_and_required_outputs():
    rng = np.random.default_rng(42)
    participants = [f"p{index}" for index in range(6)]
    groups = pd.Series([participant for participant in participants for _ in range(3)])
    features = pd.DataFrame(
        rng.normal(size=(len(groups), len(MODEL_FEATURES))), columns=MODEL_FEATURES
    )
    target = pd.Series(85 + features["rmssd"] * 2 + rng.normal(0, 0.1, len(groups)))
    metadata = pd.DataFrame(
        {
            "participant_id": groups,
            "date": pd.date_range("2020-01-01", periods=len(groups)),
        }
    )
    lightweight = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("model", LinearRegression())]
    )

    metrics, folds, predictions = evaluate_grouped_model(
        features, target, groups, metadata, pipeline=lightweight, n_splits=3
    )

    assert (folds["participant_overlap_count"] == 0).all()
    for metric in ("mae", "rmse", "r2", "pearson_correlation", "spearman_correlation"):
        assert metric in metrics
    assert list(predictions.columns) == [
        "participant_id", "date", "actual_efficiency", "predicted_efficiency",
        "absolute_error", "fold",
    ]
