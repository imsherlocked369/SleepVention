"""Train and participant-group validate the XGBoost efficiency regressor."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from src.preprocessing import (
    MODEL_FEATURES,
    TARGET_NAME,
    load_oura_workbook,
    prepare_training_data,
)

LOGGER = logging.getLogger(__name__)


def build_training_pipeline() -> Pipeline:
    """Build the specified deterministic XGBoost regression pipeline."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is required for training. Install dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.03,
                    max_depth=5,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _safe_correlation(function: Callable, actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2 or np.unique(actual).size < 2 or np.unique(predicted).size < 2:
        return float("nan")
    return float(function(actual, predicted).statistic)


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    participant_count: int,
) -> dict[str, float | int]:
    """Calculate required regression and dataset summary metrics."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "pearson_correlation": _safe_correlation(pearsonr, actual, predicted),
        "spearman_correlation": _safe_correlation(spearmanr, actual, predicted),
        "number_of_rows": int(len(actual)),
        "number_of_participants": int(participant_count),
        "mean_actual_efficiency": float(np.mean(actual)),
        "mean_predicted_efficiency": float(np.mean(predicted)),
    }


def evaluate_grouped_model(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    metadata: pd.DataFrame,
    pipeline: Pipeline | None = None,
    n_splits: int = 5,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Evaluate with participant-disjoint GroupKFold splits."""
    unique_groups = groups.nunique()
    if unique_groups < n_splits:
        raise ValueError(
            f"Grouped validation requires at least {n_splits} participants; found {unique_groups}."
        )
    template = pipeline or build_training_pipeline()
    folds = GroupKFold(n_splits=n_splits)
    fold_metrics: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold_number, (train_index, test_index) in enumerate(
        folds.split(features, target, groups), start=1
    ):
        train_participants = set(groups.iloc[train_index])
        test_participants = set(groups.iloc[test_index])
        overlap = train_participants & test_participants
        if overlap:
            raise RuntimeError(
                f"Participant leakage detected in fold {fold_number}: {sorted(overlap)}"
            )

        fitted = clone(template)
        fitted.fit(features.iloc[train_index], target.iloc[train_index])
        predicted = np.asarray(fitted.predict(features.iloc[test_index]), dtype=float)
        actual = target.iloc[test_index].to_numpy(dtype=float)
        metrics = calculate_metrics(actual, predicted, len(test_participants))
        metrics["fold"] = fold_number
        metrics["participant_overlap_count"] = 0
        fold_metrics.append(metrics)

        fold_output = metadata.iloc[test_index].copy()
        fold_output["actual_efficiency"] = actual
        fold_output["predicted_efficiency"] = predicted
        fold_output["absolute_error"] = np.abs(actual - predicted)
        fold_output["fold"] = fold_number
        prediction_frames.append(fold_output)

    predictions = pd.concat(prediction_frames).sort_index()
    fold_frame = pd.DataFrame(fold_metrics)
    summary = calculate_metrics(
        predictions["actual_efficiency"].to_numpy(),
        predictions["predicted_efficiency"].to_numpy(),
        groups.nunique(),
    )
    for metric in ("mae", "rmse", "r2", "pearson_correlation", "spearman_correlation"):
        summary[f"{metric}_mean"] = float(fold_frame[metric].mean())
        summary[f"{metric}_std"] = float(fold_frame[metric].std(ddof=0))
    summary["validation"] = "participant-grouped GroupKFold"
    summary["n_splits"] = n_splits
    return summary, fold_frame, predictions


def save_evaluation_outputs(
    output_directory: Path,
    metrics: dict[str, object],
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Persist evaluation tables and diagnostic plots."""
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "xgboost_evaluation.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8"
    )
    fold_metrics.to_csv(output_directory / "xgboost_fold_metrics.csv", index=False)
    required = [
        "participant_id", "date", "actual_efficiency", "predicted_efficiency",
        "absolute_error", "fold",
    ]
    predictions.loc[:, required].to_csv(
        output_directory / "xgboost_test_predictions.csv", index=False
    )

    actual = predictions["actual_efficiency"]
    predicted = predictions["predicted_efficiency"]
    residuals = actual - predicted

    plt.figure()
    plt.scatter(actual, predicted, alpha=0.6)
    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    plt.plot(limits, limits, "--", color="black")
    plt.xlabel("Actual efficiency")
    plt.ylabel("Predicted efficiency")
    plt.tight_layout()
    plt.savefig(output_directory / "actual_vs_predicted.png", dpi=150)
    plt.close()

    plt.figure()
    plt.scatter(predicted, residuals, alpha=0.6)
    plt.axhline(0, linestyle="--", color="black")
    plt.xlabel("Predicted efficiency")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(output_directory / "residual_plot.png", dpi=150)
    plt.close()

    plt.figure()
    plt.hist(predictions["absolute_error"], bins=20)
    plt.xlabel("Absolute error")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_directory / "error_distribution.png", dpi=150)
    plt.close()


def train_and_save(
    data_path: Path,
    sheet_name: str,
    output_model: Path,
    output_directory: Path,
) -> None:
    """Run grouped evaluation, fit all rows, and persist the production bundle."""
    raw = load_oura_workbook(data_path, sheet_name)
    features, target, groups, metadata = prepare_training_data(raw)
    metrics, fold_metrics, predictions = evaluate_grouped_model(
        features, target, groups, metadata
    )
    save_evaluation_outputs(output_directory, metrics, fold_metrics, predictions)

    pipeline = build_training_pipeline()
    pipeline.fit(features, target)
    bundle = {
        "pipeline": pipeline,
        "feature_names": MODEL_FEATURES,
        "target_name": TARGET_NAME,
        "metrics": metrics,
        "model_version": "1.0.0",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(features)),
        "training_participants": int(groups.nunique()),
    }
    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_model)
    LOGGER.info("Saved trained model bundle to %s", output_model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sheet", default="in")
    parser.add_argument(
        "--output-model", type=Path,
        default=Path("models/xgboost_efficiency_model.joblib"),
    )
    parser.add_argument("--output-directory", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    train_and_save(args.data, args.sheet, args.output_model, args.output_directory)


if __name__ == "__main__":
    main()
