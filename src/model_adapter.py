"""Runtime adapter for a persisted XGBoost efficiency model bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.model_explainer import explain_xgboost_prediction
from src.preprocessing import MODEL_FEATURES, engineer_single_record_features


class EfficiencyModelLoadError(RuntimeError):
    """Raised when a persisted efficiency model cannot be used."""


class XGBoostEfficiencyAdapter:
    """Load once and provide prediction and SHAP explanation operations."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained efficiency model not found: {self.model_path}")
        try:
            bundle = joblib.load(self.model_path)
        except Exception as exc:
            raise EfficiencyModelLoadError(f"Unable to load model bundle: {exc}") from exc
        required = {"pipeline", "feature_names", "target_name", "metrics"}
        if not isinstance(bundle, dict) or not required.issubset(bundle):
            raise EfficiencyModelLoadError("Model bundle is malformed or incomplete.")
        if bundle["feature_names"] != MODEL_FEATURES:
            raise EfficiencyModelLoadError(
                "Saved feature order does not match the runtime MODEL_FEATURES order."
            )
        if bundle["target_name"] != "efficiency":
            raise EfficiencyModelLoadError("Model bundle target must be 'efficiency'.")
        pipeline = bundle["pipeline"]
        if not hasattr(pipeline, "predict") or not hasattr(pipeline, "named_steps"):
            raise EfficiencyModelLoadError("Model bundle does not contain a fitted sklearn pipeline.")
        self.bundle: dict[str, Any] = bundle
        self.pipeline = pipeline
        self.feature_names = list(bundle["feature_names"])
        self.metrics = dict(bundle["metrics"])

    @property
    def model_mae(self) -> float | None:
        """Return persisted cross-validation MAE when available."""
        value = self.metrics.get("mae_mean", self.metrics.get("mae"))
        return float(value) if value is not None else None

    def _feature_frame(self, raw_record: dict[str, object]) -> tuple[dict[str, float], pd.DataFrame]:
        engineered = engineer_single_record_features(raw_record)
        if list(engineered) != self.feature_names:
            raise EfficiencyModelLoadError("Generated feature order does not match saved feature order.")
        return engineered, pd.DataFrame([engineered], columns=self.feature_names)

    def predict_efficiency(self, raw_record: dict[str, object]) -> float:
        """Predict efficiency and return a native Python float."""
        _, frame = self._feature_frame(raw_record)
        prediction = np.asarray(self.pipeline.predict(frame)).reshape(-1)
        if prediction.size != 1 or not np.isfinite(prediction[0]):
            raise EfficiencyModelLoadError("Model returned an invalid efficiency prediction.")
        return float(prediction[0])

    def explain_prediction(self, raw_record: dict[str, object]) -> dict[str, object]:
        """Explain one prediction using the fitted imputer and XGBoost model."""
        engineered, frame = self._feature_frame(raw_record)
        try:
            imputer = self.pipeline.named_steps["imputer"]
            model = self.pipeline.named_steps["model"]
            transformed = imputer.transform(frame)
        except (KeyError, AttributeError) as exc:
            raise EfficiencyModelLoadError(
                "Pipeline must contain fitted 'imputer' and 'model' steps."
            ) from exc
        prediction = float(np.asarray(model.predict(transformed)).reshape(-1)[0])
        return explain_xgboost_prediction(
            model, transformed, engineered, self.feature_names, prediction
        )
