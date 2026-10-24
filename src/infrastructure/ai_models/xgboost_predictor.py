"""Adaptador XGBoost. Sustituye IPredictorModel (LSP) sin filtrar detalles al dominio."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.core.exceptions import PredictionError
from src.core.interfaces.i_predictor_model import IPredictorModel
from src.infrastructure.ai_models.feature_engineering import (
    FEATURE_COLUMNS,
    build_feature_frame,
)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - entorno mínimo
    XGBClassifier = None  # type: ignore[misc, assignment]


class XgboostPredictor(IPredictorModel):
    """Carga un pipeline serializado o entrena un XGBoost clínico de respaldo."""

    def __init__(
        self,
        model_path: Path,
        training_frame: pd.DataFrame | None = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._training_frame = training_frame
        self._pipeline: Pipeline | None = None

    def predict_dropout_probability(self, features: pd.DataFrame) -> np.ndarray:
        if features is None or features.empty:
            raise PredictionError("No se recibieron características para predecir.")

        try:
            matrix = build_feature_frame(features)
            pipeline = self._get_pipeline(reference=matrix)
            probabilities = pipeline.predict_proba(matrix)[:, 1]
            clipped = np.clip(probabilities.astype(float), 0.0, 1.0)
            if np.isnan(clipped).any():
                raise PredictionError("El modelo devolvió valores no numéricos.")
            return clipped
        except PredictionError:
            raise
        except Exception as exc:
            raise PredictionError(f"Fallo al inferir deserción: {exc}") from exc

    def _get_pipeline(self, reference: pd.DataFrame) -> Pipeline:
        if self._pipeline is not None:
            return self._pipeline

        if self._model_path.exists():
            try:
                loaded = joblib.load(self._model_path)
                if not hasattr(loaded, "predict_proba"):
                    raise PredictionError("El artefacto serializado no expone predict_proba.")
                self._pipeline = loaded
                return self._pipeline
            except PredictionError:
                raise
            except Exception as exc:
                raise PredictionError(
                    f"No se pudo cargar {self._model_path.name}: {exc}"
                ) from exc

        self._pipeline = self._fit_or_mock(reference)
        try:
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._pipeline, self._model_path)
        except Exception:
            pass
        return self._pipeline

    def _fit_or_mock(self, reference: pd.DataFrame) -> Pipeline:
        training = (
            build_feature_frame(self._training_frame)
            if self._training_frame is not None and not self._training_frame.empty
            else self._synthetic_training_frame(len(reference) or 64)
        )
        labels = self._extract_labels(training_length=len(training))
        pipeline = self._build_pipeline(labels)
        pipeline.fit(training, labels)
        return pipeline

    def _extract_labels(self, training_length: int) -> np.ndarray:
        if self._training_frame is not None and "Desertor" in self._training_frame.columns:
            labels = pd.to_numeric(self._training_frame["Desertor"], errors="coerce")
            labels = labels.fillna(0).astype(int).to_numpy()
            if len(labels) == training_length and len(np.unique(labels)) > 1:
                return labels
        features = (
            build_feature_frame(self._training_frame)
            if self._training_frame is not None
            else self._synthetic_training_frame(training_length)
        )
        return _heuristic_labels(features)

    def _build_pipeline(self, labels: np.ndarray) -> Pipeline:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[("numeric", numeric_transformer, list(FEATURE_COLUMNS))],
            remainder="drop",
        )
        estimator = self._build_estimator(labels)
        return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])

    def _build_estimator(self, labels: np.ndarray) -> object:
        if XGBClassifier is None:
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(max_iter=500, class_weight="balanced")

        positives = max(int(np.sum(labels == 1)), 1)
        negatives = max(int(np.sum(labels == 0)), 1)
        return XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=negatives / positives,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            random_state=42,
        )

    def _synthetic_training_frame(self, rows: int) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        n = max(rows, 64)
        return pd.DataFrame(
            {
                "inasistencias": rng.integers(0, 16, size=n),
                "distancia": rng.uniform(2.0, 180.0, size=n),
                "nivel_socioeconomico": rng.integers(0, 5, size=n),
                "cirugias_previas": rng.integers(0, 7, size=n),
                "ingresos_mensuales": rng.uniform(250.0, 4000.0, size=n),
                "apoyo_psicologico": rng.integers(0, 2, size=n),
                "edad_inicio_meses": rng.integers(1, 18, size=n),
                "ratio_asistencia_terapia": rng.uniform(0.2, 1.0, size=n),
                "complicaciones_quirurgicas": rng.integers(0, 3, size=n),
                "insuficiencia_velofaringea": rng.integers(0, 2, size=n),
                "grado_hipernasalidad": rng.integers(0, 4, size=n),
                "peso_z_score": rng.normal(-0.8, 1.0, size=n),
            }
        )


def _heuristic_labels(features: pd.DataFrame) -> np.ndarray:
    score = (
        0.18 * features["inasistencias"]
        + 0.012 * features["distancia"]
        - 0.35 * features["nivel_socioeconomico"]
        + 0.08 * features["cirugias_previas"]
        - 0.0002 * features["ingresos_mensuales"]
        - 0.55 * features["apoyo_psicologico"]
        - 0.9 * features["ratio_asistencia_terapia"]
        + 0.35 * features["complicaciones_quirurgicas"]
        + 0.25 * features["insuficiencia_velofaringea"]
    )
    probability = 1.0 / (1.0 + np.exp(-(score - 1.4)))
    return (probability >= 0.5).astype(int)
