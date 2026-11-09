"""Inferencia con el modelo final (regresión logística T0). Solo carga y predice: nunca entrena.

Si falta el artefacto, falla con un mensaje que indica cómo generarlo
(``python -m src.infrastructure.ai_models.final_model``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.core.exceptions import InvalidPatientDataError, PredictionError
from src.core.interfaces.i_predictor_model import FactorContribution, IPredictorModel
from src.infrastructure.ai_models.final_model import DEFAULT_ARTIFACT
from src.infrastructure.ai_models.modeling_features import CATEGORICAL_FEATURES

_REQUIRED_KEYS = (
    "pipeline",
    "features",
    "priority_cutoffs",
    "priority_observed_rates",
    "reference_mean",
    "metadata",
)

FEATURE_LABELS: dict[str, str] = {
    "tipo_fisura": "Tipo de fisura",
    "edad_inicio_meses": "Edad al iniciar el tratamiento (meses)",
    "distancia_hospital_km": "Distancia al hospital (km)",
    "nivel_socioeconomico_ord": "Nivel socioeconómico (0 = bajo, 4 = alto)",
    "ingresos_mensuales_usd": "Ingresos mensuales (USD)",
    "apoyo_psicologico": "Recibe apoyo psicológico (1 = sí)",
    "dificultad_alimentacion_preop": "Dificultad de alimentación antes de la cirugía (1 = sí)",
}


class LogisticT0Predictor(IPredictorModel):
    """Adaptador del artefacto ``modelo_t0.joblib`` al puerto ``IPredictorModel``."""

    def __init__(self, artifact_path: Path | None = None) -> None:
        self._artifact_path = Path(artifact_path) if artifact_path else DEFAULT_ARTIFACT
        self._artifact: dict[str, Any] | None = None

    def load(self) -> dict[str, Any]:
        if self._artifact is not None:
            return self._artifact
        if not self._artifact_path.exists():
            raise PredictionError(
                f"No existe el modelo {self._artifact_path}. "
                "Genéralo con: python -m src.infrastructure.ai_models.final_model"
            )
        try:
            artifact = joblib.load(self._artifact_path)
        except Exception as exc:
            raise PredictionError(f"No se pudo cargar {self._artifact_path.name}: {exc}") from exc
        missing = [key for key in _REQUIRED_KEYS if not isinstance(artifact, dict) or key not in artifact]
        if missing:
            raise PredictionError(f"El artefacto {self._artifact_path.name} no tiene {missing}.")
        self._artifact = artifact
        return artifact

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.load()["features"])

    @property
    def priority_cutoffs(self) -> dict[str, float]:
        return dict(self.load()["priority_cutoffs"])

    @property
    def priority_observed_rates(self) -> dict[str, float]:
        return dict(self.load()["priority_observed_rates"])

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.load()["metadata"])

    def predict_dropout_probability(self, features: pd.DataFrame) -> np.ndarray:
        matrix = self._matrix(features)
        try:
            probabilities = self.load()["pipeline"].predict_proba(matrix)[:, 1]
        except Exception as exc:
            raise PredictionError(f"Fallo al inferir deserción: {exc}") from exc
        probabilities = np.asarray(probabilities, dtype=float)
        if not np.isfinite(probabilities).all():
            raise PredictionError("El modelo devolvió valores no numéricos.")
        return np.clip(probabilities, 0.0, 1.0)

    def explain_prediction(self, features: pd.DataFrame, top: int = 3) -> list[FactorContribution]:
        """Descomposición exacta en log-odds de la primera fila frente al paciente promedio.

        log-odds(paciente) = log-odds(paciente promedio) + suma de contribuciones.
        Las columnas one-hot de ``tipo_fisura`` se suman en una sola contribución.
        """
        contributions = self.contributions(features)
        row = self._matrix(features).iloc[0]
        ranked = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        return [
            FactorContribution(
                variable=column,
                label=FEATURE_LABELS.get(column, column),
                value=_plain(row[column]),
                contribution=round(float(value), 4),
            )
            for column, value in ranked[:top]
        ]

    def contributions(self, features: pd.DataFrame) -> dict[str, float]:
        """Contribución en log-odds de cada variable de entrada, para la primera fila."""
        artifact = self.load()
        pipeline = artifact["pipeline"]
        transformed = np.asarray(pipeline.named_steps["prep"].transform(self._matrix(features).iloc[[0]]))
        coefficients = pipeline.named_steps["clf"].coef_[0]
        per_column = (transformed[0] - artifact["reference_mean"]) * coefficients
        totals = {column: 0.0 for column in artifact["features"]}
        for name, value in zip(pipeline.named_steps["prep"].get_feature_names_out(), per_column):
            totals[_source_column(name, artifact["features"])] += float(value)
        return totals

    def reference_log_odds(self) -> float:
        """Log-odds del paciente promedio (punto de partida de la explicación)."""
        artifact = self.load()
        classifier = artifact["pipeline"].named_steps["clf"]
        return float(classifier.intercept_[0] + artifact["reference_mean"] @ classifier.coef_[0])

    def _matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        if features is None or features.empty:
            raise InvalidPatientDataError("No se recibieron características para predecir.")
        columns = list(self.load()["features"])
        missing = [column for column in columns if column not in features.columns]
        if missing:
            raise InvalidPatientDataError(f"Faltan columnas para el modelo T0: {missing}")
        matrix = features.loc[:, columns].copy()
        numeric = [column for column in columns if column not in CATEGORICAL_FEATURES]
        matrix[numeric] = matrix[numeric].apply(pd.to_numeric, errors="coerce")
        return matrix


def _source_column(transformed_name: str, columns: list[str]) -> str:
    """``num__distancia_hospital_km`` -> distancia; ``cat__tipo_fisura_Completa`` -> tipo_fisura."""
    name = transformed_name.split("__", 1)[1]
    if name in columns:
        return name
    for column in columns:
        if name.startswith(f"{column}_"):
            return column
    raise PredictionError(f"No se reconoce la columna transformada {transformed_name}.")


def _plain(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value
