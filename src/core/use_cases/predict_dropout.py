"""Caso de uso: probabilidad de deserción y etiqueta de riesgo Bajo/Medio/Alto."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.core.exceptions import InvalidPatientDataError, PredictionError
from src.core.interfaces.i_patient_repository import IPatientRepository
from src.core.interfaces.i_predictor_model import IPredictorModel


@dataclass(frozen=True)
class DropoutPrediction:
    patient_id: str
    dropout_probability: float
    risk_level: str
    patient_snapshot: dict[str, Any]


class PredictDropoutUseCase:
    """Orquesta repositorio + modelo. No conoce CSV, XGBoost ni FastAPI."""

    def __init__(
        self,
        patient_repository: IPatientRepository,
        predictor_model: IPredictorModel,
        low_risk_threshold: float = 0.33,
        high_risk_threshold: float = 0.66,
    ) -> None:
        if not 0.0 <= low_risk_threshold < high_risk_threshold <= 1.0:
            raise ValueError("Los umbrales de riesgo son inconsistentes.")
        self._patient_repository = patient_repository
        self._predictor_model = predictor_model
        self._low_risk_threshold = low_risk_threshold
        self._high_risk_threshold = high_risk_threshold

    def execute(self, patient_id: str) -> DropoutPrediction:
        record = self._patient_repository.get_patient_by_id(patient_id)
        features = self._format_features(record)
        probabilities = self._predictor_model.predict_dropout_probability(features)
        probability = self._extract_scalar(probabilities)
        resolved_id = str(record.get("id_paciente") or record.get("ID_Paciente") or patient_id)
        return DropoutPrediction(
            patient_id=resolved_id,
            dropout_probability=probability,
            risk_level=self._classify_risk(probability),
            patient_snapshot=record,
        )

    def _format_features(self, record: dict[str, Any]) -> pd.DataFrame:
        if not record:
            raise InvalidPatientDataError("La ficha del paciente está vacía.")
        frame = pd.DataFrame([record])
        if frame.empty:
            raise InvalidPatientDataError("No se pudieron tabular las características.")
        return frame

    def _extract_scalar(self, probabilities: Any) -> float:
        try:
            value = float(probabilities[0])
        except Exception as exc:
            raise PredictionError("El modelo no devolvió un vector de probabilidades.") from exc
        if not 0.0 <= value <= 1.0:
            raise PredictionError(f"Probabilidad fuera de rango: {value}")
        return round(value, 4)

    def _classify_risk(self, probability: float) -> str:
        if probability < self._low_risk_threshold:
            return "Bajo"
        if probability < self._high_risk_threshold:
            return "Medio"
        return "Alto"
