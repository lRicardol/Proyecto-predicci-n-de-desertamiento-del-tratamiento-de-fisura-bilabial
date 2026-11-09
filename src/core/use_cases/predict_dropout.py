"""Caso de uso: probabilidad de deserción y prioridad de seguimiento (alta / media / habitual).

La salida no es un "riesgo Bajo/Medio/Alto": con una deserción de base del 71 %, hasta el grupo
de menor puntaje abandona cerca de la mitad de las veces (reports/interpretabilidad.md §6).
Por eso se habla de prioridad de seguimiento, definida por puntos de corte del puntaje.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from src.core.exceptions import InvalidPatientDataError, PredictionError
from src.core.interfaces.i_patient_repository import IPatientRepository
from src.core.interfaces.i_predictor_model import FactorContribution, IPredictorModel

PRIORITY_LEVELS: tuple[str, ...] = ("alta", "media", "habitual")

PRIORITY_MESSAGES: dict[str, str] = {
    "alta": (
        "Este paciente está entre los de mayor probabilidad estimada de abandono. "
        "Se sugiere priorizar un contacto de seguimiento. No es un diagnóstico ni un pronóstico."
    ),
    "media": "Probabilidad estimada intermedia. Seguimiento según la rutina del servicio.",
    "habitual": (
        "Probabilidad estimada menor que la del resto de pacientes, aunque no es baja en términos "
        "absolutos. Mantener el seguimiento habitual."
    ),
}

DISCLAIMER: str = (
    "Herramienta de apoyo a la gestión del seguimiento, entrenada con datos sintéticos. "
    "La decisión clínica y de contacto la toma siempre el personal de salud con la historia completa."
)


@dataclass(frozen=True)
class PriorityPolicy:
    """Cortes del puntaje: >= ``high_cutoff`` es alta; < ``low_cutoff`` es habitual."""

    high_cutoff: float
    low_cutoff: float
    observed_rates: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_cutoff < self.high_cutoff <= 1.0:
            raise ValueError("Los cortes de prioridad son inconsistentes.")

    def classify(self, probability: float) -> str:
        if probability >= self.high_cutoff:
            return "alta"
        if probability < self.low_cutoff:
            return "habitual"
        return "media"


@dataclass(frozen=True)
class DropoutPrediction:
    patient_id: str
    dropout_probability: float
    priority_level: str
    priority_message: str
    group_observed_rate: float | None
    top_factors: list[FactorContribution]
    model_inputs: dict[str, Any]
    prediction_date: str
    warnings: list[str]
    disclaimer: str
    patient_snapshot: dict[str, Any]


class PredictDropoutUseCase:
    """Orquesta repositorio + modelo. No conoce CSV, scikit-learn ni FastAPI."""

    def __init__(
        self,
        patient_repository: IPatientRepository,
        predictor_model: IPredictorModel,
        priority_policy: PriorityPolicy,
        training_cohort: str | None = "v2",
        today: Callable[[], date] = date.today,
    ) -> None:
        self._patient_repository = patient_repository
        self._predictor_model = predictor_model
        self._priority_policy = priority_policy
        self._training_cohort = training_cohort
        self._today = today

    def execute(self, patient_id: str) -> DropoutPrediction:
        record = self._patient_repository.get_patient_by_id(patient_id)
        features = self._format_features(record)
        probability = self._extract_scalar(self._predictor_model.predict_dropout_probability(features))
        level = self._priority_policy.classify(probability)
        resolved_id = str(record.get("id_paciente") or record.get("ID_Paciente") or patient_id)
        inputs = {name: record.get(name) for name in self._predictor_model.feature_names}
        return DropoutPrediction(
            patient_id=resolved_id,
            dropout_probability=probability,
            priority_level=level,
            priority_message=PRIORITY_MESSAGES[level],
            group_observed_rate=self._priority_policy.observed_rates.get(level),
            top_factors=self._predictor_model.explain_prediction(features),
            model_inputs=inputs,
            prediction_date=self._today().isoformat(),
            warnings=self._warnings(record),
            disclaimer=DISCLAIMER,
            patient_snapshot=record,
        )

    def _warnings(self, record: dict[str, Any]) -> list[str]:
        cohort = record.get("cohorte")
        if self._training_cohort and cohort and cohort != self._training_cohort:
            return [
                f"El paciente pertenece a la cohorte {cohort}; el modelo se entrenó con la cohorte "
                f"{self._training_cohort}. La estimación es solo orientativa."
            ]
        return []

    def _format_features(self, record: dict[str, Any]) -> pd.DataFrame:
        if not record:
            raise InvalidPatientDataError("La ficha del paciente está vacía.")
        return pd.DataFrame([record])

    def _extract_scalar(self, probabilities: Any) -> float:
        try:
            value = float(probabilities[0])
        except Exception as exc:
            raise PredictionError("El modelo no devolvió un vector de probabilidades.") from exc
        if not 0.0 <= value <= 1.0:
            raise PredictionError(f"Probabilidad fuera de rango: {value}")
        return round(value, 4)
