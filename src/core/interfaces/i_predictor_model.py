"""Contrato atómico del modelo de deserción (ISP: no incluye analítica conversacional)."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FactorContribution:
    """Aporte de una variable a la predicción, comparado con el paciente promedio.

    ``contribution`` está en log-odds: positivo sube la probabilidad estimada, negativo la baja.
    Es una asociación del modelo, no una causa.
    """

    variable: str
    label: str
    value: object
    contribution: float


class IPredictorModel(ABC):
    """Puerto Strategy: cualquier modelo sustituible debe devolver P(deserción)."""

    @abstractmethod
    def predict_dropout_probability(
        self, features: Sequence[Mapping[str, Any]]
    ) -> Sequence[float]:
        """Retorna probabilidades en el intervalo [0.0, 1.0], una por fila."""

    def explain_prediction(
        self, features: Sequence[Mapping[str, Any]], top: int = 3
    ) -> list[FactorContribution]:
        """Variables que más pesan en la predicción de la primera fila. Opcional: vacío por defecto."""
        return []

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Columnas que el modelo usa como entrada. Vacío si el modelo no lo declara."""
        return ()
