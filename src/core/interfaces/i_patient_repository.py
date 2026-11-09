"""Contrato de acceso a datos de pacientes. Sin lógica de predicción ni de analítica."""

from abc import ABC, abstractmethod
from typing import Any


class IPatientRepository(ABC):
    """Puerto de persistencia. Devuelve diccionarios, no marcos de datos."""

    @abstractmethod
    def get_patient_by_id(self, patient_id: str) -> dict[str, Any]:
        """Devuelve un paciente como diccionario. Debe fallar si el ID no existe."""

    @abstractmethod
    def count_patients(self) -> int:
        """Número de pacientes con identificador único."""

    @abstractmethod
    def list_patient_ids(self, limit: int | None = None) -> list[str]:
        """Identificadores estables, en el orden del archivo."""
