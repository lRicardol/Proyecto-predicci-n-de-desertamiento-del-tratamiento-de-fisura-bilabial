"""Contrato de acceso a datos de pacientes. Sin lógica de predicción ni de LLM."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class IPatientRepository(ABC):
    """Puerto de persistencia para fichas clínicas y socioeconómicas."""

    @abstractmethod
    def get_all_patients_data(self) -> pd.DataFrame:
        """Devuelve el conjunto consolidado de pacientes (una fila por ID)."""

    @abstractmethod
    def get_patient_by_id(self, patient_id: str) -> dict[str, Any]:
        """Devuelve un paciente como diccionario. Debe fallar si el ID no existe."""
