"""Contratos atómicos del dominio (ISP + DIP)."""

from src.core.interfaces.i_llm_analyst import ILLMAnalyst
from src.core.interfaces.i_patient_repository import IPatientRepository
from src.core.interfaces.i_predictor_model import IPredictorModel

__all__ = ["ILLMAnalyst", "IPatientRepository", "IPredictorModel"]
