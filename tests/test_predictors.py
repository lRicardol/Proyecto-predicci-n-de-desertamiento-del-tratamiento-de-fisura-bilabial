"""Pruebas del adaptador predictivo y del repositorio CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import PatientNotFoundError
from src.infrastructure.ai_models.xgboost_predictor import XgboostPredictor
from src.infrastructure.repositories.csv_patient_repository import CsvPatientRepository

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def test_repository_loads_real_raw_datasets() -> None:
    repository = CsvPatientRepository(RAW_DIR)
    frame = repository.get_all_patients_data()
    assert not frame.empty
    assert "ID_Paciente" in frame.columns
    assert "Desertor" in frame.columns
    assert "Citas_Perdidas_NoShow" in frame.columns
    patient = repository.get_patient_by_id("PAC-1001")
    assert patient["ID_Paciente"] == "PAC-1001"


def test_repository_unknown_patient() -> None:
    repository = CsvPatientRepository(RAW_DIR)
    with pytest.raises(PatientNotFoundError):
        repository.get_patient_by_id("PAC-NO-EXISTE")


def test_xgboost_predictor_returns_coherent_probabilities(tmp_path: Path) -> None:
    training = pd.DataFrame(
        {
            "Citas_Perdidas_NoShow": [1, 12, 2, 14],
            "Distancia_Hospital_KM": [8.0, 90.0, 12.0, 110.0],
            "Nivel_Socioeconomico": ["Alto", "Bajo", "Medio", "Bajo"],
            "Cirugias_Programadas": [2, 5, 3, 6],
            "Ingresos_Mensuales_USD": [3000, 400, 1800, 350],
            "Apoyo_Psicologico": [1, 0, 1, 0],
            "Desertor": [0, 1, 0, 1],
        }
    )
    predictor = XgboostPredictor(
        model_path=tmp_path / "model.joblib",
        training_frame=training,
    )
    high_risk = pd.DataFrame(
        {
            "Citas_Perdidas_NoShow": [15],
            "Distancia_Hospital_KM": [140.0],
            "Nivel_Socioeconomico": ["Bajo"],
            "Cirugias_Programadas": [6],
            "Ingresos_Mensuales_USD": [300],
            "Apoyo_Psicologico": [0],
        }
    )
    probabilities = predictor.predict_dropout_probability(high_risk)
    assert probabilities.shape == (1,)
    assert 0.0 <= float(probabilities[0]) <= 1.0
    assert np.isfinite(probabilities).all()
