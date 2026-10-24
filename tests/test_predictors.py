"""Pruebas del predictor XGBoost y del repositorio sobre el CSV procesado."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import PatientNotFoundError
from src.infrastructure.ai_models.xgboost_predictor import XgboostPredictor, train_xgboost
from src.infrastructure.repositories.csv_patient_repository import CsvPatientRepository

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "dataset_final_limpio.csv"


def test_repository_reads_processed_csv() -> None:
    repository = CsvPatientRepository(PROCESSED)
    frame = repository.get_all_patients_data()
    assert len(frame) == 3500
    assert "id_paciente" in frame.columns
    assert "desertor" in frame.columns
    patient = repository.get_patient_by_id("PAC-1001")
    assert patient["id_paciente"] == "PAC-1001"


def test_repository_unknown_patient() -> None:
    repository = CsvPatientRepository(PROCESSED)
    with pytest.raises(PatientNotFoundError):
        repository.get_patient_by_id("PAC-NO-EXISTE")


def test_xgboost_trains_and_predicts(tmp_path: Path) -> None:
    full = pd.read_csv(PROCESSED)
    frame = pd.concat(
        [
            full[full["desertor"] == 0].head(80),
            full[full["desertor"] == 1].head(120),
        ]
    ).reset_index(drop=True)
    model_path = tmp_path / "model.joblib"
    metrics = train_xgboost(frame=frame, model_path=model_path, test_size=0.25)
    assert model_path.exists()
    assert 0.0 < metrics["scale_pos_weight"] < 2.0
    predictor = XgboostPredictor(model_path=model_path, training_frame=frame)
    sample = frame.tail(1)
    probabilities = predictor.predict_dropout_probability(sample)
    assert probabilities.shape == (1,)
    assert 0.0 <= float(probabilities[0]) <= 1.0
    assert np.isfinite(probabilities).all()
