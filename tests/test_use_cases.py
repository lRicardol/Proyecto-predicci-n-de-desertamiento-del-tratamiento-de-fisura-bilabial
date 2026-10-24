"""Pruebas de casos de uso con dobles de prueba (DIP / LSP)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import AnalyticsError, PatientNotFoundError
from src.core.interfaces.i_llm_analyst import ILLMAnalyst
from src.core.interfaces.i_patient_repository import IPatientRepository
from src.core.interfaces.i_predictor_model import IPredictorModel
from src.core.use_cases.generate_analytics import GenerateAnalyticsUseCase
from src.core.use_cases.predict_dropout import PredictDropoutUseCase
from src.infrastructure.llm_agents.langchain_analyst import LangchainAnalyst


class FakePatientRepository(IPatientRepository):
    def __init__(self, records: dict[str, dict[str, Any]]) -> None:
        self._records = records

    def get_all_patients_data(self) -> pd.DataFrame:
        return pd.DataFrame(self._records.values())

    def get_patient_by_id(self, patient_id: str) -> dict[str, Any]:
        if patient_id not in self._records:
            raise PatientNotFoundError(patient_id)
        return self._records[patient_id]


class FakePredictor(IPredictorModel):
    def __init__(self, probability: float) -> None:
        self._probability = probability

    def predict_dropout_probability(self, features: pd.DataFrame) -> np.ndarray:
        assert not features.empty
        return np.array([self._probability], dtype=float)


class FakeAnalyst(ILLMAnalyst):
    def analyze_data_and_plot(
        self,
        query: str,
        data_path: str,
        output_image_path: str,
    ) -> str:
        Path(output_image_path).write_bytes(b"png-fake")
        return f"ok:{query}:{data_path}"


def test_predict_dropout_classifies_high_risk() -> None:
    use_case = PredictDropoutUseCase(
        patient_repository=FakePatientRepository(
            {
                "PAC-1001": {
                    "ID_Paciente": "PAC-1001",
                    "Citas_Perdidas_NoShow": 12,
                    "Distancia_Hospital_KM": 80,
                    "Nivel_Socioeconomico": "Bajo",
                }
            }
        ),
        predictor_model=FakePredictor(0.91),
    )
    result = use_case.execute("PAC-1001")
    assert result.risk_level == "Alto"
    assert result.dropout_probability == 0.91


def test_predict_dropout_classifies_low_risk() -> None:
    use_case = PredictDropoutUseCase(
        patient_repository=FakePatientRepository(
            {"PAC-1000": {"ID_Paciente": "PAC-1000", "Citas_Perdidas_NoShow": 1}}
        ),
        predictor_model=FakePredictor(0.12),
    )
    result = use_case.execute("PAC-1000")
    assert result.risk_level == "Bajo"


def test_generate_analytics_persists_chart(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("ID_Paciente,Desertor\nPAC-1,1\n", encoding="utf-8")
    charts = tmp_path / "charts"
    use_case = GenerateAnalyticsUseCase(
        llm_analyst=FakeAnalyst(),
        dataset_path=dataset,
        charts_dir=charts,
    )
    result = use_case.execute("Grafica la relación entre inasistencias y deserción")
    assert Path(result.image_path).exists()
    assert result.image_url.startswith("/static/generated_charts/")


def test_generate_analytics_rejects_empty_query(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("ID_Paciente,Desertor\nPAC-1,1\n", encoding="utf-8")
    use_case = GenerateAnalyticsUseCase(
        llm_analyst=FakeAnalyst(),
        dataset_path=dataset,
        charts_dir=tmp_path / "charts",
    )
    with pytest.raises(AnalyticsError):
        use_case.execute("  ")


def test_local_langchain_analyst_writes_png(tmp_path: Path) -> None:
    csv_path = tmp_path / "dataset_desercion_fisura_bilabial.csv"
    pd.DataFrame(
        {
            "ID_Paciente": ["PAC-1", "PAC-2"],
            "Citas_Perdidas_NoShow": [2, 11],
            "Desertor": [0, 1],
        }
    ).to_csv(csv_path, index=False)
    output = tmp_path / "chart.png"
    analyst = LangchainAnalyst(api_key="")
    text = analyst.analyze_data_and_plot(
        query="Grafica la relación entre inasistencias y deserción",
        data_path=str(csv_path),
        output_image_path=str(output),
    )
    assert output.exists()
    assert output.stat().st_size > 0
    assert "inasistencias" in text.lower() or "no-show" in text.lower() or "no-shows" in text.lower()
