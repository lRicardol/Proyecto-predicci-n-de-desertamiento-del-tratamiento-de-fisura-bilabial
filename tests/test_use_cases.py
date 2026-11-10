"""Pruebas de casos de uso con dobles de prueba (DIP / LSP)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import pytest

from src.core.exceptions import (
    AnalyticsError,
    InvalidPatientDataError,
    PatientNotFoundError,
    PredictionError,
)
from src.core.interfaces.i_llm_analyst import ILLMAnalyst
from src.core.interfaces.i_patient_repository import IPatientRepository
from src.core.interfaces.i_predictor_model import FactorContribution, IPredictorModel
from src.core.use_cases.generate_analytics import GenerateAnalyticsUseCase
from src.core.use_cases.predict_dropout import (
    PRIORITY_MESSAGES,
    PredictDropoutUseCase,
    PriorityPolicy,
)
from src.infrastructure.llm_agents.local_analyst import LocalStatisticalAnalyst


class FakePatientRepository(IPatientRepository):
    def __init__(self, records: dict[str, dict[str, Any]]) -> None:
        self._records = records

    def get_patient_by_id(self, patient_id: str) -> dict[str, Any]:
        if patient_id not in self._records:
            raise PatientNotFoundError(patient_id)
        return self._records[patient_id]

    def count_patients(self) -> int:
        return len(self._records)

    def list_patient_ids(self, limit: int | None = None) -> list[str]:
        identifiers = list(self._records)
        return identifiers if limit is None else identifiers[:limit]


FACTOR = FactorContribution(
    variable="distancia_hospital_km",
    label="Distancia al hospital (km)",
    value=120.0,
    contribution=0.66,
)


class FakePredictor(IPredictorModel):
    def __init__(self, probability: float) -> None:
        self._probability = probability

    def predict_dropout_probability(
        self, features: Sequence[Mapping[str, Any]]
    ) -> Sequence[float]:
        assert len(features) == 1
        return [self._probability]

    def explain_prediction(
        self, features: Sequence[Mapping[str, Any]], top: int = 3
    ) -> list[FactorContribution]:
        return [FACTOR]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("distancia_hospital_km",)


class FakeAnalyst(ILLMAnalyst):
    def analyze_data_and_plot(
        self,
        query: str,
        data_path: str,
        output_image_path: str,
    ) -> str:
        Path(output_image_path).write_bytes(b"png-fake")
        return f"ok:{query}:{data_path}"


POLICY = PriorityPolicy(
    high_cutoff=0.83,
    low_cutoff=0.62,
    observed_rates={"alta": 0.9, "media": 0.77, "habitual": 0.49},
)


def _use_case(record: dict[str, Any], probability: float, **kwargs: Any) -> PredictDropoutUseCase:
    return PredictDropoutUseCase(
        patient_repository=FakePatientRepository({record["id_paciente"]: record}),
        predictor_model=FakePredictor(probability),
        priority_policy=POLICY,
        today=lambda: date(2026, 11, 9),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("probability", "level"),
    [(0.91, "alta"), (0.83, "alta"), (0.75, "media"), (0.62, "media"), (0.40, "habitual")],
)
def test_predict_dropout_assigns_priority(probability: float, level: str) -> None:
    record = {"id_paciente": "PAC-2001", "cohorte": "v2", "distancia_hospital_km": 50.0}
    result = _use_case(record, probability).execute("PAC-2001")
    assert result.priority_level == level
    assert result.dropout_probability == probability
    assert result.priority_message == PRIORITY_MESSAGES[level]
    assert result.group_observed_rate == POLICY.observed_rates[level]


def test_predict_dropout_reports_factors_inputs_and_date() -> None:
    record = {"id_paciente": "PAC-2001", "cohorte": "v2", "distancia_hospital_km": 120.0}
    result = _use_case(record, 0.9, scenario="T0", model_version="t0-logistica-1").execute("PAC-2001")
    assert result.top_factors == [FACTOR]
    assert result.model_inputs == {"distancia_hospital_km": 120.0}
    assert result.prediction_date == "2026-11-09"
    assert result.warnings == []
    assert result.scenario == "T0"
    assert result.model_version == "t0-logistica-1"
    assert "datos sintéticos" in result.disclaimer


def test_predict_dropout_warns_outside_training_cohort() -> None:
    record = {"id_paciente": "PAC-1001", "cohorte": "v1", "distancia_hospital_km": 10.0}
    result = _use_case(record, 0.9).execute("PAC-1001")
    assert len(result.warnings) == 1
    assert "v1" in result.warnings[0]


def test_predict_dropout_unknown_patient() -> None:
    record = {"id_paciente": "PAC-2001", "distancia_hospital_km": 10.0}
    with pytest.raises(PatientNotFoundError):
        _use_case(record, 0.5).execute("PAC-9999")


def test_predict_dropout_rejects_missing_features() -> None:
    record = {"id_paciente": "PAC-2001", "cohorte": "v2"}
    with pytest.raises(InvalidPatientDataError, match="Faltan características"):
        _use_case(record, 0.5).execute("PAC-2001")


def test_predict_dropout_rejects_out_of_range_probability() -> None:
    record = {"id_paciente": "PAC-2001", "distancia_hospital_km": 10.0}
    with pytest.raises(PredictionError):
        _use_case(record, 1.5).execute("PAC-2001")


@pytest.mark.parametrize(("high", "low"), [(0.5, 0.6), (0.5, 0.5), (1.2, 0.3)])
def test_priority_policy_rejects_inconsistent_cutoffs(high: float, low: float) -> None:
    with pytest.raises(ValueError):
        PriorityPolicy(high_cutoff=high, low_cutoff=low)


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


def test_generate_analytics_uses_unique_chart_names(tmp_path: Path) -> None:
    """Dos consultas seguidas no pisan el mismo PNG (sin chart_output.png fijo)."""
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("ID_Paciente,Desertor\nPAC-1,1\n", encoding="utf-8")
    use_case = GenerateAnalyticsUseCase(
        llm_analyst=FakeAnalyst(),
        dataset_path=dataset,
        charts_dir=tmp_path / "charts",
    )
    first = use_case.execute("consulta uno")
    second = use_case.execute("consulta dos")
    assert Path(first.image_path).name != Path(second.image_path).name
    assert Path(first.image_path).exists() and Path(second.image_path).exists()


def test_local_analyst_unknown_query_fails_with_examples(tmp_path: Path) -> None:
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame({"id_paciente": ["PAC-1"], "desertor": [1]}).to_csv(csv_path, index=False)
    analyst = LocalStatisticalAnalyst()
    with pytest.raises(AnalyticsError, match="No se reconoció"):
        analyst.analyze_data_and_plot(
            query="¿Cuál es el clima de mañana?",
            data_path=str(csv_path),
            output_image_path=str(tmp_path / "chart.png"),
        )


def test_local_analyst_group_comparison(tmp_path: Path) -> None:
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "id_paciente": [f"PAC-{i}" for i in range(6)],
            "tipo_fisura": ["Completa"] * 3 + ["Incompleta"] * 3,
            "desertor": [1, 1, 0, 0, 0, 1],
        }
    ).to_csv(csv_path, index=False)
    output = tmp_path / "groups.png"
    text = LocalStatisticalAnalyst().analyze_data_and_plot(
        query="Compara las tasas de deserción entre grupos",
        data_path=str(csv_path),
        output_image_path=str(output),
    )
    assert output.exists()
    assert "asociaci" in text.lower()


def test_local_analyst_writes_png(tmp_path: Path) -> None:
    csv_path = tmp_path / "dataset_final_limpio.csv"
    pd.DataFrame(
        {
            "id_paciente": ["PAC-1", "PAC-2"],
            "citas_perdidas_noshow": [2, 11],
            "desertor": [0, 1],
            "cohorte": ["v1", "v1"],
        }
    ).to_csv(csv_path, index=False)
    output = tmp_path / "chart_output.png"
    analyst = LocalStatisticalAnalyst()
    text = analyst.analyze_data_and_plot(
        query="Grafica la relación entre inasistencias y deserción",
        data_path=str(csv_path),
        output_image_path=str(output),
    )
    assert output.exists()
    assert output.stat().st_size > 0
    assert "inasistencias" in text.lower() or "no-show" in text.lower()


def test_predict_scenario_does_not_use_repository() -> None:
    record = {"id_paciente": "PAC-2001", "distancia_hospital_km": 10.0}
    result = _use_case(record, 0.9).execute_scenario({"distancia_hospital_km": 120.0})
    assert result.patient_id == "escenario"
    assert result.model_inputs == {"distancia_hospital_km": 120.0}
    assert result.priority_level == "alta"
    assert result.warnings == []


def test_predict_scenario_requires_model_features() -> None:
    record = {"id_paciente": "PAC-2001", "distancia_hospital_km": 10.0}
    with pytest.raises(InvalidPatientDataError):
        _use_case(record, 0.9).execute_scenario({"otra_variable": 1})


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Comparar grupos de deserción por tipo de fisura", "tipo_fisura"),
        ("Comparar grupos de deserción por nivel socioeconómico", "nivel_socioeconomico"),
        ("Comparar grupos de deserción por cohorte", "cohorte"),
        ("Compara las tasas de deserción entre grupos", "tipo_fisura"),
    ],
)
def test_local_analyst_group_comparison_uses_requested_variable(
    tmp_path: Path, query: str, expected: str
) -> None:
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "id_paciente": [f"PAC-{i}" for i in range(8)],
            "tipo_fisura": ["Completa", "Incompleta"] * 4,
            "nivel_socioeconomico": ["Bajo", "Alto"] * 4,
            "cohorte": ["v1"] * 4 + ["v2"] * 4,
            "desertor": [1, 0, 1, 1, 0, 1, 0, 1],
        }
    ).to_csv(csv_path, index=False)
    text = LocalStatisticalAnalyst().analyze_data_and_plot(
        query=query, data_path=str(csv_path), output_image_path=str(tmp_path / "g.png")
    )
    assert f"por {expected}" in text


def test_local_analyst_group_comparison_missing_variable_fails(tmp_path: Path) -> None:
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame({"tipo_fisura": ["Completa", "Incompleta"], "desertor": [1, 0]}).to_csv(csv_path, index=False)
    with pytest.raises(AnalyticsError, match="cohorte"):
        LocalStatisticalAnalyst().analyze_data_and_plot(
            query="Comparar grupos de deserción por cohorte",
            data_path=str(csv_path),
            output_image_path=str(tmp_path / "g.png"),
        )
