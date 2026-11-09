"""Pruebas de la API HTTP con el modelo final T0 (artefacto versionado del repositorio)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from main import app
from src.infrastructure.ai_models.modeling_features import FEATURES_T0


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def test_prediction_returns_priority(client: TestClient) -> None:
    response = client.get("/api/predictions/PAC-2001")
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "PAC-2001"
    assert 0.0 <= data["dropout_probability"] <= 1.0
    assert data["priority_level"] in {"alta", "media", "habitual"}
    assert data["priority_message"]
    assert 0.0 < data["group_observed_dropout_rate"] < 1.0
    assert set(data["model_inputs"]) == set(FEATURES_T0)
    assert len(data["top_factors"]) == 3
    assert data["warnings"] == []
    assert data["scenario"] == "T0"
    assert data["model_version"]
    assert "risk_level" not in data
    assert "patient_snapshot" not in data


def test_prediction_post_matches_get(client: TestClient) -> None:
    by_get = client.get("/api/predictions/PAC-2002").json()
    by_post = client.post("/api/predictions", json={"patient_id": "PAC-2002"}).json()
    assert by_post["dropout_probability"] == by_get["dropout_probability"]


def test_prediction_v1_patient_is_not_in_modeling_cohort(client: TestClient) -> None:
    """El flujo predictivo solo usa la cohorte v2: un ID de v1 no existe para el modelo."""
    assert client.get("/api/predictions/PAC-1001").status_code == 404


def test_prediction_rejects_invalid_patient_id(client: TestClient) -> None:
    assert client.get("/api/predictions/ID%20con%20espacios").status_code == 422
    response = client.post("/api/predictions", json={"patient_id": ""})
    assert response.status_code == 422


def test_prediction_unknown_patient_is_404(client: TestClient) -> None:
    assert client.get("/api/predictions/PAC-NO-EXISTE").status_code == 404


def test_health_reports_model_metadata(client: TestClient) -> None:
    data = client.get("/api/health").json()
    assert data["model_scenario"] == "T0"
    assert data["model_version"]


def test_prediction_reports_model_metadata(client: TestClient) -> None:
    data = client.get("/api/predictions/PAC-2001").json()
    assert data["scenario"] == "T0"
    assert data["model_version"]


def test_analytics_unknown_query_is_422(client: TestClient) -> None:
    response = client.post("/api/analytics", json={"query": "clima de mañana"})
    assert response.status_code == 422
    assert "No se reconoció" in response.json()["detail"]


def test_health_suggests_v2_patients(client: TestClient) -> None:
    data = client.get("/api/health").json()
    assert data["sample_patient_ids"][0] == "PAC-2001"
