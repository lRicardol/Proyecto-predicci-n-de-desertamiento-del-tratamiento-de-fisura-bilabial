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
    assert "risk_level" not in data


def test_prediction_post_matches_get(client: TestClient) -> None:
    by_get = client.get("/api/predictions/PAC-2002").json()
    by_post = client.post("/api/predictions", json={"patient_id": "PAC-2002"}).json()
    assert by_post["dropout_probability"] == by_get["dropout_probability"]


def test_prediction_warns_for_v1_patient(client: TestClient) -> None:
    data = client.get("/api/predictions/PAC-1001").json()
    assert any("v1" in warning for warning in data["warnings"])


def test_prediction_unknown_patient_is_404(client: TestClient) -> None:
    assert client.get("/api/predictions/PAC-NO-EXISTE").status_code == 404


def test_health_suggests_v2_patients(client: TestClient) -> None:
    data = client.get("/api/health").json()
    assert data["sample_patient_ids"][0] == "PAC-2001"
