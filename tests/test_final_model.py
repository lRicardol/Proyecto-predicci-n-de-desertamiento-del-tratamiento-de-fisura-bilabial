"""Pruebas del modelo final (regresión logística T0): entrenamiento, artefacto e inferencia."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import InvalidPatientDataError, PredictionError
from src.infrastructure.ai_models.final_model import (
    DEFAULT_ARTIFACT,
    priority_groups,
    save_artifact,
    train_final_model,
)
from src.infrastructure.ai_models.logistic_predictor import LogisticT0Predictor
from src.infrastructure.ai_models.modeling_features import EXCLUDED_LEAKAGE, FEATURES_T0
from src.infrastructure.data.etl import build_modeling_frame


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return build_modeling_frame(cohort="v2")


@pytest.fixture(scope="module")
def saved_artifact(frame: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    artifact = train_final_model(frame, seed=7)
    path, _ = save_artifact(artifact, tmp_path_factory.mktemp("modelo") / "modelo_t0.joblib")
    return path


def test_artifact_uses_only_t0_features(saved_artifact: Path) -> None:
    predictor = LogisticT0Predictor(saved_artifact)
    assert predictor.feature_names == FEATURES_T0
    assert not set(predictor.feature_names) & set(EXCLUDED_LEAKAGE)


def test_priority_cutoffs_and_rates_are_ordered(saved_artifact: Path) -> None:
    predictor = LogisticT0Predictor(saved_artifact)
    cutoffs = predictor.priority_cutoffs
    rates = predictor.priority_observed_rates
    assert 0.0 < cutoffs["habitual"] < cutoffs["alta"] < 1.0
    assert rates["alta"] > rates["media"] > rates["habitual"]


def test_metadata_json_is_written_without_pipeline(saved_artifact: Path) -> None:
    summary = json.loads(saved_artifact.with_suffix(".json").read_text(encoding="utf-8"))
    assert "pipeline" not in summary
    assert summary["pacientes"] == 3000
    assert summary["cohorte"] == "v2"
    assert summary["fuera_de_muestra"]["roc_auc"] > 0.65  # muy por encima del 0.5 de la línea base


def test_predictions_are_probabilities(saved_artifact: Path, frame: pd.DataFrame) -> None:
    probabilities = LogisticT0Predictor(saved_artifact).predict_dropout_probability(frame.head(50))
    assert probabilities.shape == (50,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_explanation_is_exact_in_log_odds(saved_artifact: Path, frame: pd.DataFrame) -> None:
    predictor = LogisticT0Predictor(saved_artifact)
    patient = frame.iloc[[10]]
    probability = float(predictor.predict_dropout_probability(patient)[0])
    total = predictor.reference_log_odds() + sum(predictor.contributions(patient).values())
    assert total == pytest.approx(np.log(probability / (1 - probability)), abs=1e-6)


def test_explain_prediction_returns_top_factors(saved_artifact: Path, frame: pd.DataFrame) -> None:
    factors = LogisticT0Predictor(saved_artifact).explain_prediction(frame.iloc[[0]], top=3)
    assert len(factors) == 3
    assert all(factor.variable in FEATURES_T0 for factor in factors)
    magnitudes = [abs(factor.contribution) for factor in factors]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_predictor_rejects_leakage_columns(saved_artifact: Path, frame: pd.DataFrame) -> None:
    """Las columnas con fuga no se ignoran en silencio: la predicción T0 las rechaza."""
    predictor = LogisticT0Predictor(saved_artifact)
    patient = frame.iloc[[0]].copy()
    base = predictor.predict_dropout_probability(patient)[0]
    patient["citas_perdidas_noshow"] = 99
    with pytest.raises(InvalidPatientDataError, match="fuga"):
        predictor.predict_dropout_probability(patient)
    patient = frame.iloc[[0]].copy()
    patient["citas_terapia_asistidas"] = 30
    with pytest.raises(InvalidPatientDataError, match="fuga"):
        predictor.predict_dropout_probability(patient)
    assert predictor.predict_dropout_probability(frame.iloc[[0]])[0] == base


def test_predictor_rejects_missing_columns(saved_artifact: Path) -> None:
    with pytest.raises(InvalidPatientDataError, match="Faltan columnas"):
        LogisticT0Predictor(saved_artifact).predict_dropout_probability(
            pd.DataFrame({"distancia_hospital_km": [10.0]})
        )


def test_predictor_fails_clearly_without_artifact(tmp_path: Path) -> None:
    with pytest.raises(PredictionError, match="final_model"):
        LogisticT0Predictor(tmp_path / "no_existe.joblib").load()


def test_priority_groups() -> None:
    groups = priority_groups(np.array([0.9, 0.83, 0.7, 0.62, 0.5]), high_cutoff=0.83, low_cutoff=0.62)
    assert groups.tolist() == ["alta", "alta", "media", "media", "habitual"]


def test_committed_artifact_loads() -> None:
    """El artefacto versionado en el repo es el que sirve la API."""
    predictor = LogisticT0Predictor(DEFAULT_ARTIFACT)
    assert predictor.metadata["escenario"] == "T0"
    assert predictor.feature_names == FEATURES_T0
