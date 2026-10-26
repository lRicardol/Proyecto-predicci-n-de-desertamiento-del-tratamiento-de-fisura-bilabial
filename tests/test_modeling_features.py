"""Pruebas del contrato de features sin fuga (escenarios T0 / T1)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.core.exceptions import InvalidPatientDataError
from src.infrastructure.ai_models.modeling_features import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    EXCLUDED_LEAKAGE,
    FEATURES_T0,
    FEATURES_T1,
    ORDINAL_FEATURES,
    SCENARIOS,
    select_features,
    select_target,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "dataset_final_limpio.csv"


def test_t0_is_subset_of_t1() -> None:
    assert set(FEATURES_T0) < set(FEATURES_T1)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_scenarios_exclude_leakage(scenario: str) -> None:
    assert not set(SCENARIOS[scenario]) & set(EXCLUDED_LEAKAGE)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_scenarios_have_no_duplicates(scenario: str) -> None:
    assert len(SCENARIOS[scenario]) == len(set(SCENARIOS[scenario]))


def test_typed_features_belong_to_t1() -> None:
    typed = set(CATEGORICAL_FEATURES) | set(ORDINAL_FEATURES) | set(BINARY_FEATURES)
    assert typed <= set(FEATURES_T1)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_processed_dataset_satisfies_contract(scenario: str) -> None:
    frame = pd.read_csv(PROCESSED)
    selected = select_features(frame, scenario)
    assert list(selected.columns) == list(SCENARIOS[scenario])
    assert len(selected) == len(frame)
    assert not set(selected.columns) & set(EXCLUDED_LEAKAGE)


def test_select_features_rejects_missing_columns() -> None:
    frame = pd.DataFrame({"tipo_fisura": ["Completa"]})
    with pytest.raises(InvalidPatientDataError, match="Faltan columnas"):
        select_features(frame, "T0")


def test_select_features_rejects_unknown_scenario() -> None:
    with pytest.raises(ValueError, match="Escenario desconocido"):
        select_features(pd.DataFrame(), "T9")


def test_select_features_does_not_mutate_input() -> None:
    frame = pd.read_csv(PROCESSED).head(5)
    before = frame.copy()
    selected = select_features(frame, "T1")
    selected.iloc[0, 0] = "modificado"
    pd.testing.assert_frame_equal(frame, before)


def test_select_target_validates_binary() -> None:
    assert select_target(pd.DataFrame({"desertor": [0, 1, 1]})).tolist() == [0, 1, 1]
    with pytest.raises(InvalidPatientDataError):
        select_target(pd.DataFrame({"desertor": [0, 2]}))
    with pytest.raises(InvalidPatientDataError):
        select_target(pd.DataFrame({"desertor": [0, None]}))
    with pytest.raises(InvalidPatientDataError):
        select_target(pd.DataFrame({"otra": [0]}))
