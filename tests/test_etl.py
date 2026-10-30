"""Pruebas del ETL sin fuga (frente 2). Usan CSV mínimos en tmp_path: rápidas y sin acoplarse al dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.core.exceptions import DataAccessError
from src.infrastructure.ai_models.modeling_features import (
    EXCLUDED_LEAKAGE,
    FEATURES_T0,
    FEATURES_T1,
    ID_COLUMN,
    TARGET,
    select_features,
)
from src.infrastructure.data.etl import build_modeling_frame, quality_report


def _write_raw(directory: Path, desercion_ids: list[str], extra_ids: list[str] | None = None) -> None:
    all_ids = desercion_ids + (extra_ids or [])
    pd.DataFrame(
        {
            "ID_Paciente": desercion_ids,
            "Tipo_Fisura_Bilateral": ["Bilateral Completa", "Incompleta", "Asimétrica"][: len(desercion_ids)],
            "Edad_Inicio_Meses": [6, 4, 8][: len(desercion_ids)],
            "Distancia_Hospital_KM": [10.0, None, 80.5][: len(desercion_ids)],
            "Nivel_Socioeconomico": ["Bajo", "Medio-Bajo", "Alto"][: len(desercion_ids)],
            "Ingresos_Mensuales_USD": [500, 900, 3000][: len(desercion_ids)],
            "Apoyo_Psicologico": [0, 1, 1][: len(desercion_ids)],
            "Citas_Perdidas_NoShow": [9, 2, 0][: len(desercion_ids)],  # fuga: no debe pasar
            "Desertor": [1, 0, 1][: len(desercion_ids)],
        }
    ).to_csv(directory / "dataset_v2_desercion_socioeconomica.csv", index=False)
    n = len(all_ids)
    pd.DataFrame(
        {
            "ID_Paciente": all_ids,
            "Edad_Queiloplastia_Meses": [4.0] * n,
            "Edad_Palatoplastia_Meses": [12.0] * n,
            "Dificultad_Alimentacion_Preop": [1] * n,
            "Dehiscencia_Herida_Postop": [0] * n,
            "Fistula_Palatina_Residual": [0] * n,
            "Peso_Z_Score_2Anos": [-1.0] * n,  # fuga: no debe pasar
        }
    ).to_csv(directory / "dataset_v2_hitos_quirurgicos.csv", index=False)
    pd.DataFrame(
        {
            "ID_Paciente": all_ids,
            "Citas_Terapia_Programadas": [30] * n,  # fuga: no debe pasar
            "Citas_Terapia_Asistidas": [20] * n,  # fuga: no debe pasar
            "Hipoacusia_Conductiva": [0] * n,
            "Grado_Hipernasalidad": [1] * n,
            "Insuficiencia_Velofaringea_IVF": [0] * n,
        }
    ).to_csv(directory / "dataset_v2_terapia_lenguaje.csv", index=False)


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    _write_raw(tmp_path, ["PAC-2000", "PAC-2001", "PAC-2002"], extra_ids=["PAC-2003"])
    return tmp_path


def test_frame_has_only_contract_columns(raw_dir: Path) -> None:
    frame = build_modeling_frame(raw_dir)
    assert list(frame.columns) == [ID_COLUMN, *FEATURES_T1, TARGET]
    leaked = set(frame.columns) & (set(EXCLUDED_LEAKAGE) - {ID_COLUMN, TARGET})
    assert not leaked


def test_scenarios_can_be_selected(raw_dir: Path) -> None:
    frame = build_modeling_frame(raw_dir)
    assert list(select_features(frame, "T0").columns) == list(FEATURES_T0)
    assert list(select_features(frame, "T1").columns) == list(FEATURES_T1)


def test_unlabeled_patients_are_dropped(raw_dir: Path) -> None:
    frame = build_modeling_frame(raw_dir)
    assert len(frame) == 3
    assert "PAC-2003" not in set(frame[ID_COLUMN])


def test_missing_values_are_not_imputed(raw_dir: Path) -> None:
    frame = build_modeling_frame(raw_dir)
    assert frame["distancia_hospital_km"].isna().sum() == 1
    assert quality_report(frame)["columnas_con_nulos"] == {"distancia_hospital_km": 1}


def test_categories_are_harmonized(raw_dir: Path) -> None:
    frame = build_modeling_frame(raw_dir)
    assert frame["tipo_fisura"].tolist() == ["Completa", "Incompleta", "Asimetrica"]
    assert frame["nivel_socioeconomico_ord"].tolist() == [0.0, 1.0, 4.0]


def test_duplicated_ids_fail(raw_dir: Path) -> None:
    path = raw_dir / "dataset_v2_hitos_quirurgicos.csv"
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(DataAccessError):
        build_modeling_frame(raw_dir)


def test_missing_raw_file_fails(tmp_path: Path) -> None:
    with pytest.raises(DataAccessError):
        build_modeling_frame(tmp_path)


def test_unknown_cohort_fails(raw_dir: Path) -> None:
    with pytest.raises(ValueError):
        build_modeling_frame(raw_dir, cohort="v3")


def test_real_v2_cohort_is_isolated() -> None:
    frame = build_modeling_frame()
    ids = frame[ID_COLUMN].str.removeprefix("PAC-").astype(int)
    assert len(frame) == 3000
    assert ids.min() >= 2000
    assert frame[TARGET].sum() == 2142
