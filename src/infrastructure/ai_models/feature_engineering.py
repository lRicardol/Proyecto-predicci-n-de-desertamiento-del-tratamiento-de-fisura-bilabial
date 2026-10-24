"""Mapeo de la ficha procesada a la matriz de XGBoost. SRP: solo features."""

from __future__ import annotations

from typing import Any

import pandas as pd

MODEL_FEATURES: tuple[str, ...] = (
    "citas_perdidas_noshow",
    "distancia_hospital_km",
    "nivel_socioeconomico_ord",
    "complicaciones_quirurgicas",
)

TARGET_COLUMN = "desertor"

SES_ORDINAL: dict[str, int] = {
    "bajo": 0,
    "medio-bajo": 1,
    "medio bajo": 1,
    "medio": 2,
    "medio-alto": 3,
    "medio alto": 3,
    "alto": 4,
}


def patient_record_to_frame(record: dict[str, Any]) -> pd.DataFrame:
    return extract_model_matrix(pd.DataFrame([record]))


def extract_model_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    """Acepta el CSV limpio (snake_case) o alias legacy y devuelve las 4 features clave."""
    if raw is None or raw.empty:
        raise ValueError("No hay filas para construir la matriz de características.")

    frame = raw.copy()
    frame.columns = [str(col).strip() for col in frame.columns]

    noshow = _numeric(
        frame,
        ["citas_perdidas_noshow", "Citas_Perdidas_NoShow", "inasistencias"],
    )
    distancia = _numeric(
        frame,
        ["distancia_hospital_km", "Distancia_Hospital_KM", "distancia"],
    )
    ses = _socioeconomic_ordinal(frame)
    complicaciones = _numeric(
        frame,
        ["complicaciones_quirurgicas"],
    )
    if complicaciones.isna().all():
        dehiscencia = _numeric(frame, ["dehiscencia_herida_postop", "Dehiscencia_Herida_Postop"])
        fistula = _numeric(frame, ["fistula_palatina_residual", "Fistula_Palatina_Residual"])
        complicaciones = dehiscencia.fillna(0) + fistula.fillna(0)

    matrix = pd.DataFrame(
        {
            "citas_perdidas_noshow": noshow.fillna(0.0),
            "distancia_hospital_km": distancia.fillna(0.0),
            "nivel_socioeconomico_ord": ses.fillna(2.0),
            "complicaciones_quirurgicas": complicaciones.fillna(0.0),
        }
    )
    return matrix[list(MODEL_FEATURES)].astype(float)


def extract_target(raw: pd.DataFrame) -> pd.Series:
    for column in ("desertor", "Desertor"):
        if column in raw.columns:
            return pd.to_numeric(raw[column], errors="coerce").fillna(0).astype(int)
    raise ValueError("El dataset no contiene la columna objetivo desertor.")


def _numeric(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series([float("nan")] * len(frame), index=frame.index, dtype="float64")


def _socioeconomic_ordinal(frame: pd.DataFrame) -> pd.Series:
    if "nivel_socioeconomico_ord" in frame.columns:
        return pd.to_numeric(frame["nivel_socioeconomico_ord"], errors="coerce")

    for column in ("nivel_socioeconomico", "Nivel_Socioeconomico"):
        if column not in frame.columns:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            return pd.to_numeric(frame[column], errors="coerce")
        return (
            frame[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(SES_ORDINAL)
            .astype(float)
        )
    return pd.Series([2.0] * len(frame), index=frame.index, dtype="float64")
