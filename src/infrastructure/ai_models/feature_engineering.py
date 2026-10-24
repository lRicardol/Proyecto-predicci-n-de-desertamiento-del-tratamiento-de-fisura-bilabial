"""Transformación de fichas clínicas a matriz numérica. SRP: solo features."""

from __future__ import annotations

from typing import Any

import pandas as pd

SOCIOECONOMIC_ORDER: dict[str, int] = {
    "bajo": 0,
    "medio-bajo": 1,
    "medio bajo": 1,
    "medio": 2,
    "medio-alto": 3,
    "medio alto": 3,
    "alto": 4,
}

FEATURE_COLUMNS: tuple[str, ...] = (
    "inasistencias",
    "distancia",
    "nivel_socioeconomico",
    "cirugias_previas",
    "ingresos_mensuales",
    "apoyo_psicologico",
    "edad_inicio_meses",
    "ratio_asistencia_terapia",
    "complicaciones_quirurgicas",
    "insuficiencia_velofaringea",
    "grado_hipernasalidad",
    "peso_z_score",
)


def patient_record_to_frame(record: dict[str, Any]) -> pd.DataFrame:
    """Convierte un dict de repositorio en una fila de características."""
    return build_feature_frame(pd.DataFrame([record]))


def build_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres heterogéneos de las cohortes v1 y v2."""
    if raw.empty:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))

    frame = raw.copy()
    inasistencias = _first_numeric(frame, ["Citas_Perdidas_NoShow", "inasistencias"])
    distancia = _first_numeric(frame, ["Distancia_Hospital_KM", "distancia"])
    ingresos = _first_numeric(frame, ["Ingresos_Mensuales_USD", "ingresos_mensuales"])
    apoyo = _first_numeric(frame, ["Apoyo_Psicologico", "apoyo_psicologico"])
    edad = _first_numeric(frame, ["Edad_Inicio_Meses", "edad_inicio_meses"])
    cirugias = _first_numeric(frame, ["Cirugias_Programadas", "cirugias_previas"])
    dehiscencia = _first_numeric(frame, ["Dehiscencia_Herida_Postop"])
    fistula = _first_numeric(frame, ["Fistula_Palatina_Residual"])
    terapia_prog = _first_numeric(frame, ["Citas_Terapia_Programadas"])
    terapia_asist = _first_numeric(frame, ["Citas_Terapia_Asistidas"])
    ivf = _first_numeric(frame, ["Insuficiencia_Velofaringea_IVF"])
    hipernasalidad = _first_numeric(frame, ["Grado_Hipernasalidad"])
    peso = _first_numeric(frame, ["Peso_Z_Score_2Anos", "peso_z_score"])
    ses = _encode_socioeconomic(frame)

    ratio = (terapia_asist / terapia_prog.replace(0, pd.NA)).fillna(0.0).clip(0.0, 1.0)
    complicaciones = (dehiscencia.fillna(0) + fistula.fillna(0)).clip(lower=0)

    features = pd.DataFrame(
        {
            "inasistencias": inasistencias.fillna(0.0),
            "distancia": distancia.fillna(0.0),
            "nivel_socioeconomico": ses,
            "cirugias_previas": cirugias.fillna(0.0),
            "ingresos_mensuales": ingresos.fillna(0.0),
            "apoyo_psicologico": apoyo.fillna(0.0),
            "edad_inicio_meses": edad.fillna(0.0),
            "ratio_asistencia_terapia": ratio,
            "complicaciones_quirurgicas": complicaciones,
            "insuficiencia_velofaringea": ivf.fillna(0.0),
            "grado_hipernasalidad": hipernasalidad.fillna(0.0),
            "peso_z_score": peso.fillna(0.0),
        }
    )
    return features[list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _first_numeric(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series([0.0] * len(frame), index=frame.index, dtype="float64")


def _encode_socioeconomic(frame: pd.DataFrame) -> pd.Series:
    if "nivel_socioeconomico" in frame.columns and pd.api.types.is_numeric_dtype(
        frame["nivel_socioeconomico"]
    ):
        return pd.to_numeric(frame["nivel_socioeconomico"], errors="coerce").fillna(2.0)

    source = None
    for column in ("Nivel_Socioeconomico", "nivel_socioeconomico"):
        if column in frame.columns:
            source = frame[column]
            break
    if source is None:
        return pd.Series([2.0] * len(frame), index=frame.index, dtype="float64")

    encoded = (
        source.astype(str)
        .str.strip()
        .str.lower()
        .map(SOCIOECONOMIC_ORDER)
        .fillna(2.0)
        .astype(float)
    )
    return encoded
