"""Contrato de features sin fuga para modelar deserción. SRP: solo qué columnas entran.

Escenarios (ver reports/definicion_problema.md):
- T0: información disponible al ingreso del paciente.
- T1: T0 + hitos quirúrgicos primarios y hallazgos clínicos posteriores a la palatoplastia.

Las columnas de EXCLUDED_LEAKAGE describen el tratamiento completo (asistencia acumulada,
peso a los 2 años) o identifican la cohorte; no pueden conocerse al momento de predecir.
Este módulo no imputa, no codifica ni entrena: eso pertenece al pipeline de modelado.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from src.core.exceptions import InvalidPatientDataError
from src.infrastructure.ai_models.feature_engineering import SES_ORDINAL

TARGET: Final[str] = "desertor"
ID_COLUMN: Final[str] = "id_paciente"

FEATURES_T0: Final[tuple[str, ...]] = (
    "tipo_fisura",
    "edad_inicio_meses",
    "distancia_hospital_km",
    "nivel_socioeconomico_ord",
    "ingresos_mensuales_usd",
    "apoyo_psicologico",
    "dificultad_alimentacion_preop",
)

FEATURES_T1: Final[tuple[str, ...]] = FEATURES_T0 + (
    "edad_queiloplastia_meses",
    "edad_palatoplastia_meses",
    "dehiscencia_herida_postop",
    "fistula_palatina_residual",
    "hipoacusia_conductiva",
    "grado_hipernasalidad",
    "insuficiencia_velofaringea_ivf",
)

SCENARIOS: Final[dict[str, tuple[str, ...]]] = {
    "T0": FEATURES_T0,
    "T1": FEATURES_T1,
}

# Columna -> motivo de exclusión. Documentado en reports/diccionario_datos.md.
EXCLUDED_LEAKAGE: Final[dict[str, str]] = {
    "citas_terapia_asistidas": "acumulado del tratamiento; separa la etiqueta casi perfecto en v2",
    "ratio_asistencia_terapia": "derivada de citas_terapia_asistidas; AUC 1.0 en v2",
    "citas_perdidas_noshow": "acumulado del tratamiento; AUC 0.991 en v1",
    "citas_asistidas": "acumulado; en v2 no existe y el ETL la imputa con la mediana de v1",
    "citas_asistidas_imputada": "indicador de cohorte",
    "cirugias_programadas": "en v2 se estima con otra fórmula; proxy de cohorte",
    "citas_terapia_programadas": "depende de la duración real del tratamiento (pendiente de validar)",
    "peso_z_score_2anos": "medido a los 2 años; posterior al momento de predicción",
    "cohorte": "identifica el origen del dataset, no al paciente",
    ID_COLUMN: "identificador",
    TARGET: "variable objetivo",
}

CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("tipo_fisura",)
ORDINAL_FEATURES: Final[dict[str, tuple[int, ...]]] = {
    "nivel_socioeconomico_ord": tuple(sorted(set(SES_ORDINAL.values()))),
    "grado_hipernasalidad": (0, 1, 2, 3),
}
BINARY_FEATURES: Final[tuple[str, ...]] = (
    "apoyo_psicologico",
    "dificultad_alimentacion_preop",
    "dehiscencia_herida_postop",
    "fistula_palatina_residual",
    "hipoacusia_conductiva",
    "insuficiencia_velofaringea_ivf",
)


def features_for(scenario: str) -> tuple[str, ...]:
    try:
        return SCENARIOS[scenario]
    except KeyError as exc:
        raise ValueError(
            f"Escenario desconocido {scenario!r}. Opciones: {sorted(SCENARIOS)}"
        ) from exc


def select_features(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Devuelve solo las columnas del escenario; falla si faltan o si hay fuga."""
    columns = features_for(scenario)
    leaked = [col for col in columns if col in EXCLUDED_LEAKAGE]
    if leaked:
        raise InvalidPatientDataError(f"El escenario {scenario} incluye columnas excluidas: {leaked}")
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise InvalidPatientDataError(f"Faltan columnas para {scenario}: {missing}")
    return frame.loc[:, list(columns)].copy()


def select_target(frame: pd.DataFrame) -> pd.Series:
    if TARGET not in frame.columns:
        raise InvalidPatientDataError(f"El dataset no contiene la columna objetivo {TARGET}.")
    target = pd.to_numeric(frame[TARGET], errors="coerce")
    if target.isna().any() or not target.isin((0, 1)).all():
        raise InvalidPatientDataError(f"{TARGET} debe ser binaria (0/1) y sin faltantes.")
    return target.astype(int)
