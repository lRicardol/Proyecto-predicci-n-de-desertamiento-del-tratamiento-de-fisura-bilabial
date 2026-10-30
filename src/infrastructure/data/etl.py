"""ETL sin fuga: de los CSV crudos al marco de modelado. SRP: leer, armonizar, validar.

Diferencias con ``notebooks/00_procesamiento_dataset_unificado.py`` (ver reports/auditoria.md §3.5):

- Una sola cohorte por llamada (v2 para modelar; v1 solo descriptiva). Nunca se concatenan.
- Las columnas excluidas (``EXCLUDED_LEAKAGE``) no se leen: ni no-show, ni asistencia a terapia,
  ni peso Z. No hay nada que "quitar después".
- No inventa columnas (``citas_asistidas``, ``cirugias_programadas``) ni indicadores de cohorte.
- No imputa. Los faltantes quedan como NaN; la imputación va dentro del ``Pipeline`` y se ajusta
  solo con train.
- Solo conserva pacientes con etiqueta ``Desertor`` (join desde el archivo de deserción).

El marco resultante cumple el contrato de ``modeling_features.py``: ``id_paciente`` +
``FEATURES_T1`` + ``desertor``. Para T0 basta ``select_features(frame, "T0")``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import pandas as pd

from src.core.exceptions import DataAccessError, InvalidPatientDataError
from src.infrastructure.ai_models.feature_engineering import SES_ORDINAL
from src.infrastructure.ai_models.modeling_features import (
    EXCLUDED_LEAKAGE,
    FEATURES_T1,
    ID_COLUMN,
    TARGET,
    select_features,
    select_target,
)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"

_RAW_ID: Final[str] = "ID_Paciente"

# cohorte -> (deserción, hitos quirúrgicos, terapia). La de deserción v2 tiene un alias histórico.
RAW_FILES: Final[dict[str, tuple[tuple[str, ...], str, str]]] = {
    "v1": (
        ("dataset_desercion_fisura_bilabial.csv",),
        "dataset_hitos_quirurgicos_complicaciones.csv",
        "dataset_terapia_lenguaje_ivf.csv",
    ),
    "v2": (
        ("dataset_v2_desercion_socioeconomica.csv", "dataset_desercion_socioeconomica.csv"),
        "dataset_v2_hitos_quirurgicos.csv",
        "dataset_v2_terapia_lenguaje.csv",
    ),
}

# Única lista de columnas crudas que se leen (nombre crudo -> nombre procesado).
_COLUMNS_DESERCION: Final[dict[str, str]] = {
    _RAW_ID: ID_COLUMN,
    "Tipo_Fisura_Bilateral": "tipo_fisura",
    "Edad_Inicio_Meses": "edad_inicio_meses",
    "Distancia_Hospital_KM": "distancia_hospital_km",
    "Nivel_Socioeconomico": "nivel_socioeconomico",
    "Ingresos_Mensuales_USD": "ingresos_mensuales_usd",
    "Apoyo_Psicologico": "apoyo_psicologico",
    "Desertor": TARGET,
}
_COLUMNS_CIRUGIA: Final[dict[str, str]] = {
    _RAW_ID: ID_COLUMN,
    "Edad_Queiloplastia_Meses": "edad_queiloplastia_meses",
    "Edad_Palatoplastia_Meses": "edad_palatoplastia_meses",
    "Dificultad_Alimentacion_Preop": "dificultad_alimentacion_preop",
    "Dehiscencia_Herida_Postop": "dehiscencia_herida_postop",
    "Fistula_Palatina_Residual": "fistula_palatina_residual",
}
_COLUMNS_TERAPIA: Final[dict[str, str]] = {
    _RAW_ID: ID_COLUMN,
    "Hipoacusia_Conductiva": "hipoacusia_conductiva",
    "Grado_Hipernasalidad": "grado_hipernasalidad",
    "Insuficiencia_Velofaringea_IVF": "insuficiencia_velofaringea_ivf",
}

_TIPO_FISURA: Final[dict[str, str]] = {
    "completa": "Completa",
    "incompleta": "Incompleta",
    "asimetrica": "Asimetrica",
    "asimétrica": "Asimetrica",
}
_NUMERIC_FEATURES: Final[tuple[str, ...]] = tuple(
    col for col in FEATURES_T1 if col not in ("tipo_fisura", "nivel_socioeconomico_ord")
)


def build_modeling_frame(raw_dir: Path | None = None, cohort: str = "v2") -> pd.DataFrame:
    """Marco de modelado de una cohorte: ``id_paciente`` + ``FEATURES_T1`` + ``desertor``.

    Sin imputación y sin columnas de ``EXCLUDED_LEAKAGE``. Úsese ``cohort="v2"`` para modelar;
    ``"v1"`` solo tiene 30 no desertores y no sirve para entrenar ni evaluar.
    """
    if cohort not in RAW_FILES:
        raise ValueError(f"Cohorte desconocida {cohort!r}. Opciones: {sorted(RAW_FILES)}")
    directory = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    desercion_names, cirugia_name, terapia_name = RAW_FILES[cohort]

    desercion = _read_raw(_resolve(directory, desercion_names), _COLUMNS_DESERCION)
    cirugia = _read_raw(_resolve(directory, (cirugia_name,)), _COLUMNS_CIRUGIA)
    terapia = _read_raw(_resolve(directory, (terapia_name,)), _COLUMNS_TERAPIA)

    # La etiqueta es el contrato mínimo: se parte de deserción y se descartan IDs sin etiqueta.
    frame = desercion.merge(cirugia, on=ID_COLUMN, how="left", validate="one_to_one")
    frame = frame.merge(terapia, on=ID_COLUMN, how="left", validate="one_to_one")
    if len(frame) != len(desercion):
        raise DataAccessError(f"El merge de {cohort} cambió el número de filas.")

    frame["tipo_fisura"] = _harmonize_tipo_fisura(frame["tipo_fisura"])
    frame["nivel_socioeconomico_ord"] = _encode_ses(frame["nivel_socioeconomico"])
    for column in _NUMERIC_FEATURES + (TARGET,):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    ordered = [ID_COLUMN, *FEATURES_T1, TARGET]
    frame = frame.loc[:, ordered].sort_values(ID_COLUMN).reset_index(drop=True)
    validate_modeling_frame(frame)
    return frame


def validate_modeling_frame(frame: pd.DataFrame) -> None:
    """Falla si el marco rompe el contrato: fuga, IDs repetidos o etiqueta no binaria."""
    leaked = sorted(set(frame.columns) & (set(EXCLUDED_LEAKAGE) - {ID_COLUMN, TARGET}))
    if leaked:
        raise InvalidPatientDataError(f"El marco contiene columnas excluidas: {leaked}")
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise InvalidPatientDataError("id_paciente tiene nulos o duplicados.")
    select_target(frame)  # binaria 0/1 y sin faltantes
    select_features(frame, "T1")  # T1 incluye T0


def quality_report(frame: pd.DataFrame) -> dict[str, object]:
    """Resumen para la bitácora: filas, tasa de deserción y faltantes por columna."""
    nulls = frame.isna().sum()
    return {
        "filas": int(len(frame)),
        "tasa_desercion": float(frame[TARGET].mean()),
        "no_desertores": int((frame[TARGET] == 0).sum()),
        "columnas_con_nulos": {col: int(n) for col, n in nulls.items() if n > 0},
    }


def _resolve(directory: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    raise DataAccessError(f"No existe {names[0]} (ni alias) en {directory}.")


def _read_raw(path: Path, columns: dict[str, str]) -> pd.DataFrame:
    """Lee solo las columnas permitidas: lo demás nunca entra al marco."""
    try:
        frame = pd.read_csv(path, usecols=lambda name: str(name).strip() in columns)
    except Exception as exc:
        raise DataAccessError(f"No se pudo leer {path.name}: {exc}") from exc
    frame.columns = [str(col).strip() for col in frame.columns]
    missing = [raw for raw in columns if raw not in frame.columns]
    if missing:
        raise DataAccessError(f"{path.name} no contiene las columnas {missing}.")
    frame = frame.rename(columns=columns)
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str).str.strip()
    if frame[ID_COLUMN].duplicated().any():
        raise DataAccessError(f"{path.name} tiene id_paciente duplicados.")
    return frame


def _harmonize_tipo_fisura(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(r"^bilateral\s+", "", regex=True, case=False)
        .str.lower()
    )
    return cleaned.map(_TIPO_FISURA)  # etiqueta desconocida -> NaN, no "Desconocido"


def _encode_ses(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().map(SES_ORDINAL).astype("float64")


def main() -> int:
    try:
        frame = build_modeling_frame()
    except (DataAccessError, InvalidPatientDataError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(f"Marco de modelado v2: {frame.shape[0]} filas x {frame.shape[1]} columnas")
    print(quality_report(frame))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
