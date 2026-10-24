"""Repositorio CSV: única clase responsable de leer y consolidar data/raw."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.core.exceptions import DataAccessError, PatientNotFoundError
from src.core.interfaces.i_patient_repository import IPatientRepository

ID_COLUMN = "ID_Paciente"
TARGET_COLUMN = "Desertor"

PRIMARY_FILES: tuple[str, ...] = (
    "dataset_desercion_fisura_bilabial.csv",
    "dataset_v2_desercion_socioeconomica.csv",
)

AUXILIARY_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        "dataset_hitos_quirurgicos_complicaciones.csv",
        "dataset_v2_hitos_quirurgicos.csv",
    ),
    (
        "dataset_terapia_lenguaje_ivf.csv",
        "dataset_v2_terapia_lenguaje.csv",
    ),
)


class CsvPatientRepository(IPatientRepository):
    """Lee CSVs de fisura bilabial, une cohortes v1/v2 y limpia nulos básicos."""

    def __init__(self, data_raw_dir: Path) -> None:
        self._data_raw_dir = Path(data_raw_dir)
        self._cache: pd.DataFrame | None = None

    def get_all_patients_data(self) -> pd.DataFrame:
        try:
            if self._cache is None:
                self._cache = self._load_and_consolidate()
            return self._cache.copy()
        except (PatientNotFoundError, DataAccessError):
            raise
        except Exception as exc:
            raise DataAccessError(f"No fue posible consolidar los CSV: {exc}") from exc

    def get_patient_by_id(self, patient_id: str) -> dict[str, Any]:
        if not patient_id or not str(patient_id).strip():
            raise PatientNotFoundError("El identificador de paciente está vacío.")

        normalized_id = str(patient_id).strip()
        frame = self.get_all_patients_data()
        matches = frame[frame[ID_COLUMN].astype(str) == normalized_id]
        if matches.empty:
            raise PatientNotFoundError(
                f"No existe el paciente '{normalized_id}' en data/raw."
            )

        record = matches.iloc[0].to_dict()
        return {key: _json_safe(value) for key, value in record.items()}

    def _load_and_consolidate(self) -> pd.DataFrame:
        if not self._data_raw_dir.exists():
            raise DataAccessError(f"No existe el directorio {self._data_raw_dir}.")

        primary = self._concat_existing(PRIMARY_FILES)
        if primary.empty or ID_COLUMN not in primary.columns:
            raise DataAccessError(
                "No se encontraron datasets primarios de deserción con ID_Paciente."
            )

        consolidated = primary.drop_duplicates(subset=[ID_COLUMN], keep="first")
        for group in AUXILIARY_GROUPS:
            auxiliary = self._concat_existing(group)
            if auxiliary.empty or ID_COLUMN not in auxiliary.columns:
                continue
            auxiliary = auxiliary.drop_duplicates(subset=[ID_COLUMN], keep="first")
            extra_columns = [col for col in auxiliary.columns if col != ID_COLUMN]
            overlap = [col for col in extra_columns if col in consolidated.columns]
            auxiliary = auxiliary.drop(columns=overlap, errors="ignore")
            consolidated = consolidated.merge(auxiliary, on=ID_COLUMN, how="left")

        return self._clean_frame(consolidated)

    def _concat_existing(self, filenames: tuple[str, ...]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for name in filenames:
            path = self._data_raw_dir / name
            if not path.exists():
                continue
            try:
                loaded = pd.read_csv(path)
            except Exception as exc:
                raise DataAccessError(f"Error leyendo {path.name}: {exc}") from exc
            if loaded.empty:
                continue
            frames.append(loaded)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)

    def _clean_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        cleaned = frame.copy()
        cleaned.columns = [str(col).strip() for col in cleaned.columns]
        cleaned = cleaned.replace(r"^\s*$", pd.NA, regex=True)
        cleaned = cleaned.dropna(axis=1, how="all")
        cleaned = cleaned.dropna(how="all")

        if ID_COLUMN in cleaned.columns:
            cleaned[ID_COLUMN] = cleaned[ID_COLUMN].astype(str).str.strip()
            cleaned = cleaned[cleaned[ID_COLUMN].ne("") & cleaned[ID_COLUMN].ne("nan")]

        numeric_candidates = [
            "Edad_Inicio_Meses",
            "Distancia_Hospital_KM",
            "Ingresos_Mensuales_USD",
            "Apoyo_Psicologico",
            "Cirugias_Programadas",
            "Citas_Asistidas",
            "Citas_Perdidas_NoShow",
            "Citas_Terapia_Programadas",
            "Citas_Terapia_Asistidas",
            "Hipoacusia_Conductiva",
            "Grado_Hipernasalidad",
            "Insuficiencia_Velofaringea_IVF",
            "Edad_Queiloplastia_Meses",
            "Edad_Palatoplastia_Meses",
            "Dificultad_Alimentacion_Preop",
            "Dehiscencia_Herida_Postop",
            "Fistula_Palatina_Residual",
            "Peso_Z_Score_2Anos",
            TARGET_COLUMN,
        ]
        for column in numeric_candidates:
            if column in cleaned.columns:
                cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

        numeric_columns = cleaned.select_dtypes(include=["number"]).columns
        for column in numeric_columns:
            if cleaned[column].isna().any():
                cleaned[column] = cleaned[column].fillna(cleaned[column].median())

        categorical_columns = cleaned.select_dtypes(include=["object", "string"]).columns
        for column in categorical_columns:
            if column == ID_COLUMN:
                continue
            mode = cleaned[column].mode(dropna=True)
            fill_value = str(mode.iloc[0]) if not mode.empty else "Desconocido"
            cleaned[column] = cleaned[column].fillna(fill_value).astype(str)

        if TARGET_COLUMN in cleaned.columns:
            cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].fillna(0).astype(int)

        return cleaned.reset_index(drop=True)


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value
