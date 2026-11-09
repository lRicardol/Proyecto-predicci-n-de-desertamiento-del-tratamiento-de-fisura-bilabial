"""Repositorio CSV del marco de modelado v2 (una fila por paciente, sin imputar)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.core.exceptions import DataAccessError, PatientNotFoundError
from src.core.interfaces.i_patient_repository import IPatientRepository
from src.infrastructure.ai_models.modeling_features import FEATURES_T0

ID_COLUMN = "id_paciente"


class CsvPatientRepository(IPatientRepository):
    """Una fila por paciente desde el CSV de modelado (no el histórico unificado)."""

    def __init__(self, dataset_path: Path) -> None:
        path = Path(dataset_path)
        if path.is_dir():
            path = path / "dataset_modelado_v2.csv"
        self._dataset_path = path
        self._cache: pd.DataFrame | None = None

    def count_patients(self) -> int:
        return int(len(self._frame()))

    def list_patient_ids(self, limit: int | None = None) -> list[str]:
        identifiers = self._frame()[ID_COLUMN].astype(str).tolist()
        if limit is None:
            return identifiers
        return identifiers[:limit]

    def get_all_patients_data(self) -> pd.DataFrame:
        """Copia del marco. Uso interno y de pruebas; el puerto de dominio no lo expone."""
        return self._frame().copy()

    def _frame(self) -> pd.DataFrame:
        try:
            if self._cache is None:
                self._cache = self._load()
            return self._cache
        except (PatientNotFoundError, DataAccessError):
            raise
        except Exception as exc:
            raise DataAccessError(f"No fue posible leer el CSV procesado: {exc}") from exc

    def get_patient_by_id(self, patient_id: str) -> dict[str, Any]:
        if not patient_id or not str(patient_id).strip():
            raise PatientNotFoundError("El identificador de paciente está vacío.")
        normalized = str(patient_id).strip().upper()
        frame = self._frame()
        matches = frame[frame[ID_COLUMN].astype(str).str.upper() == normalized]
        if matches.empty:
            raise PatientNotFoundError(
                f"No existe el paciente '{patient_id}' en {self._dataset_path.name}."
            )
        record = matches.iloc[0].to_dict()
        return {key: _json_safe(value) for key, value in record.items()}

    def _load(self) -> pd.DataFrame:
        if not self._dataset_path.exists():
            raise DataAccessError(
                f"No existe {self._dataset_path}. "
                "Genéralo con: python -m src.infrastructure.data.etl"
            )
        try:
            frame = pd.read_csv(self._dataset_path)
        except Exception as exc:
            raise DataAccessError(f"Error leyendo {self._dataset_path.name}: {exc}") from exc
        if frame.empty:
            raise DataAccessError("El dataset procesado está vacío.")
        frame.columns = [str(col).strip() for col in frame.columns]
        if ID_COLUMN not in frame.columns:
            raise DataAccessError(f"{self._dataset_path.name} no tiene columna {ID_COLUMN}.")
        frame[ID_COLUMN] = frame[ID_COLUMN].astype(str).str.strip()
        duplicated = frame[ID_COLUMN].duplicated()
        if duplicated.any():
            raise DataAccessError(
                f"{self._dataset_path.name} tiene {int(duplicated.sum())} identificadores "
                "duplicados. No se eliminan en silencio."
            )
        missing = [column for column in FEATURES_T0 if column not in frame.columns]
        if missing:
            raise DataAccessError(f"{self._dataset_path.name} no tiene las columnas T0 {missing}.")
        return frame.reset_index(drop=True)


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value
