"""Repositorio CSV: lee exclusivamente dataset_final_limpio.csv."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.core.exceptions import DataAccessError, PatientNotFoundError
from src.core.interfaces.i_patient_repository import IPatientRepository

ID_COLUMN = "id_paciente"


class CsvPatientRepository(IPatientRepository):
    """Una fila por paciente desde el CSV unificado de data/processed."""

    def __init__(self, dataset_path: Path) -> None:
        path = Path(dataset_path)
        if path.is_dir():
            path = path / "dataset_final_limpio.csv"
        self._dataset_path = path
        self._cache: pd.DataFrame | None = None

    def get_all_patients_data(self) -> pd.DataFrame:
        try:
            if self._cache is None:
                self._cache = self._load()
            return self._cache.copy()
        except (PatientNotFoundError, DataAccessError):
            raise
        except Exception as exc:
            raise DataAccessError(f"No fue posible leer el CSV procesado: {exc}") from exc

    def get_patient_by_id(self, patient_id: str) -> dict[str, Any]:
        if not patient_id or not str(patient_id).strip():
            raise PatientNotFoundError("El identificador de paciente está vacío.")
        normalized = str(patient_id).strip().upper()
        frame = self.get_all_patients_data()
        matches = frame[frame[ID_COLUMN].astype(str).str.upper() == normalized]
        if matches.empty:
            raise PatientNotFoundError(
                f"No existe el paciente '{patient_id}' en dataset_final_limpio.csv."
            )
        record = matches.iloc[0].to_dict()
        return {key: _json_safe(value) for key, value in record.items()}

    def _load(self) -> pd.DataFrame:
        if not self._dataset_path.exists():
            raise DataAccessError(
                f"No existe {self._dataset_path}. Ejecuta el ETL de notebooks/."
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
        return frame.drop_duplicates(subset=[ID_COLUMN], keep="first").reset_index(drop=True)


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value
