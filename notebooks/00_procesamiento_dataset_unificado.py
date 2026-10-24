"""ETL de los 6 CSV reales en data/raw hacia data/processed/dataset_final_limpio.csv.

Cohorte v1 (PAC-1000): deserción + hitos quirúrgicos + terapia IVF.
Cohorte v2 (PAC-2000): deserción socioeconómica + hitos + terapia.
La llave primaria es ID_Paciente. Solo se conservan pacientes con etiqueta Desertor.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import pandas as pd

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH: Final[Path] = PROCESSED_DIR / "dataset_final_limpio.csv"

ID_COL: Final[str] = "ID_Paciente"

RAW_FILES: Final[dict[str, str]] = {
    "desercion_v1": "dataset_desercion_fisura_bilabial.csv",
    "cirugia_v1": "dataset_hitos_quirurgicos_complicaciones.csv",
    "terapia_v1": "dataset_terapia_lenguaje_ivf.csv",
    "desercion_v2": "dataset_v2_desercion_socioeconomica.csv",
    "cirugia_v2": "dataset_v2_hitos_quirurgicos.csv",
    "terapia_v2": "dataset_v2_terapia_lenguaje.csv",
}

FILENAME_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "dataset_v2_desercion_socioeconomica.csv": (
        "dataset_desercion_socioeconomica.csv",
        "dataset_v2_desercion_socioeconomica.csv",
    ),
}

SES_ORDINAL: Final[dict[str, int]] = {
    "bajo": 0,
    "medio-bajo": 1,
    "medio bajo": 1,
    "medio": 2,
    "medio-alto": 3,
    "medio alto": 3,
    "alto": 4,
}


class DatasetProcessingError(RuntimeError):
    """Fallo controlado del pipeline de ingeniería de datos."""


def resolve_raw_path(filename: str) -> Path:
    """Resuelve el nombre canónico o alias (p. ej. dataset_desercion_socioeconomica.csv)."""
    candidates = (filename, *FILENAME_ALIASES.get(filename, ()))
    for name in candidates:
        path = RAW_DIR / name
        if path.exists():
            return path
    raise DatasetProcessingError(
        f"No existe el archivo real {filename} ni sus alias en {RAW_DIR}"
    )


def load_csv(filename: str) -> pd.DataFrame:
    path = resolve_raw_path(filename)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise DatasetProcessingError(f"No se pudo leer {filename}: {exc}") from exc
    if frame.empty:
        raise DatasetProcessingError(f"{filename} está vacío.")
    if ID_COL not in frame.columns:
        raise DatasetProcessingError(f"{filename} no contiene {ID_COL}.")
    frame.columns = [str(col).strip() for col in frame.columns]
    frame[ID_COL] = frame[ID_COL].astype(str).str.strip()
    frame = frame.drop_duplicates(subset=[ID_COL], keep="first")
    return frame


def merge_cohort(
    desercion: pd.DataFrame,
    cirugia: pd.DataFrame,
    terapia: pd.DataFrame,
    cohorte: str,
) -> pd.DataFrame:
    """Left join desde deserción: la etiqueta Desertor es el contrato mínimo."""
    n_before = len(desercion)
    merged = desercion.merge(cirugia, on=ID_COL, how="left", validate="one_to_one")
    merged = merged.merge(terapia, on=ID_COL, how="left", validate="one_to_one")
    if len(merged) != n_before:
        raise DatasetProcessingError(
            f"El merge de {cohorte} duplicó filas: {n_before} -> {len(merged)}"
        )
    merged["cohorte"] = cohorte
    return merged


def harmonize_tipo_fisura(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(r"^Bilateral\s+", "", regex=True)
        .str.replace("Asimétrica", "Asimetrica")
        .str.replace("Asimetrica", "Asimetrica")
    )
    mapping = {
        "Completa": "Completa",
        "Asimetrica": "Asimetrica",
        "Incompleta": "Incompleta",
    }
    return cleaned.map(lambda value: mapping.get(value, "Desconocido"))


def encode_socioeconomic(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(SES_ORDINAL)
    )


def clean_and_engineer(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["tipo_fisura"] = harmonize_tipo_fisura(working["Tipo_Fisura_Bilateral"])
    working["nivel_socioeconomico"] = working["Nivel_Socioeconomico"].astype(str).str.strip()
    working["nivel_socioeconomico_ord"] = encode_socioeconomic(working["nivel_socioeconomico"])

    numeric_map = {
        "edad_inicio_meses": "Edad_Inicio_Meses",
        "distancia_hospital_km": "Distancia_Hospital_KM",
        "ingresos_mensuales_usd": "Ingresos_Mensuales_USD",
        "apoyo_psicologico": "Apoyo_Psicologico",
        "cirugias_programadas": "Cirugias_Programadas",
        "citas_asistidas": "Citas_Asistidas",
        "citas_perdidas_noshow": "Citas_Perdidas_NoShow",
        "edad_queiloplastia_meses": "Edad_Queiloplastia_Meses",
        "edad_palatoplastia_meses": "Edad_Palatoplastia_Meses",
        "dificultad_alimentacion_preop": "Dificultad_Alimentacion_Preop",
        "dehiscencia_herida_postop": "Dehiscencia_Herida_Postop",
        "fistula_palatina_residual": "Fistula_Palatina_Residual",
        "peso_z_score_2anos": "Peso_Z_Score_2Anos",
        "citas_terapia_programadas": "Citas_Terapia_Programadas",
        "citas_terapia_asistidas": "Citas_Terapia_Asistidas",
        "hipoacusia_conductiva": "Hipoacusia_Conductiva",
        "grado_hipernasalidad": "Grado_Hipernasalidad",
        "insuficiencia_velofaringea_ivf": "Insuficiencia_Velofaringea_IVF",
        "desertor": "Desertor",
    }
    for new_name, old_name in numeric_map.items():
        if old_name in working.columns:
            working[new_name] = pd.to_numeric(working[old_name], errors="coerce")
        else:
            working[new_name] = pd.NA

    # Protocolo quirúrgico labio-paladar: queiloplastia + palatoplastia = 2 hitos.
    # Si v2 no trae Cirugias_Programadas, se estima con hitos + complicaciones.
    protocol_surgeries = (
        working["edad_queiloplastia_meses"].notna().astype(int)
        + working["edad_palatoplastia_meses"].notna().astype(int)
    )
    complication_reops = (
        working["dehiscencia_herida_postop"].fillna(0)
        + working["fistula_palatina_residual"].fillna(0)
    )
    estimated = protocol_surgeries + complication_reops
    working["cirugias_programadas"] = working["cirugias_programadas"].fillna(estimated)

    working["complicaciones_quirurgicas"] = (
        working["dehiscencia_herida_postop"].fillna(0)
        + working["fistula_palatina_residual"].fillna(0)
    ).clip(lower=0)
    working["intervalo_cirugias_meses"] = (
        working["edad_palatoplastia_meses"] - working["edad_queiloplastia_meses"]
    )
    terapia_prog = working["citas_terapia_programadas"].replace(0, pd.NA)
    working["ratio_asistencia_terapia"] = (
        working["citas_terapia_asistidas"] / terapia_prog
    ).clip(lower=0, upper=1)

    # Imputación: numéricas por mediana de cohorte (evita mezclar escalas v1/v2).
    # Categóricas ya armonizadas; SES ordinal no imputado con la etiqueta Desertor.
    numeric_cols = [
        "edad_inicio_meses",
        "distancia_hospital_km",
        "ingresos_mensuales_usd",
        "apoyo_psicologico",
        "cirugias_programadas",
        "citas_asistidas",
        "citas_perdidas_noshow",
        "edad_queiloplastia_meses",
        "edad_palatoplastia_meses",
        "intervalo_cirugias_meses",
        "dificultad_alimentacion_preop",
        "dehiscencia_herida_postop",
        "fistula_palatina_residual",
        "complicaciones_quirurgicas",
        "peso_z_score_2anos",
        "citas_terapia_programadas",
        "citas_terapia_asistidas",
        "ratio_asistencia_terapia",
        "hipoacusia_conductiva",
        "grado_hipernasalidad",
        "insuficiencia_velofaringea_ivf",
        "nivel_socioeconomico_ord",
    ]
    for column in numeric_cols:
        working[column] = pd.to_numeric(working[column], errors="coerce")
        working[column] = working.groupby("cohorte")[column].transform(
            lambda serie: serie.fillna(serie.median())
        )
        if working[column].isna().any():
            working[column] = working[column].fillna(working[column].median())

    if "Citas_Asistidas" in frame.columns:
        working["citas_asistidas_imputada"] = frame["Citas_Asistidas"].isna().astype(int)
    else:
        working["citas_asistidas_imputada"] = 1

    fisura_dummies = pd.get_dummies(working["tipo_fisura"], prefix="fisura")
    for expected in ("fisura_Completa", "fisura_Asimetrica", "fisura_Incompleta"):
        if expected not in fisura_dummies.columns:
            fisura_dummies[expected] = 0
    working = pd.concat([working, fisura_dummies], axis=1)

    working["desertor"] = working["desertor"].fillna(0).astype(int)
    working = working[working[ID_COL].ne("") & working[ID_COL].ne("nan")]

    ordered = [
        ID_COL,
        "cohorte",
        "tipo_fisura",
        "fisura_Completa",
        "fisura_Asimetrica",
        "fisura_Incompleta",
        "edad_inicio_meses",
        "distancia_hospital_km",
        "nivel_socioeconomico",
        "nivel_socioeconomico_ord",
        "ingresos_mensuales_usd",
        "apoyo_psicologico",
        "cirugias_programadas",
        "citas_asistidas",
        "citas_asistidas_imputada",
        "citas_perdidas_noshow",
        "edad_queiloplastia_meses",
        "edad_palatoplastia_meses",
        "intervalo_cirugias_meses",
        "dificultad_alimentacion_preop",
        "dehiscencia_herida_postop",
        "fistula_palatina_residual",
        "complicaciones_quirurgicas",
        "peso_z_score_2anos",
        "citas_terapia_programadas",
        "citas_terapia_asistidas",
        "ratio_asistencia_terapia",
        "hipoacusia_conductiva",
        "grado_hipernasalidad",
        "insuficiencia_velofaringea_ivf",
        "desertor",
    ]
    missing = [col for col in ordered if col not in working.columns]
    if missing:
        raise DatasetProcessingError(f"Faltan columnas finales: {missing}")
    clean = working[ordered].rename(columns={ID_COL: "id_paciente"})
    bool_cols = [col for col in clean.columns if str(clean[col].dtype) == "boolean" or clean[col].dtype == bool]
    for col in bool_cols:
        clean[col] = clean[col].astype(int)
    dummy_cols = ["fisura_Completa", "fisura_Asimetrica", "fisura_Incompleta"]
    clean[dummy_cols] = clean[dummy_cols].astype(int)
    return clean.sort_values("id_paciente").reset_index(drop=True)


def build_unified_dataset() -> pd.DataFrame:
    tables = {key: load_csv(name) for key, name in RAW_FILES.items()}

    extra_v1 = (~tables["cirugia_v1"][ID_COL].isin(tables["desercion_v1"][ID_COL])).sum()
    cohort_v1 = merge_cohort(
        tables["desercion_v1"],
        tables["cirugia_v1"],
        tables["terapia_v1"],
        cohorte="v1",
    )
    cohort_v2 = merge_cohort(
        tables["desercion_v2"],
        tables["cirugia_v2"],
        tables["terapia_v2"],
        cohorte="v2",
    )
    unified = pd.concat([cohort_v1, cohort_v2], ignore_index=True, sort=False)
    if unified["ID_Paciente"].duplicated().any():
        raise DatasetProcessingError("Hay ID_Paciente duplicados entre cohortes.")

    clean = clean_and_engineer(unified)
    clean.attrs["pacientes_quirurgicos_v1_sin_etiqueta"] = int(extra_v1)
    return clean


def save_processed(frame: pd.DataFrame, path: Path = OUTPUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


def main() -> int:
    try:
        dataset = build_unified_dataset()
        output = save_processed(dataset)
    except DatasetProcessingError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"Dataset unificado: {output}")
    print(f"Filas: {len(dataset)} | Columnas: {len(dataset.columns)}")
    print(f"Cohortes: {dataset['cohorte'].value_counts().to_dict()}")
    print(f"Tasa de deserción: {dataset['desertor'].mean():.3f}")
    print(f"Nulos restantes: {int(dataset.isna().sum().sum())}")
    excluded = dataset.attrs.get("pacientes_quirurgicos_v1_sin_etiqueta", 0)
    print(
        "Pacientes v1 con cirugía/terapia pero sin Desertor (excluidos): "
        f"{excluded}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
