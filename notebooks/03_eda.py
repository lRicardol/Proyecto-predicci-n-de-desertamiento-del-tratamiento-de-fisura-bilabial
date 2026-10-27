"""EDA reproducible del proyecto de deserción en fisura labiopalatina.

Este script es el entregable del frente 3 de `reports/auditoria.md`.
No entrena modelos, no imputa, no modifica datos y no toca la API.
Genera figuras en `reports/figures/` y tablas en `reports/eda_tables.md`.

Uso:
    .\\.venv\\Scripts\\python.exe notebooks/03_eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_auc_score

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.ai_models.modeling_features import (
    EXCLUDED_LEAKAGE,
    FEATURES_T0,
    FEATURES_T1,
    ID_COLUMN,
    TARGET,
    select_features,
    select_target,
)

PROCESSED_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "dataset_final_limpio.csv"
RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
FIGURES_DIR: Final[Path] = PROJECT_ROOT / "reports" / "figures"
TABLES_PATH: Final[Path] = PROJECT_ROOT / "reports" / "eda_tables.md"

RAW_FILES: Final[dict[str, str]] = {
    "desercion_v1": "dataset_desercion_fisura_bilabial.csv",
    "cirugia_v1": "dataset_hitos_quirurgicos_complicaciones.csv",
    "terapia_v1": "dataset_terapia_lenguaje_ivf.csv",
    "desercion_v2": "dataset_v2_desercion_socioeconomica.csv",
    "cirugia_v2": "dataset_v2_hitos_quirurgicos.csv",
    "terapia_v2": "dataset_v2_terapia_lenguaje.csv",
}

PAIR: Final[dict[int, str]] = {0: "No desertor", 1: "Desertor"}
COHORT_COLORS: Final[dict[str, str]] = {"v1": "#4C72B0", "v2": "#DD8452"}
TARGET_COLORS: Final[dict[str, str]] = {"No desertor": "#55A868", "Desertor": "#C44E52"}

SES_LABELS: Final[dict[int, str]] = {
    0: "Bajo (0)",
    1: "Medio-bajo (1)",
    2: "Medio (2)",
    3: "Medio-alto (3)",
    4: "Alto (4)",
}

AVAILABILITY_VARS: Final[tuple[str, ...]] = (
    "Tipo_Fisura_Bilateral",
    "Edad_Inicio_Meses",
    "Distancia_Hospital_KM",
    "Nivel_Socioeconomico",
    "Ingresos_Mensuales_USD",
    "Apoyo_Psicologico",
    "Cirugias_Programadas",
    "Citas_Asistidas",
    "Citas_Perdidas_NoShow",
    "Edad_Queiloplastia_Meses",
    "Edad_Palatoplastia_Meses",
    "Dificultad_Alimentacion_Preop",
    "Dehiscencia_Herida_Postop",
    "Fistula_Palatina_Residual",
    "Peso_Z_Score_2Anos",
    "Citas_Terapia_Programadas",
    "Citas_Terapia_Asistidas",
    "Hipoacusia_Conductiva",
    "Grado_Hipernasalidad",
    "Insuficiencia_Velofaringea_IVF",
    "Desertor",
)

T0_LABELS: Final[dict[str, str]] = {
    "tipo_fisura": "Tipo de fisura",
    "edad_inicio_meses": "Edad de inicio (meses)",
    "distancia_hospital_km": "Distancia al hospital (km)",
    "nivel_socioeconomico_ord": "Nivel socioeconómico (ordinal)",
    "ingresos_mensuales_usd": "Ingresos mensuales (USD)",
    "apoyo_psicologico": "Apoyo psicológico",
    "dificultad_alimentacion_preop": "Dificultad de alimentación preop.",
}

T1_LABELS: Final[dict[str, str]] = {
    **T0_LABELS,
    "edad_queiloplastia_meses": "Edad de queiloplastia (meses)",
    "edad_palatoplastia_meses": "Edad de palatoplastia (meses)",
    "dehiscencia_herida_postop": "Dehiscencia de la herida",
    "fistula_palatina_residual": "Fístula palatina residual",
    "hipoacusia_conductiva": "Hipoacusia conductiva",
    "grado_hipernasalidad": "Grado de hipernasalidad",
    "insuficiencia_velofaringea_ivf": "Insuficiencia velofaríngea (IVF)",
}

LEAKAGE_LABELS: Final[dict[str, str]] = {
    "citas_perdidas_noshow": "Citas perdidas (no-show)",
    "citas_terapia_asistidas": "Citas de terapia asistidas",
    "ratio_asistencia_terapia": "Ratio de asistencia a terapia",
    "citas_asistidas": "Citas asistidas",
    "cirugias_programadas": "Cirugías programadas",
    "peso_z_score_2anos": "Peso Z a los 2 años",
    "grado_hipernasalidad": "Grado de hipernasalidad",
    "insuficiencia_velofaringea_ivf": "Insuficiencia velofaríngea (IVF)",
}

MISSING_VARIABLES: Final[tuple[str, ...]] = (
    "Fecha de ingreso y fecha de cierre del tratamiento",
    "Fechas de citas: programadas, asistidas, canceladas y no-show",
    "Motivo de cierre: abandono, alta, traslado, suspensión médica, remisión",
    "Barreras reportadas por la familia",
    "Transporte y tiempo de desplazamiento",
    "Satisfacción y percepción de progreso",
    "Relación con el personal de salud",
    "Intención de continuar el tratamiento",
    "Motivación, autoeficacia y estado de ánimo",
    "Estrés, ansiedad y apoyo familiar",
    "Horario y disponibilidad para asistir",
)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza de Wilson para una proporción."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def _auc(target: pd.Series, values: pd.Series) -> float:
    """AUC univariado. Devuelve NaN si falta una clase o no hay variación."""
    data = pd.DataFrame({"target": target, "value": values}).dropna()
    if data["target"].nunique() < 2 or data["value"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(data["target"], data["value"]))


def _smd(a: pd.Series, b: pd.Series) -> float:
    """Diferencia estandarizada de medias (v1 - v2)."""
    a = a.dropna()
    b = b.dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def _format_cell(value: object) -> str:
    """Normaliza celdas para Markdown, incluidas listas y valores nulos."""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return ", ".join(map(str, value)).replace("|", "\\|")
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|")


def _md_table(frame: pd.DataFrame, float_format: str = ".3f") -> str:
    """Convierte un DataFrame en una tabla Markdown sin depender de tabulate."""
    if frame.empty:
        return "_Sin datos._\n"
    table = frame.copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(
                lambda value: "" if pd.isna(value) else format(float(value), float_format)
            )
        else:
            table[column] = table[column].map(_format_cell)
    header = "| " + " | ".join(map(str, table.columns)) + " |"
    separator = "|" + "|".join(["---"] * len(table.columns)) + "|"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows]) + "\n"


def _save_figure(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _style_axis(ax: plt.Axes, title: str, ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel)


def _rate_table(
    frame: pd.DataFrame,
    group_column: str,
    label: str,
    order: list[object] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = frame.groupby(group_column, dropna=False, observed=True)[TARGET]
    keys = order if order is not None else list(grouped.groups.keys())
    for key in keys:
        values = grouped.get_group(key) if key in grouped.groups else pd.Series(dtype=int)
        n = int(len(values))
        k = int(values.sum())
        p = k / n if n else float("nan")
        low, high = _wilson_ci(k, n)
        rows.append(
            {
                "Grupo": key,
                "n": n,
                "Desertores": k,
                "Tasa": p,
                "IC95 inferior": low,
                "IC95 superior": high,
                "Variable": label,
            }
        )
    return pd.DataFrame(rows)


def _bar_rate(
    ax: plt.Axes,
    frame: pd.DataFrame,
    group_column: str,
    title: str,
    order: list[object] | None = None,
    label_map: dict[object, str] | None = None,
    rotate: int = 0,
) -> None:
    data = _rate_table(frame, group_column, title, order=order)
    if data.empty:
        ax.set_visible(False)
        return
    positions = np.arange(len(data))
    rates = data["Tasa"].to_numpy(dtype=float)
    errors = np.vstack(
        [
            np.maximum(0.0, rates - data["IC95 inferior"].to_numpy(dtype=float)),
            np.maximum(0.0, data["IC95 superior"].to_numpy(dtype=float) - rates),
        ]
    )
    bars = ax.bar(
        positions,
        rates,
        color="#4C72B0",
        alpha=0.85,
        yerr=errors,
        capsize=3,
        edgecolor="white",
    )
    labels = [
        label_map.get(value, str(value)) if label_map else str(value)
        for value in data["Grupo"]
    ]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Tasa de deserción")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    _style_axis(ax, title)
    for bar, rate, n in zip(bars, rates, data["n"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(1.04, rate + 0.035),
            f"{rate:.0%}\nn={n}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _hist_by_target(
    ax: plt.Axes,
    frame: pd.DataFrame,
    column: str,
    title: str,
    log_scale: bool = False,
) -> None:
    sns.histplot(
        data=frame,
        x=column,
        hue="desenlace",
        bins=30,
        stat="density",
        common_norm=False,
        element="step",
        fill=True,
        alpha=0.35,
        palette=TARGET_COLORS,
        ax=ax,
        legend=True,
    )
    if log_scale:
        ax.set_xscale("log")
    _style_axis(ax, title, "Densidad")
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("")


def _count_by_target(
    ax: plt.Axes,
    frame: pd.DataFrame,
    column: str,
    title: str,
) -> None:
    sns.countplot(
        data=frame,
        x=column,
        hue="desenlace",
        palette=TARGET_COLORS,
        ax=ax,
    )
    _style_axis(ax, title, "Pacientes")
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("")


# ---------------------------------------------------------------------------
# Carga y preparación
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"No existe {PROCESSED_PATH}. Ejecuta el ETL antes del EDA."
        )
    processed = pd.read_csv(PROCESSED_PATH)
    raw = {
        key: pd.read_csv(RAW_DIR / name)
        for key, name in RAW_FILES.items()
        if (RAW_DIR / name).exists()
    }
    missing_raw = [name for name in RAW_FILES.values() if not (RAW_DIR / name).exists()]
    if missing_raw:
        raise FileNotFoundError(f"Faltan archivos raw: {missing_raw}")

    # Validar el contrato de features antes de cualquier análisis.
    for scenario in ("T0", "T1"):
        select_features(processed, scenario)

    processed["cohorte"] = processed["cohorte"].astype(str)
    processed["desenlace"] = select_target(processed).map(PAIR)
    v1 = processed[processed["cohorte"] == "v1"].copy()
    v2 = processed[processed["cohorte"] == "v2"].copy()
    return processed, v1, v2, raw


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------
def quality_table(
    processed: pd.DataFrame,
    v1: pd.DataFrame,
    v2: pd.DataFrame,
    raw: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    v1_labeled_ids = set(raw["desercion_v1"]["ID_Paciente"].astype(str))
    v1_extra_ids = set(raw["cirugia_v1"]["ID_Paciente"].astype(str)) - v1_labeled_ids
    v2_ids = set(raw["desercion_v2"]["ID_Paciente"].astype(str))
    v2_extra_ids = set(raw["cirugia_v2"]["ID_Paciente"].astype(str)) - v2_ids

    date_columns = [
        column
        for column in processed.columns
        if any(token in column.lower() for token in ("fecha", "date", "cierre", "visita"))
    ]
    raw_null_count = int(
        sum(int(frame.isna().sum().sum()) for frame in raw.values())
    )
    raw_labels_v1 = sorted(raw["desercion_v1"]["Tipo_Fisura_Bilateral"].dropna().unique())
    raw_labels_v2 = sorted(raw["desercion_v2"]["Tipo_Fisura_Bilateral"].dropna().unique())
    ses_v1 = sorted(v1["nivel_socioeconomico"].dropna().unique())
    ses_v2 = sorted(v2["nivel_socioeconomico"].dropna().unique())
    age_inconsistency = int(
        (
            processed["edad_inicio_meses"]
            > processed["edad_queiloplastia_meses"]
        ).sum()
    )
    negative_interval = int((processed["intervalo_cirugias_meses"] < 0).sum())

    rows = [
        {
            "Chequeo": "Pacientes en el CSV procesado",
            "Resultado": len(processed),
            "Detalle": "v1=500, v2=3000; una fila por paciente.",
        },
        {
            "Chequeo": "IDs únicos / duplicados",
            "Resultado": f"{processed[ID_COLUMN].nunique()} / {processed[ID_COLUMN].duplicated().sum()}",
            "Detalle": "La llave no se repite.",
        },
        {
            "Chequeo": "Solapamiento de IDs v1 vs v2",
            "Resultado": len(set(v1[ID_COLUMN]) & set(v2[ID_COLUMN])),
            "Detalle": "Las cohortes no comparten pacientes.",
        },
        {
            "Chequeo": "Pacientes v1 con cirugía/terapia sin etiqueta",
            "Resultado": len(v1_extra_ids),
            "Detalle": "Se excluyen del modelado por no tener Desertor.",
        },
        {
            "Chequeo": "Pacientes v2 con cirugía/terapia sin etiqueta",
            "Resultado": len(v2_extra_ids),
            "Detalle": "No se usan.",
        },
        {
            "Chequeo": "Nulos en archivos raw",
            "Resultado": raw_null_count,
            "Detalle": "Los seis CSV vienen completos.",
        },
        {
            "Chequeo": "Nulos en el CSV procesado",
            "Resultado": int(processed.isna().sum().sum()),
            "Detalle": "0 por construcción: el ETL imputó.",
        },
        {
            "Chequeo": "Columnas de fecha",
            "Resultado": len(date_columns),
            "Detalle": "No hay fechas de ingreso, citas, cirugías ni cierre.",
        },
        {
            "Chequeo": "Estructura longitudinal",
            "Resultado": "Una fila por paciente",
            "Detalle": "No hay historial de citas ni de etapas.",
        },
        {
            "Chequeo": "Etiqueta binaria",
            "Resultado": sorted(processed[TARGET].unique().tolist()),
            "Detalle": "Sin fecha ni motivo de cierre.",
        },
        {
            "Chequeo": "No desertores en v1",
            "Resultado": int((v1[TARGET] == 0).sum()),
            "Detalle": "Insuficiente para entrenar o evaluar v1.",
        },
        {
            "Chequeo": "Citas_Asistidas en v2",
            "Resultado": "Constante 11",
            "Detalle": "La columna no existe en v2; el ETL la imputa.",
        },
        {
            "Chequeo": "Cirugías_Programadas en v2",
            "Resultado": "2 a 4 (estimadas)",
            "Detalle": "En v1 es 2 a 5; el ETL usa otra fórmula.",
        },
        {
            "Chequeo": "Niveles de SES",
            "Resultado": f"v1={ses_v1} · v2={ses_v2}",
            "Detalle": "Escalas de 3 y 4 niveles; no son equivalentes.",
        },
        {
            "Chequeo": "Etiquetas de tipo de fisura",
            "Resultado": f"v1={raw_labels_v1} · v2={raw_labels_v2}",
            "Detalle": "El ETL armoniza 'Bilateral ...' con el resto.",
        },
        {
            "Chequeo": "Edad de inicio > edad de queiloplastia",
            "Resultado": age_inconsistency,
            "Detalle": "Inconsistencia temporal a revisar.",
        },
        {
            "Chequeo": "Intervalo de cirugías negativo",
            "Resultado": negative_interval,
            "Detalle": "Palatoplastia antes de queiloplastia.",
        },
    ]
    return pd.DataFrame(rows)


def balance_table(processed: pd.DataFrame, v1: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, frame in (("v1", v1), ("v2", v2), ("Total", processed)):
        n = len(frame)
        k = int(frame[TARGET].sum())
        p = k / n if n else float("nan")
        low, high = _wilson_ci(k, n)
        rows.append(
            {
                "Cohorte": label,
                "Pacientes": n,
                "Desertores": k,
                "No desertores": n - k,
                "Tasa": p,
                "IC95 inferior": low,
                "IC95 superior": high,
            }
        )
    return pd.DataFrame(rows)


def availability_table(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variable in AVAILABILITY_VARS:
        in_v1 = int(
            any(variable in frame.columns for key, frame in raw.items() if key.endswith("_v1"))
        )
        in_v2 = int(
            any(variable in frame.columns for key, frame in raw.items() if key.endswith("_v2"))
        )
        rows.append({"Variable": variable, "v1": in_v1, "v2": in_v2})
    return pd.DataFrame(rows)


def cohort_effect_table(processed: pd.DataFrame) -> pd.DataFrame:
    variables = (
        "citas_perdidas_noshow",
        "citas_terapia_asistidas",
        "peso_z_score_2anos",
        "dificultad_alimentacion_preop",
        "distancia_hospital_km",
        "ingresos_mensuales_usd",
        "edad_inicio_meses",
        "grado_hipernasalidad",
        "nivel_socioeconomico_ord",
    )
    rows: list[dict[str, object]] = []
    for variable in variables:
        v1_values = processed.loc[processed["cohorte"] == "v1", variable]
        v2_values = processed.loc[processed["cohorte"] == "v2", variable]
        rows.append(
            {
                "Variable": variable,
                "v1 media": v1_values.mean(),
                "v2 media": v2_values.mean(),
                "v1 mediana": v1_values.median(),
                "v2 mediana": v2_values.median(),
                "Diferencia": v1_values.mean() - v2_values.mean(),
                "SMD (v1-v2)": _smd(v1_values, v2_values),
                "|SMD|": abs(_smd(v1_values, v2_values)),
            }
        )
    return pd.DataFrame(rows)


def auc_table(processed: pd.DataFrame) -> pd.DataFrame:
    variables = [
        *FEATURES_T0,
        *[column for column in FEATURES_T1 if column not in FEATURES_T0],
        "citas_perdidas_noshow",
        "citas_terapia_asistidas",
        "ratio_asistencia_terapia",
        "citas_asistidas",
        "cirugias_programadas",
        "peso_z_score_2anos",
    ]
    variables = list(dict.fromkeys(variables))
    rows: list[dict[str, object]] = []
    for variable in variables:
        if variable == "tipo_fisura":
            continue
        v1_mask = processed["cohorte"] == "v1"
        v2_mask = processed["cohorte"] == "v2"
        auc_v1 = _auc(processed.loc[v1_mask, TARGET], processed.loc[v1_mask, variable])
        auc_v2 = _auc(processed.loc[v2_mask, TARGET], processed.loc[v2_mask, variable])
        if variable in EXCLUDED_LEAKAGE:
            decision = "Excluir: " + EXCLUDED_LEAKAGE[variable]
        elif variable in FEATURES_T0:
            decision = "T0"
        elif variable in FEATURES_T1:
            decision = "T1"
        else:
            decision = "Revisar"
        rows.append(
            {
                "Variable": variable,
                "AUC v1": auc_v1,
                "AUC v2": auc_v2,
                "Señal v2": abs(auc_v2 - 0.5) if pd.notna(auc_v2) else np.nan,
                "Decisión": decision,
            }
        )
    return pd.DataFrame(rows).sort_values("Señal v2", ascending=False, na_position="last")


def feature_summary_table(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in features:
        series = frame[feature]
        numeric = pd.to_numeric(series, errors="coerce")
        if feature in ("tipo_fisura",):
            summary = ", ".join(
                f"{key}: {value}"
                for key, value in series.value_counts().sort_index().items()
            )
            kind = "categórica"
        elif numeric.nunique(dropna=True) <= 5:
            summary = ", ".join(
                f"{int(key)}: {value}"
                for key, value in numeric.value_counts().sort_index().items()
            )
            kind = "ordinal/binaria"
        else:
            summary = (
                f"mediana {numeric.median():.1f} "
                f"[{numeric.quantile(0.25):.1f}, {numeric.quantile(0.75):.1f}]"
            )
            kind = "continua"
        rows.append(
            {
                "Feature": feature,
                "Tipo": kind,
                "n": int(series.notna().sum()),
                "Faltantes": int(series.isna().sum()),
                "Resumen v2": summary,
                "AUC v2": _auc(frame[TARGET], numeric),
            }
        )
    return pd.DataFrame(rows)


def artifact_table(processed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort, frame in processed.groupby("cohorte"):
        rows.append(
            {
                "Cohorte": cohort,
                "citas_asistidas únicos": frame["citas_asistidas"].nunique(),
                "citas_asistidas rango": (
                    f"{frame['citas_asistidas'].min():.0f}–{frame['citas_asistidas'].max():.0f}"
                ),
                "cirugias_programadas rango": (
                    f"{frame['cirugias_programadas'].min():.0f}–{frame['cirugias_programadas'].max():.0f}"
                ),
                "citas_asistidas_imputada=1": int(frame["citas_asistidas_imputada"].sum()),
            }
        )
    return pd.DataFrame(rows)


def correlation_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    variables = [
        column for column in FEATURES_T1 if column != "tipo_fisura"
    ] + [TARGET]
    correlation = frame[variables].corr(method="spearman")
    rows: list[dict[str, object]] = []
    for i, first in enumerate(variables):
        for second in variables[i + 1 :]:
            value = correlation.loc[first, second]
            if pd.notna(value) and abs(value) >= 0.30:
                rows.append(
                    {
                        "Variable 1": first,
                        "Variable 2": second,
                        "Spearman": value,
                        "|r|": abs(value),
                    }
                )
    return pd.DataFrame(rows).sort_values("|r|", ascending=False, na_position="last")


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def figure_balance(
    processed: pd.DataFrame, v1: pd.DataFrame, v2: pd.DataFrame
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    cohorts = ["v1", "v2", "Total"]
    frames = [v1, v2, processed]
    no_dropout = [int((frame[TARGET] == 0).sum()) for frame in frames]
    dropout = [int((frame[TARGET] == 1).sum()) for frame in frames]
    positions = np.arange(len(cohorts))
    axes[0].bar(positions, no_dropout, color=TARGET_COLORS["No desertor"], label="No desertor")
    axes[0].bar(
        positions,
        dropout,
        bottom=no_dropout,
        color=TARGET_COLORS["Desertor"],
        label="Desertor",
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(cohorts)
    axes[0].set_ylabel("Pacientes")
    axes[0].legend(frameon=False, loc="upper left")
    _style_axis(axes[0], "Balance de clases por cohorte")
    for position, total, positives in zip(positions, no_dropout, dropout):
        axes[0].text(position, total + 40, f"{total + positives}", ha="center", fontsize=9)

    rates = [k / n for k, n in zip(dropout, (len(v1), len(v2), len(processed)))]
    lows = [_wilson_ci(k, n)[0] for k, n in zip(dropout, (len(v1), len(v2), len(processed)))]
    highs = [_wilson_ci(k, n)[1] for k, n in zip(dropout, (len(v1), len(v2), len(processed)))]
    errors = np.vstack(
        [
            np.maximum(0, np.array(rates) - np.array(lows)),
            np.maximum(0, np.array(highs) - np.array(rates)),
        ]
    )
    bars = axes[1].bar(
        positions,
        rates,
        color=["#C44E52", "#C44E52", "#8172B2"],
        yerr=errors,
        capsize=4,
        alpha=0.9,
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(cohorts)
    axes[1].set_ylim(0, 1.08)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes[1].set_ylabel("Tasa de deserción")
    _style_axis(axes[1], "Tasa de deserción con IC95")
    for bar, rate in zip(bars, rates):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.03,
            f"{rate:.1%}",
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    return _save_figure(fig, "01_balance_clases.png")


def figure_availability(availability: pd.DataFrame) -> Path:
    matrix = availability.set_index("Variable")[["v1", "v2"]]
    annotations = matrix.replace({1: "Sí", 0: "No"})
    fig, ax = plt.subplots(figsize=(6.2, 8.2))
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cbar=False,
        cmap=ListedColormap(["#F2F2F2", "#4C72B0"]),
        linewidths=0.6,
        linecolor="white",
        ax=ax,
    )
    ax.set_title(
        "Disponibilidad de columnas por cohorte\n(1 = existe en el archivo raw)",
        fontsize=12,
        loc="left",
        fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    return _save_figure(fig, "02_disponibilidad_columnas.png")


def figure_cohort_effect(processed: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    continuous = [
        ("citas_perdidas_noshow", "Citas perdidas (no-show)", False),
        ("citas_terapia_asistidas", "Citas de terapia asistidas", False),
        ("peso_z_score_2anos", "Peso Z a los 2 años", False),
        ("distancia_hospital_km", "Distancia al hospital (km)", True),
        ("ingresos_mensuales_usd", "Ingresos mensuales (USD)", True),
        ("edad_inicio_meses", "Edad de inicio (meses)", False),
    ]
    for ax, (variable, title, log_scale) in zip(axes.ravel(), continuous):
        sns.boxplot(
            data=processed,
            x="cohorte",
            y=variable,
            hue="cohorte",
            palette=COHORT_COLORS,
            legend=False,
            showfliers=False,
            ax=ax,
        )
        if log_scale:
            ax.set_yscale("log")
        _style_axis(ax, title, "")
        ax.set_xlabel("")

    ax = axes.ravel()[6]
    difficulty = (
        processed.groupby("cohorte")["dificultad_alimentacion_preop"]
        .agg(["count", "mean"])
        .reindex(["v1", "v2"])
    )
    bars = ax.bar(
        difficulty.index,
        difficulty["mean"],
        color=[COHORT_COLORS["v1"], COHORT_COLORS["v2"]],
        alpha=0.85,
    )
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylabel("Proporción con dificultad")
    _style_axis(ax, "Dificultad de alimentación preop. (%)")
    for bar, (_, row) in zip(bars, difficulty.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row["mean"] + 0.03,
            f"{row['mean']:.1%}\nn={int(row['count'])}",
            ha="center",
            fontsize=8,
        )

    ax = axes.ravel()[7]
    shares = (
        pd.crosstab(processed["cohorte"], processed["grado_hipernasalidad"], normalize="index")
        .reindex(["v1", "v2"])
    )
    shares.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=sns.color_palette("viridis", n_colors=shares.shape[1]),
        edgecolor="white",
    )
    ax.set_xticklabels(["v1", "v2"], rotation=0)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("")
    ax.set_ylabel("Proporción")
    _style_axis(ax, "Grado de hipernasalidad por cohorte")
    ax.legend(title="Grado", fontsize=8, title_fontsize=8, frameon=False)

    fig.suptitle(
        "Efecto cohorte: v1 y v2 no vienen de la misma distribución",
        fontsize=15,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    return _save_figure(fig, "03_efecto_cohorte.png")


def figure_leakage_noshow(processed: pd.DataFrame, v1: pd.DataFrame, v2: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    sns.boxplot(
        data=processed,
        x="cohorte",
        y="citas_perdidas_noshow",
        hue="desenlace",
        palette=TARGET_COLORS,
        showfliers=False,
        ax=axes[0],
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Citas perdidas (no-show)")
    _style_axis(axes[0], "No-show por cohorte y desenlace")
    axes[0].legend(title="", frameon=False, fontsize=8)

    _bar_rate(
        axes[1],
        v1.assign(
            noshow_bin=pd.cut(
                v1["citas_perdidas_noshow"],
                bins=[-1, 1, 3, 6, 9, 30],
                labels=["0–1", "2–3", "4–6", "7–9", "10+"],
            )
        ),
        "noshow_bin",
        "v1: deserción por no-show",
    )
    _bar_rate(
        axes[2],
        v2.assign(
            noshow_bin=pd.cut(
                v2["citas_perdidas_noshow"],
                bins=[-1, 1, 3, 6, 9, 30],
                labels=["0–1", "2–3", "4–6", "7–9", "10+"],
            )
        ),
        "noshow_bin",
        "v2: deserción por no-show",
    )
    fig.suptitle(
        "Fuga: el no-show acumulado describe el tratamiento completo, no el momento de predicción",
        fontsize=13,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    return _save_figure(fig, "04_fuga_noshow.png")


def figure_leakage_therapy(v2: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    sns.histplot(
        data=v2,
        x="ratio_asistencia_terapia",
        hue="desenlace",
        bins=30,
        multiple="layer",
        stat="count",
        common_norm=False,
        palette=TARGET_COLORS,
        alpha=0.55,
        ax=axes[0],
    )
    axes[0].set_ylabel("Pacientes")
    _style_axis(axes[0], "Ratio de asistencia a terapia (v2)")

    sns.scatterplot(
        data=v2,
        x="citas_terapia_programadas",
        y="citas_terapia_asistidas",
        hue="desenlace",
        palette=TARGET_COLORS,
        alpha=0.55,
        s=16,
        ax=axes[1],
    )
    axes[1].set_xlabel("Citas de terapia programadas")
    axes[1].set_ylabel("Citas de terapia asistidas")
    _style_axis(axes[1], "Asistencia vs. programación (v2)")
    axes[1].legend(title="", frameon=False, fontsize=8)

    sns.boxplot(
        data=v2,
        x="desenlace",
        y="citas_terapia_asistidas",
        hue="desenlace",
        palette=TARGET_COLORS,
        legend=False,
        showfliers=False,
        ax=axes[2],
    )
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Citas de terapia asistidas")
    _style_axis(axes[2], "Citas asistidas por desenlace (v2)")
    fig.suptitle(
        "Fuga: el ratio de asistencia separa las clases casi perfectamente en v2",
        fontsize=13,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    return _save_figure(fig, "05_fuga_terapia.png")


def figure_etl_artifacts(processed: pd.DataFrame, v1: pd.DataFrame, v2: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].hist(v1["citas_asistidas"], bins=20, color=COHORT_COLORS["v1"], alpha=0.75, label="v1")
    axes[0].axvline(
        11,
        color=COHORT_COLORS["v2"],
        linewidth=2.5,
        label="v2: constante 11",
    )
    axes[0].set_xlabel("Citas asistidas")
    axes[0].set_ylabel("Pacientes")
    axes[0].legend(frameon=False)
    _style_axis(axes[0], "Citas_Asistidas: v2 no existe")

    sns.countplot(
        data=processed,
        x="cirugias_programadas",
        hue="cohorte",
        palette=COHORT_COLORS,
        ax=axes[1],
    )
    axes[1].set_xlabel("Cirugías programadas")
    axes[1].set_ylabel("Pacientes")
    _style_axis(axes[1], "Cirugías_Programadas: v2 estimado")
    axes[1].legend(title="", frameon=False)

    shares = (
        pd.crosstab(processed["cohorte"], processed["citas_asistidas_imputada"])
        .reindex(["v1", "v2"])
        .fillna(0)
    )
    shares.plot(
        kind="bar",
        stacked=True,
        ax=axes[2],
        color=["#55A868", "#C44E52"],
        edgecolor="white",
    )
    axes[2].set_xticklabels(["v1", "v2"], rotation=0)
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Pacientes")
    _style_axis(axes[2], "Flag de imputación = cohorte")
    axes[2].legend(
        title="citas_asistidas_imputada",
        fontsize=8,
        title_fontsize=8,
        frameon=False,
    )

    fig.suptitle(
        "Artefactos del ETL que identifican la cohorte y no deben usarse como features",
        fontsize=13,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    return _save_figure(fig, "06_artefactos_etl.png")


def figure_t0_distributions(v2: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes_flat = axes.ravel()
    categorical = {
        "tipo_fisura",
        "nivel_socioeconomico_ord",
        "apoyo_psicologico",
        "dificultad_alimentacion_preop",
    }
    for ax, feature in zip(axes_flat, FEATURES_T0):
        if feature in categorical:
            _count_by_target(ax, v2, feature, T0_LABELS[feature])
        else:
            log_scale = feature in ("distancia_hospital_km", "ingresos_mensuales_usd")
            _hist_by_target(
                ax,
                v2,
                feature,
                T0_LABELS[feature] + (" (escala log)" if log_scale else ""),
                log_scale=log_scale,
            )
        ax.tick_params(axis="x", labelsize=8)
    for ax in axes_flat[len(FEATURES_T0) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Distribución de las features T0 en v2, por desenlace",
        fontsize=15,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    return _save_figure(fig, "07_t0_distribuciones.png")


def figure_t0_rates(v2: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes_flat = axes.ravel()

    _bar_rate(
        axes_flat[0],
        v2.assign(ses=v2["nivel_socioeconomico_ord"]),
        "ses",
        "Tasa por nivel socioeconómico",
        order=[0, 1, 2, 4],
        label_map=SES_LABELS,
        rotate=20,
    )

    distance_bins = pd.cut(
        v2["distancia_hospital_km"],
        bins=[0, 20, 40, 80, 150, 400],
        labels=["0–20", "20–40", "40–80", "80–150", "150+"],
        include_lowest=True,
    )
    _bar_rate(
        axes_flat[1],
        v2.assign(distancia_bin=distance_bins),
        "distancia_bin",
        "Tasa por distancia (km)",
        order=["0–20", "20–40", "40–80", "80–150", "150+"],
    )

    income_bins = pd.qcut(
        v2["ingresos_mensuales_usd"],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    )
    _bar_rate(
        axes_flat[2],
        v2.assign(ingreso_bin=income_bins),
        "ingreso_bin",
        "Tasa por ingresos (cuartiles)",
        order=["Q1", "Q2", "Q3", "Q4"],
    )

    age_bins = pd.cut(
        v2["edad_inicio_meses"],
        bins=[0, 3, 6, 9, 12],
        labels=["1–3", "4–6", "7–9", "10–11"],
        include_lowest=True,
    )
    _bar_rate(
        axes_flat[3],
        v2.assign(edad_bin=age_bins),
        "edad_bin",
        "Tasa por edad de inicio (meses)",
        order=["1–3", "4–6", "7–9", "10–11"],
    )

    _bar_rate(
        axes_flat[4],
        v2,
        "tipo_fisura",
        "Tasa por tipo de fisura",
        order=["Completa", "Incompleta", "Asimetrica"],
    )
    _bar_rate(
        axes_flat[5],
        v2,
        "apoyo_psicologico",
        "Tasa según apoyo psicológico",
        order=[0, 1],
        label_map={0: "No", 1: "Sí"},
    )
    _bar_rate(
        axes_flat[6],
        v2,
        "dificultad_alimentacion_preop",
        "Tasa según dificultad de alimentación",
        order=[0, 1],
        label_map={0: "No", 1: "Sí"},
    )
    axes_flat[7].set_visible(False)
    fig.suptitle(
        "Features T0 en v2: tasas de deserción con IC95",
        fontsize=15,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    return _save_figure(fig, "08_t0_tasas.png")


def figure_t1_speech(v2: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.ravel()

    _bar_rate(
        axes_flat[0],
        v2,
        "grado_hipernasalidad",
        "Hipernasalidad por grado",
        order=[0, 1, 2, 3],
    )
    _bar_rate(
        axes_flat[1],
        v2,
        "insuficiencia_velofaringea_ivf",
        "Insuficiencia velofaríngea (IVF)",
        order=[0, 1],
        label_map={0: "No", 1: "Sí"},
    )
    _bar_rate(
        axes_flat[2],
        v2,
        "hipoacusia_conductiva",
        "Hipoacusia conductiva",
        order=[0, 1],
        label_map={0: "No", 1: "Sí"},
    )
    _bar_rate(
        axes_flat[3],
        v2,
        "dehiscencia_herida_postop",
        "Dehiscencia de la herida",
        order=[0, 1],
        label_map={0: "No", 1: "Sí"},
    )
    _bar_rate(
        axes_flat[4],
        v2,
        "fistula_palatina_residual",
        "Fístula palatina residual",
        order=[0, 1],
        label_map={0: "No", 1: "Sí"},
    )
    palate_bins = pd.cut(
        v2["edad_palatoplastia_meses"],
        bins=[8, 12, 15, 18, 22],
        labels=["9–12", "12–15", "15–18", "18–21"],
        include_lowest=True,
    )
    _bar_rate(
        axes_flat[5],
        v2.assign(paladar_bin=palate_bins),
        "paladar_bin",
        "Edad de palatoplastia (meses)",
        order=["9–12", "12–15", "15–18", "18–21"],
    )
    fig.suptitle(
        "Features T1 en v2: hipernasalidad e IVF concentran la señal (posible fuga)",
        fontsize=15,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    return _save_figure(fig, "09_t1_habla.png")


def figure_correlation(v2: pd.DataFrame, pairs: pd.DataFrame) -> Path:
    variables = [
        column for column in FEATURES_T1 if column != "tipo_fisura"
    ] + [TARGET]
    correlation = v2[variables].corr(method="spearman")
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        correlation,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Spearman"},
        ax=ax,
    )
    ax.set_title(
        "Correlación de Spearman en v2 (diagnóstico, no selección automática)",
        fontsize=12,
        loc="left",
        fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    path = _save_figure(fig, "10_correlacion.png")
    if not pairs.empty:
        pairs.to_csv(FIGURES_DIR / "10_correlaciones_altas.csv", index=False)
    return path


def figure_age_inconsistency(processed: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.scatterplot(
        data=processed,
        x="edad_queiloplastia_meses",
        y="edad_inicio_meses",
        hue="cohorte",
        palette=COHORT_COLORS,
        alpha=0.55,
        s=20,
        ax=axes[0],
    )
    limit = max(
        processed["edad_queiloplastia_meses"].max(),
        processed["edad_inicio_meses"].max(),
    ) + 1
    axes[0].plot([0, limit], [0, limit], linestyle="--", color="black", linewidth=1)
    axes[0].set_xlabel("Edad de queiloplastia (meses)")
    axes[0].set_ylabel("Edad de inicio (meses)")
    _style_axis(axes[0], "Edad de inicio vs. primera cirugía")
    axes[0].legend(title="", frameon=False)

    difference = processed["edad_inicio_meses"] - processed["edad_queiloplastia_meses"]
    sns.histplot(
        data=processed.assign(diferencia=difference),
        x="diferencia",
        hue="cohorte",
        bins=30,
        stat="count",
        common_norm=False,
        element="step",
        fill=True,
        alpha=0.35,
        palette=COHORT_COLORS,
        ax=axes[1],
    )
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Edad de inicio − edad de queiloplastia (meses)")
    axes[1].set_ylabel("Pacientes")
    _style_axis(axes[1], "Inconsistencia temporal")

    above = int((difference > 0).sum())
    axes[1].text(
        0.02,
        0.95,
        f"{above} pacientes ({above / len(processed):.1%}) inician después de la queiloplastia",
        transform=axes[1].transAxes,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    return _save_figure(fig, "11_edad_inicio_vs_cirugia.png")


def figure_ses_cohort(processed: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    rates = (
        processed.groupby(["cohorte", "nivel_socioeconomico"], observed=True)[TARGET]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )
    rates["tasa"] = rates["mean"]
    rates["n"] = rates["count"]
    order = ["Bajo", "Medio-Bajo", "Medio", "Alto"]
    rate_pivot = rates.pivot(index="nivel_socioeconomico", columns="cohorte", values="tasa").reindex(order)
    count_pivot = rates.pivot(index="nivel_socioeconomico", columns="cohorte", values="n").reindex(order)

    positions = np.arange(len(order))
    width = 0.38
    for offset, cohort in zip((-width / 2, width / 2), ("v1", "v2")):
        values = rate_pivot.get(cohort, pd.Series(index=order, dtype=float))
        bars = axes[0].bar(
            positions + offset,
            values.fillna(0.0),
            width=width,
            color=COHORT_COLORS[cohort],
            label=cohort,
            alpha=0.9,
        )
        for bar, value, count in zip(bars, values, count_pivot.get(cohort, pd.Series(index=order, dtype=float))):
            if pd.isna(value) or pd.isna(count):
                continue
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.0%}\nn={int(count)}",
                ha="center",
                fontsize=7,
            )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(order)
    axes[0].set_ylim(0, 1.16)
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Tasa de deserción")
    _style_axis(axes[0], "Tasa por SES y cohorte")
    axes[0].legend(title="", frameon=False)

    counts = (
        processed.groupby(["cohorte", "nivel_socioeconomico"], observed=True)
        .size()
        .reset_index(name="n")
    )
    sns.barplot(
        data=counts,
        x="nivel_socioeconomico",
        y="n",
        hue="cohorte",
        order=order,
        palette=COHORT_COLORS,
        ax=axes[1],
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Pacientes")
    _style_axis(axes[1], "Tamaño de cada celda SES")
    axes[1].legend(title="", frameon=False)
    fig.suptitle(
        "Las escalas de SES no son equivalentes: v1 tiene 3 niveles y v2 tiene 4",
        fontsize=13,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    return _save_figure(fig, "12_ses_cohorte.png")


# ---------------------------------------------------------------------------
# Informe de tablas
# ---------------------------------------------------------------------------
def write_tables(
    quality: pd.DataFrame,
    balance: pd.DataFrame,
    availability: pd.DataFrame,
    cohort_effect: pd.DataFrame,
    aucs: pd.DataFrame,
    artifacts: pd.DataFrame,
    t0_summary: pd.DataFrame,
    t1_summary: pd.DataFrame,
    t0_rates: pd.DataFrame,
    t1_rates: pd.DataFrame,
    pairs: pd.DataFrame,
    figures: list[Path],
) -> None:
    lines = [
        "# Tablas del EDA (generadas automáticamente)",
        "",
        "Generado por `notebooks/03_eda.py`. No editar a mano.",
        "",
        "## 1. Calidad y estructura",
        "",
        _md_table(quality),
        "## 2. Balance de clases",
        "",
        _md_table(balance),
        "## 3. Disponibilidad de columnas por cohorte",
        "",
        _md_table(availability),
        "## 4. Efecto cohorte",
        "",
        "SMD = diferencia estandarizada de medias (v1 − v2). |SMD| > 0,2 indica diferencia relevante.",
        "",
        _md_table(cohort_effect),
        "## 5. AUC univariado",
        "",
        "0,5 = sin señal; cerca de 0 o 1 = separación. Es asociación, no causalidad.",
        "",
        _md_table(aucs),
        "## 6. Artefactos del ETL",
        "",
        _md_table(artifacts),
        "## 7. Resumen de features T0 (v2)",
        "",
        _md_table(t0_summary),
        "## 8. Resumen de features T1 (v2)",
        "",
        _md_table(t1_summary),
        "## 9. Tasas de deserción por grupo (T0, v2)",
        "",
        _md_table(t0_rates),
        "## 10. Tasas de deserción por grupo (T1, v2)",
        "",
        _md_table(t1_rates),
        "## 11. Correlaciones altas (|Spearman| ≥ 0,30, v2)",
        "",
        _md_table(pairs) if not pairs.empty else "_No hay pares con |r| ≥ 0,30._\n",
        "## 12. Figuras generadas",
        "",
    ]
    for path in figures:
        lines.append(f"- `{path.relative_to(PROJECT_ROOT).as_posix()}`")
    if not pairs.empty:
        lines.append(
            "- `reports/figures/10_correlaciones_altas.csv` (pares con |Spearman| ≥ 0,30)"
        )
    lines.append("")
    TABLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLES_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    processed, v1, v2, raw = load_data()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    quality = quality_table(processed, v1, v2, raw)
    balance = balance_table(processed, v1, v2)
    availability = availability_table(raw)
    cohort_effect = cohort_effect_table(processed)
    aucs = auc_table(processed)
    artifacts = artifact_table(processed)
    t0_summary = feature_summary_table(v2, FEATURES_T0)
    t1_summary = feature_summary_table(v2, FEATURES_T1)
    t0_rates = pd.concat(
        [
            _rate_table(v2, "nivel_socioeconomico_ord", "SES ordinal", order=[0, 1, 2, 4]),
            _rate_table(v2, "tipo_fisura", "Tipo de fisura", order=["Completa", "Incompleta", "Asimetrica"]),
            _rate_table(v2, "apoyo_psicologico", "Apoyo psicológico", order=[0, 1]),
            _rate_table(v2, "dificultad_alimentacion_preop", "Dificultad alimentación", order=[0, 1]),
        ],
        ignore_index=True,
    )
    t1_rates = pd.concat(
        [
            _rate_table(v2, "grado_hipernasalidad", "Grado de hipernasalidad", order=[0, 1, 2, 3]),
            _rate_table(v2, "insuficiencia_velofaringea_ivf", "IVF", order=[0, 1]),
            _rate_table(v2, "hipoacusia_conductiva", "Hipoacusia conductiva", order=[0, 1]),
            _rate_table(v2, "dehiscencia_herida_postop", "Dehiscencia", order=[0, 1]),
            _rate_table(v2, "fistula_palatina_residual", "Fístula residual", order=[0, 1]),
        ],
        ignore_index=True,
    )
    pairs = correlation_pairs(v2)

    figures = [
        figure_balance(processed, v1, v2),
        figure_availability(availability),
        figure_cohort_effect(processed),
        figure_leakage_noshow(processed, v1, v2),
        figure_leakage_therapy(v2),
        figure_etl_artifacts(processed, v1, v2),
        figure_t0_distributions(v2),
        figure_t0_rates(v2),
        figure_t1_speech(v2),
        figure_correlation(v2, pairs),
        figure_age_inconsistency(processed),
        figure_ses_cohort(processed),
    ]

    write_tables(
        quality,
        balance,
        availability,
        cohort_effect,
        aucs,
        artifacts,
        t0_summary,
        t1_summary,
        t0_rates,
        t1_rates,
        pairs,
        figures,
    )

    print("=== EDA · deserción en fisura labiopalatina ===")
    print(f"Filas: {len(processed)} | v1={len(v1)} | v2={len(v2)}")
    print(f"Figuras: {len(figures)} en {FIGURES_DIR}")
    print(f"Tablas: {TABLES_PATH}")
    print("\nHallazgos de control:")
    print(f"- No desertores en v1: {int((v1[TARGET] == 0).sum())}")
    print(
        "- Citas_Asistidas en v2: "
        f"{v2['citas_asistidas'].nunique()} valor(es), "
        f"constante={v2['citas_asistidas'].nunique() == 1}"
    )
    ratio_auc = _auc(v2[TARGET], v2["ratio_asistencia_terapia"])
    print(f"- AUC v2 de ratio_asistencia_terapia: {ratio_auc:.3f}")
    age_bad = int((processed["edad_inicio_meses"] > processed["edad_queiloplastia_meses"]).sum())
    print(f"- Edad de inicio posterior a queiloplastia: {age_bad} pacientes")
    print("- Revisar reports/eda.md para la interpretación.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
