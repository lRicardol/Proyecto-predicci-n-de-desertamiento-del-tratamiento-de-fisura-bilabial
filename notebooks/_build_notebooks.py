"""Genera los notebooks de Fase 1 y Fase 2 (formato profesional con Markdown + código)."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent


def md(source: str) -> dict:
    return new_markdown_cell(source.strip() + "\n")


def code(source: str) -> dict:
    return new_code_cell(source.strip() + "\n")


SETUP = """
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
plt.rcParams.update({
    "figure.figsize": (10, 5.5),
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 110,
})

CANDIDATES = [Path.cwd(), Path.cwd().parent]
ROOT = next(
    path for path in CANDIDATES
    if (path / "data" / "processed" / "dataset_final_limpio.csv").exists()
    or (path / "data" / "raw" / "dataset_desercion_fisura_bilabial.csv").exists()
)
PROCESSED = ROOT / "data" / "processed" / "dataset_final_limpio.csv"
RAW = ROOT / "data" / "raw"
print(f"Raíz del proyecto: {ROOT}")
"""


def build_00() -> None:
    nb = new_notebook(
        cells=[
            md("""
# 00 · Ingeniería de datos: unificación de los 6 CSV reales

**Proyecto:** predicción de deserción del tratamiento de fisura labiopalatina.

Este notebook documenta el pipeline ETL que consume los archivos **reales** de `data/raw/`
(no datos simulados). La implementación ejecutable está en
`notebooks/00_procesamiento_dataset_unificado.py`.

## Objetivo
Construir `data/processed/dataset_final_limpio.csv` con **una fila por paciente**,
etiqueta `desertor` y variables clínicas, quirúrgicas, de terapia y socioeconómicas
alineadas entre las cohortes v1 (PAC-1000) y v2 (PAC-2000).

## Fuentes
| Archivo | Rol | Llave |
|---|---|---|
| `dataset_desercion_fisura_bilabial.csv` | Deserción y contexto v1 | `ID_Paciente` |
| `dataset_hitos_quirurgicos_complicaciones.csv` | Cirugía v1 | `ID_Paciente` |
| `dataset_terapia_lenguaje_ivf.csv` | Terapia / IVF v1 | `ID_Paciente` |
| `dataset_v2_desercion_socioeconomica.csv` (alias: `dataset_desercion_socioeconomica.csv`) | Deserción v2 | `ID_Paciente` |
| `dataset_v2_hitos_quirurgicos.csv` | Cirugía v2 | `ID_Paciente` |
| `dataset_v2_terapia_lenguaje.csv` | Terapia v2 | `ID_Paciente` |

## Criterios de merge y nulos
- Join **left** desde las tablas con `Desertor`: sin etiqueta no hay caso de modelado.
- Pacientes v1 con cirugía/terapia pero **sin** ficha de deserción se excluyen (no se imputa el target).
- Numéricas: imputación por **mediana de cohorte** (evita mezclar escalas v1/v2).
- `Cirugias_Programadas` no existe en v2: se estima con 2 hitos protocolarios + complicaciones.
- Categóricas: `tipo_fisura` armonizado (se retira el prefijo "Bilateral") y SES ordinal 0–4.
            """),
            code(SETUP),
            code("""
from importlib.machinery import SourceFileLoader

etl = SourceFileLoader(
    "etl_fisura",
    str(ROOT / "notebooks" / "00_procesamiento_dataset_unificado.py"),
).load_module()

inventario = []
for clave, archivo in etl.RAW_FILES.items():
    path = etl.resolve_raw_path(archivo)
    tmp = pd.read_csv(path)
    inventario.append({
        "clave": clave,
        "archivo": path.name,
        "filas": len(tmp),
        "columnas": tmp.shape[1],
        "ids_unicos": tmp["ID_Paciente"].nunique(),
        "tiene_desertor": "Desertor" in tmp.columns,
    })
inventario_df = pd.DataFrame(inventario)
inventario_df
"""),
            md("""
## Inventario y calidad de llaves

Se verifica que cada CSV tenga `ID_Paciente`, que no haya IDs vacíos y que el merge
v1/v2 no duplique pacientes entre cohortes.
            """),
            code("""
dataset = etl.build_unified_dataset()
output = etl.save_processed(dataset)
print(f"Guardado en: {output}")
print(f"Forma final: {dataset.shape}")
print("Nulos por columna:")
print(dataset.isna().sum().sort_values(ascending=False).head(10))
print("\\nCohortes:", dataset["cohorte"].value_counts().to_dict())
print("Tasa de deserción global:", round(float(dataset["desertor"].mean()), 4))
print("Tasa por cohorte:")
print(dataset.groupby("cohorte")["desertor"].mean())
dataset.head()
"""),
            md("""
## Conclusiones del procesamiento
- El dataset analítico es el producto de **6 archivos reales**, no de un mock.
- La unidad de análisis es el paciente (`id_paciente`).
- Las variables están en `snake_case`, con dummies de fisura y SES ordinal listos para XGBoost.
- El desbalance de `desertor` es estructural (más marcado en v1) y debe tratarse con `scale_pos_weight` en modelado, no borrando casos.
            """),
        ]
    )
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    path = HERE / "00_procesamiento_dataset_unificado.ipynb"
    nbformat.write(nb, path)
    print(f"escrito {path.name}")


def build_01() -> None:
    nb = new_notebook(
        cells=[
            md("""
# 01 · Exploración clínica, hitos quirúrgicos y terapia de lenguaje

## Introducción
La rehabilitación de la fisura labiopalatina no se reduce a una cirugía: combina
**queiloplastia**, **palatoplastia**, control de complicaciones (dehiscencia, fístula)
y un programa prolongado de **terapia de lenguaje**, a menudo con
insuficiencia velofaríngea (IVF) e hipernasalidad.

Este notebook usa el dataset unificado `data/processed/dataset_final_limpio.csv`,
derivado de los seis CSV de `data/raw/`.

## Preguntas
1. ¿A qué edad ocurren los hitos quirúrgicos y cuál es el intervalo entre ellos?
2. ¿Cómo se relacionan las complicaciones con el número de cirugías?
3. ¿La asistencia a terapia, la IVF y la hipernasalidad describen trayectorias distintas?
            """),
            code(SETUP),
            code("""
if not PROCESSED.exists():
    raise FileNotFoundError(
        "No existe dataset_final_limpio.csv. Ejecuta primero "
        "notebooks/00_procesamiento_dataset_unificado.py"
    )
df = pd.read_csv(PROCESSED)
print(df.shape)
print(df.dtypes)
df.head()
"""),
            md("""
## Variables analizadas
| Grupo | Variables |
|---|---|
| Identidad | `id_paciente`, `cohorte`, `tipo_fisura` |
| Hitos quirúrgicos | `edad_queiloplastia_meses`, `edad_palatoplastia_meses`, `intervalo_cirugias_meses`, `cirugias_programadas` |
| Complicaciones | `dehiscencia_herida_postop`, `fistula_palatina_residual`, `complicaciones_quirurgicas`, `dificultad_alimentacion_preop` |
| Terapia | `citas_terapia_programadas`, `citas_terapia_asistidas`, `ratio_asistencia_terapia` |
| Habla / oído | `grado_hipernasalidad`, `insuficiencia_velofaringea_ivf`, `hipoacusia_conductiva` |
| Crecimiento | `peso_z_score_2anos` |
            """),
            code("""
clinicas = [
    "edad_queiloplastia_meses",
    "edad_palatoplastia_meses",
    "intervalo_cirugias_meses",
    "cirugias_programadas",
    "complicaciones_quirurgicas",
    "citas_terapia_programadas",
    "citas_terapia_asistidas",
    "ratio_asistencia_terapia",
    "grado_hipernasalidad",
    "peso_z_score_2anos",
]
df[clinicas].describe().T.round(2)
"""),
            md("""
## Hitos quirúrgicos: edades y ventana entre labio y paladar
            """),
            code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.kdeplot(
    data=df, x="edad_queiloplastia_meses", hue="cohorte",
    fill=True, common_norm=False, ax=axes[0],
)
axes[0].set_title("Edad de queiloplastia (meses)")
sns.kdeplot(
    data=df, x="edad_palatoplastia_meses", hue="cohorte",
    fill=True, common_norm=False, ax=axes[1],
)
axes[1].set_title("Edad de palatoplastia (meses)")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(
    data=df, x="cohorte", y="intervalo_cirugias_meses",
    hue="tipo_fisura", ax=ax,
)
ax.set_title("Intervalo queiloplastia → palatoplastia según tipo de fisura")
ax.set_ylabel("Meses entre cirugías")
plt.show()
"""),
            md("""
## Número de cirugías y complicaciones
`cirugias_programadas` en v1 viene del CSV de deserción; en v2 se reconstruye
con hitos protocolarios más reoperaciones asociadas a dehiscencia o fístula.
            """),
            code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(
    data=df, x="cirugias_programadas", hue="cohorte", ax=axes[0],
)
axes[0].set_title("Distribución del número de cirugías")
sns.barplot(
    data=df,
    x="complicaciones_quirurgicas",
    y="cirugias_programadas",
    hue="cohorte",
    errorbar=("ci", 95),
    ax=axes[1],
)
axes[1].set_title("Cirugías medias según carga de complicaciones")
axes[1].set_xlabel("N.º de complicaciones (dehiscencia + fístula)")
fig.tight_layout()
plt.show()

comp_rate = (
    df.assign(
        dehiscencia=df["dehiscencia_herida_postop"].astype(int),
        fistula=df["fistula_palatina_residual"].astype(int),
    )
    .groupby("cohorte")[["dehiscencia", "fistula", "dificultad_alimentacion_preop"]]
    .mean()
    .round(3)
)
comp_rate
"""),
            md("""
## Terapia de lenguaje, IVF e hipernasalidad
            """),
            code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(
    data=df.sample(min(len(df), 800), random_state=42),
    x="citas_terapia_programadas",
    y="citas_terapia_asistidas",
    hue="insuficiencia_velofaringea_ivf",
    alpha=0.65,
    ax=axes[0],
)
axes[0].plot([0, 50], [0, 50], ls="--", c="gray", lw=1)
axes[0].set_title("Asistencia vs citas programadas de terapia")

sns.violinplot(
    data=df,
    x="grado_hipernasalidad",
    y="ratio_asistencia_terapia",
    hue="insuficiencia_velofaringea_ivf",
    split=True,
    inner="quartile",
    ax=axes[1],
)
axes[1].set_title("Adherencia a terapia según hipernasalidad e IVF")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(9, 5))
heat = pd.crosstab(
    df["hipoacusia_conductiva"].astype(int),
    df["insuficiencia_velofaringea_ivf"].astype(int),
    normalize="index",
)
sns.heatmap(heat, annot=True, fmt=".1%", cmap="YlGnBu", ax=ax)
ax.set_xlabel("IVF (0/1)")
ax.set_ylabel("Hipoacusia conductiva (0/1)")
ax.set_title("Co-ocurrencia hipoacusia × insuficiencia velofaríngea")
plt.show()
"""),
            md("""
## Estado nutricional a los 2 años y tipo de fisura
            """),
            code("""
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxenplot(
    data=df, x="tipo_fisura", y="peso_z_score_2anos", hue="cohorte", ax=ax,
)
ax.axhline(0, color="gray", ls="--", lw=1)
ax.set_title("Z-score de peso a los 2 años según morfología de la fisura")
plt.show()

resumen = (
    df.groupby(["cohorte", "tipo_fisura"], as_index=False)
    .agg(
        n=("id_paciente", "count"),
        queiloplastia_mediana=("edad_queiloplastia_meses", "median"),
        palatoplastia_mediana=("edad_palatoplastia_meses", "median"),
        ratio_terapia=("ratio_asistencia_terapia", "mean"),
        tasa_ivf=("insuficiencia_velofaringea_ivf", "mean"),
        z_peso=("peso_z_score_2anos", "mean"),
    )
    .round(3)
)
resumen
"""),
            md("""
## Conclusiones clínicas
1. Los hitos de **labio y paladar** se concentran en el primer y segundo año de vida;
   el intervalo entre cirugías es un indicador de continuidad del protocolo, no de
   deserción por sí mismo.
2. La **carga de complicaciones** (fístula, dehiscencia) se asocia con más procedimientos:
   es una variable de exposición clínica, no un sustituto del no-show.
3. La **asistencia a terapia** y la IVF describen la fase larga del tratamiento:
   pacientes con hipernasalidad alta e IVF requieren seguimiento fonoaudiológico
   prolongado, donde la deserción es más costosa funcionalmente.
4. El z-score ponderal a los 2 años resume dificultad alimentaria preoperatoria y
   acceso a controles; debe leerse junto al contexto socioeconómico (notebook 02).
            """),
        ]
    )
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    path = HERE / "01_exploracion_clinica_y_terapia.ipynb"
    nbformat.write(nb, path)
    print(f"escrito {path.name}")


def build_02() -> None:
    nb = new_notebook(
        cells=[
            md("""
# 02 · Análisis predictivo de deserción

## Introducción
La variable objetivo `desertor` indica si el paciente abandonó el itinerario
terapéutico. En fisura labiopalatina la deserción suele concentrarse cuando
aumentan las **citas perdidas (no-show)**, la **distancia al centro** y la
**vulnerabilidad socioeconómica**, y cuando falta apoyo psicológico.

Este notebook trabaja exclusivamente con `dataset_final_limpio.csv`
(unión real de las 6 fuentes de `data/raw/`).

## Hipótesis
- H1: los desertores acumulan más `citas_perdidas_noshow`.
- H2: la tasa de deserción decrece al subir `nivel_socioeconomico_ord`.
- H3: mayor `distancia_hospital_km` se asocia a más deserción, sobre todo sin apoyo psicológico.
            """),
            code(SETUP),
            code("""
df = pd.read_csv(PROCESSED)
print(df.shape)
print(df["desertor"].value_counts().rename({0: "continúa", 1: "deserta"}))
print("Tasa global:", round(float(df["desertor"].mean()), 4))
print(df.groupby("cohorte")["desertor"].agg(["count", "mean"]))
df.head()
"""),
            md("""
## Variables del modelo conceptual
| Bloque | Variables |
|---|---|
| Objetivo | `desertor` |
| Adherencia | `citas_perdidas_noshow`, `citas_asistidas`, `ratio_asistencia_terapia` |
| Acceso | `distancia_hospital_km`, `nivel_socioeconomico`, `ingresos_mensuales_usd`, `apoyo_psicologico` |
| Clínica | `cirugias_programadas`, `complicaciones_quirurgicas`, `tipo_fisura` |
            """),
            md("""
## Desbalance de la etiqueta
v1 está fuertemente sesgada hacia deserción; v2 es más heterogénea.
Eso obliga a `scale_pos_weight` en XGBoost; no se submuestrea el CSV analítico.
            """),
            code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
order = ["continúa", "deserta"]
counts = df["desertor"].map({0: "continúa", 1: "deserta"}).value_counts().reindex(order)
axes[0].bar(counts.index, counts.values, color=["#4c78a8", "#f58518"])
axes[0].set_title("Conteo global de deserción")
axes[0].set_ylabel("Pacientes")
sns.barplot(
    data=df, x="cohorte", y="desertor", errorbar=("ci", 95), ax=axes[1], color="#54a24b",
)
axes[1].set_ylim(0, 1)
axes[1].set_title("Tasa de deserción por cohorte")
fig.tight_layout()
plt.show()
"""),
            md("""
## Boxplot detallado: citas perdidas (no-show) según deserción

Gráfico solicitado de forma explícita: distribución de inasistencias condicionada a `desertor`,
con cuartiles, atípicos y media muestral.
            """),
            code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
palette = {0: "#4c78a8", 1: "#f58518"}
sns.boxplot(
    data=df,
    x="desertor",
    y="citas_perdidas_noshow",
    hue="desertor",
    palette=palette,
    width=0.55,
    fliersize=3,
    legend=False,
    ax=axes[0],
)
sns.stripplot(
    data=df.sample(min(len(df), 600), random_state=7),
    x="desertor",
    y="citas_perdidas_noshow",
    color="black",
    alpha=0.18,
    size=3,
    ax=axes[0],
)
means = df.groupby("desertor")["citas_perdidas_noshow"].mean()
axes[0].scatter([0, 1], means.reindex([0, 1]), color="red", s=60, zorder=5, label="media")
axes[0].set_xticks([0, 1], ["Continúa (0)", "Deserta (1)"])
axes[0].set_ylabel("Citas perdidas (no-show)")
axes[0].set_title("Boxplot: inasistencias según deserción")
axes[0].legend(loc="upper left")

sns.boxplot(
    data=df,
    x="cohorte",
    y="citas_perdidas_noshow",
    hue="desertor",
    palette=palette,
    ax=axes[1],
)
axes[1].set_title("No-shows por cohorte y deserción")
axes[1].legend(title="Desertor")
fig.tight_layout()
plt.show()

stats_noshow = (
    df.groupby("desertor")["citas_perdidas_noshow"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .round(3)
)
stats_noshow.index = stats_noshow.index.map({0: "continúa", 1: "deserta"})
stats_noshow
"""),
            md("""
## Factor socioeconómico y distancia al centro
            """),
            code("""
ses_order = ["Bajo", "Medio-Bajo", "Medio", "Alto"]
present = [nivel for nivel in ses_order if nivel in set(df["nivel_socioeconomico"])]
tasas = (
    df.groupby("nivel_socioeconomico")["desertor"]
    .agg(n="count", tasa="mean", noshow="mean")
    .reindex(present)
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(
    data=df, x="nivel_socioeconomico", y="desertor",
    order=present, errorbar=("ci", 95), ax=axes[0], color="#e45756",
)
axes[0].set_ylim(0, 1)
axes[0].set_title("Tasa de deserción por nivel socioeconómico")
axes[0].tick_params(axis="x", rotation=15)

sns.kdeplot(
    data=df, x="distancia_hospital_km", hue="desertor",
    fill=True, common_norm=False, ax=axes[1], palette=palette,
)
axes[1].set_title("Distancia al hospital según deserción")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(9, 5))
sns.pointplot(
    data=df,
    x="nivel_socioeconomico",
    y="desertor",
    hue="apoyo_psicologico",
    order=present,
    errorbar=("ci", 95),
    ax=ax,
)
ax.set_ylim(0, 1)
ax.set_title("Deserción: nivel socioeconómico × apoyo psicológico")
ax.tick_params(axis="x", rotation=15)
plt.show()
tasas.round(3)
"""),
            md("""
## Correlación con la etiqueta y entre predictores
Se usan solo numéricas del modelo conceptual. Una correlación alta entre no-show y
`desertor` no implica causalidad, pero sí prioridad clínica de seguimiento de inasistencias.
            """),
            code("""
features = [
    "citas_perdidas_noshow",
    "distancia_hospital_km",
    "nivel_socioeconomico_ord",
    "ingresos_mensuales_usd",
    "apoyo_psicologico",
    "cirugias_programadas",
    "complicaciones_quirurgicas",
    "ratio_asistencia_terapia",
    "edad_inicio_meses",
    "desertor",
]
corr = df[features].corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap="vlag", center=0,
    linewidths=0.4, ax=ax,
)
ax.set_title("Matriz de correlación (Pearson)")
plt.show()

asoc = (
    corr["desertor"]
    .drop(labels=["desertor"])
    .sort_values(key=lambda s: s.abs(), ascending=False)
    .to_frame("corr_desertor")
)
asoc
"""),
            code("""
fig, ax = plt.subplots(figsize=(9, 6))
muestra = df.sample(min(len(df), 900), random_state=21)
sns.scatterplot(
    data=muestra,
    x="distancia_hospital_km",
    y="citas_perdidas_noshow",
    hue="desertor",
    style="cohorte",
    alpha=0.7,
    ax=ax,
    palette=palette,
)
ax.set_title("Plano de riesgo: distancia × inasistencias")
plt.show()
"""),
            md("""
## Conclusiones predictivas
1. El **boxplot de no-shows** es la señal más accionable: los desertores concentran
   más citas perdidas, con cola larga de inasistencias reiteradas.
2. El **gradiente socioeconómico** y los **ingresos** se alinean con la deserción;
   el apoyo psicológico atenúa la tasa en varios estratos.
3. La **distancia al hospital** desplaza la masa de desertores hacia trayectos más largos;
   combinada con no-shows define un plano de riesgo de seguimiento.
4. El desbalance de clases (sobre todo v1) impide usar accuracy como métrica única.
   El backend debe entrenar XGBoost con `scale_pos_weight` sobre este CSV limpio.
            """),
        ]
    )
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    path = HERE / "02_analisis_predictivo_desercion.ipynb"
    nbformat.write(nb, path)
    print(f"escrito {path.name}")


if __name__ == "__main__":
    build_00()
    build_01()
    build_02()
