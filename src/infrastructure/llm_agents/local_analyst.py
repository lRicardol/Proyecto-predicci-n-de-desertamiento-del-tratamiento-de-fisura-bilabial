"""Analista estadístico local: reglas de consulta + Matplotlib/Seaborn. Sin APIs."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.core.exceptions import AnalyticsError
from src.core.interfaces.i_llm_analyst import IStatisticalAnalyst

SES_ORDER = ("Bajo", "Medio-Bajo", "Medio", "Alto")
PALETTE = {0: "#4c78a8", 1: "#e45756"}


_EXAMPLES = (
    "inasistencias y deserción",
    "nivel socioeconómico",
    "distancia al hospital",
    "complicaciones quirúrgicas",
    "terapia de lenguaje",
    "distribución de deserción",
    "matriz de correlación",
    "comparar grupos de deserción",
)

_RETROSPECTIVE = (
    " Aviso: esta variable resume el tratamiento ya ocurrido; no sirve para predecir al ingreso "
    "y la asociación no implica causalidad."
)


class LocalStatisticalAnalyst(IStatisticalAnalyst):
    """Mapea lenguaje natural a gráficos equivalentes a los notebooks 01 y 02."""

    def analyze_data_and_plot(
        self,
        query: str,
        data_path: str,
        output_image_path: str,
    ) -> str:
        cleaned = (query or "").strip()
        if not cleaned:
            raise AnalyticsError("La consulta analítica no puede estar vacía.")

        csv_path = Path(data_path)
        image_path = Path(output_image_path)
        if not csv_path.exists():
            raise AnalyticsError(f"No existe el dataset {csv_path}.")

        try:
            frame = pd.read_csv(csv_path)
        except Exception as exc:
            raise AnalyticsError(f"No se pudo leer {csv_path.name}: {exc}") from exc
        if frame.empty:
            raise AnalyticsError("El dataset analítico está vacío.")

        image_path.parent.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="talk")

        try:
            narrative = self._dispatch(_fold_accents(cleaned), frame, image_path)
        except AnalyticsError:
            raise
        except Exception as exc:
            plt.close("all")
            raise AnalyticsError(f"No se pudo generar el gráfico: {exc}") from exc

        if not image_path.exists() or image_path.stat().st_size == 0:
            raise AnalyticsError(f"No se persistió el gráfico en {image_path}.")
        return narrative

    def _dispatch(self, query: str, frame: pd.DataFrame, image_path: Path) -> str:
        intents = (
            (("inasistencia", "noshow", "no-show", "no show", "citas perdidas"), self._plot_noshow),
            (("socioeconom", "ses", "ingreso", "pobreza"), self._plot_socioeconomic),
            (("distancia", "km", "viaje"), self._plot_distance),
            (("cirug", "complic", "fistula", "dehisc", "quirurg"), self._plot_surgery),
            (("terapia", "lenguaje", "ivf", "hipernasal"), self._plot_therapy),
            (("correl", "heatmap", "matriz"), self._plot_correlation),
            (("compar", "grupo"), self._plot_group_comparison),
            (("distrib", "etiqueta", "cuantos desert", "tasa de desert"), self._plot_dropout_overview),
        )
        words = set(re.findall(r"\w+", query))
        for keywords, handler in intents:
            if any(_matches(keyword, query, words) for keyword in keywords):
                return handler(frame, image_path)
        examples = "; ".join(_EXAMPLES)
        raise AnalyticsError(
            "No se reconoció la consulta. Pruebe con: " + examples + "."
        )

    def _plot_noshow(self, frame: pd.DataFrame, image_path: Path) -> str:
        _require(frame, ["citas_perdidas_noshow", "desertor"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        sns.boxplot(
            data=frame,
            x="desertor",
            y="citas_perdidas_noshow",
            hue="desertor",
            palette=PALETTE,
            width=0.55,
            fliersize=3,
            legend=False,
            ax=axes[0],
        )
        sample = frame.sample(min(len(frame), 600), random_state=7)
        sns.stripplot(
            data=sample,
            x="desertor",
            y="citas_perdidas_noshow",
            color="black",
            alpha=0.18,
            size=3,
            ax=axes[0],
        )
        means = frame.groupby("desertor")["citas_perdidas_noshow"].mean()
        axes[0].scatter(
            [0, 1],
            means.reindex([0, 1]),
            color="#b91c1c",
            s=70,
            zorder=5,
            label="media",
        )
        axes[0].set_xticks([0, 1], ["Continúa (0)", "Deserta (1)"])
        axes[0].set_title("Asociación: inasistencias según deserción (retrospectiva)")
        axes[0].set_ylabel("Citas perdidas (no-show)")
        axes[0].legend(loc="upper left")

        if "cohorte" in frame.columns:
            sns.boxplot(
                data=frame,
                x="cohorte",
                y="citas_perdidas_noshow",
                hue="desertor",
                palette=PALETTE,
                ax=axes[1],
            )
            axes[1].set_title("No-shows por cohorte y deserción")
            axes[1].legend(title="Desertor")
        else:
            axes[1].axis("off")

        _save(fig, image_path)
        return (
            "Boxplot de inasistencias: "
            f"media en continuadores = {means.get(0, float('nan')):.2f}; "
            f"en desertores = {means.get(1, float('nan')):.2f}. "
            f"n={len(frame)}. Unidad: citas perdidas acumuladas."
            + _RETROSPECTIVE
        )

    def _plot_socioeconomic(self, frame: pd.DataFrame, image_path: Path) -> str:
        _require(frame, ["nivel_socioeconomico", "desertor"])
        present = [nivel for nivel in SES_ORDER if nivel in set(frame["nivel_socioeconomico"])]
        tasas = frame.groupby("nivel_socioeconomico")["desertor"].mean()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
        sns.barplot(
            data=frame,
            x="nivel_socioeconomico",
            y="desertor",
            order=present,
            errorbar=("ci", 95),
            color="#e45756",
            ax=axes[0],
        )
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Tasa de deserción")
        axes[0].set_title("Asociación entre nivel socioeconómico y deserción")

        if "apoyo_psicologico" in frame.columns:
            sns.pointplot(
                data=frame,
                x="nivel_socioeconomico",
                y="desertor",
                hue="apoyo_psicologico",
                order=present,
                errorbar=("ci", 95),
                ax=axes[1],
            )
            axes[1].set_ylim(0, 1)
            axes[1].set_title("SES × apoyo psicológico")
        else:
            axes[1].axis("off")

        _save(fig, image_path)
        rendered = ", ".join(f"{idx}: {tasas.get(idx, float('nan')):.1%}" for idx in present)
        return (
            f"Tasa de deserción por nivel socioeconómico (n={len(frame)}). {rendered}. "
            "Es una asociación, no un efecto causal."
        )

    def _plot_distance(self, frame: pd.DataFrame, image_path: Path) -> str:
        _require(frame, ["distancia_hospital_km", "desertor"])
        fig, ax = plt.subplots(figsize=(10, 5.4))
        sns.kdeplot(
            data=frame,
            x="distancia_hospital_km",
            hue="desertor",
            fill=True,
            common_norm=False,
            palette=PALETTE,
            ax=ax,
        )
        ax.set_title("Distribución de distancia al centro médico")
        ax.set_xlabel("Distancia (km)")
        _save(fig, image_path)
        means = frame.groupby("desertor")["distancia_hospital_km"].mean()
        return (
            "Densidad de distancia al hospital. "
            f"Media continuadores = {means.get(0, float('nan')):.1f} km; "
            f"desertores = {means.get(1, float('nan')):.1f} km."
        )

    def _plot_surgery(self, frame: pd.DataFrame, image_path: Path) -> str:
        _require(frame, ["complicaciones_quirurgicas", "cirugias_programadas", "desertor"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
        sns.countplot(data=frame, x="cirugias_programadas", hue="desertor", palette=PALETTE, ax=axes[0])
        axes[0].set_title("Cirugías programadas según deserción")
        sns.barplot(
            data=frame,
            x="complicaciones_quirurgicas",
            y="cirugias_programadas",
            hue="desertor",
            palette=PALETTE,
            errorbar=("ci", 95),
            ax=axes[1],
        )
        axes[1].set_title("Carga operatoria vs complicaciones")
        axes[1].set_xlabel("N.º de complicaciones (dehiscencia + fístula)")
        _save(fig, image_path)
        rate = float(frame["complicaciones_quirurgicas"].gt(0).mean())
        return (
            f"El {rate:.1%} de pacientes tiene al menos una complicación quirúrgica. "
            "El gráfico contrapone carga operatoria y deserción."
        )

    def _plot_therapy(self, frame: pd.DataFrame, image_path: Path) -> str:
        _require(frame, ["ratio_asistencia_terapia", "grado_hipernasalidad"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
        hue = "insuficiencia_velofaringea_ivf" if "insuficiencia_velofaringea_ivf" in frame.columns else None
        if {"citas_terapia_programadas", "citas_terapia_asistidas"}.issubset(frame.columns):
            sample = frame.sample(min(len(frame), 800), random_state=42)
            sns.scatterplot(
                data=sample,
                x="citas_terapia_programadas",
                y="citas_terapia_asistidas",
                hue=hue or "grado_hipernasalidad",
                alpha=0.65,
                ax=axes[0],
            )
            axes[0].plot([0, 50], [0, 50], ls="--", c="gray", lw=1)
            axes[0].set_title("Asistencia vs citas programadas de terapia")
        else:
            axes[0].axis("off")
        sns.violinplot(
            data=frame,
            x="grado_hipernasalidad",
            y="ratio_asistencia_terapia",
            hue=hue,
            split=bool(hue),
            inner="quartile",
            ax=axes[1],
        )
        axes[1].set_title("Adherencia a terapia según hipernasalidad")
        axes[1].set_ylabel("Ratio de asistencia")
        _save(fig, image_path)
        mean_ratio = float(frame["ratio_asistencia_terapia"].mean())
        return (
            f"Asistencia media a terapia = {mean_ratio:.1%}. "
            "Se muestra la distribución según grado de hipernasalidad."
        )

    def _plot_correlation(self, frame: pd.DataFrame, image_path: Path) -> str:
        cols = [
            "citas_perdidas_noshow",
            "distancia_hospital_km",
            "nivel_socioeconomico_ord",
            "complicaciones_quirurgicas",
            "desertor",
        ]
        extras = [
            "ingresos_mensuales_usd",
            "apoyo_psicologico",
            "cirugias_programadas",
            "ratio_asistencia_terapia",
            "edad_inicio_meses",
        ]
        selected = cols + [col for col in extras if col in frame.columns]
        _require(frame, cols)
        # Spearman: las variables son continuas, ordinales y binarias; no se asume linealidad.
        corr = frame[selected].corr(method="spearman", numeric_only=True)
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            corr,
            annot=False,
            cmap="vlag",
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.4,
            cbar_kws={"label": "Correlación de Spearman"},
            ax=ax,
        )
        ax.set_title("Matriz de correlación (Spearman)")
        _save(fig, image_path)
        corr_noshow = float(corr.loc["citas_perdidas_noshow", "desertor"])
        return f"Matriz de correlación de Spearman. No-show vs deserción: rho = {corr_noshow:.2f}."

    def _plot_group_comparison(self, frame: pd.DataFrame, image_path: Path) -> str:
        _require(frame, ["desertor"])
        group_column = next(
            (column for column in ("tipo_fisura", "nivel_socioeconomico", "cohorte") if column in frame.columns),
            None,
        )
        if group_column is None:
            raise AnalyticsError(
                "No hay columna de agrupación (tipo_fisura, nivel_socioeconomico o cohorte) en el CSV."
            )
        rates = frame.groupby(group_column, dropna=False)["desertor"].agg(["mean", "count"])
        rates = rates[rates["count"] > 0]
        fig, ax = plt.subplots(figsize=(9, 5.2))
        sns.barplot(
            data=frame,
            x=group_column,
            y="desertor",
            errorbar=("ci", 95),
            color="#4c78a8",
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Tasa de deserción")
        ax.set_title(f"Asociación: tasa de deserción por {group_column.replace('_', ' ')}")
        _save(fig, image_path)
        rendered = "; ".join(
            f"{index}: {row['mean']:.1%} (n={int(row['count'])})" for index, row in rates.iterrows()
        )
        return (
            f"Comparación de grupos por {group_column} (n={len(frame)}). {rendered}. "
            "Son asociaciones observadas, no efectos causales."
        )

    def _plot_dropout_overview(self, frame: pd.DataFrame, image_path: Path) -> str:
        _require(frame, ["desertor"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        counts = frame["desertor"].map({0: "continúa", 1: "deserta"}).value_counts().reindex(
            ["continúa", "deserta"]
        )
        axes[0].bar(counts.index, counts.values, color=["#4c78a8", "#e45756"])
        axes[0].set_ylabel("Pacientes")
        axes[0].set_title("Distribución de deserción")
        if "cohorte" in frame.columns:
            sns.barplot(
                data=frame,
                x="cohorte",
                y="desertor",
                errorbar=("ci", 95),
                color="#54a24b",
                ax=axes[1],
            )
            axes[1].set_ylim(0, 1)
            axes[1].set_title("Tasa de deserción por cohorte")
        else:
            axes[1].axis("off")
        _save(fig, image_path)
        total = int(counts.sum())
        deserters = int(counts.get("deserta", 0))
        return (
            f"Distribución de la etiqueta: {deserters}/{total} desertores "
            f"({deserters / total:.1%}). "
            "Prueba consultas como 'inasistencias', 'nivel socioeconómico' o 'distancia'."
        )


def _fold_accents(text: str) -> str:
    """Minúsculas y sin tildes, para que "socioeconómico" y "socioeconomico" den lo mismo."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _matches(keyword: str, query: str, words: set[str]) -> bool:
    """"ses" es una sigla: solo cuenta como palabra completa (no dentro de "sesión")."""
    if keyword == "ses":
        return keyword in words
    return keyword in query


def _require(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise AnalyticsError(f"El CSV limpio no tiene las columnas {missing}.")


def _save(fig: plt.Figure, image_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(image_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
