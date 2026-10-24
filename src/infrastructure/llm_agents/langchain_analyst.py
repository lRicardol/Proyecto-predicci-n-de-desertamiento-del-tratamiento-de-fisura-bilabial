"""Analista LangChain. Implementa ILLMAnalyst y persiste gráficos en static/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.core.exceptions import AnalyticsError
from src.core.interfaces.i_llm_analyst import ILLMAnalyst

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "analyst_system_prompt.txt"


class LangchainAnalyst(ILLMAnalyst):
    """Agente pandas de LangChain con respaldo estadístico local si no hay API key."""

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        provider: str = "openai",
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._model_name = model_name
        self._temperature = temperature
        self._provider = provider.lower()

    def analyze_data_and_plot(
        self,
        query: str,
        data_path: str,
        output_image_path: str,
    ) -> str:
        if not query or not query.strip():
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
        narrative = ""
        if self._api_key:
            try:
                narrative = self._run_langchain_agent(query, frame, image_path)
            except Exception as exc:
                narrative = (
                    "El agente LLM no estuvo disponible "
                    f"({exc}). Se genera un análisis local de respaldo."
                )

        if not image_path.exists():
            local_text = self._run_local_analyst(query, frame, image_path)
            narrative = f"{narrative}\n\n{local_text}".strip() if narrative else local_text

        if not image_path.exists():
            raise AnalyticsError(
                f"No se persistió el gráfico en {image_path}."
            )
        return narrative.strip()

    def _run_langchain_agent(
        self,
        query: str,
        frame: pd.DataFrame,
        image_path: Path,
    ) -> str:
        from langchain_experimental.agents.agent_toolkits import (
            create_pandas_dataframe_agent,
        )
        from langchain_openai import ChatOpenAI

        system_prompt = _load_prompt().format(output_image_path=str(image_path))
        llm = ChatOpenAI(
            model=self._model_name,
            temperature=self._temperature,
            api_key=self._api_key,
        )
        agent = create_pandas_dataframe_agent(
            llm,
            frame,
            verbose=False,
            allow_dangerous_code=True,
            agent_executor_kwargs={"handle_parsing_errors": True},
            prefix=system_prompt,
        )
        instruction = (
            f"{query}\n\n"
            f"Si generas un gráfico, guárdalo en: {image_path}"
        )
        result = agent.invoke({"input": instruction})
        if isinstance(result, dict):
            return str(result.get("output") or result)
        return str(result)

    def _run_local_analyst(
        self,
        query: str,
        frame: pd.DataFrame,
        image_path: Path,
    ) -> str:
        normalized = query.lower()
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 5.5))

        if "inasist" in normalized or "noshow" in normalized or "no-show" in normalized:
            summary = self._plot_no_show(frame, ax)
        elif "distancia" in normalized:
            summary = self._plot_distance(frame, ax)
        elif "socioeconom" in normalized or "ingreso" in normalized:
            summary = self._plot_socioeconomic(frame, ax)
        elif "cirug" in normalized or "fistula" in normalized or "dehisc" in normalized:
            summary = self._plot_surgery(frame, ax)
        else:
            summary = self._plot_dropout_overview(frame, ax)

        fig.tight_layout()
        fig.savefig(image_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return summary

    def _plot_no_show(self, frame: pd.DataFrame, ax: plt.Axes) -> str:
        if "Citas_Perdidas_NoShow" not in frame.columns or "Desertor" not in frame.columns:
            return self._plot_dropout_overview(frame, ax)
        working = frame[["Citas_Perdidas_NoShow", "Desertor"]].dropna()
        sns.boxplot(
            data=working,
            x="Desertor",
            y="Citas_Perdidas_NoShow",
            hue="Desertor",
            ax=ax,
            palette="Set2",
            legend=False,
        )
        ax.set_title("Inasistencias según deserción del tratamiento")
        ax.set_xlabel("Desertor (0 = continúa, 1 = deserta)")
        ax.set_ylabel("Citas perdidas (no-show)")
        means = working.groupby("Desertor")["Citas_Perdidas_NoShow"].mean()
        return (
            "Análisis local: las inasistencias se asocian con la deserción. "
            f"Media de no-shows en continuadores: {means.get(0, float('nan')):.2f}; "
            f"en desertores: {means.get(1, float('nan')):.2f}. "
            f"n={len(working)}."
        )

    def _plot_distance(self, frame: pd.DataFrame, ax: plt.Axes) -> str:
        if "Distancia_Hospital_KM" not in frame.columns or "Desertor" not in frame.columns:
            return self._plot_dropout_overview(frame, ax)
        working = frame[["Distancia_Hospital_KM", "Desertor"]].dropna()
        sns.histplot(
            data=working,
            x="Distancia_Hospital_KM",
            hue="Desertor",
            bins=20,
            kde=True,
            ax=ax,
            palette="coolwarm",
        )
        ax.set_title("Distancia al hospital y deserción")
        mean_km = working["Distancia_Hospital_KM"].mean()
        return (
            f"Análisis local: distancia media al hospital = {mean_km:.1f} km. "
            "Se compara la distribución entre quienes desertan y quienes continúan."
        )

    def _plot_socioeconomic(self, frame: pd.DataFrame, ax: plt.Axes) -> str:
        if "Nivel_Socioeconomico" not in frame.columns or "Desertor" not in frame.columns:
            return self._plot_dropout_overview(frame, ax)
        rates = (
            frame.groupby("Nivel_Socioeconomico")["Desertor"]
            .mean()
            .sort_values(ascending=False)
        )
        rates.plot(kind="bar", color="#c44e52", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Tasa de deserción")
        ax.set_title("Deserción por nivel socioeconómico")
        rendered = ", ".join(f"{idx}: {val:.1%}" for idx, val in rates.items())
        return f"Análisis local de deserción por nivel socioeconómico. {rendered}."

    def _plot_surgery(self, frame: pd.DataFrame, ax: plt.Axes) -> str:
        columns = [
            col
            for col in ("Fistula_Palatina_Residual", "Dehiscencia_Herida_Postop", "Desertor")
            if col in frame.columns
        ]
        if "Desertor" not in columns or len(columns) < 2:
            return self._plot_dropout_overview(frame, ax)
        melted = frame[columns].melt(id_vars="Desertor")
        sns.barplot(
            data=melted,
            x="variable",
            y="value",
            hue="Desertor",
            ax=ax,
            palette="muted",
        )
        ax.set_title("Complicaciones quirúrgicas según deserción")
        ax.set_ylabel("Proporción / valor medio")
        return (
            "Análisis local: se contrastan fístula palatina residual y dehiscencia "
            "postoperatoria entre desertores y continuadores."
        )

    def _plot_dropout_overview(self, frame: pd.DataFrame, ax: plt.Axes) -> str:
        if "Desertor" not in frame.columns:
            frame.dtypes.value_counts().plot(kind="bar", ax=ax, color="#4c72b0")
            ax.set_title("Composición de tipos de columna del dataset")
            return (
                f"El dataset tiene {len(frame)} filas y {len(frame.columns)} columnas. "
                "No hay columna Desertor; se muestra un resumen estructural."
            )
        counts = frame["Desertor"].value_counts().sort_index()
        labels = {0: "Continúa", 1: "Deserta"}
        counts.index = [labels.get(int(idx), str(idx)) for idx in counts.index]
        counts.plot(kind="bar", color=["#55a868", "#c44e52"], ax=ax)
        ax.set_title("Distribución de deserción")
        ax.set_ylabel("Pacientes")
        total = int(counts.sum())
        deserters = int(counts.get("Deserta", 0))
        rate = deserters / total if total else 0.0
        return (
            f"Análisis local: {total} pacientes, {deserters} desertores "
            f"({rate:.1%}). Gráfico de distribución guardado."
        )


def _load_prompt() -> str:
    if not PROMPT_PATH.exists():
        return (
            "Eres un analista clínico. Guarda cualquier gráfico en {output_image_path}."
        )
    return PROMPT_PATH.read_text(encoding="utf-8")
