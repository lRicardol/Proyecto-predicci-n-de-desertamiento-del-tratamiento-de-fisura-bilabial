"""Caso de uso: consulta en lenguaje natural → analista local → chart_output.png."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time

from src.core.exceptions import AnalyticsError
from src.core.interfaces.i_llm_analyst import ILLMAnalyst

CHART_FILENAME = "chart_output.png"


@dataclass(frozen=True)
class AnalyticsResult:
    query: str
    narrative: str
    image_path: str
    image_url: str


class GenerateAnalyticsUseCase:
    def __init__(
        self,
        llm_analyst: ILLMAnalyst,
        dataset_path: Path,
        charts_dir: Path,
    ) -> None:
        self._llm_analyst = llm_analyst
        self._dataset_path = Path(dataset_path)
        self._charts_dir = Path(charts_dir)

    def execute(self, query: str) -> AnalyticsResult:
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            raise AnalyticsError("La consulta en lenguaje natural es obligatoria.")
        if not self._dataset_path.exists():
            raise AnalyticsError(f"No existe el dataset {self._dataset_path}.")

        self._charts_dir.mkdir(parents=True, exist_ok=True)
        image_path = self._charts_dir / CHART_FILENAME
        try:
            narrative = self._llm_analyst.analyze_data_and_plot(
                query=cleaned_query,
                data_path=str(self._dataset_path),
                output_image_path=str(image_path),
            )
        except AnalyticsError:
            raise
        except Exception as exc:
            raise AnalyticsError(f"Fallo del analista local: {exc}") from exc

        if not image_path.exists():
            raise AnalyticsError("El analista no almacenó chart_output.png.")
        cache_bust = int(time())
        return AnalyticsResult(
            query=cleaned_query,
            narrative=narrative,
            image_path=str(image_path),
            image_url=f"/static/generated_charts/{CHART_FILENAME}?t={cache_bust}",
        )
