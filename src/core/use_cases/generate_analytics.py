"""Caso de uso: consulta en lenguaje natural → analista local → PNG con nombre único."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time
from uuid import uuid4

from src.core.exceptions import AnalyticsError
from src.core.interfaces.i_llm_analyst import IStatisticalAnalyst

_MAX_CHARTS = 24


@dataclass(frozen=True)
class AnalyticsResult:
    query: str
    narrative: str
    image_path: str
    image_url: str


class GenerateAnalyticsUseCase:
    def __init__(
        self,
        llm_analyst: IStatisticalAnalyst,
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
        filename = f"chart_{uuid4().hex}.png"
        image_path = self._charts_dir / filename
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
            raise AnalyticsError("El analista no almacenó el gráfico.")
        self._prune_charts()
        cache_bust = int(time())
        return AnalyticsResult(
            query=cleaned_query,
            narrative=narrative,
            image_path=str(image_path),
            image_url=f"/static/generated_charts/{filename}?t={cache_bust}",
        )

    def _prune_charts(self) -> None:
        charts = sorted(self._charts_dir.glob("chart_*.png"), key=lambda path: path.stat().st_mtime)
        for stale in charts[:-_MAX_CHARTS]:
            stale.unlink(missing_ok=True)
