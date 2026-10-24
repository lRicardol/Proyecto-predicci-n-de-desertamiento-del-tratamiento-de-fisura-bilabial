"""Caso de uso: analítica conversacional y persistencia del gráfico."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import uuid

from src.core.exceptions import AnalyticsError
from src.core.interfaces.i_llm_analyst import ILLMAnalyst


@dataclass(frozen=True)
class AnalyticsResult:
    query: str
    narrative: str
    image_path: str
    image_url: str


class GenerateAnalyticsUseCase:
    """Delega al puerto ILLMAnalyst y garantiza que el PNG quede en static/."""

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
        image_path = self._charts_dir / self._build_filename(cleaned_query)

        narrative = self._llm_analyst.analyze_data_and_plot(
            query=cleaned_query,
            data_path=str(self._dataset_path),
            output_image_path=str(image_path),
        )
        if not image_path.exists():
            raise AnalyticsError(
                "El analista no almacenó el gráfico en el directorio estático."
            )
        return AnalyticsResult(
            query=cleaned_query,
            narrative=narrative,
            image_path=str(image_path),
            image_url=f"/static/generated_charts/{image_path.name}",
        )

    def _build_filename(self, query: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", query.lower()).strip("-")[:40]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        token = uuid.uuid4().hex[:8]
        safe_slug = slug or "consulta"
        return f"{stamp}_{safe_slug}_{token}.png"
