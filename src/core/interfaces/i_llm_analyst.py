"""Contrato atómico del analista estadístico local (ISP: no incluye predicción)."""

from abc import ABC, abstractmethod


class ILLMAnalyst(ABC):
    """Puerto Strategy para consultas en lenguaje natural y gráficos locales."""

    @abstractmethod
    def analyze_data_and_plot(
        self,
        query: str,
        data_path: str,
        output_image_path: str,
    ) -> str:
        """Analiza el CSV indicado, persiste un gráfico PNG y retorna un informe."""
