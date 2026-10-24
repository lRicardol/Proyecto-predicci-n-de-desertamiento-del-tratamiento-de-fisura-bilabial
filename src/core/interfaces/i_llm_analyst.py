"""Contrato atómico del analista conversacional (ISP: no incluye predicción)."""

from abc import ABC, abstractmethod


class ILLMAnalyst(ABC):
    """Puerto Strategy para consultas estadísticas en lenguaje natural y gráficos."""

    @abstractmethod
    def analyze_data_and_plot(
        self,
        query: str,
        data_path: str,
        output_image_path: str,
    ) -> str:
        """Analiza el CSV indicado, persiste un gráfico y retorna un informe textual."""
