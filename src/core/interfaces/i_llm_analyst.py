"""Contrato del intérprete estadístico local (palabras clave, no un modelo de lenguaje)."""

from abc import ABC, abstractmethod


class IStatisticalAnalyst(ABC):
    """Puerto para consultas en lenguaje natural y gráficos locales."""

    @abstractmethod
    def analyze_data_and_plot(
        self,
        query: str,
        data_path: str,
        output_image_path: str,
    ) -> str:
        """Analiza el CSV indicado, persiste un gráfico PNG y retorna un informe."""


# Nombre histórico. La implementación no llama a un LLM.
ILLMAnalyst = IStatisticalAnalyst
