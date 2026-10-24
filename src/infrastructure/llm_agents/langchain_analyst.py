"""Compatibilidad: el analista LangChain se sustituyó por el motor local."""

from src.infrastructure.llm_agents.local_analyst import LocalStatisticalAnalyst

LangchainAnalyst = LocalStatisticalAnalyst
