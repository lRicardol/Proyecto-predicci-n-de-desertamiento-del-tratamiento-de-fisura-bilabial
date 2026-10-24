"""Configuración de entorno. Único lugar que lee variables externas (DIP en el borde)."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Parámetros de API, rutas locales y umbrales clínicos de riesgo."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")

    data_raw_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    data_processed_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")
    charts_dir: Path = Field(default=PROJECT_ROOT / "static" / "generated_charts")
    static_dir: Path = Field(default=PROJECT_ROOT / "static")
    model_artifact_path: Path = Field(
        default=PROJECT_ROOT
        / "src"
        / "infrastructure"
        / "ai_models"
        / "serialized"
        / "xgboost_dropout_pipeline.joblib"
    )
    primary_dataset_name: str = Field(
        default="dataset_desercion_fisura_bilabial.csv"
    )

    low_risk_threshold: float = Field(default=0.33, ge=0.0, le=1.0)
    high_risk_threshold: float = Field(default=0.66, ge=0.0, le=1.0)

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)

    def ensure_runtime_directories(self) -> None:
        """Crea directorios de trabajo si el despliegue arranca en frío."""
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.model_artifact_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton de configuración para inyección en FastAPI."""
    settings = Settings()
    settings.ensure_runtime_directories()
    return settings
