"""Configuración local. Sin API keys ni servicios externos."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_raw_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    data_processed_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")
    processed_dataset_path: Path = Field(
        default=PROJECT_ROOT / "data" / "processed" / "dataset_final_limpio.csv"
    )
    charts_dir: Path = Field(default=PROJECT_ROOT / "static" / "generated_charts")
    static_dir: Path = Field(default=PROJECT_ROOT / "static")
    model_artifact_path: Path = Field(
        default=PROJECT_ROOT
        / "src"
        / "infrastructure"
        / "ai_models"
        / "serialized"
        / "model.joblib"
    )
    primary_dataset_name: str = Field(default="dataset_final_limpio.csv")

    low_risk_threshold: float = Field(default=0.33, ge=0.0, le=1.0)
    high_risk_threshold: float = Field(default=0.66, ge=0.0, le=1.0)

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)

    def ensure_runtime_directories(self) -> None:
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.model_artifact_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_runtime_directories()
    return settings
