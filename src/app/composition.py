"""Composition root: única capa que instancia infraestructura concreta."""

from functools import lru_cache

import pandas as pd

from config import get_settings
from src.core.use_cases.generate_analytics import GenerateAnalyticsUseCase
from src.core.use_cases.predict_dropout import PredictDropoutUseCase
from src.infrastructure.ai_models.xgboost_predictor import XgboostPredictor
from src.infrastructure.llm_agents.local_analyst import LocalStatisticalAnalyst
from src.infrastructure.repositories.csv_patient_repository import CsvPatientRepository


@lru_cache(maxsize=1)
def get_patient_repository() -> CsvPatientRepository:
    settings = get_settings()
    return CsvPatientRepository(settings.processed_dataset_path)


@lru_cache(maxsize=1)
def get_predictor_model() -> XgboostPredictor:
    settings = get_settings()
    return XgboostPredictor(
        model_path=settings.model_artifact_path,
        dataset_path=settings.processed_dataset_path,
    )


@lru_cache(maxsize=1)
def get_local_analyst() -> LocalStatisticalAnalyst:
    return LocalStatisticalAnalyst()


def get_predict_dropout_use_case() -> PredictDropoutUseCase:
    settings = get_settings()
    return PredictDropoutUseCase(
        patient_repository=get_patient_repository(),
        predictor_model=get_predictor_model(),
        low_risk_threshold=settings.low_risk_threshold,
        high_risk_threshold=settings.high_risk_threshold,
    )


def warmup_predictor() -> None:
    """Carga model.joblib o entrena XGBoost la primera vez que arranca la API."""
    get_predictor_model().predict_dropout_probability(
        pd.DataFrame(
            [
                {
                    "citas_perdidas_noshow": 0,
                    "distancia_hospital_km": 0,
                    "nivel_socioeconomico_ord": 2,
                    "complicaciones_quirurgicas": 0,
                }
            ]
        )
    )


def get_generate_analytics_use_case() -> GenerateAnalyticsUseCase:
    settings = get_settings()
    return GenerateAnalyticsUseCase(
        llm_analyst=get_local_analyst(),
        dataset_path=settings.processed_dataset_path,
        charts_dir=settings.charts_dir,
    )
