"""Composition root. Único módulo de aplicación que instancia infraestructura."""

from functools import lru_cache

from config import Settings, get_settings
from src.core.use_cases.generate_analytics import GenerateAnalyticsUseCase
from src.core.use_cases.predict_dropout import PredictDropoutUseCase
from src.infrastructure.ai_models.xgboost_predictor import XgboostPredictor
from src.infrastructure.llm_agents.langchain_analyst import LangchainAnalyst
from src.infrastructure.repositories.csv_patient_repository import CsvPatientRepository


@lru_cache(maxsize=1)
def get_patient_repository() -> CsvPatientRepository:
    settings = get_settings()
    return CsvPatientRepository(settings.data_raw_dir)


@lru_cache(maxsize=1)
def get_predictor_model() -> XgboostPredictor:
    settings = get_settings()
    repository = get_patient_repository()
    return XgboostPredictor(
        model_path=settings.model_artifact_path,
        training_frame=repository.get_all_patients_data(),
    )


def get_predict_dropout_use_case() -> PredictDropoutUseCase:
    settings = get_settings()
    return PredictDropoutUseCase(
        patient_repository=get_patient_repository(),
        predictor_model=get_predictor_model(),
        low_risk_threshold=settings.low_risk_threshold,
        high_risk_threshold=settings.high_risk_threshold,
    )


def get_generate_analytics_use_case() -> GenerateAnalyticsUseCase:
    settings: Settings = get_settings()
    analyst = LangchainAnalyst(
        api_key=settings.openai_api_key,
        model_name=settings.llm_model,
        temperature=settings.llm_temperature,
        provider=settings.llm_provider,
    )
    dataset_path = settings.data_raw_dir / settings.primary_dataset_name
    return GenerateAnalyticsUseCase(
        llm_analyst=analyst,
        dataset_path=dataset_path,
        charts_dir=settings.charts_dir,
    )
