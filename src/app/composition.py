"""Composition root: única capa que instancia infraestructura concreta."""

from functools import lru_cache

from config import get_settings
from src.core.use_cases.generate_analytics import GenerateAnalyticsUseCase
from src.core.use_cases.predict_dropout import PredictDropoutUseCase, PriorityPolicy
from src.infrastructure.ai_models.logistic_predictor import LogisticT0Predictor
from src.infrastructure.llm_agents.local_analyst import LocalStatisticalAnalyst
from src.infrastructure.repositories.csv_patient_repository import CsvPatientRepository


@lru_cache(maxsize=1)
def get_patient_repository() -> CsvPatientRepository:
    settings = get_settings()
    return CsvPatientRepository(settings.modeling_dataset_path)


@lru_cache(maxsize=1)
def get_predictor_model() -> LogisticT0Predictor:
    settings = get_settings()
    return LogisticT0Predictor(artifact_path=settings.model_artifact_path)


@lru_cache(maxsize=1)
def get_local_analyst() -> LocalStatisticalAnalyst:
    return LocalStatisticalAnalyst()


def get_priority_policy() -> PriorityPolicy:
    """Los cortes viajan dentro del artefacto: modelo y cortes siempre son de la misma versión."""
    predictor = get_predictor_model()
    cutoffs = predictor.priority_cutoffs
    return PriorityPolicy(
        high_cutoff=cutoffs["alta"],
        low_cutoff=cutoffs["habitual"],
        observed_rates=predictor.priority_observed_rates,
    )


def get_predict_dropout_use_case() -> PredictDropoutUseCase:
    predictor = get_predictor_model()
    return PredictDropoutUseCase(
        patient_repository=get_patient_repository(),
        predictor_model=predictor,
        priority_policy=get_priority_policy(),
        training_cohort=predictor.metadata.get("cohorte"),
        scenario=str(predictor.metadata.get("escenario") or "T0"),
        model_version=str(predictor.metadata.get("version") or ""),
    )


def warmup_predictor() -> None:
    """Carga modelo_t0.joblib al arrancar. No entrena: si falta, la API no arranca y dice por qué."""
    get_predictor_model().load()


def get_generate_analytics_use_case() -> GenerateAnalyticsUseCase:
    settings = get_settings()
    return GenerateAnalyticsUseCase(
        llm_analyst=get_local_analyst(),
        dataset_path=settings.processed_dataset_path,
        charts_dir=settings.charts_dir,
    )
