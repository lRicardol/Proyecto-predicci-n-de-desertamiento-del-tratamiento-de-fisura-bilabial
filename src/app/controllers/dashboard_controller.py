"""Controlador del panel: salud, vista HTML y analítica conversacional."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse

from config import get_settings
from src.app.composition import (
    get_generate_analytics_use_case,
    get_patient_repository,
    get_predictor_model,
)
from src.app.views.schemas import AnalyticsRequest, AnalyticsResponse, HealthResponse
from src.core.exceptions import DomainError
from src.core.use_cases.generate_analytics import GenerateAnalyticsUseCase

dashboard_router = APIRouter(tags=["dashboard"])
DASHBOARD_TEMPLATE = Path(__file__).resolve().parent.parent / "views" / "dashboard.html"


@dashboard_router.get("/", response_class=HTMLResponse)
def dashboard_home() -> HTMLResponse:
    if not DASHBOARD_TEMPLATE.exists():
        raise HTTPException(status_code=500, detail="No existe la plantilla del dashboard.")
    return HTMLResponse(DASHBOARD_TEMPLATE.read_text(encoding="utf-8"))


@dashboard_router.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    repository = get_patient_repository()
    identifiers = repository.list_patient_ids()
    present = set(identifiers)
    preferred = [pid for pid in ("PAC-2001", "PAC-2002", "PAC-2003") if pid in present]
    if not preferred:
        preferred = identifiers[:3]
    metadata = get_predictor_model().metadata
    return HealthResponse(
        status="ok",
        patients_loaded=repository.count_patients(),
        dataset=settings.modeling_dataset_path.name,
        sample_patient_ids=preferred[:3],
        model_scenario=str(metadata.get("escenario") or ""),
        model_version=str(metadata.get("version") or ""),
    )


@dashboard_router.post("/api/analytics", response_model=AnalyticsResponse)
def generate_analytics(
    payload: AnalyticsRequest,
    use_case: GenerateAnalyticsUseCase = Depends(get_generate_analytics_use_case),
) -> AnalyticsResponse:
    try:
        result = use_case.execute(payload.query)
    except DomainError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fallo inesperado de analítica: {exc}",
        ) from exc
    return AnalyticsResponse(
        query=result.query,
        narrative=result.narrative,
        image_url=result.image_url,
    )
