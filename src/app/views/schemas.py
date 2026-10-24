"""DTOs HTTP. No contienen reglas de negocio."""

from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    patient_id: str = Field(..., min_length=1, examples=["PAC-1001"])


class PredictionResponse(BaseModel):
    patient_id: str
    dropout_probability: float
    risk_level: str
    patient_snapshot: dict[str, Any]


class AnalyticsRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        examples=["Grafica la relación entre inasistencias y deserción"],
    )


class AnalyticsResponse(BaseModel):
    query: str
    narrative: str
    image_url: str


class HealthResponse(BaseModel):
    status: str
    patients_loaded: int
    dataset: str
