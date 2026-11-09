"""DTOs HTTP. No contienen reglas de negocio."""

from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    patient_id: str = Field(..., min_length=1, examples=["PAC-1001"])


class FactorResponse(BaseModel):
    variable: str
    label: str
    value: Any = None
    contribution: float = Field(..., description="Log-odds frente al paciente promedio; > 0 sube la estimación.")
    direction: str


class PredictionResponse(BaseModel):
    patient_id: str
    dropout_probability: float
    priority_level: str = Field(..., examples=["alta", "media", "habitual"])
    priority_message: str
    group_observed_dropout_rate: float | None = None
    top_factors: list[FactorResponse] = Field(default_factory=list)
    model_inputs: dict[str, Any] = Field(default_factory=dict)
    prediction_date: str
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str
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
    sample_patient_ids: list[str] = Field(default_factory=list)
