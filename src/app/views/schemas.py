"""DTOs HTTP. No contienen reglas de negocio."""

from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    patient_id: str = Field(
        ...,
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
        examples=["PAC-2001"],
    )


class FactorResponse(BaseModel):
    variable: str
    label: str
    value: Any = None
    contribution: float = Field(..., description="Log-odds frente al paciente promedio; > 0 sube la estimación.")
    direction: str


class PredictionResponse(BaseModel):
    patient_id: str
    dropout_probability: float = Field(..., ge=0.0, le=1.0)
    priority_level: str = Field(..., examples=["alta", "media", "habitual"])
    priority_message: str
    group_observed_dropout_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    top_factors: list[FactorResponse] = Field(default_factory=list)
    model_inputs: dict[str, Any] = Field(default_factory=dict)
    prediction_date: str
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str
    scenario: str = Field(..., examples=["T0"])
    model_version: str


class AnalyticsRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
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
