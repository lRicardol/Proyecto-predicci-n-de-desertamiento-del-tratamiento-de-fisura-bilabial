"""DTOs HTTP. No contienen reglas de negocio."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    patient_id: str = Field(
        ...,
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
        examples=["PAC-2001"],
    )


class ScenarioRequest(BaseModel):
    """Paciente hipotético: las 7 variables del modelo T0, con los rangos del entrenamiento.

    Los límites son los observados en la cohorte v2 (no se extrapola fuera de lo que el modelo vio).
    ``nivel_socioeconomico_ord`` solo admite los niveles presentes en los datos: 0 bajo,
    1 medio-bajo, 2 medio y 4 alto (el 3 no aparece en la cohorte).
    """

    model_config = ConfigDict(extra="forbid")

    tipo_fisura: Literal["Completa", "Incompleta", "Asimetrica"]
    edad_inicio_meses: int = Field(..., ge=1, le=11)
    distancia_hospital_km: float = Field(..., ge=5, le=300)
    nivel_socioeconomico_ord: Literal[0, 1, 2, 4]
    ingresos_mensuales_usd: int = Field(..., ge=250, le=6000)
    apoyo_psicologico: Literal[0, 1]
    dificultad_alimentacion_preop: Literal[0, 1]


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
    model_scenario: str = ""
    model_version: str = ""
