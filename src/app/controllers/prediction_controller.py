"""Controlador HTTP de predicción. Solo traduce request/response."""

from fastapi import APIRouter, Depends, HTTPException, status

from src.app.composition import get_predict_dropout_use_case
from src.app.views.schemas import FactorResponse, PredictionRequest, PredictionResponse
from src.core.exceptions import DomainError, PatientNotFoundError
from src.core.use_cases.predict_dropout import PredictDropoutUseCase

prediction_router = APIRouter(prefix="/predictions", tags=["predicción"])


@prediction_router.get("/{patient_id}", response_model=PredictionResponse)
def predict_dropout_get(
    patient_id: str,
    use_case: PredictDropoutUseCase = Depends(get_predict_dropout_use_case),
) -> PredictionResponse:
    return _execute(patient_id, use_case)


@prediction_router.post("", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_dropout_post(
    payload: PredictionRequest,
    use_case: PredictDropoutUseCase = Depends(get_predict_dropout_use_case),
) -> PredictionResponse:
    return _execute(payload.patient_id, use_case)


def _execute(patient_id: str, use_case: PredictDropoutUseCase) -> PredictionResponse:
    try:
        result = use_case.execute(patient_id)
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DomainError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return PredictionResponse(
        patient_id=result.patient_id,
        dropout_probability=result.dropout_probability,
        priority_level=result.priority_level,
        priority_message=result.priority_message,
        group_observed_dropout_rate=result.group_observed_rate,
        top_factors=[
            FactorResponse(
                variable=factor.variable,
                label=factor.label,
                value=factor.value,
                contribution=factor.contribution,
                direction="aumenta la estimación" if factor.contribution > 0 else "reduce la estimación",
            )
            for factor in result.top_factors
        ],
        model_inputs=result.model_inputs,
        prediction_date=result.prediction_date,
        warnings=result.warnings,
        disclaimer=result.disclaimer,
        patient_snapshot=result.patient_snapshot,
    )
