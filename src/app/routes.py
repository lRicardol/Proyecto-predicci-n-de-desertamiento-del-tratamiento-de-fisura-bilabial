"""Registro de rutas. No contiene lógica de negocio ni de infraestructura."""

from fastapi import FastAPI

from src.app.controllers.dashboard_controller import dashboard_router
from src.app.controllers.prediction_controller import prediction_router


def register_routes(app: FastAPI) -> None:
    """Monta los controladores MVC sobre la aplicación FastAPI."""
    app.include_router(dashboard_router)
    app.include_router(prediction_router, prefix="/api")
