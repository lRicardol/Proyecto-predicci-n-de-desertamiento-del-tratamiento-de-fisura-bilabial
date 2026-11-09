"""Punto de entrada: composition root HTTP + estáticos + ciclo de vida."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import get_settings
from src.app.composition import warmup_predictor
from src.app.routes import register_routes


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    settings.ensure_runtime_directories()
    warmup_predictor()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="Predicción de deserción · Fisura bilabial",
        description=(
            "Prototipo académico con datos sintéticos. Regresión logística T0 "
            "y analítica local, sin APIs externas. No es una herramienta diagnóstica."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    # Solo orígenes locales: el dashboard se sirve same-origin; esto cubre
    # herramientas de desarrollo locales sin abrir la API a la red.
    application.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    application.mount(
        "/static",
        StaticFiles(directory=str(settings.static_dir)),
        name="static",
    )
    register_routes(application)
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    runtime = get_settings()
    uvicorn.run(
        "main:app",
        host=runtime.host,
        port=runtime.port,
        reload=runtime.debug,
    )
