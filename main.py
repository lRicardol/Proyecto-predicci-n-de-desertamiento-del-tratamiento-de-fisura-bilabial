"""Punto de entrada: composition root HTTP + estáticos + ciclo de vida."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import get_settings
from src.app.routes import register_routes


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    settings.ensure_runtime_directories()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="Predicción de deserción · Fisura bilabial",
        description=(
            "API de analítica predictiva y conversacional para deserción "
            "del tratamiento de fisura labiopalatina."
        ),
        version="1.0.0",
        lifespan=lifespan,
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
