"""Modelo final de producción: regresión logística en T0 (ver reports/comparacion_modelos.md).

SRP: entrenar una vez con toda la cohorte v2 y guardar un artefacto versionado. No compara
modelos (eso es ``experiments.py``) ni sirve predicciones (eso es ``logistic_predictor.py``).

El artefacto (``serialized/modelo_t0.joblib``) es un diccionario con:

- ``pipeline``: imputación + codificación + escalado + regresión logística, ajustado con v2.
- ``features``: columnas de entrada (``FEATURES_T0``).
- ``priority_cutoffs``: puntajes que separan prioridad alta / media / habitual. Salen de los
  percentiles 80 y 30 de las predicciones *fuera de muestra* (validación cruzada), no de las
  del modelo sobre sus propios datos (reports/interpretabilidad.md §6.3).
- ``priority_observed_rates``: deserción observada en cada grupo de prioridad (fuera de muestra).
- ``reference_mean``: paciente promedio en el espacio transformado, para explicar predicciones.
- ``metadata``: fecha, tamaño, hiperparámetros, métricas fuera de muestra y versiones.

Uso: ``python -m src.infrastructure.ai_models.final_model``
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.infrastructure.ai_models.evaluation_metrics import DEFAULT_MIN_RECALL
from src.infrastructure.ai_models.experiments import (
    HOLDOUT_SELECTION_FOLDS,
    fit_and_select,
    model_specs,
)
from src.infrastructure.ai_models.modeling_features import select_features, select_target

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT: Final[Path] = Path(__file__).resolve().parent / "serialized" / "modelo_t0.joblib"

MODEL_NAME: Final[str] = "regresion_logistica"
SCENARIO: Final[str] = "T0"
ARTIFACT_VERSION: Final[str] = "t0-logistica-1"
# Punto de partida de trabajo, pendiente de validar con la capacidad real de Fisulab:
# alta = 20 % superior, habitual = 30 % inferior, media = el resto.
HIGH_PRIORITY_QUANTILE: Final[float] = 0.80
LOW_PRIORITY_QUANTILE: Final[float] = 0.30


def train_final_model(frame: pd.DataFrame, seed: int = 42) -> dict[str, Any]:
    """Ajusta la regresión logística T0 con todo ``frame`` y arma el artefacto (sin guardarlo)."""
    X = select_features(frame, SCENARIO)
    y = select_target(frame)
    pipeline, params, threshold = fit_and_select(
        model_specs()[MODEL_NAME],
        list(X.columns),
        X,
        y,
        inner_splits=HOLDOUT_SELECTION_FOLDS,
        min_recall=DEFAULT_MIN_RECALL,
        seed=seed,
        n_jobs=1,
    )

    cv = StratifiedKFold(n_splits=HOLDOUT_SELECTION_FOLDS, shuffle=True, random_state=seed)
    out_of_sample = cross_val_predict(clone(pipeline), X, y, cv=cv, method="predict_proba")[:, 1]
    high_cutoff = float(np.quantile(out_of_sample, HIGH_PRIORITY_QUANTILE))
    low_cutoff = float(np.quantile(out_of_sample, LOW_PRIORITY_QUANTILE))
    groups = priority_groups(out_of_sample, high_cutoff, low_cutoff)
    observed = {
        level: float(y.to_numpy()[groups == level].mean()) for level in ("alta", "media", "habitual")
    }

    transformed = pipeline.named_steps["prep"].transform(X)
    return {
        "pipeline": pipeline,
        "features": list(X.columns),
        "priority_cutoffs": {"alta": high_cutoff, "habitual": low_cutoff},
        "priority_observed_rates": observed,
        "reference_mean": np.asarray(transformed, dtype=float).mean(axis=0),
        "metadata": {
            "version": ARTIFACT_VERSION,
            "modelo": MODEL_NAME,
            "escenario": SCENARIO,
            "entrenado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cohorte": "v2",
            "pacientes": int(len(X)),
            "tasa_desercion": float(y.mean()),
            "hiperparametros": params,
            "umbral_recall_minimo": {"min_recall": DEFAULT_MIN_RECALL, "umbral": float(threshold)},
            "fuera_de_muestra": {
                "roc_auc": float(roc_auc_score(y, out_of_sample)),
                "pr_auc": float(average_precision_score(y, out_of_sample)),
                "pliegues": HOLDOUT_SELECTION_FOLDS,
            },
            "cuantiles_prioridad": {"alta": HIGH_PRIORITY_QUANTILE, "habitual": LOW_PRIORITY_QUANTILE},
            "semilla": seed,
            "python": platform.python_version(),
            "bibliotecas": _library_versions(),
            "datos": "sintéticos; no extrapolable a pacientes reales",
        },
    }


def priority_groups(scores: np.ndarray, high_cutoff: float, low_cutoff: float) -> np.ndarray:
    """alta si puntaje >= corte alto; habitual si < corte bajo; media en el resto."""
    scores = np.asarray(scores, dtype=float)
    return np.where(scores >= high_cutoff, "alta", np.where(scores < low_cutoff, "habitual", "media"))


def save_artifact(artifact: dict[str, Any], path: Path = DEFAULT_ARTIFACT) -> tuple[Path, Path]:
    """Guarda el ``.joblib`` y, al lado, un ``.json`` legible con todo menos el pipeline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    summary = {
        "features": artifact["features"],
        "priority_cutoffs": artifact["priority_cutoffs"],
        "priority_observed_rates": artifact["priority_observed_rates"],
        **artifact["metadata"],
    }
    json_path = path.with_suffix(".json")
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    return path, json_path


def _library_versions() -> dict[str, str]:
    import importlib.metadata as metadata

    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "joblib"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "no instalado"
    return versions


def main(argv: Sequence[str] | None = None) -> int:
    from src.core.exceptions import DataAccessError, InvalidPatientDataError
    from src.infrastructure.data.etl import build_modeling_frame

    parser = argparse.ArgumentParser(description="Entrena y guarda el modelo final (regresión logística T0).")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        artifact = train_final_model(build_modeling_frame(args.raw_dir), seed=args.seed)
    except (DataAccessError, InvalidPatientDataError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    for path in save_artifact(artifact, args.output):
        print(f"  escrito: {path}")
    metadata = artifact["metadata"]
    print(json.dumps(
        {
            "hiperparametros": metadata["hiperparametros"],
            "fuera_de_muestra": metadata["fuera_de_muestra"],
            "priority_cutoffs": artifact["priority_cutoffs"],
            "priority_observed_rates": artifact["priority_observed_rates"],
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
