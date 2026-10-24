"""Entrenamiento e inferencia XGBoost sobre dataset_final_limpio.csv. Sin mocks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.core.exceptions import PredictionError
from src.core.interfaces.i_predictor_model import IPredictorModel
from src.infrastructure.ai_models.feature_engineering import (
    MODEL_FEATURES,
    extract_model_matrix,
    extract_target,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "dataset_final_limpio.csv"
DEFAULT_MODEL = Path(__file__).resolve().parent / "serialized" / "model.joblib"


class XgboostPredictor(IPredictorModel):
    """Carga model.joblib o entrena XGBoost real sobre el CSV procesado."""

    def __init__(
        self,
        model_path: Path | None = None,
        dataset_path: Path | None = None,
        training_frame: pd.DataFrame | None = None,
    ) -> None:
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL
        self._dataset_path = Path(dataset_path) if dataset_path else DEFAULT_DATASET
        self._training_frame = training_frame
        self._model: XGBClassifier | None = None

    def predict_dropout_probability(self, features: pd.DataFrame) -> np.ndarray:
        if features is None or features.empty:
            raise PredictionError("No se recibieron características para predecir.")
        try:
            matrix = extract_model_matrix(features)
            model = self._load_or_train()
            probabilities = model.predict_proba(matrix)[:, 1]
            clipped = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
            if np.isnan(clipped).any():
                raise PredictionError("El modelo devolvió valores no numéricos.")
            return clipped
        except PredictionError:
            raise
        except Exception as exc:
            raise PredictionError(f"Fallo al inferir deserción: {exc}") from exc

    def _load_or_train(self) -> XGBClassifier:
        if self._model is not None:
            return self._model
        if self._model_path.exists():
            loaded = joblib.load(self._model_path)
            if not hasattr(loaded, "predict_proba"):
                raise PredictionError("El artefacto serializado no expone predict_proba.")
            self._model = loaded
            return self._model
        metrics = train_xgboost(
            dataset_path=self._dataset_path,
            model_path=self._model_path,
            frame=self._training_frame,
        )
        _print_training_report(metrics)
        self._model = joblib.load(self._model_path)
        return self._model


def train_xgboost(
    dataset_path: Path = DEFAULT_DATASET,
    model_path: Path = DEFAULT_MODEL,
    frame: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Entrena, evalúa, imprime métricas y persiste serialized/model.joblib."""
    source = frame if frame is not None else _read_processed(dataset_path)
    features = extract_model_matrix(source)
    target = extract_target(source)
    if len(features) != len(target):
        raise PredictionError("Features y target no tienen la misma longitud.")
    if target.nunique() < 2:
        raise PredictionError("El target desertor no tiene las dos clases.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise PredictionError("El split de entrenamiento perdió una de las clases.")

    # Deserción real ≈ 74.6% (clase positiva mayoritaria).
    # XGBoost: scale_pos_weight = n_neg / n_pos = (1 - p) / p
    scale_pos_weight = n_neg / n_pos
    dropout_rate = n_pos / (n_pos + n_neg)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        random_state=random_state,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics: dict[str, Any] = {
        "n_rows": int(len(source)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_pos_train": n_pos,
        "n_neg_train": n_neg,
        "dropout_rate_train": float(dropout_rate),
        "scale_pos_weight": float(scale_pos_weight),
        "features": list(MODEL_FEATURES),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, digits=3, zero_division=0
        ),
        "model_path": str(model_path),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return metrics


def _read_processed(dataset_path: Path) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise PredictionError(
            f"No existe el dataset procesado {path}. "
            "Ejecuta notebooks/00_procesamiento_dataset_unificado.py"
        )
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise PredictionError(f"No se pudo leer {path}: {exc}") from exc
    if frame.empty:
        raise PredictionError(f"{path.name} está vacío.")
    return frame


def _print_training_report(metrics: dict[str, Any]) -> None:
    matrix = metrics["confusion_matrix"]
    print("=== Entrenamiento XGBoost · deserción fisura bilabial ===")
    print(f"Filas: {metrics.get('n_rows', metrics['n_train'] + metrics['n_test'])}")
    print(f"Features: {metrics['features']}")
    print(f"Train/Test: {metrics['n_train']} / {metrics['n_test']}")
    print(
        "Desbalance train: "
        f"desertores={metrics['n_pos_train']} "
        f"({metrics['dropout_rate_train']:.1%}) | "
        f"continuadores={metrics['n_neg_train']}"
    )
    print(
        "scale_pos_weight = n_neg / n_pos = "
        f"{metrics['scale_pos_weight']:.4f}  "
        f"(referencia de desbalance ~74.6% deserción)"
    )
    print(f"Precisión: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print("Matriz de confusión [[TN, FP], [FN, TP]]:")
    print(matrix)
    print(metrics["classification_report"])
    print(f"Modelo guardado en: {metrics['model_path']}")


if __name__ == "__main__":
    report = train_xgboost()
    _print_training_report(report)
