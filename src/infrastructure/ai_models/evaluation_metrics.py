"""Métricas de evaluación para los experimentos de deserción. SRP: solo calcula, no entrena.

Convención: la clase positiva es ``desertor = 1``. Un paciente se marca como "alerta" cuando
``score >= umbral``. Por eso:

- recall        = desertores detectados / desertores reales.
- especificidad = continuadores no alertados / continuadores reales.

El umbral se elige con ``choose_threshold`` y datos de entrenamiento, nunca con el test.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

DEFAULT_MIN_RECALL: Final[float] = 0.90
TOP_FRACTIONS: Final[tuple[float, ...]] = (0.10, 0.20, 0.30)

RANKING_METRICS: Final[tuple[str, ...]] = ("roc_auc", "pr_auc", "brier")
THRESHOLD_METRICS: Final[tuple[str, ...]] = (
    "recall",
    "especificidad",
    "precision",
    "proporcion_alertas",
)
TOP_K_METRICS: Final[tuple[str, ...]] = tuple(
    f"{kind}_top{round(fraction * 100)}"
    for fraction in TOP_FRACTIONS
    for kind in ("captura", "precision")
)
METRIC_NAMES: Final[tuple[str, ...]] = RANKING_METRICS + THRESHOLD_METRICS + TOP_K_METRICS


def _as_arrays(y_true: object, scores: object) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    s = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    if y.ndim != 1 or y.shape != s.shape:
        raise ValueError("y_true y scores deben ser vectores de la misma longitud.")
    if y.size == 0:
        raise ValueError("No hay observaciones para evaluar.")
    if np.isnan(s).any():
        raise ValueError("scores contiene valores no numéricos.")
    return y, s


def choose_threshold(
    y_true: object, scores: object, min_recall: float = DEFAULT_MIN_RECALL
) -> float:
    """Umbral que maximiza la especificidad entre los que logran ``recall >= min_recall``.

    Entre umbrales con la misma especificidad se toma el más alto (alerta más estricta).
    Con puntajes constantes el único umbral posible marca a todos como alerta.
    """
    if not 0.0 < min_recall <= 1.0:
        raise ValueError("min_recall debe estar en (0, 1].")
    y, s = _as_arrays(y_true, scores)
    if y.min() == y.max():
        raise ValueError("Se necesitan las dos clases para elegir un umbral.")
    fpr, tpr, thresholds = roc_curve(y, s, drop_intermediate=False)
    feasible = tpr >= min_recall - 1e-12
    best_fpr = fpr[feasible].min()
    candidates = np.flatnonzero(feasible & np.isclose(fpr, best_fpr))
    return float(thresholds[candidates[0]])  # umbrales decrecientes: el primero es el más alto


def threshold_metrics(y_true: object, scores: object, threshold: float) -> dict[str, float]:
    y, s = _as_arrays(y_true, scores)
    alert = s >= threshold
    positive = y == 1
    tp = int((alert & positive).sum())
    fn = int((~alert & positive).sum())
    fp = int((alert & ~positive).sum())
    tn = int((~alert & ~positive).sum())
    return {
        "recall": tp / (tp + fn) if tp + fn else math.nan,
        "especificidad": tn / (tn + fp) if tn + fp else math.nan,
        "precision": tp / (tp + fp) if tp + fp else math.nan,
        "proporcion_alertas": float(alert.mean()),
    }


def top_k_metrics(
    y_true: object, scores: object, fractions: tuple[float, ...] = TOP_FRACTIONS
) -> dict[str, float]:
    """Captura y precisión al revisar solo el ``fraction`` de pacientes con mayor puntaje.

    ``captura`` = desertores dentro del grupo revisado / desertores totales.
    Los empates en el corte se reparten en proporción (valor esperado con desempate aleatorio),
    para que un puntaje constante no dependa del orden de las filas.
    """
    y, s = _as_arrays(y_true, scores)
    total_positive = int(y.sum())
    ordered = np.sort(s)[::-1]
    result: dict[str, float] = {}
    for fraction in fractions:
        k = max(1, math.ceil(fraction * len(y)))
        cutoff = ordered[k - 1]
        above = s > cutoff
        tied = s == cutoff
        needed = k - int(above.sum())
        tp = float(y[above].sum()) + float(y[tied].mean()) * needed
        label = round(fraction * 100)
        result[f"captura_top{label}"] = tp / total_positive if total_positive else math.nan
        result[f"precision_top{label}"] = tp / k
    return result


def evaluate_scores(y_true: object, scores: object, threshold: float) -> dict[str, float]:
    """Todas las métricas de ``METRIC_NAMES`` para un conjunto de puntajes y un umbral fijo."""
    y, s = _as_arrays(y_true, scores)
    if y.min() == y.max():
        raise ValueError("Se necesitan las dos clases para calcular ROC-AUC y PR-AUC.")
    metrics: dict[str, float] = {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "brier": float(brier_score_loss(y, s)),
    }
    metrics.update(threshold_metrics(y, s, threshold))
    metrics.update(top_k_metrics(y, s))
    return {name: metrics[name] for name in METRIC_NAMES}


def calibration_table(y_true: object, scores: object, n_bins: int = 10) -> pd.DataFrame:
    """Tabla de calibración con cajas de igual tamaño (por cuantiles de puntaje)."""
    y, s = _as_arrays(y_true, scores)
    bins = pd.qcut(pd.Series(s).rank(method="first"), q=min(n_bins, len(s)), labels=False)
    table = (
        pd.DataFrame({"caja": bins, "score": s, "y": y})
        .groupby("caja")
        .agg(n=("y", "size"), score_medio=("score", "mean"), frecuencia_observada=("y", "mean"))
        .reset_index()
    )
    return table


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Error de calibración esperado: diferencia media ponderada entre puntaje y frecuencia."""
    gap = (table["frecuencia_observada"] - table["score_medio"]).abs()
    return float((gap * table["n"]).sum() / table["n"].sum())


def bootstrap_intervals(
    y_true: object,
    scores: object,
    threshold: float,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Intervalo percentil (1 - alpha) por remuestreo, con el umbral fijo.

    Solo es válido para predicciones sobre datos que no intervinieron en el ajuste (holdout).
    """
    y, s = _as_arrays(y_true, scores)
    rng = np.random.default_rng(seed)
    draws: list[dict[str, float]] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        draws.append(evaluate_scores(y[idx], s[idx], threshold))
    if not draws:
        raise ValueError("Ningún remuestreo contiene las dos clases.")
    frame = pd.DataFrame(draws)
    lower = frame.quantile(alpha / 2)
    upper = frame.quantile(1 - alpha / 2)
    return {name: (float(lower[name]), float(upper[name])) for name in METRIC_NAMES}


def subgroup_metrics(
    y_true: object, scores: object, alerts: object, groups: pd.Series
) -> pd.DataFrame:
    """Desempeño por subgrupo. ``alerts`` puede ser 0/1 o la fracción de repeticiones en alerta."""
    y, s = _as_arrays(y_true, scores)
    frame = pd.DataFrame(
        {
            "grupo": np.asarray(groups),
            "y": y,
            "score": s,
            "alerta": np.asarray(alerts, dtype=float),
        }
    ).dropna(subset=["grupo"])
    rows = []
    for group, part in frame.groupby("grupo", sort=True):
        positives = part[part["y"] == 1]
        negatives = part[part["y"] == 0]
        both = part["y"].nunique() == 2
        rows.append(
            {
                "grupo": str(group),
                "n": int(len(part)),
                "prevalencia": float(part["y"].mean()),
                "roc_auc": float(roc_auc_score(part["y"], part["score"])) if both else math.nan,
                "recall": float(positives["alerta"].mean()) if len(positives) else math.nan,
                "especificidad": (
                    float(1.0 - negatives["alerta"].mean()) if len(negatives) else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)
