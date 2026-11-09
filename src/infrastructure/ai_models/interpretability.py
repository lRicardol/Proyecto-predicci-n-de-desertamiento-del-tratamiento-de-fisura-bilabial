"""Interpretabilidad del modelo de referencia (frente 5). SRP: explicar, no comparar ni servir.

La explicación es exacta porque el modelo es lineal en log-odds:
log-odds(paciente) = log-odds(paciente promedio) + suma de contribuciones.

Uso: python -m src.infrastructure.ai_models.interpretability
Escribe en reports/modelos/:
    interpretabilidad_coeficientes.csv, interpretabilidad_permutacion.csv,
    interpretabilidad_marginal.csv, interpretabilidad_pacientes.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.infrastructure.ai_models.experiments import (
    ExperimentConfig, fit_and_select, model_specs, scenario_matrix, split_development_holdout,
)
from src.infrastructure.ai_models.modeling_features import CATEGORICAL_FEATURES, select_target

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "reports" / "modelos"
REFERENCE_MODEL: Final[str] = "regresion_logistica"
COMPARISON_MODELS: Final[tuple[str, ...]] = ("regresion_logistica", "random_forest", "xgboost")


def _feature_names(pipeline: Pipeline) -> list[str]:
    return [name.split("__", 1)[1] for name in pipeline.named_steps["prep"].get_feature_names_out()]


def _scales(pipeline: Pipeline, columns: Sequence[str]) -> dict[str, float]:
    numeric = [column for column in columns if column not in CATEGORICAL_FEATURES]
    scaler = pipeline.named_steps["prep"].named_transformers_["num"].named_steps.get("scaler")
    return {column: 1.0 for column in numeric} if scaler is None else dict(zip(numeric, scaler.scale_))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _logit(probability: float) -> float:
    clipped = float(np.clip(probability, 1e-9, 1 - 1e-9))
    return float(np.log(clipped / (1 - clipped)))


def coefficient_table(
    pipeline: Pipeline, columns: Sequence[str], X: pd.DataFrame, y: pd.Series,
    n_bootstrap: int = 200, seed: int = 42,
) -> pd.DataFrame:
    """OR por 1 DE, IC 95 % bootstrap y estabilidad de signo. El OR por 10 km es or_por_unidad ** 10."""
    if X.empty or y.empty:
        raise ValueError("coefficient_table necesita datos de entrenamiento no vacíos.")
    names = _feature_names(pipeline)
    scales = _scales(pipeline, columns)
    point = pipeline.named_steps["clf"].coef_[0]
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        positions = rng.integers(0, len(X), len(X))
        sample_y = y.iloc[positions]
        if sample_y.nunique() < 2:
            continue
        fitted = clone(pipeline).fit(X.iloc[positions], sample_y)
        draws.append(fitted.named_steps["clf"].coef_[0])
    if len(draws) < 2:
        raise ValueError(
            f"Bootstrap insuficiente: solo {len(draws)} remuestreos válidos de {n_bootstrap}."
        )
    bootstrap = np.vstack(draws)
    sd = np.array([scales.get(name, np.nan) for name in names])
    per_unit = np.where(np.isnan(sd), point, point / np.where(np.isnan(sd), 1.0, sd))
    lower, upper = np.nanpercentile(bootstrap, [2.5, 97.5], axis=0)
    return pd.DataFrame({
        "variable": names,
        "coeficiente": point,
        "or_por_de": np.exp(point),
        "or_por_unidad": np.exp(per_unit),
        "unidad": ["vs categoría de referencia" if np.isnan(value) else "1 unidad" for value in sd],
        "ic95_inf": np.exp(lower),
        "ic95_sup": np.exp(upper),
        "estabilidad_signo": np.mean(np.sign(bootstrap) == np.sign(point), axis=0),
        "n_bootstrap": len(bootstrap),
    })


def permutation_table(
    frame: pd.DataFrame, scenario: str, config: ExperimentConfig,
    models: Sequence[str] = COMPARISON_MODELS, n_repeats: int = 20,
) -> pd.DataFrame:
    """Caída de ROC-AUC en el holdout al permutar cada variable. Ranking comparable entre modelos."""
    development, holdout = split_development_holdout(frame, config.holdout_size, config.seed)
    y_dev, y_holdout = select_target(development), select_target(holdout)
    X_dev, X_holdout = scenario_matrix(development, scenario), scenario_matrix(holdout, scenario)
    rows = []
    for name in models:
        pipeline, _, _ = fit_and_select(
            model_specs()[name], list(X_dev.columns), X_dev, y_dev,
            inner_splits=config.inner_splits, min_recall=config.min_recall,
            seed=config.seed, n_jobs=config.n_jobs,
        )
        result = permutation_importance(
            pipeline, X_holdout, y_holdout, scoring="roc_auc",
            n_repeats=n_repeats, random_state=config.seed, n_jobs=config.n_jobs,
        )
        for column, mean, deviation in zip(
            X_holdout.columns, result.importances_mean, result.importances_std
        ):
            rows.append({
                "escenario": scenario, "modelo": name, "variable": column,
                "caida_auc_media": mean, "caida_auc_sd": deviation,
            })
    return pd.DataFrame(rows)


def marginal_table(
    pipeline: Pipeline, X: pd.DataFrame, scenario: str,
    variables: Sequence[tuple[str, Sequence[Any]]],
) -> pd.DataFrame:
    """Probabilidad media si todos los pacientes tuvieran ese valor (efecto marginal, no causal)."""
    rows = []
    for column, values in variables:
        for value in values:
            counterfactual = X.copy()
            counterfactual[column] = value
            rows.append({
                "escenario": scenario, "variable": column, "valor": value,
                "probabilidad_media": float(pipeline.predict_proba(counterfactual)[:, 1].mean()),
                "n": int(len(counterfactual)),
            })
    return pd.DataFrame(rows)


def patient_table(
    pipeline: Pipeline, X: pd.DataFrame, patient_ids: pd.Series, top: int = 3,
) -> pd.DataFrame:
    """Descomposición exacta en log-odds: 5 pacientes de mayor y 5 de menor puntaje."""
    if X.empty:
        raise ValueError("patient_table necesita al menos un paciente.")
    names = _feature_names(pipeline)
    coefficient = pipeline.named_steps["clf"].coef_[0]
    transformed = pipeline.named_steps["prep"].transform(X)
    contributions = (transformed - transformed.mean(axis=0)) * coefficient
    probabilities = pipeline.predict_proba(X)[:, 1]
    ranked = np.argsort(-probabilities)
    rows = []
    for position in np.concatenate([ranked[:5], ranked[-5:]]):
        logit = _logit(probabilities[position])
        for feature in np.argsort(-np.abs(contributions[position]))[:top]:
            rows.append({
                "id_paciente": patient_ids.iloc[position],
                "probabilidad": probabilities[position],
                "variable": names[feature],
                "contribucion_log_odds": contributions[position, feature],
                "delta_probabilidad": probabilities[position]
                - _sigmoid(logit - contributions[position, feature]),
            })
    return pd.DataFrame(rows)


def plot_odds_ratios(table: pd.DataFrame, path: Path) -> None:
    from matplotlib.figure import Figure  # sin pyplot, igual que experiments.py

    ordered = table.sort_values("or_por_de")
    figure = Figure(figsize=(7, 0.45 * len(ordered) + 1.5))
    axes = figure.subplots()
    positions = np.arange(len(ordered))
    axes.errorbar(
        ordered["or_por_de"], positions,
        xerr=[ordered["or_por_de"] - ordered["ic95_inf"], ordered["ic95_sup"] - ordered["or_por_de"]],
        fmt="o", ecolor="gray", capsize=3,
    )
    axes.axvline(1.0, linestyle="--", linewidth=1)
    axes.set_xscale("log")
    axes.set_yticks(positions, [name.replace("_", " ") for name in ordered["variable"]], fontsize=8)
    axes.set_xlabel("Odds ratio por 1 DE (escala log)")
    axes.set_title("Regresión logística T0 — coeficientes con IC 95 % bootstrap")
    figure.tight_layout()
    figure.savefig(path, dpi=130)


def main(argv: Sequence[str] | None = None) -> int:
    from src.infrastructure.data.etl import build_modeling_frame

    parser = argparse.ArgumentParser(description="Interpretabilidad del modelo de referencia (frente 5).")
    parser.add_argument("--scenario", default="T0", choices=("T0", "T1", "T1_sin_habla"))
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--repeats-permutation", type=int, default=20)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    config = ExperimentConfig()
    frame = build_modeling_frame(args.raw_dir)
    development, holdout = split_development_holdout(frame, config.holdout_size, config.seed)
    X_dev, y_dev = scenario_matrix(development, args.scenario), select_target(development)
    X_holdout = scenario_matrix(holdout, args.scenario)

    pipeline, params, threshold = fit_and_select(
        model_specs()[REFERENCE_MODEL], list(X_dev.columns), X_dev, y_dev,
        inner_splits=config.inner_splits, min_recall=config.min_recall,
        seed=config.seed, n_jobs=config.n_jobs,
    )

    ses_values = sorted(X_dev["nivel_socioeconomico_ord"].dropna().unique())
    distance_values = np.round(
        np.quantile(X_dev["distancia_hospital_km"].dropna(), np.linspace(0.1, 0.9, 9)), 1
    )
    tables = {
        "interpretabilidad_coeficientes.csv": coefficient_table(
            pipeline, list(X_dev.columns), X_dev, y_dev, n_bootstrap=args.bootstrap
        ),
        "interpretabilidad_permutacion.csv": permutation_table(
            frame, args.scenario, config, n_repeats=args.repeats_permutation
        ),
        "interpretabilidad_marginal.csv": marginal_table(
            pipeline, X_dev, args.scenario,
            [("nivel_socioeconomico_ord", ses_values), ("distancia_hospital_km", distance_values)],
        ),
        "interpretabilidad_pacientes.csv": patient_table(
            pipeline, X_holdout, holdout.loc[X_holdout.index, "id_paciente"]
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        path = args.output_dir / filename
        table.round(6).to_csv(path, index=False, encoding="utf-8")
        print(f"  escrito: {path}")

    figures = args.output_dir / "figuras"
    figures.mkdir(parents=True, exist_ok=True)
    plot_odds_ratios(tables["interpretabilidad_coeficientes.csv"], figures / "interpretabilidad_or.png")

    print(json.dumps({"escenario": args.scenario, "hiperparametros": params,
                      "umbral": threshold}, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())