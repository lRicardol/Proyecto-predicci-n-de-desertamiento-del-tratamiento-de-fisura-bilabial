"""Experimentos de modelado de deserción (frente 4). SRP: comparar modelos, no servirlos.

Diseño (ver reports/definicion_problema.md §5 y reports/comparacion_modelos.md):

- Población: una sola cohorte (v2). El marco viene de ``build_modeling_frame`` y las matrices
  de ``select_features``: ninguna columna de ``EXCLUDED_LEAKAGE`` puede entrar.
- Escenarios: ``T0`` (ingreso), ``T1`` (tras cirugías primarias) y ``T1_sin_habla`` (T1 sin
  hipernasalidad ni IVF, que podrían medirse después de la terapia).
- Holdout del 20 % apartado al inicio; solo se evalúa una vez, al final.
- Sobre el 80 % restante: validación cruzada repetida estratificada (externa) con búsqueda de
  hiperparámetros en validación cruzada interna (anidada). Imputación, codificación y
  escalado van dentro del ``Pipeline``, ajustados solo con los datos de entrenamiento.
- El umbral exige ``recall >= min_recall`` sobre desertores y maximiza la especificidad. Se
  elige con predicciones fuera de muestra del entrenamiento, nunca con el pliegue evaluado.
- Sin pesos de clase (``class_weight`` / ``scale_pos_weight``): el desbalance se maneja con el
  umbral, para no distorsionar la calibración.

Uso: ``python -m src.infrastructure.ai_models.experiments`` (opciones con ``--help``).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.infrastructure.ai_models.evaluation_metrics import (
    DEFAULT_MIN_RECALL,
    METRIC_NAMES,
    bootstrap_intervals,
    calibration_table,
    choose_threshold,
    evaluate_scores,
    expected_calibration_error,
    subgroup_metrics,
)
from src.infrastructure.ai_models.modeling_features import (
    CATEGORICAL_FEATURES,
    select_features,
    select_target,
)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "reports" / "modelos"

SPEECH_FEATURES: Final[tuple[str, ...]] = (
    "grado_hipernasalidad",
    "insuficiencia_velofaringea_ivf",
)
SCENARIO_NAMES: Final[tuple[str, ...]] = ("T0", "T1", "T1_sin_habla")
HOLDOUT_SELECTION_FOLDS: Final[int] = 5


@dataclass(frozen=True)
class ExperimentConfig:
    scenarios: tuple[str, ...] = SCENARIO_NAMES
    models: tuple[str, ...] | None = None  # None = todos
    repeats: int = 3
    outer_splits: int = 5
    inner_splits: int = 3
    holdout_size: float = 0.2
    min_recall: float = DEFAULT_MIN_RECALL
    n_bootstrap: int = 1000
    seed: int = 42
    n_jobs: int = -1
    quick: bool = False  # 1 repetición, rejillas mínimas y bosques pequeños: solo para probar

    def __post_init__(self) -> None:
        if self.repeats < 1 or self.outer_splits < 2 or self.inner_splits < 2:
            raise ValueError("repeats >= 1 y outer_splits / inner_splits >= 2.")
        if not 0.0 < self.holdout_size < 1.0:
            raise ValueError("holdout_size debe estar en (0, 1).")
        if not 0.0 < self.min_recall <= 1.0:
            raise ValueError("min_recall debe estar en (0, 1].")
        unknown = sorted(set(self.scenarios) - set(SCENARIO_NAMES))
        if unknown:
            raise ValueError(f"Escenarios desconocidos {unknown}. Opciones: {list(SCENARIO_NAMES)}")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: Callable[[int], BaseEstimator]  # semilla -> clasificador sin ajustar
    grid: dict[str, list[Any]] = field(default_factory=dict)  # claves con prefijo ``clf__``
    scale: bool = False  # estandarizar numéricas (solo modelos sensibles a la escala)


@dataclass
class ExperimentResults:
    folds: pd.DataFrame
    summary: pd.DataFrame
    oof: pd.DataFrame
    params: pd.DataFrame
    holdout: pd.DataFrame
    calibration: pd.DataFrame
    calibration_summary: pd.DataFrame
    subgroups: pd.DataFrame
    metadata: dict[str, Any]


# --------------------------------------------------------------------------- modelos


def _build_xgboost(seed: int) -> BaseEstimator:
    from xgboost import XGBClassifier  # import diferido: solo este modelo lo necesita

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
        verbosity=0,
    )


def _trim_grid(grid: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Rejilla mínima para pruebas rápidas: 2 valores del primer parámetro, 1 de los demás."""
    trimmed: dict[str, list[Any]] = {}
    for position, (key, values) in enumerate(grid.items()):
        trimmed[key] = list(values[:2] if position == 0 else values[:1])
    return trimmed


def model_specs(quick: bool = False) -> dict[str, ModelSpec]:
    """Catálogo de modelos. Ninguno usa ``class_weight`` ni ``scale_pos_weight``."""
    n_trees = 50 if quick else 200
    specs = [
        ModelSpec(
            "dummy_mayoritaria", lambda seed: DummyClassifier(strategy="most_frequent")
        ),
        ModelSpec("dummy_prior", lambda seed: DummyClassifier(strategy="prior")),
        ModelSpec(
            "dummy_estratificado",
            lambda seed: DummyClassifier(strategy="stratified", random_state=seed),
        ),
        ModelSpec(
            "regresion_logistica",
            lambda seed: LogisticRegression(max_iter=2000, random_state=seed),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
            scale=True,
        ),
        ModelSpec(
            "arbol",
            lambda seed: DecisionTreeClassifier(random_state=seed),
            {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [10, 30, 60]},
        ),
        ModelSpec(
            "random_forest",
            lambda seed: RandomForestClassifier(n_estimators=n_trees, n_jobs=1, random_state=seed),
            {"clf__max_depth": [4, 8, None], "clf__min_samples_leaf": [5, 20]},
        ),
        ModelSpec(
            "xgboost",
            _build_xgboost,
            {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [2, 4],
                "clf__learning_rate": [0.05, 0.1],
            },
        ),
    ]
    if quick:
        specs = [
            ModelSpec(s.name, s.build, _trim_grid(s.grid), s.scale) for s in specs
        ]
    return {spec.name: spec for spec in specs}


# --------------------------------------------------------------------------- datos


def scenario_matrix(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Matriz de un escenario, siempre a través del contrato de ``modeling_features``."""
    if scenario == "T0":
        return select_features(frame, "T0")
    if scenario == "T1":
        return select_features(frame, "T1")
    if scenario == "T1_sin_habla":
        return select_features(frame, "T1").drop(columns=list(SPEECH_FEATURES))
    raise ValueError(f"Escenario desconocido {scenario!r}. Opciones: {list(SCENARIO_NAMES)}")


def split_development_holdout(
    frame: pd.DataFrame, holdout_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aparta el holdout estratificado. El índice original se conserva en ambas partes."""
    target = select_target(frame)
    development, holdout = train_test_split(
        frame, test_size=holdout_size, stratify=target, random_state=seed
    )
    return development.sort_index(), holdout.sort_index()


def build_pipeline(estimator: BaseEstimator, columns: Sequence[str], scale: bool) -> Pipeline:
    """Imputación + codificación (+ escalado) + clasificador, todo ajustable solo con train."""
    categorical = [column for column in columns if column in CATEGORICAL_FEATURES]
    numeric = [column for column in columns if column not in categorical]
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers: list[tuple[str, Any, list[str]]] = [("num", Pipeline(numeric_steps), numeric)]
    if categorical:
        categorical_steps = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_steps, categorical))
    return Pipeline(
        [("prep", ColumnTransformer(transformers, remainder="drop")), ("clf", estimator)]
    )


# --------------------------------------------------------------------------- ajuste


def fit_and_select(
    spec: ModelSpec,
    columns: Sequence[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    inner_splits: int,
    min_recall: float,
    seed: int,
    n_jobs: int,
) -> tuple[Pipeline, dict[str, Any], float]:
    """Búsqueda interna de hiperparámetros y umbral, usando solo ``X_train`` / ``y_train``.

    El umbral sale de predicciones fuera de muestra (validación cruzada interna) del modelo ya
    elegido; así no depende de lo bien que el modelo se ajusta a sus propios datos.
    """
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    pipeline = build_pipeline(spec.build(seed), columns, spec.scale)
    if spec.grid:
        search = GridSearchCV(
            pipeline, spec.grid, scoring="roc_auc", cv=inner_cv, n_jobs=n_jobs, refit=True
        )
        search.fit(X_train, y_train)
        best, best_params = search.best_estimator_, dict(search.best_params_)
    else:
        best, best_params = clone(pipeline).fit(X_train, y_train), {}
    out_of_sample = cross_val_predict(
        clone(best), X_train, y_train, cv=inner_cv, method="predict_proba", n_jobs=n_jobs
    )[:, 1]
    return best, best_params, choose_threshold(y_train, out_of_sample, min_recall)


def _positive_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def cross_validate_model(
    spec: ModelSpec,
    scenario: str,
    X: pd.DataFrame,
    y: pd.Series,
    config: ExperimentConfig,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]]]:
    """Validación cruzada externa repetida con ajuste interno en cada pliegue."""
    outer = RepeatedStratifiedKFold(
        n_splits=config.outer_splits, n_repeats=config.repeats, random_state=config.seed
    )
    fold_rows: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []
    param_rows: list[dict[str, Any]] = []
    for number, (train_idx, valid_idx) in enumerate(outer.split(X, y)):
        repeat, fold = divmod(number, config.outer_splits)
        model, params, threshold = fit_and_select(
            spec,
            list(X.columns),
            X.iloc[train_idx],
            y.iloc[train_idx],
            inner_splits=config.inner_splits,
            min_recall=config.min_recall,
            seed=config.seed + number,
            n_jobs=config.n_jobs,
        )
        y_valid = y.iloc[valid_idx]
        scores = _positive_scores(model, X.iloc[valid_idx])
        key = {"escenario": scenario, "modelo": spec.name, "repeticion": repeat, "pliegue": fold}
        fold_rows.append(
            {**key, "umbral": threshold, **evaluate_scores(y_valid, scores, threshold)}
        )
        param_rows.append({**key, "hiperparametros": json.dumps(params, default=str)})
        oof_parts.append(
            pd.DataFrame(
                {
                    "escenario": scenario,
                    "modelo": spec.name,
                    "repeticion": repeat,
                    "fila": X.index[valid_idx],
                    "y": y_valid.to_numpy(),
                    "score": scores,
                    "alerta": (scores >= threshold).astype(int),
                }
            )
        )
    return fold_rows, oof_parts, param_rows


def evaluate_holdout(
    spec: ModelSpec,
    scenario: str,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    """Ajusta con todo el desarrollo (búsqueda y umbral incluidos) y evalúa el holdout una vez."""
    model, params, threshold = fit_and_select(
        spec,
        list(X_dev.columns),
        X_dev,
        y_dev,
        inner_splits=HOLDOUT_SELECTION_FOLDS,
        min_recall=config.min_recall,
        seed=config.seed,
        n_jobs=config.n_jobs,
    )
    scores = _positive_scores(model, X_holdout)
    estimate = evaluate_scores(y_holdout, scores, threshold)
    intervals = bootstrap_intervals(
        y_holdout, scores, threshold, n_boot=config.n_bootstrap, seed=config.seed
    )
    return [
        {
            "escenario": scenario,
            "modelo": spec.name,
            "metrica": metric,
            "valor": estimate[metric],
            "ic95_inf": intervals[metric][0],
            "ic95_sup": intervals[metric][1],
            "umbral": threshold,
            "hiperparametros": json.dumps(params, default=str),
        }
        for metric in METRIC_NAMES
    ]


# --------------------------------------------------------------------------- resúmenes


def summarize_folds(folds: pd.DataFrame) -> pd.DataFrame:
    """Media, desviación y percentiles 2,5 / 97,5 de cada métrica entre los pliegues externos.

    Es variación entre pliegues, no un intervalo de confianza: los pliegues comparten datos.
    """
    long = folds.melt(
        id_vars=["escenario", "modelo"],
        value_vars=list(METRIC_NAMES),
        var_name="metrica",
        value_name="valor",
    )
    grouped = long.groupby(["escenario", "modelo", "metrica"], sort=False)["valor"]
    summary = grouped.agg(
        media="mean",
        sd="std",
        p2_5=lambda values: values.quantile(0.025),
        p97_5=lambda values: values.quantile(0.975),
        n_pliegues="count",
    ).reset_index()
    return summary


def build_calibration(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabla de calibración y ECE con las predicciones fuera de muestra de todas las repeticiones."""
    tables, errors = [], []
    for (scenario, model), part in oof.groupby(["escenario", "modelo"], sort=False):
        table = calibration_table(part["y"], part["score"])
        tables.append(table.assign(escenario=scenario, modelo=model))
        errors.append({"escenario": scenario, "modelo": model, "ece": expected_calibration_error(table)})
    return pd.concat(tables, ignore_index=True), pd.DataFrame(errors)


def subgroup_definitions(development: pd.DataFrame) -> dict[str, pd.Series]:
    """Subgrupos de equidad: nivel socioeconómico y cuartil de distancia al hospital."""
    ses = development["nivel_socioeconomico_ord"].map(
        lambda value: f"SES {int(value)}" if pd.notna(value) else None
    )
    quartile = pd.qcut(development["distancia_hospital_km"], q=4, labels=False, duplicates="drop")
    distance = quartile.map(lambda value: f"Q{int(value) + 1} distancia" if pd.notna(value) else None)
    return {"nivel_socioeconomico": ses, "distancia_hospital": distance}


def build_subgroups(oof: pd.DataFrame, development: pd.DataFrame) -> pd.DataFrame:
    """Desempeño por subgrupo con la puntuación media de cada paciente entre repeticiones."""
    per_patient = (
        oof.groupby(["escenario", "modelo", "fila"], sort=False)
        .agg(y=("y", "first"), score=("score", "mean"), alerta=("alerta", "mean"))
        .reset_index()
    )
    definitions = subgroup_definitions(development)
    tables = []
    for (scenario, model), part in per_patient.groupby(["escenario", "modelo"], sort=False):
        for variable, labels in definitions.items():
            table = subgroup_metrics(part["y"], part["score"], part["alerta"], labels.loc[part["fila"]])
            tables.append(table.assign(escenario=scenario, modelo=model, variable=variable))
    return pd.concat(tables, ignore_index=True)


# --------------------------------------------------------------------------- orquestación


def run_experiments(frame: pd.DataFrame, config: ExperimentConfig | None = None) -> ExperimentResults:
    config = config or ExperimentConfig()
    if config.quick:
        config = replace(config, repeats=1)
    catalog = model_specs(config.quick)
    names = config.models if config.models is not None else tuple(catalog)
    unknown = sorted(set(names) - set(catalog))
    if unknown:
        raise ValueError(f"Modelos desconocidos {unknown}. Opciones: {list(catalog)}")

    started = time.time()
    development, holdout = split_development_holdout(frame, config.holdout_size, config.seed)
    y_dev, y_holdout = select_target(development), select_target(holdout)

    fold_rows: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []
    param_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    seconds: dict[str, float] = {}
    for scenario in config.scenarios:
        X_dev = scenario_matrix(development, scenario)
        X_holdout = scenario_matrix(holdout, scenario)
        for name in names:
            tic = time.time()
            folds, oofs, params = cross_validate_model(catalog[name], scenario, X_dev, y_dev, config)
            fold_rows += folds
            oof_parts += oofs
            param_rows += params
            holdout_rows += evaluate_holdout(
                catalog[name], scenario, X_dev, y_dev, X_holdout, y_holdout, config
            )
            seconds[f"{scenario}/{name}"] = round(time.time() - tic, 1)
            print(f"[{scenario}] {name}: {seconds[f'{scenario}/{name}']} s", flush=True)

    folds_frame = pd.DataFrame(fold_rows)
    oof = pd.concat(oof_parts, ignore_index=True)
    calibration, calibration_summary = build_calibration(oof)
    metadata = _metadata(config, development, holdout, names, seconds, time.time() - started)
    return ExperimentResults(
        folds=folds_frame,
        summary=summarize_folds(folds_frame),
        oof=oof,
        params=pd.DataFrame(param_rows),
        holdout=pd.DataFrame(holdout_rows),
        calibration=calibration,
        calibration_summary=calibration_summary,
        subgroups=build_subgroups(oof, development),
        metadata=metadata,
    )


def _library_versions() -> dict[str, str]:
    import importlib.metadata as metadata

    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "xgboost"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "no instalado"
    return versions


def _metadata(
    config: ExperimentConfig,
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    names: Sequence[str],
    seconds: dict[str, float],
    total_seconds: float,
) -> dict[str, Any]:
    return {
        "configuracion": {**config.__dict__, "scenarios": list(config.scenarios), "models": list(names)},
        "pacientes_desarrollo": int(len(development)),
        "pacientes_holdout": int(len(holdout)),
        "tasa_desercion_desarrollo": float(select_target(development).mean()),
        "tasa_desercion_holdout": float(select_target(holdout).mean()),
        "segundos_por_modelo": seconds,
        "segundos_totales": round(total_seconds, 1),
        "python": platform.python_version(),
        "bibliotecas": _library_versions(),
    }


# --------------------------------------------------------------------------- salidas


def _plot_calibration(calibration: pd.DataFrame, scenario: str, path: Path) -> None:
    from matplotlib.figure import Figure  # sin pyplot: no depende del backend

    figure = Figure(figsize=(6, 6))
    axes = figure.subplots()
    axes.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibración perfecta")
    for model, part in calibration[calibration["escenario"] == scenario].groupby("modelo", sort=False):
        axes.plot(part["score_medio"], part["frecuencia_observada"], marker="o", label=model)
    axes.set_xlabel("Probabilidad predicha (media por caja)")
    axes.set_ylabel("Frecuencia observada de desertores")
    axes.set_title(f"Calibración fuera de muestra — {scenario}")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)


def write_results(results: ExperimentResults, output_dir: Path) -> list[Path]:
    """Escribe tablas CSV, metadatos y figuras de calibración. Devuelve las rutas creadas."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figuras"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "cv_resultados_por_pliegue.csv": results.folds,
        "cv_resumen.csv": results.summary,
        "cv_hiperparametros.csv": results.params,
        "holdout_resultados.csv": results.holdout,
        "calibracion_cajas.csv": results.calibration,
        "calibracion_resumen.csv": results.calibration_summary,
        "subgrupos.csv": results.subgroups,
    }
    written: list[Path] = []
    for filename, table in tables.items():
        path = output_dir / filename
        table.round(6).to_csv(path, index=False, encoding="utf-8")
        written.append(path)
    metadata_path = output_dir / "metadatos.json"
    metadata_path.write_text(
        json.dumps(results.metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    written.append(metadata_path)
    for scenario in results.calibration["escenario"].unique():
        figure_path = figures_dir / f"calibracion_{scenario}.png"
        _plot_calibration(results.calibration, str(scenario), figure_path)
        written.append(figure_path)
    return written


# --------------------------------------------------------------------------- CLI


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparación de modelos de deserción (frente 4).")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIO_NAMES), choices=SCENARIO_NAMES)
    parser.add_argument("--models", nargs="+", default=None, help="Por defecto, todos.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-recall", type=float, default=DEFAULT_MIN_RECALL)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quick", action="store_true", help="Prueba rápida; no usar para reportar.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from src.core.exceptions import DataAccessError, InvalidPatientDataError
    from src.infrastructure.data.etl import build_modeling_frame  # import diferido: evita dependencia circular de capas

    args = _parse_args(argv)
    try:
        config = ExperimentConfig(
            scenarios=tuple(args.scenarios),
            models=tuple(args.models) if args.models else None,
            repeats=args.repeats,
            min_recall=args.min_recall,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
            n_jobs=args.n_jobs,
            quick=args.quick,
        )
        frame = build_modeling_frame(args.raw_dir)
        results = run_experiments(frame, config)
    except (DataAccessError, InvalidPatientDataError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    for path in write_results(results, args.output_dir):
        print(f"  escrito: {path}")
    print(f"Listo en {results.metadata['segundos_totales']} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
