"""Pruebas de los experimentos de modelado (frente 4). Datos sintéticos: rápidas y sin acoplarse al CSV."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.infrastructure.ai_models.evaluation_metrics import METRIC_NAMES
from src.infrastructure.ai_models.experiments import (
    SCENARIO_NAMES,
    SPEECH_FEATURES,
    ExperimentConfig,
    build_pipeline,
    cross_validate_model,
    fit_and_select,
    main,
    model_specs,
    run_experiments,
    scenario_matrix,
    split_development_holdout,
    write_results,
)
from src.infrastructure.ai_models.modeling_features import (
    EXCLUDED_LEAKAGE,
    FEATURES_T0,
    FEATURES_T1,
    TARGET,
    select_target,
)

LEAK_COLUMN = "citas_perdidas_noshow"


def make_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Marco sintético con el contrato T1 y una columna con fuga que nunca debe usarse."""
    rng = np.random.default_rng(seed)
    ses = rng.choice([0, 1, 2, 4], n, p=[0.45, 0.35, 0.15, 0.05])
    distance = np.clip(rng.gamma(2.0, 25.0, n), 5, 300)
    speech = rng.integers(0, 4, n)
    logit = 0.9 + 0.012 * (distance - 50) - 0.3 * ses + 0.5 * (speech - 1)
    desertor = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    frame = pd.DataFrame(
        {
            "id_paciente": [f"PAC-{2000 + i}" for i in range(n)],
            "tipo_fisura": rng.choice(["Completa", "Incompleta", "Asimetrica"], n),
            "edad_inicio_meses": rng.integers(1, 12, n),
            "distancia_hospital_km": distance,
            "nivel_socioeconomico_ord": ses.astype(float),
            "ingresos_mensuales_usd": rng.integers(250, 6000, n),
            "apoyo_psicologico": rng.integers(0, 2, n),
            "dificultad_alimentacion_preop": rng.integers(0, 2, n),
            "edad_queiloplastia_meses": rng.uniform(3, 9, n),
            "edad_palatoplastia_meses": rng.uniform(9, 20, n),
            "dehiscencia_herida_postop": rng.integers(0, 2, n),
            "fistula_palatina_residual": rng.integers(0, 2, n),
            "hipoacusia_conductiva": rng.integers(0, 2, n),
            "grado_hipernasalidad": speech,
            "insuficiencia_velofaringea_ivf": rng.integers(0, 2, n),
            TARGET: desertor,
        }
    )
    frame[LEAK_COLUMN] = desertor * 8 + rng.integers(0, 2, n)  # separa la etiqueta casi perfecto
    return frame


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_frame()


def quick_config(**overrides: object) -> ExperimentConfig:
    base = dict(quick=True, repeats=1, n_bootstrap=20, n_jobs=1, seed=7)
    base.update(overrides)
    return ExperimentConfig(**base)  # type: ignore[arg-type]


def xgboost_works() -> bool:
    try:
        from xgboost import XGBClassifier

        XGBClassifier(n_estimators=2, verbosity=0).fit(np.arange(20).reshape(10, 2), [0, 1] * 5)
    except Exception:
        return False
    return True


# --------------------------------------------------------------------------- datos y escenarios


def test_scenarios_follow_the_contract_and_ignore_leakage_columns(frame: pd.DataFrame) -> None:
    assert LEAK_COLUMN in frame.columns and LEAK_COLUMN in EXCLUDED_LEAKAGE
    assert list(scenario_matrix(frame, "T0").columns) == list(FEATURES_T0)
    assert list(scenario_matrix(frame, "T1").columns) == list(FEATURES_T1)
    without_speech = scenario_matrix(frame, "T1_sin_habla")
    assert not set(SPEECH_FEATURES) & set(without_speech.columns)
    assert set(without_speech.columns) == set(FEATURES_T1) - set(SPEECH_FEATURES)
    for scenario in SCENARIO_NAMES:
        assert not set(scenario_matrix(frame, scenario).columns) & set(EXCLUDED_LEAKAGE)


def test_unknown_scenario_fails(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Escenario desconocido"):
        scenario_matrix(frame, "T9")
    with pytest.raises(ValueError, match="Escenarios desconocidos"):
        ExperimentConfig(scenarios=("T9",))


def test_holdout_is_disjoint_stratified_and_keeps_index(frame: pd.DataFrame) -> None:
    development, holdout = split_development_holdout(frame, 0.2, seed=1)
    assert len(development) + len(holdout) == len(frame)
    assert len(holdout) == pytest.approx(0.2 * len(frame), abs=1)
    assert not set(development.index) & set(holdout.index)
    assert select_target(holdout).mean() == pytest.approx(select_target(development).mean(), abs=0.03)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(repeats=0)
    with pytest.raises(ValueError):
        ExperimentConfig(min_recall=1.5)
    with pytest.raises(ValueError):
        ExperimentConfig(holdout_size=1.0)


# --------------------------------------------------------------------------- modelos y pipeline


def test_models_use_no_class_weights() -> None:
    for quick in (False, True):
        for name, spec in model_specs(quick).items():
            if name == "xgboost" and not xgboost_works():
                continue  # en un entorno sin XGBoost real se omite; en el del proyecto sí se comprueba
            params = spec.build(0).get_params()
            assert params.get("class_weight") is None, name
            assert params.get("scale_pos_weight") in (None, 1), name
            assert not any("weight" in key for key in spec.grid), name


def test_catalog_contains_baselines_and_requested_models() -> None:
    assert set(model_specs()) == {
        "dummy_mayoritaria",
        "dummy_prior",
        "dummy_estratificado",
        "regresion_logistica",
        "arbol",
        "random_forest",
        "xgboost",
    }


def test_quick_grids_are_small() -> None:
    for spec in model_specs(quick=True).values():
        combos = int(np.prod([len(values) for values in spec.grid.values()])) if spec.grid else 1
        assert combos <= 2, spec.name


def test_pipeline_handles_missing_values_and_unseen_categories(frame: pd.DataFrame) -> None:
    X = scenario_matrix(frame, "T1")
    y = select_target(frame)
    train, valid = X.iloc[:180].copy(), X.iloc[180:].copy()
    train.iloc[0, train.columns.get_loc("distancia_hospital_km")] = np.nan
    train.iloc[1, train.columns.get_loc("tipo_fisura")] = None
    valid["tipo_fisura"] = "CategoriaNueva"
    for spec in model_specs(quick=True).values():
        if spec.name == "xgboost" and not xgboost_works():
            continue
        pipeline = build_pipeline(spec.build(0), list(X.columns), spec.scale)
        pipeline.fit(train, y.iloc[:180])
        assert np.isfinite(pipeline.predict_proba(valid)[:, 1]).all(), spec.name


def test_preprocessing_lives_inside_the_pipeline_and_scales_only_when_asked() -> None:
    specs = model_specs(quick=True)
    columns = list(FEATURES_T1)

    def numeric_branch(name: str):
        spec = specs[name]
        pipeline = build_pipeline(spec.build(0), columns, spec.scale)
        assert list(pipeline.named_steps) == ["prep", "clf"]
        return {step: trans for step, trans, _ in pipeline.named_steps["prep"].transformers}["num"]

    assert "scaler" in numeric_branch("regresion_logistica").named_steps
    forest = numeric_branch("random_forest")
    assert "scaler" not in forest.named_steps
    assert "imputer" in forest.named_steps


# --------------------------------------------------------------------------- ajuste y validación


def test_fit_and_select_returns_threshold_and_grid_params(frame: pd.DataFrame) -> None:
    spec = model_specs(quick=True)["arbol"]
    X, y = scenario_matrix(frame, "T0"), select_target(frame)
    model, params, threshold = fit_and_select(
        spec, list(X.columns), X, y, inner_splits=3, min_recall=0.9, seed=0, n_jobs=1
    )
    assert 0.0 < threshold <= 1.0
    assert set(params) <= set(spec.grid)
    assert model.predict_proba(X).shape == (len(X), 2)


@pytest.mark.parametrize("name", ["dummy_prior", "regresion_logistica", "arbol", "random_forest"])
def test_cross_validation_covers_each_patient_once_per_repeat(frame: pd.DataFrame, name: str) -> None:
    X, y = scenario_matrix(frame, "T0"), select_target(frame)
    config = quick_config(repeats=2, outer_splits=4)
    folds, oofs, params = cross_validate_model(model_specs(quick=True)[name], "T0", X, y, config)
    oof = pd.concat(oofs, ignore_index=True)
    assert len(folds) == len(params) == 2 * 4
    assert set(folds[0]) >= {"escenario", "modelo", "repeticion", "pliegue", "umbral", *METRIC_NAMES}
    for _, part in oof.groupby("repeticion"):
        assert sorted(part["fila"]) == sorted(X.index)
    assert set(oof["alerta"]) <= {0, 1}


def test_cross_validation_is_reproducible(frame: pd.DataFrame) -> None:
    X, y = scenario_matrix(frame, "T0"), select_target(frame)
    spec = model_specs(quick=True)["regresion_logistica"]
    first = cross_validate_model(spec, "T0", X, y, quick_config(outer_splits=3))[0]
    second = cross_validate_model(spec, "T0", X, y, quick_config(outer_splits=3))[0]
    assert first == second


def test_leakage_column_never_inflates_the_results(frame: pd.DataFrame) -> None:
    """La columna con fuga separa la etiqueta casi perfecto; si entrara, el AUC sería ~1."""
    results = run_experiments(
        frame, quick_config(scenarios=("T1",), models=("regresion_logistica", "arbol"), outer_splits=3)
    )
    auc = results.summary.query("metrica == 'roc_auc'")["media"]
    assert (auc < 0.95).all()


# --------------------------------------------------------------------------- orquestación


@pytest.fixture(scope="module")
def results(frame: pd.DataFrame):
    config = quick_config(
        scenarios=("T0", "T1_sin_habla"),
        models=("dummy_prior", "regresion_logistica", "arbol"),
        outer_splits=3,
    )
    return run_experiments(frame, config)


def test_results_have_one_row_per_scenario_model_and_metric(results) -> None:
    assert len(results.folds) == 2 * 3 * 3  # escenarios x modelos x pliegues (1 repetición)
    assert len(results.summary) == 2 * 3 * len(METRIC_NAMES)
    assert len(results.holdout) == 2 * 3 * len(METRIC_NAMES)
    assert not results.holdout.duplicated(["escenario", "modelo", "metrica"]).any()
    assert set(results.holdout["metrica"]) == set(METRIC_NAMES)
    assert (results.holdout["ic95_inf"] <= results.holdout["ic95_sup"]).all()
    assert set(results.calibration_summary["modelo"]) == {"dummy_prior", "regresion_logistica", "arbol"}
    assert {"nivel_socioeconomico", "distancia_hospital"} == set(results.subgroups["variable"])


def test_holdout_patients_never_appear_in_cross_validation(frame: pd.DataFrame, results) -> None:
    _, holdout = split_development_holdout(frame, 0.2, seed=7)
    assert not set(results.oof["fila"]) & set(holdout.index)
    assert results.metadata["pacientes_holdout"] == len(holdout)
    assert results.metadata["pacientes_desarrollo"] == len(frame) - len(holdout)


def test_dummy_prior_is_a_chance_level_baseline(results) -> None:
    auc = results.summary.query("modelo == 'dummy_prior' and metrica == 'roc_auc'")["media"]
    assert auc.eq(0.5).all()


def test_write_results_creates_tables_metadata_and_figures(results, tmp_path: Path) -> None:
    written = {path.name for path in write_results(results, tmp_path)}
    assert {
        "cv_resultados_por_pliegue.csv",
        "cv_resumen.csv",
        "cv_hiperparametros.csv",
        "holdout_resultados.csv",
        "calibracion_cajas.csv",
        "calibracion_resumen.csv",
        "subgrupos.csv",
        "metadatos.json",
        "calibracion_T0.png",
        "calibracion_T1_sin_habla.png",
    } <= written
    metadata = json.loads((tmp_path / "metadatos.json").read_text(encoding="utf-8"))
    assert metadata["configuracion"]["min_recall"] == 0.9
    assert (tmp_path / "figuras" / "calibracion_T0.png").stat().st_size > 0


def test_unknown_model_fails(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Modelos desconocidos"):
        run_experiments(frame, quick_config(models=("modelo_inexistente",)))


@pytest.mark.skipif(not xgboost_works(), reason="XGBoost real no disponible en este entorno")
def test_xgboost_runs_through_the_same_protocol(frame: pd.DataFrame) -> None:
    results = run_experiments(
        frame, quick_config(scenarios=("T0",), models=("xgboost",), outer_splits=3)
    )
    assert len(results.folds) == 3
    assert results.holdout["metrica"].nunique() == len(METRIC_NAMES)


# --------------------------------------------------------------------------- CLI y datos reales


def test_cli_rejects_unknown_models(tmp_path: Path) -> None:
    assert main(["--quick", "--models", "no_existe", "--output-dir", str(tmp_path)]) == 1


def test_cli_smoke_on_real_v2_cohort(tmp_path: Path) -> None:
    code = main(
        [
            "--quick",
            "--scenarios",
            "T0",
            "--models",
            "dummy_prior",
            "arbol",
            "--n-jobs",
            "1",
            "--bootstrap",
            "20",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    metadata = json.loads((tmp_path / "metadatos.json").read_text(encoding="utf-8"))
    assert metadata["pacientes_desarrollo"] + metadata["pacientes_holdout"] == 3000
    assert metadata["pacientes_holdout"] == 600
