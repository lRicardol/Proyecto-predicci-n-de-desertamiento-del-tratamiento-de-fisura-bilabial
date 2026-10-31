"""Pruebas de las métricas de evaluación (frente 4): umbral, top-k, calibración y subgrupos."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.infrastructure.ai_models.evaluation_metrics import (
    METRIC_NAMES,
    bootstrap_intervals,
    calibration_table,
    choose_threshold,
    evaluate_scores,
    expected_calibration_error,
    subgroup_metrics,
    threshold_metrics,
    top_k_metrics,
)

Y = [1, 1, 1, 1, 0, 0, 0, 0]
SCORES = [0.9, 0.8, 0.7, 0.3, 0.6, 0.2, 0.1, 0.05]


def test_threshold_reaches_min_recall_with_best_specificity() -> None:
    threshold = choose_threshold(Y, SCORES, min_recall=0.9)
    metrics = threshold_metrics(Y, SCORES, threshold)
    assert threshold == pytest.approx(0.3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["especificidad"] == pytest.approx(0.75)


def test_threshold_relaxes_with_lower_min_recall() -> None:
    threshold = choose_threshold(Y, SCORES, min_recall=0.75)
    metrics = threshold_metrics(Y, SCORES, threshold)
    assert threshold == pytest.approx(0.7)
    assert metrics["recall"] == pytest.approx(0.75)
    assert metrics["especificidad"] == pytest.approx(1.0)


def test_threshold_with_constant_scores_flags_everyone() -> None:
    scores = [0.5] * len(Y)
    threshold = choose_threshold(Y, scores)
    metrics = threshold_metrics(Y, scores, threshold)
    assert threshold == pytest.approx(0.5)
    assert metrics["recall"] == 1.0
    assert metrics["especificidad"] == 0.0


def test_threshold_requires_both_classes_and_valid_recall() -> None:
    with pytest.raises(ValueError, match="dos clases"):
        choose_threshold([1, 1, 1], [0.2, 0.5, 0.9])
    with pytest.raises(ValueError, match="min_recall"):
        choose_threshold(Y, SCORES, min_recall=0.0)
    with pytest.raises(ValueError, match="misma longitud"):
        choose_threshold(Y, SCORES[:-1])


def test_threshold_metrics_handle_empty_denominators() -> None:
    metrics = threshold_metrics([1, 1], [0.9, 0.8], threshold=0.95)
    assert metrics["recall"] == 0.0
    assert math.isnan(metrics["especificidad"])
    assert math.isnan(metrics["precision"])
    assert metrics["proporcion_alertas"] == 0.0


def test_top_k_metrics_on_distinct_scores() -> None:
    y = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    scores = np.linspace(1.0, 0.1, 10)
    result = top_k_metrics(y, scores)
    assert result["captura_top10"] == pytest.approx(1 / 3)
    assert result["precision_top10"] == pytest.approx(1.0)
    assert result["captura_top20"] == pytest.approx(2 / 3)
    assert result["captura_top30"] == pytest.approx(2 / 3)  # el tercer paciente es continuador
    assert result["precision_top30"] == pytest.approx(2 / 3)


def test_top_k_ties_do_not_depend_on_row_order() -> None:
    rng = np.random.default_rng(0)
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.full(10, 0.4)
    expected = top_k_metrics(y, scores)
    assert expected["captura_top10"] == pytest.approx(0.1)  # 0.3 desertores esperados de 3
    for _ in range(5):
        order = rng.permutation(10)
        assert top_k_metrics(y[order], scores[order]) == pytest.approx(expected)


def test_evaluate_scores_returns_every_metric() -> None:
    result = evaluate_scores(Y, SCORES, threshold=0.5)
    assert list(result) == list(METRIC_NAMES)
    perfect = evaluate_scores([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)
    assert perfect["roc_auc"] == 1.0
    assert perfect["pr_auc"] == pytest.approx(1.0)
    assert perfect["recall"] == 1.0 and perfect["especificidad"] == 1.0


def test_evaluate_scores_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="dos clases"):
        evaluate_scores([1, 1], [0.2, 0.9], threshold=0.5)


def test_calibration_error_is_zero_when_scores_match_frequency() -> None:
    y = [0, 1] * 50
    table = calibration_table(y, [0.5] * 100, n_bins=5)
    assert table["n"].sum() == 100
    assert expected_calibration_error(table) == pytest.approx(0.0, abs=1e-9)


def test_calibration_error_detects_overconfidence() -> None:
    y = [0, 1] * 50
    table = calibration_table(y, [0.95] * 100, n_bins=5)
    assert expected_calibration_error(table) == pytest.approx(0.45, abs=1e-9)


def test_bootstrap_intervals_bracket_the_estimate_and_are_reproducible() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    scores = np.clip(y * 0.3 + rng.normal(0.4, 0.2, 300), 0, 1)
    threshold = choose_threshold(y, scores)
    estimate = evaluate_scores(y, scores, threshold)
    first = bootstrap_intervals(y, scores, threshold, n_boot=200, seed=3)
    second = bootstrap_intervals(y, scores, threshold, n_boot=200, seed=3)
    assert first == second
    low, high = first["roc_auc"]
    assert low <= estimate["roc_auc"] <= high


def test_subgroup_metrics_reports_each_group() -> None:
    y = [1, 0, 1, 0, 1, 1]
    scores = [0.9, 0.2, 0.8, 0.6, 0.7, 0.4]
    alerts = [1, 0, 1, 1, 1, 0]
    groups = pd.Series(["a", "a", "a", "a", "b", "b"])
    table = subgroup_metrics(y, scores, alerts, groups).set_index("grupo")
    assert table.loc["a", "n"] == 4
    assert table.loc["a", "recall"] == pytest.approx(1.0)
    assert table.loc["a", "especificidad"] == pytest.approx(0.5)
    assert table.loc["a", "roc_auc"] == pytest.approx(1.0)
    assert math.isnan(table.loc["b", "roc_auc"])  # el grupo b no tiene continuadores
    assert math.isnan(table.loc["b", "especificidad"])
    assert table.loc["b", "recall"] == pytest.approx(0.5)
