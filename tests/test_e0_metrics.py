from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aeollm_e0.metrics import (
    DIMS,
    PredictionValidationError,
    evaluate_predictions,
    pairwise_accuracy,
    validate_predictions,
)


def _labels() -> pd.DataFrame:
    rows = []
    for question_id in (1, 2):
        for index, score in enumerate((1.0, 2.0, 3.0)):
            row = {"questionId": question_id, "answerId": f"q{question_id}_{index}"}
            row.update({dim: score for dim in DIMS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_pairwise_accuracy_matches_official_tie_semantics() -> None:
    accuracy, correct, total = pairwise_accuracy(
        np.asarray([1.0, 1.0, 2.0]), np.asarray([1.0, 2.0, 2.0])
    )
    assert (correct, total) == (1, 3)
    assert accuracy == pytest.approx(1 / 3)


def test_strict_validation_rejects_missing_key_and_out_of_range() -> None:
    labels = _labels()
    missing = labels.iloc[:-1].copy()
    with pytest.raises(PredictionValidationError, match="missing 1"):
        validate_predictions(missing, labels)

    invalid = labels.copy()
    invalid.loc[0, "insight"] = 10.1
    with pytest.raises(PredictionValidationError, match="outside"):
        validate_predictions(invalid, labels)


def test_perfect_predictions_have_perfect_official_metrics() -> None:
    labels = _labels()
    weights = {question_id: {dim: 0.25 for dim in DIMS} for question_id in (1, 2)}
    summary, details = evaluate_predictions(labels.copy(), labels, weights)
    assert len(details) == 2
    assert summary.iloc[0]["spearman"] == pytest.approx(1.0)
    assert summary.iloc[0]["kendall"] == pytest.approx(1.0)
    assert summary.iloc[0]["accuracy"] == pytest.approx(1.0)
    assert summary.iloc[0]["pair_total"] == 6
