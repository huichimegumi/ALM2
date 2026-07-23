from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aeollm_e0.metrics import DIMS
from aeollm_e1.e1_7_pipeline import (
    _query_policy,
    nested_loqo_query_ridge_predictions,
)


def _toy_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    matched_rows: list[dict[str, object]] = []
    criterion_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for question_id in range(1, 5):
        for document_index in range(4):
            key = {"questionId": question_id, "answerId": f"Q{question_id}D{document_index}"}
            criterion_rows.append({**key, "signal": float(document_index)})
            matched_rows.append({**key, "signal": float((question_id + document_index) % 2)})
            label_rows.append(
                {
                    **key,
                    **{
                        dimension: 2.0 + 2.0 * document_index
                        for dimension in DIMS
                    },
                }
            )
    return (
        {
            "matched_full": pd.DataFrame(matched_rows),
            "criterion_only": pd.DataFrame(criterion_rows),
        },
        pd.DataFrame(label_rows),
    )


def test_nested_query_selection_uses_only_outer_training_questions() -> None:
    frames, labels = _toy_data()
    candidates = {
        dimension: ["matched_full", "criterion_only"]
        for dimension in DIMS
    }
    features = {
        query: {dimension: ["signal"] for dimension in DIMS}
        for query in frames
    }
    prediction, selections, candidate_scores = nested_loqo_query_ridge_predictions(
        frames,
        labels,
        candidates,
        features,
        alphas=(0.1, 1.0),
    )
    assert len(prediction) == len(labels)
    assert np.isfinite(prediction[DIMS].to_numpy()).all()
    assert set(selections["train_documents"]) == {12}
    assert set(selections["test_documents"]) == {4}
    assert set(selections["query_source"]) == {"criterion_only"}
    assert len(candidate_scores) == 4 * len(DIMS) * 2 * 2


def test_fixed_query_policy_routes_only_alignment_dimensions() -> None:
    policy = _query_policy(["criterion_only"], ["matched_full"])
    assert policy["comprehensiveness"] == ["criterion_only"]
    assert policy["instruction_following"] == ["matched_full"]
    assert policy["insight"] == ["matched_full"]
    assert policy["readability"] == ["matched_full"]


def test_query_frames_must_have_identical_keys() -> None:
    frames, labels = _toy_data()
    frames["criterion_only"] = frames["criterion_only"].iloc[:-1]
    candidates = {dimension: ["matched_full"] for dimension in DIMS}
    features = {
        query: {dimension: ["signal"] for dimension in DIMS}
        for query in frames
    }
    with pytest.raises(ValueError, match="not aligned"):
        nested_loqo_query_ridge_predictions(
            frames, labels, candidates, features, alphas=(1.0,)
        )
