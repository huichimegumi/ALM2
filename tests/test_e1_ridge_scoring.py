from __future__ import annotations

import numpy as np
import pandas as pd

from aeollm_e0.metrics import DIMS
from aeollm_e1.ridge_scoring import (
    build_model_feature_columns,
    nested_loqo_ridge_predictions,
    own_dimension_rubric_columns,
)


def test_dimension_feature_mapping_uses_only_its_own_rubric() -> None:
    groups = {
        "global": ["global_0", "global_1"],
        "rubric_primary": [f"rubric_{dimension}_score" for dimension in DIMS],
    }
    mapping = build_model_feature_columns(groups, ["length"])
    assert mapping["insight"]["rubric"] == ["rubric_insight_score"]
    assert "rubric_readability_score" not in mapping["insight"]["all"]
    assert mapping["insight"]["all"] == ["global_0", "global_1", "rubric_insight_score", "length"]
    assert own_dimension_rubric_columns(groups, "readability") == ["rubric_readability_score"]


def test_nested_loqo_predicts_every_document_without_key_leakage() -> None:
    rows = []
    labels = []
    for question_id in range(1, 5):
        for document_index in range(4):
            quality = document_index / 3
            row = {"questionId": question_id, "answerId": f"Q{question_id}D{document_index}"}
            target = {"questionId": question_id, "answerId": row["answerId"]}
            for dim_index, dimension in enumerate(DIMS):
                row[f"feature_{dimension}"] = quality + dim_index * 0.01
                target[dimension] = 2.0 + 6.0 * quality + dim_index * 0.1
            rows.append(row)
            labels.append(target)
    frame = pd.DataFrame(rows)
    truth = pd.DataFrame(labels)
    mapping = {dimension: [f"feature_{dimension}"] for dimension in DIMS}
    prediction, selections = nested_loqo_ridge_predictions(
        frame, truth, mapping, alphas=(0.1, 1.0)
    )
    assert len(prediction) == 16
    assert len(selections) == 4 * len(DIMS)
    assert np.isfinite(prediction[DIMS].to_numpy()).all()
    assert prediction[DIMS].min().min() >= 0
    assert prediction[DIMS].max().max() <= 10
    assert set(selections["test_documents"]) == {4}
