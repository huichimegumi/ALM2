from __future__ import annotations

import numpy as np
import pandas as pd

from aeollm_e0.metrics import DIMS
from aeollm_e0.modeling import loqo_ridge_predictions


def test_loqo_ridge_produces_one_prediction_per_input() -> None:
    rows = []
    for question_id in range(1, 7):
        for index in range(4):
            feature = question_id * 0.1 + index
            row = {
                "questionId": question_id,
                "answerId": f"{question_id}_{index}",
                "feature": feature,
            }
            row.update({dim: feature + offset for offset, dim in enumerate(DIMS)})
            rows.append(row)
    frame = pd.DataFrame(rows)
    predictions, selections = loqo_ridge_predictions(
        frame[["questionId", "answerId", "feature"]],
        frame[["questionId", "answerId", *DIMS]],
        numeric_columns=["feature"],
    )
    assert len(predictions) == len(frame)
    assert np.isfinite(predictions[DIMS].to_numpy()).all()
    assert set(selections["held_out_question"]) == set(range(1, 7))
