from __future__ import annotations

import pandas as pd
import pytest

from aeollm_e0.statistics import paired_question_bootstrap, question_bootstrap


def _details(correct: list[int], total: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reportId": [1, 2],
            "pair_correct": correct,
            "pair_total": total,
            "accuracy": [c / t for c, t in zip(correct, total, strict=True)],
            "spearman": [0.1, 0.2],
            "kendall": [0.0, 0.1],
        }
    )


def test_accuracy_bootstrap_estimate_uses_pooled_pairs_not_question_macro() -> None:
    details = _details([9, 0], [10, 100])
    result = question_bootstrap(
        {"model": details}, n_resamples=100, seed=7
    )
    accuracy = result[result["metric"] == "accuracy"].iloc[0]
    assert accuracy["estimate"] == pytest.approx(9 / 110)
    assert accuracy["estimate"] != pytest.approx((0.9 + 0.0) / 2)


def test_paired_accuracy_reports_net_correct_pairs() -> None:
    left = _details([9, 60], [10, 100])
    right = _details([8, 55], [10, 100])
    result = paired_question_bootstrap(
        {"left": left, "right": right},
        [("left", "right")],
        n_resamples=100,
        seed=7,
    )
    accuracy = result[result["metric"] == "accuracy"].iloc[0]
    assert accuracy["mean_delta"] == pytest.approx(6 / 110)
    assert accuracy["net_correct_pairs"] == 6
    assert accuracy["positive_questions"] == 2
