from __future__ import annotations

import pandas as pd

from aeollm_e1.e2_a01_pipeline import _dimension_gate_records


def _row(
    reference: str,
    metric: str,
    delta: float,
    probability: float,
    positive: int,
) -> dict[str, object]:
    return {
        "candidate": "diagonal_matched_separate_hybrid",
        "reference": reference,
        "metric": metric,
        "mean_delta": delta,
        "ci_low": delta - 0.1,
        "ci_high": delta + 0.1,
        "probability_delta_gt_zero": probability,
        "positive_questions": positive,
        "tied_questions": 0,
        "n_questions": 10,
    }


def test_dimension_gates_are_independent_and_require_baseline_noninferiority() -> None:
    rows = []
    for dimension in ("comprehensiveness", "instruction_following"):
        metric = f"spearman_{dimension}"
        rows.append(
            _row(
                "fixed_matched_hybrid",
                metric,
                0.05,
                0.95,
                8,
            )
        )
        rows.append(
            _row(
                "ridge_global_structure",
                metric,
                0.02 if dimension == "comprehensiveness" else -0.01,
                0.7,
                6,
            )
        )
    gates = _dimension_gate_records(pd.DataFrame(rows)).set_index("dimension")
    assert bool(gates.loc["comprehensiveness", "passed"])
    assert not bool(gates.loc["instruction_following", "passed"])
