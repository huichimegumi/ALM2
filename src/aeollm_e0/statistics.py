from __future__ import annotations

import numpy as np
import pandas as pd

PRIMARY_METRICS = ["spearman", "kendall", "accuracy"]


def question_bootstrap(
    details_by_model: dict[str, pd.DataFrame],
    *,
    n_resamples: int = 5000,
    seed: int = 20260721,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for model_name, details in details_by_model.items():
        values = details.set_index("reportId")
        question_ids = values.index.to_numpy()
        for metric in PRIMARY_METRICS:
            observed = float(values[metric].mean())
            samples = np.empty(n_resamples, dtype=float)
            for index in range(n_resamples):
                selected = rng.choice(question_ids, size=len(question_ids), replace=True)
                samples[index] = float(values.loc[selected, metric].mean())
            finite = samples[np.isfinite(samples)]
            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "estimate": observed,
                    "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
                    "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
                    "n_questions": int(len(question_ids)),
                    "n_resamples": int(n_resamples),
                }
            )
    return pd.DataFrame(rows)


def paired_question_bootstrap(
    details_by_model: dict[str, pd.DataFrame],
    comparisons: list[tuple[str, str]],
    *,
    n_resamples: int = 5000,
    seed: int = 20260721,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for candidate, reference in comparisons:
        if candidate not in details_by_model or reference not in details_by_model:
            continue
        left = details_by_model[candidate].set_index("reportId")
        right = details_by_model[reference].set_index("reportId")
        common = left.index.intersection(right.index).to_numpy()
        for metric in PRIMARY_METRICS:
            deltas = left.loc[common, metric].to_numpy() - right.loc[common, metric].to_numpy()
            deltas = deltas[np.isfinite(deltas)]
            if not len(deltas):
                continue
            samples = np.empty(n_resamples, dtype=float)
            for index in range(n_resamples):
                selected = rng.integers(0, len(deltas), size=len(deltas))
                samples[index] = float(deltas[selected].mean())
            rows.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    "mean_delta": float(deltas.mean()),
                    "ci_low": float(np.quantile(samples, 0.025)),
                    "ci_high": float(np.quantile(samples, 0.975)),
                    "probability_delta_gt_zero": float(np.mean(samples > 0)),
                    "n_questions": int(len(common)),
                }
            )
    return pd.DataFrame(rows)
