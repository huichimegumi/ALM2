from __future__ import annotations

import numpy as np
import pandas as pd

PRIMARY_METRICS = ["accuracy", "spearman", "kendall"]


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
        sampled_indices = rng.integers(
            0, len(question_ids), size=(n_resamples, len(question_ids))
        )
        for metric in PRIMARY_METRICS:
            observed = (
                float(values["pair_correct"].sum() / values["pair_total"].sum())
                if metric == "accuracy"
                else float(values[metric].mean())
            )
            if metric == "accuracy":
                correct = values["pair_correct"].to_numpy(dtype=float)
                total = values["pair_total"].to_numpy(dtype=float)
                samples = correct[sampled_indices].sum(axis=1) / total[
                    sampled_indices
                ].sum(axis=1)
            else:
                metric_values = values[metric].to_numpy(dtype=float)
                samples = metric_values[sampled_indices].mean(axis=1)
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
            question_deltas = (
                left.loc[common, metric].to_numpy()
                - right.loc[common, metric].to_numpy()
            )
            finite = np.isfinite(question_deltas)
            finite_questions = common[finite]
            question_deltas = question_deltas[finite]
            if not len(question_deltas):
                continue
            samples = np.empty(n_resamples, dtype=float)
            sampled_indices = rng.integers(
                0,
                len(finite_questions),
                size=(n_resamples, len(finite_questions)),
            )
            if metric == "accuracy":
                left_finite = left.loc[finite_questions]
                right_finite = right.loc[finite_questions]
                left_correct = left_finite["pair_correct"].to_numpy(dtype=float)
                left_total = left_finite["pair_total"].to_numpy(dtype=float)
                right_correct = right_finite["pair_correct"].to_numpy(dtype=float)
                right_total = right_finite["pair_total"].to_numpy(dtype=float)
                samples = (
                    left_correct[sampled_indices].sum(axis=1)
                    / left_total[sampled_indices].sum(axis=1)
                    - right_correct[sampled_indices].sum(axis=1)
                    / right_total[sampled_indices].sum(axis=1)
                )
            else:
                samples = question_deltas[sampled_indices].mean(axis=1)
            observed = (
                float(
                    left.loc[finite_questions, "pair_correct"].sum()
                    / left.loc[finite_questions, "pair_total"].sum()
                    - right.loc[finite_questions, "pair_correct"].sum()
                    / right.loc[finite_questions, "pair_total"].sum()
                )
                if metric == "accuracy"
                else float(question_deltas.mean())
            )
            rows.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    "mean_delta": observed,
                    "ci_low": float(np.quantile(samples, 0.025)),
                    "ci_high": float(np.quantile(samples, 0.975)),
                    "probability_delta_gt_zero": float(np.mean(samples > 0)),
                    "positive_questions": int(np.sum(question_deltas > 0)),
                    "tied_questions": int(np.sum(question_deltas == 0)),
                    "net_correct_pairs": (
                        int(
                            left.loc[finite_questions, "pair_correct"].sum()
                            - right.loc[finite_questions, "pair_correct"].sum()
                        )
                        if metric == "accuracy"
                        else np.nan
                    ),
                    "n_questions": int(len(finite_questions)),
                }
            )
    return pd.DataFrame(rows)
