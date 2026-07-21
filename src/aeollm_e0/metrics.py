from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

DIMS = ["comprehensiveness", "insight", "instruction_following", "readability"]
SCORE_COLUMNS = {dim: f"{dim}_score" for dim in DIMS}
KEY_COLUMNS = ["questionId", "answerId"]


class PredictionValidationError(ValueError):
    """Raised when predictions would be silently accepted by the official evaluator."""


def normalize_labels(frame: pd.DataFrame) -> pd.DataFrame:
    required = {*KEY_COLUMNS, *SCORE_COLUMNS.values()}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"label file missing columns: {missing}")
    out = frame.copy()
    out["questionId"] = pd.to_numeric(out["questionId"], errors="raise").astype(int)
    out["answerId"] = out["answerId"].astype(str).str.strip()
    for dim, column in SCORE_COLUMNS.items():
        out[dim] = pd.to_numeric(out[column], errors="raise")
    return out[[*KEY_COLUMNS, *DIMS]]


def normalize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    rename = {column: dim for dim, column in SCORE_COLUMNS.items() if column in out.columns}
    out = out.rename(columns=rename)
    required = {*KEY_COLUMNS, *DIMS}
    missing = sorted(required - set(out.columns))
    if missing:
        raise PredictionValidationError(f"prediction file missing columns: {missing}")
    out["questionId"] = pd.to_numeric(out["questionId"], errors="raise").astype(int)
    out["answerId"] = out["answerId"].astype(str).str.strip()
    for dim in DIMS:
        out[dim] = pd.to_numeric(out[dim], errors="coerce")
    return out[[*KEY_COLUMNS, *DIMS]]


def validate_predictions(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    require_complete: bool = True,
    score_min: float = 0.0,
    score_max: float = 10.0,
) -> pd.DataFrame:
    """Validate constraints the official evaluator otherwise handles permissively.

    The official script inner-joins keys, deduplicates rows, and renormalizes weights
    over non-null dimensions. E0 rejects those cases so methods are compared on the
    same documents and all four required dimensions.
    """
    pred = normalize_predictions(predictions)
    truth = normalize_labels(labels) if any(c.endswith("_score") for c in labels.columns) else labels.copy()
    truth = truth[[*KEY_COLUMNS, *DIMS]].copy()

    duplicate_count = int(pred.duplicated(KEY_COLUMNS, keep=False).sum())
    if duplicate_count:
        raise PredictionValidationError(f"duplicate prediction keys: {duplicate_count} rows")
    if pred["answerId"].eq("").any():
        raise PredictionValidationError("prediction contains empty answerId")
    values = pred[DIMS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PredictionValidationError("prediction contains missing, NaN, or infinite dimension scores")
    if ((values < score_min) | (values > score_max)).any():
        bad = int(((values < score_min) | (values > score_max)).sum())
        raise PredictionValidationError(
            f"prediction contains {bad} scores outside [{score_min}, {score_max}]"
        )

    pred_keys = set(map(tuple, pred[KEY_COLUMNS].itertuples(index=False, name=None)))
    truth_keys = set(map(tuple, truth[KEY_COLUMNS].itertuples(index=False, name=None)))
    extra = pred_keys - truth_keys
    missing = truth_keys - pred_keys
    if extra:
        raise PredictionValidationError(f"prediction contains {len(extra)} unknown keys")
    if require_complete and missing:
        by_question: dict[int, int] = {}
        for question_id, _ in missing:
            by_question[int(question_id)] = by_question.get(int(question_id), 0) + 1
        raise PredictionValidationError(
            f"prediction is missing {len(missing)} label keys; by question={by_question}"
        )
    return pred.sort_values(KEY_COLUMNS).reset_index(drop=True)


def load_dimension_weights(rubric_dir: Path, question_ids: list[int]) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for question_id in question_ids:
        candidates = [
            rubric_dir / f"criterion{question_id}.json",
            rubric_dir / f"criteria{question_id}.json",
        ]
        rubric_path = next((path for path in candidates if path.exists()), None)
        if rubric_path is None:
            raise FileNotFoundError(f"rubric not found for question {question_id}: {candidates}")
        payload = json.loads(rubric_path.read_text(encoding="utf-8"))
        raw = payload.get("dimension_weight", {}) or {}
        weights = {dim: float(raw.get(dim, 0.0)) for dim in DIMS}
        if sum(weights.values()) <= 0:
            weights = {dim: 1.0 for dim in DIMS}
        result[int(question_id)] = weights
    return result


def weighted_total(values: np.ndarray, weights: Mapping[str, float]) -> np.ndarray:
    vector = np.asarray([float(weights.get(dim, 0.0)) for dim in DIMS], dtype=float)
    if vector.sum() <= 0:
        vector = np.ones(len(DIMS), dtype=float)
    vector /= vector.sum()
    return np.asarray(values, dtype=float) @ vector


def pairwise_accuracy(gt_scores: np.ndarray, pred_scores: np.ndarray) -> tuple[float, int, int]:
    """Exact AEOLLM-2 official accuracy, including exact handling of ties."""
    correct = 0
    total = 0
    for i, j in itertools.combinations(range(len(gt_scores)), 2):
        gold_preference = np.sign(gt_scores[i] - gt_scores[j])
        predicted_preference = np.sign(pred_scores[i] - pred_scores[j])
        if np.isnan(gold_preference) or np.isnan(predicted_preference):
            continue
        total += 1
        if gold_preference == predicted_preference:
            correct += 1
    return (float(correct / total) if total else np.nan, correct, total)


def _safe_rank_metrics(gold: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float, int, int]:
    if len(gold) < 3:
        return np.nan, np.nan, np.nan, 0, 0
    if np.ptp(gold) == 0 or np.ptp(prediction) == 0:
        spearman = np.nan
        kendall = np.nan
    else:
        spearman = float(spearmanr(gold, prediction)[0])
        kendall = float(kendalltau(gold, prediction)[0])
    accuracy, correct, total = pairwise_accuracy(gold, prediction)
    return spearman, kendall, accuracy, correct, total


def evaluate_predictions(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    dimension_weights: Mapping[int, Mapping[str, float]],
    *,
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate predictions with the official per-question aggregation semantics."""
    truth = normalize_labels(labels) if any(c.endswith("_score") for c in labels.columns) else labels.copy()
    truth = truth[[*KEY_COLUMNS, *DIMS]]
    pred = validate_predictions(predictions, truth, require_complete=True) if strict else normalize_predictions(predictions)
    merged = pred.merge(truth, on=KEY_COLUMNS, how="inner", suffixes=("_pred", "_gt"))

    detail_rows: list[dict[str, float | int]] = []
    all_correct = 0
    all_pairs = 0
    for question_id in sorted(truth["questionId"].unique()):
        sub = merged[merged["questionId"] == question_id]
        weights = dimension_weights[int(question_id)]
        pred_matrix = sub[[f"{dim}_pred" for dim in DIMS]].to_numpy(dtype=float)
        gold_matrix = sub[[f"{dim}_gt" for dim in DIMS]].to_numpy(dtype=float)
        pred_total = weighted_total(pred_matrix, weights)
        gold_total = weighted_total(gold_matrix, weights)
        sp, kt, acc, correct, pairs = _safe_rank_metrics(gold_total, pred_total)
        row: dict[str, float | int] = {
            "reportId": int(question_id),
            "n_documents": int(len(sub)),
            "spearman": sp,
            "kendall": kt,
            "accuracy": acc,
            "pair_correct": correct,
            "pair_total": pairs,
        }
        all_correct += correct
        all_pairs += pairs
        for index, dim in enumerate(DIMS):
            dim_sp, dim_kt, dim_acc, _, _ = _safe_rank_metrics(gold_matrix[:, index], pred_matrix[:, index])
            row[f"spearman_{dim}"] = dim_sp
            row[f"kendall_{dim}"] = dim_kt
            row[f"accuracy_{dim}"] = dim_acc
        detail_rows.append(row)

    details = pd.DataFrame(detail_rows).sort_values("reportId").reset_index(drop=True)
    summary = {
        "spearman": float(details["spearman"].mean()),
        "kendall": float(details["kendall"].mean()),
        "accuracy": float(all_correct / all_pairs) if all_pairs else np.nan,
        "n_questions": int(len(details)),
        "n_documents": int(len(merged)),
        "pair_total": int(all_pairs),
    }
    for dim in DIMS:
        for metric in ("spearman", "kendall", "accuracy"):
            summary[f"{metric}_{dim}"] = float(details[f"{metric}_{dim}"].mean())
    return pd.DataFrame([summary]), details


def to_submission_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    pred = normalize_predictions(predictions)
    out = pd.DataFrame({"taskId": 0, "questionId": pred["questionId"], "answerId": pred["answerId"]})
    for dim in DIMS:
        out[SCORE_COLUMNS[dim]] = pred[dim]
    return out
