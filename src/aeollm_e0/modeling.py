from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .metrics import DIMS, KEY_COLUMNS, pairwise_accuracy

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
METADATA_COLUMNS = ["model_id", "prompt_variant"]


def _numeric_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def _mixed_pipeline(numeric: list[str], categorical: list[str], alpha: float) -> Pipeline:
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            ),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline([("features", transformer), ("ridge", Ridge(alpha=alpha))])


def _select_alpha(
    builder: Callable[[float], Pipeline],
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> float:
    unique_groups = np.unique(groups)
    splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
    candidates: list[tuple[float, float, float, int, float]] = []
    for alpha in ALPHAS:
        prediction = np.full_like(y, np.nan, dtype=float)
        for train_index, validation_index in splitter.split(x, y, groups):
            model = builder(alpha)
            model.fit(x.iloc[train_index], y[train_index])
            prediction[validation_index] = np.asarray(
                model.predict(x.iloc[validation_index]), dtype=float
            )
        target_matrix = y[:, None] if y.ndim == 1 else y
        prediction_matrix = prediction[:, None] if prediction.ndim == 1 else prediction
        correct = 0
        total = 0
        correlations: list[float] = []
        for question_id in unique_groups:
            mask = groups == question_id
            for column in range(target_matrix.shape[1]):
                _, group_correct, group_total = pairwise_accuracy(
                    target_matrix[mask, column],
                    prediction_matrix[mask, column],
                )
                correct += group_correct
                total += group_total
                if (
                    np.ptp(target_matrix[mask, column]) > 0
                    and np.ptp(prediction_matrix[mask, column]) > 0
                ):
                    correlations.append(
                        float(
                            spearmanr(
                                target_matrix[mask, column],
                                prediction_matrix[mask, column],
                            )[0]
                        )
                    )
        accuracy = float(correct / total) if total else float("-inf")
        spearman = float(np.mean(correlations)) if correlations else float("-inf")
        mae = float(np.mean(np.abs(prediction_matrix - target_matrix)))
        candidates.append((accuracy, spearman, -mae, -len(candidates), float(alpha)))
    return max(candidates)[-1]


def loqo_ridge_predictions(
    frame: pd.DataFrame,
    target: pd.DataFrame,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.merge(target[[*KEY_COLUMNS, *DIMS]], on=KEY_COLUMNS, validate="one_to_one")
    categorical_columns = categorical_columns or []
    feature_columns = [*numeric_columns, *categorical_columns]
    result = data[KEY_COLUMNS].copy()
    for dim in DIMS:
        result[dim] = np.nan
    selections: list[dict[str, object]] = []

    for held_out in sorted(data["questionId"].unique()):
        train_mask = data["questionId"] != held_out
        test_mask = ~train_mask
        x_train = data.loc[train_mask, feature_columns]
        x_test = data.loc[test_mask, feature_columns]
        groups = data.loc[train_mask, "questionId"].to_numpy()
        for dim in DIMS:
            y_train = data.loc[train_mask, dim].to_numpy(dtype=float)
            if categorical_columns:
                builder = lambda alpha: _mixed_pipeline(numeric_columns, categorical_columns, alpha)
            else:
                builder = _numeric_pipeline
            alpha = _select_alpha(builder, x_train, y_train, groups)
            model = builder(alpha)
            model.fit(x_train, y_train)
            result.loc[test_mask, dim] = np.clip(model.predict(x_test), 0.0, 10.0)
            selections.append({"held_out_question": int(held_out), "dimension": dim, "alpha": alpha})
    return result, pd.DataFrame(selections)


def mean_loqo_predictions(labels: pd.DataFrame) -> pd.DataFrame:
    result = labels[KEY_COLUMNS].copy()
    for dim in DIMS:
        result[dim] = np.nan
    for held_out in sorted(labels["questionId"].unique()):
        train_mask = labels["questionId"] != held_out
        test_mask = ~train_mask
        for dim in DIMS:
            result.loc[test_mask, dim] = float(labels.loc[train_mask, dim].mean())
    return result


def random_kfold_ridge_predictions(
    frame: pd.DataFrame,
    target: pd.DataFrame,
    numeric_columns: list[str],
    *,
    random_state: int = 20260721,
) -> pd.DataFrame:
    data = frame.merge(target[[*KEY_COLUMNS, *DIMS]], on=KEY_COLUMNS, validate="one_to_one")
    result = data[KEY_COLUMNS].copy()
    for dim in DIMS:
        result[dim] = np.nan
    splitter = KFold(n_splits=10, shuffle=True, random_state=random_state)
    x = data[numeric_columns]
    for train_index, test_index in splitter.split(x):
        for dim in DIMS:
            model = _numeric_pipeline(alpha=10.0)
            model.fit(x.iloc[train_index], data.iloc[train_index][dim].to_numpy(dtype=float))
            result.loc[test_index, dim] = np.clip(model.predict(x.iloc[test_index]), 0.0, 10.0)
    return result


def loqo_calibration_predictions(
    judge_predictions: pd.DataFrame,
    labels: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    data = judge_predictions.merge(labels[[*KEY_COLUMNS, *DIMS]], on=KEY_COLUMNS, suffixes=("_judge", "_gold"))
    result = data[KEY_COLUMNS].copy()
    for dim in DIMS:
        result[dim] = np.nan
    for held_out in sorted(data["questionId"].unique()):
        train_mask = data["questionId"] != held_out
        test_mask = ~train_mask
        if method == "multioutput_ridge":
            x_train = data.loc[train_mask, [f"{dim}_judge" for dim in DIMS]].to_numpy(dtype=float)
            y_train = data.loc[train_mask, [f"{dim}_gold" for dim in DIMS]].to_numpy(dtype=float)
            groups = data.loc[train_mask, "questionId"].to_numpy()
            x_frame = pd.DataFrame(x_train, columns=DIMS)
            builder = _numeric_pipeline
            alpha = _select_alpha(builder, x_frame, y_train, groups)
            model = builder(alpha).fit(x_frame, y_train)
            prediction = model.predict(
                pd.DataFrame(
                    data.loc[test_mask, [f"{dim}_judge" for dim in DIMS]].to_numpy(dtype=float),
                    columns=DIMS,
                )
            )
            result.loc[test_mask, DIMS] = np.clip(prediction, 0.0, 10.0)
            continue
        for dim in DIMS:
            x_train = data.loc[train_mask, f"{dim}_judge"].to_numpy(dtype=float)
            y_train = data.loc[train_mask, f"{dim}_gold"].to_numpy(dtype=float)
            x_test = data.loc[test_mask, f"{dim}_judge"].to_numpy(dtype=float)
            if method == "affine":
                model = Ridge(alpha=1.0).fit(x_train.reshape(-1, 1), y_train)
                prediction = model.predict(x_test.reshape(-1, 1))
            elif method == "isotonic":
                model = IsotonicRegression(out_of_bounds="clip").fit(x_train, y_train)
                prediction = model.predict(x_test)
            else:
                raise ValueError(f"unknown calibration method: {method}")
            result.loc[test_mask, dim] = np.clip(prediction, 0.0, 10.0)
    return result
