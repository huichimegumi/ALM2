from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aeollm_e0.metrics import DIMS, KEY_COLUMNS

RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
SURFACE_EXCLUDED = {
    *KEY_COLUMNS,
    "model_name",
    "model_id",
    "prompt_variant",
    "prompt_position",
}


def numeric_surface_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in SURFACE_EXCLUDED and pd.api.types.is_numeric_dtype(frame[column])
    ]


def load_feature_groups(feature_dir: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    frame = pd.read_csv(feature_dir / "document_features.csv")
    manifest = json.loads((feature_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete E1.3 feature manifest: {feature_dir}")
    groups = {name: list(columns) for name, columns in manifest["feature_groups"].items()}
    referenced = {column for columns in groups.values() for column in columns}
    missing = sorted(referenced - set(frame.columns))
    if missing:
        raise ValueError(f"manifest references missing feature columns: {missing[:10]}")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"duplicate feature keys in {feature_dir}")
    return frame, groups


def own_dimension_rubric_columns(groups: Mapping[str, Sequence[str]], dimension: str) -> list[str]:
    prefix = f"rubric_{dimension}_"
    columns = [column for column in groups["rubric_primary"] if column.startswith(prefix)]
    if not columns:
        raise ValueError(f"no primary rubric features for {dimension}")
    return columns


def build_model_feature_columns(
    e1_groups: Mapping[str, Sequence[str]], surface_columns: Sequence[str]
) -> dict[str, dict[str, list[str]]]:
    global_columns = list(e1_groups["global"])
    surface = list(surface_columns)
    result: dict[str, dict[str, list[str]]] = {}
    for dimension in DIMS:
        rubric = own_dimension_rubric_columns(e1_groups, dimension)
        result[dimension] = {
            "global": global_columns,
            "rubric": rubric,
            "structure": surface,
            "global_structure": [*global_columns, *surface],
            "global_rubric": [*global_columns, *rubric],
            "rubric_structure": [*rubric, *surface],
            "all": [*global_columns, *rubric, *surface],
        }
    return result


def _ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, solver="lsqr", tol=1e-6)),
        ]
    )


def _select_alpha(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: Sequence[float],
) -> tuple[float, float]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("inner grouped validation needs at least two questions")
    splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
    best_alpha = float(alphas[0])
    best_mae = float("inf")
    for alpha in alphas:
        fold_mae: list[float] = []
        for train_index, validation_index in splitter.split(x, y, groups):
            model = _ridge_pipeline(float(alpha)).fit(x.iloc[train_index], y[train_index])
            prediction = np.asarray(model.predict(x.iloc[validation_index]), dtype=float)
            fold_mae.append(float(np.mean(np.abs(prediction - y[validation_index]))))
        mean_mae = float(np.mean(fold_mae))
        if mean_mae < best_mae:
            best_alpha = float(alpha)
            best_mae = mean_mae
    return best_alpha, best_mae


def fit_grouped_ridge_fold(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    groups: np.ndarray,
    *,
    alphas: Sequence[float] = RIDGE_ALPHAS,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit one leakage-safe outer-fold Ridge model and return train/test predictions."""
    if not alphas or any(float(alpha) <= 0 for alpha in alphas):
        raise ValueError("all Ridge alphas must be positive")
    y_train = np.asarray(y_train, dtype=float)
    groups = np.asarray(groups)
    if len(x_train) != len(y_train) or len(groups) != len(y_train):
        raise ValueError("training features, targets, and groups must have equal length")
    alpha, inner_mae = _select_alpha(x_train, y_train, groups, alphas)
    model = _ridge_pipeline(alpha).fit(x_train, y_train)
    train_prediction = np.clip(np.asarray(model.predict(x_train), dtype=float), 0.0, 10.0)
    test_prediction = np.clip(np.asarray(model.predict(x_test), dtype=float), 0.0, 10.0)
    return train_prediction, test_prediction, alpha, inner_mae


def nested_loqo_ridge_predictions(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    feature_columns_by_dimension: Mapping[str, Sequence[str]],
    *,
    alphas: Sequence[float] = RIDGE_ALPHAS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not alphas or any(float(alpha) <= 0 for alpha in alphas):
        raise ValueError("all Ridge alphas must be positive")
    required_columns = {
        column for dimension in DIMS for column in feature_columns_by_dimension[dimension]
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"missing model feature columns: {missing[:10]}")
    data = frame.merge(labels[[*KEY_COLUMNS, *DIMS]], on=KEY_COLUMNS, validate="one_to_one")
    if len(data) != len(labels) or len(data) != len(frame):
        raise ValueError("feature and label keys do not match exactly")
    data = data.sort_values(KEY_COLUMNS).reset_index(drop=True)
    result = data[KEY_COLUMNS].copy()
    for dimension in DIMS:
        result[dimension] = np.nan
    selections: list[dict[str, object]] = []

    for held_out in sorted(data["questionId"].unique()):
        train_mask = data["questionId"] != held_out
        test_mask = ~train_mask
        groups = data.loc[train_mask, "questionId"].to_numpy(dtype=int)
        for dimension in DIMS:
            columns = list(feature_columns_by_dimension[dimension])
            x_train = data.loc[train_mask, columns]
            x_test = data.loc[test_mask, columns]
            y_train = data.loc[train_mask, dimension].to_numpy(dtype=float)
            alpha, inner_mae = _select_alpha(x_train, y_train, groups, alphas)
            model = _ridge_pipeline(alpha).fit(x_train, y_train)
            result.loc[test_mask, dimension] = np.clip(model.predict(x_test), 0.0, 10.0)
            selections.append(
                {
                    "held_out_question": int(held_out),
                    "dimension": dimension,
                    "alpha": alpha,
                    "inner_mae": inner_mae,
                    "feature_count": len(columns),
                    "train_documents": int(train_mask.sum()),
                    "test_documents": int(test_mask.sum()),
                }
            )
    if not np.isfinite(result[DIMS].to_numpy(dtype=float)).all():
        raise ValueError("nested LOQO produced non-finite predictions")
    return result, pd.DataFrame(selections)
