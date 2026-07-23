from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml

from aeollm_e0.data import sha256_file
from aeollm_e0.metrics import (
    DIMS,
    KEY_COLUMNS,
    evaluate_predictions,
    load_dimension_weights,
    normalize_labels,
    normalize_predictions,
    to_submission_frame,
    validate_predictions,
    weighted_total,
)
from aeollm_e0.statistics import paired_question_bootstrap, question_bootstrap

from .pairwise_training import (
    MLPTrainingConfig,
    config_dict,
    fit_fold_preprocessor,
    train_fold_ensemble,
)
from .ridge_scoring import load_feature_groups, numeric_surface_columns


@dataclass(frozen=True)
class E15Config:
    labels: Path
    rubric_dir: Path
    surface_features: Path
    unbounded_features: Path
    ridge_reference: Path
    output_dir: Path
    seeds: tuple[int, ...] = (20260721, 20260722, 20260723)
    bootstrap_resamples: int = 5000
    bootstrap_seed: int = 20260721


def _merge_features(e1: pd.DataFrame, surface: pd.DataFrame) -> pd.DataFrame:
    merged = e1.merge(surface, on=KEY_COLUMNS, validate="one_to_one", suffixes=("", "_surface"))
    if len(merged) != len(e1) or len(merged) != len(surface):
        raise ValueError("E1 and surface feature keys do not match")
    return merged.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _feature_sets(groups: dict[str, list[str]], surface_columns: list[str]) -> dict[str, list[str]]:
    rubric = list(groups["rubric_primary"])
    return {
        "structure": list(surface_columns),
        "rubric": rubric,
        "all": [*groups["global"], *rubric, *surface_columns],
    }


def _mlp_loqo_predictions(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    columns: list[str],
    *,
    training: MLPTrainingConfig,
    seeds: tuple[int, ...],
    use_pairwise: bool,
    preprocessing_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.merge(labels, on=KEY_COLUMNS, validate="one_to_one")
    if len(data) != len(labels) or len(data) != len(frame):
        raise ValueError("feature and label keys do not match exactly")
    result = data[KEY_COLUMNS].copy()
    result[DIMS] = np.nan
    diagnostics: list[dict[str, object]] = []
    for held_out in sorted(data["questionId"].unique()):
        train_mask = data["questionId"] != held_out
        test_mask = ~train_mask
        raw_train = data.loc[train_mask, columns].to_numpy(dtype=np.float64)
        raw_test = data.loc[test_mask, columns].to_numpy(dtype=np.float64)
        preprocessor = fit_fold_preprocessor(raw_train, training, seed=preprocessing_seed)
        x_train = preprocessor.transform(raw_train)
        x_test = preprocessor.transform(raw_test)
        prediction, fold_diagnostics = train_fold_ensemble(
            x_train,
            x_test,
            data.loc[train_mask, DIMS].to_numpy(dtype=np.float32),
            data.loc[train_mask, "questionId"].to_numpy(dtype=int),
            config=training,
            seeds=seeds,
            use_pairwise=use_pairwise,
        )
        result.loc[test_mask, DIMS] = np.clip(prediction, 0.0, 10.0)
        for row in fold_diagnostics:
            diagnostics.append(
                {
                    "held_out_question": int(held_out),
                    "pca_components": int(x_train.shape[1]),
                    **row,
                }
            )
    return result, pd.DataFrame(diagnostics)


def _errors(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    weights: dict[int, dict[str, float]],
) -> dict[str, float]:
    merged = predictions.merge(labels, on=KEY_COLUMNS, suffixes=("_pred", "_gt"))
    result: dict[str, float] = {}
    for dimension in DIMS:
        error = merged[f"{dimension}_pred"] - merged[f"{dimension}_gt"]
        result[f"mae_{dimension}"] = float(np.mean(np.abs(error)))
        result[f"rmse_{dimension}"] = float(np.sqrt(np.mean(error**2)))
    pred_total: list[float] = []
    gold_total: list[float] = []
    for question_id, group in merged.groupby("questionId", sort=True):
        pred_total.extend(
            weighted_total(
                group[[f"{dimension}_pred" for dimension in DIMS]].to_numpy(),
                weights[int(question_id)],
            )
        )
        gold_total.extend(
            weighted_total(
                group[[f"{dimension}_gt" for dimension in DIMS]].to_numpy(),
                weights[int(question_id)],
            )
        )
    error = np.asarray(pred_total) - np.asarray(gold_total)
    result["mae"] = float(np.mean(np.abs(error)))
    result["rmse"] = float(np.sqrt(np.mean(error**2)))
    return result


def _evaluate(
    name: str,
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    weights: dict[int, dict[str, float]],
    prediction_dir: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    clean = validate_predictions(predictions, labels)
    summary, details = evaluate_predictions(clean, labels, weights)
    record: dict[str, object] = {"model": name, **summary.iloc[0].to_dict()}
    record.update(_errors(clean, labels, weights))
    to_submission_frame(clean).to_csv(
        prediction_dir / f"{name}.tsv", sep="\t", index=False, float_format="%.8f"
    )
    details.insert(0, "model", name)
    return record, details


def _write_report(path: Path, metrics: pd.DataFrame, paired: pd.DataFrame) -> None:
    ranked = metrics.sort_values(["accuracy", "spearman"], ascending=False)
    lines = [
        "# E1.5 Huber and within-question pairwise results",
        "",
        "All MLP predictions are outer leave-one-question-out and averaged over three fixed seeds.",
        "Training epochs, architecture, pair weight, and preprocessing are fixed before evaluation;",
        "held-out question labels are never used for early stopping or model selection.",
        "",
        "## Main results",
        "",
        ranked[["model", "accuracy", "spearman", "kendall", "mae", "rmse"]].to_markdown(
            index=False, floatfmt=".4f"
        ),
        "",
        "## Pairwise-loss checks",
        "",
    ]
    indexed = metrics.set_index("model")
    for feature_set in ("structure", "rubric", "all"):
        pair = indexed.loc[f"mlp_{feature_set}_huber_pair"]
        reg = indexed.loc[f"mlp_{feature_set}_huber"]
        lines.append(
            f"- {feature_set}: pairwise minus Huber accuracy = "
            f"{float(pair['accuracy']) - float(reg['accuracy']):+.4f}; "
            f"Spearman = {float(pair['spearman']) - float(reg['spearman']):+.4f}."
        )
    lines.extend(
        [
            "",
            "## Paired question bootstrap",
            "",
            paired.to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Interpretation rules",
            "",
            "- Pairwise improvements must be judged against the same architecture and features.",
            "- A similar gain for structure and rubric indicates a generic ranking-objective effect.",
            "- Huber degradation with ranking improvement is an expected calibration/ranking tradeoff.",
            "- With 10 questions, paired question-bootstrap intervals remain the uncertainty unit.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_e1_5(config: E15Config, training: MLPTrainingConfig) -> pd.DataFrame:
    started = time.perf_counter()
    output = config.output_dir
    prediction_dir = output / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    labels = normalize_labels(pd.read_csv(config.labels))
    weights = load_dimension_weights(config.rubric_dir, sorted(labels["questionId"].unique()))
    e1, groups = load_feature_groups(config.unbounded_features)
    surface = pd.read_csv(config.surface_features)
    surface_columns = numeric_surface_columns(surface)
    frame = _merge_features(e1, surface)
    feature_sets = _feature_sets(groups, surface_columns)

    models: dict[str, pd.DataFrame] = {}
    diagnostic_frames: list[pd.DataFrame] = []
    for feature_name, columns in feature_sets.items():
        for use_pairwise, loss_name in ((False, "huber"), (True, "huber_pair")):
            name = f"mlp_{feature_name}_{loss_name}"
            prediction, diagnostics = _mlp_loqo_predictions(
                frame,
                labels,
                columns,
                training=training,
                seeds=config.seeds,
                use_pairwise=use_pairwise,
                preprocessing_seed=config.bootstrap_seed,
            )
            models[name] = prediction
            diagnostics.insert(0, "model", name)
            diagnostics.insert(1, "raw_feature_count", len(columns))
            diagnostic_frames.append(diagnostics)
    if config.ridge_reference.exists():
        models["ridge_all_unbounded"] = normalize_predictions(
            pd.read_csv(config.ridge_reference, sep="\t")
        )

    metric_records: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    details_by_model: dict[str, pd.DataFrame] = {}
    for name, predictions in models.items():
        record, details = _evaluate(name, predictions, labels, weights, prediction_dir)
        metric_records.append(record)
        detail_frames.append(details)
        details_by_model[name] = details.drop(columns="model")
    metrics = pd.DataFrame(metric_records).sort_values("model").reset_index(drop=True)
    metrics.to_csv(output / "model_metrics.csv", index=False, float_format="%.8f")
    pd.concat(detail_frames, ignore_index=True).to_csv(
        output / "per_question_metrics.csv", index=False, float_format="%.8f"
    )
    pd.concat(diagnostic_frames, ignore_index=True).to_csv(
        output / "training_diagnostics.csv", index=False, float_format="%.8f"
    )
    bootstrap = question_bootstrap(
        details_by_model,
        n_resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
    )
    bootstrap.to_csv(output / "bootstrap_ci.csv", index=False, float_format="%.8f")
    comparisons = [
        (f"mlp_{feature_set}_huber_pair", f"mlp_{feature_set}_huber")
        for feature_set in feature_sets
    ]
    if "ridge_all_unbounded" in models:
        comparisons.extend(
            [
                ("mlp_all_huber", "ridge_all_unbounded"),
                ("mlp_all_huber_pair", "ridge_all_unbounded"),
            ]
        )
    paired = paired_question_bootstrap(
        details_by_model,
        comparisons,
        n_resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
    )
    paired.to_csv(output / "paired_bootstrap.csv", index=False, float_format="%.8f")
    _write_report(output / "e1_5_conclusions.md", metrics, paired)
    elapsed = time.perf_counter() - started
    protocol = {
        "name": "AEOLLM-2 E1.5 fixed MLP Huber/pairwise comparison",
        "outer_split": "Leave-One-Question-Out",
        "model_selection": "none; all hyperparameters and epochs fixed",
        "primary_metric": "official weighted-total pairwise accuracy",
        "secondary_metrics": ["Spearman", "Kendall"],
        "pair_scope": "within question and dimension only",
        "seed_ensemble": list(config.seeds),
        "training": config_dict(training),
        "feature_sets": {name: len(columns) for name, columns in feature_sets.items()},
        "gpu_required": False,
        "paths": {key: str(value.resolve()) for key, value in asdict(config).items() if isinstance(value, Path)},
        "input_hashes": {
            "labels": sha256_file(config.labels),
            "surface_features": sha256_file(config.surface_features),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": str(torch.__version__),
        },
    }
    (output / "protocol.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    status = {
        "status": "complete",
        "models": len(models),
        "documents": len(labels),
        "questions": int(labels["questionId"].nunique()),
        "elapsed_seconds": elapsed,
        "gpu_used": False,
    }
    (output / "run_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run E1.5 fixed MLP Huber/pairwise ablations")
    parser.add_argument(
        "--labels",
        type=Path,
        default=root / "data/official/hf-aeollm/aeollm-2-train/train_deepresearch.csv",
    )
    parser.add_argument(
        "--rubric-dir",
        type=Path,
        default=root / "data/official/hf-aeollm/aeollm-2-train/rubric_dataset",
    )
    parser.add_argument("--surface-features", type=Path, default=root / "outputs/e0/surface_features.csv")
    parser.add_argument(
        "--unbounded-features",
        type=Path,
        default=root / "outputs/e1/features/qwen3-0.6b-unbounded",
    )
    parser.add_argument(
        "--ridge-reference",
        type=Path,
        default=root / "outputs/e1/e1_4/predictions/all_unbounded.tsv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "outputs/e1/e1_5_accuracy"
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = E15Config(
        labels=args.labels,
        rubric_dir=args.rubric_dir,
        surface_features=args.surface_features,
        unbounded_features=args.unbounded_features,
        ridge_reference=args.ridge_reference,
        output_dir=args.output_dir,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    training = MLPTrainingConfig(epochs=args.epochs)
    metrics = run_e1_5(config, training)
    print(metrics[["model", "accuracy", "spearman", "kendall", "mae"]].to_string(index=False))
    return 0
