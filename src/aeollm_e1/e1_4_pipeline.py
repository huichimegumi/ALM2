from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml

from aeollm_e0.data import sha256_file
from aeollm_e0.metrics import (
    DIMS,
    KEY_COLUMNS,
    evaluate_predictions,
    load_dimension_weights,
    normalize_labels,
    to_submission_frame,
    validate_predictions,
    weighted_total,
)
from aeollm_e0.modeling import mean_loqo_predictions
from aeollm_e0.statistics import paired_question_bootstrap, question_bootstrap

from .ridge_scoring import (
    RIDGE_ALPHAS,
    build_model_feature_columns,
    load_feature_groups,
    nested_loqo_ridge_predictions,
    numeric_surface_columns,
)


@dataclass(frozen=True)
class E14Config:
    labels: Path
    rubric_dir: Path
    surface_features: Path
    capped_features: Path
    unbounded_features: Path
    output_dir: Path
    bootstrap_resamples: int = 5000
    seed: int = 20260721


def _diagnostic_errors(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    weights: dict[int, dict[str, float]],
) -> dict[str, float]:
    merged = predictions.merge(labels, on=KEY_COLUMNS, suffixes=("_pred", "_gt"))
    result: dict[str, float] = {}
    for dimension in DIMS:
        error = merged[f"{dimension}_pred"].to_numpy() - merged[f"{dimension}_gt"].to_numpy()
        result[f"mae_{dimension}"] = float(np.mean(np.abs(error)))
        result[f"rmse_{dimension}"] = float(np.sqrt(np.mean(error**2)))
    predicted_totals: list[float] = []
    gold_totals: list[float] = []
    for question_id, group in merged.groupby("questionId", sort=True):
        predicted_totals.extend(
            weighted_total(
                group[[f"{dimension}_pred" for dimension in DIMS]].to_numpy(),
                weights[int(question_id)],
            )
        )
        gold_totals.extend(
            weighted_total(
                group[[f"{dimension}_gt" for dimension in DIMS]].to_numpy(),
                weights[int(question_id)],
            )
        )
    error = np.asarray(predicted_totals) - np.asarray(gold_totals)
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
    record.update(_diagnostic_errors(clean, labels, weights))
    to_submission_frame(clean).to_csv(
        prediction_dir / f"{name}.tsv",
        sep="\t",
        index=False,
        encoding="utf-8",
        float_format="%.8f",
    )
    details.insert(0, "model", name)
    return record, details


def _merge_feature_sources(e1: pd.DataFrame, surface: pd.DataFrame) -> pd.DataFrame:
    merged = e1.merge(surface, on=KEY_COLUMNS, validate="one_to_one", suffixes=("", "_surface"))
    if len(merged) != len(e1) or len(merged) != len(surface):
        raise ValueError("E1 and E0 surface feature keys do not match")
    return merged


def _write_report(path: Path, metrics: pd.DataFrame, paired: pd.DataFrame) -> None:
    ranked = metrics.sort_values("spearman", ascending=False)
    indexed = metrics.set_index("model")
    lines = [
        "# E1.4 nested LOQO Ridge results",
        "",
        "All predictions are outer leave-one-question-out. Ridge alpha is selected",
        "inside each outer training split using question-grouped cross-validation and MAE.",
        "E1.5 ranking losses are not used here.",
        "",
        "## Main results",
        "",
        ranked[["model", "spearman", "kendall", "accuracy", "mae", "rmse"]].to_markdown(
            index=False, floatfmt=".4f"
        ),
        "",
        "## Primary checks",
        "",
    ]
    for representation in ("capped", "unbounded"):
        rubric = indexed.loc[f"rubric_{representation}"]
        glob = indexed.loc[f"global_{representation}"]
        all_model = indexed.loc[f"all_{representation}"]
        lines.extend(
            [
                f"- {representation}: rubric minus global Spearman = "
                f"{float(rubric['spearman']) - float(glob['spearman']):+.4f}.",
                f"- {representation}: all minus rubric Spearman = "
                f"{float(all_model['spearman']) - float(rubric['spearman']):+.4f}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Paired question bootstrap",
            "",
            paired.to_markdown(index=False, floatfmt=".4f") if len(paired) else "No comparisons.",
            "",
            "## Interpretation",
            "",
            "- `rubric_*` uses only the target dimension's primary criterion-conditioned features.",
            "- `structure` reuses E0 surface features and excludes generator/prompt metadata.",
            "- `all_*` combines global embeddings, target-dimension rubric features, and structure.",
            "- Fixed top-3/top-5 and chunk-count diagnostic features are excluded.",
            "- Prefer question-level paired uncertainty over pair-level significance claims.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_e1_4(config: E14Config) -> pd.DataFrame:
    output = config.output_dir
    prediction_dir = output / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    labels = normalize_labels(pd.read_csv(config.labels))
    question_ids = sorted(labels["questionId"].unique().tolist())
    weights = load_dimension_weights(config.rubric_dir, question_ids)
    surface = pd.read_csv(config.surface_features)
    surface_columns = numeric_surface_columns(surface)
    if not surface_columns:
        raise ValueError("no numeric E0 surface features")

    representation_data: dict[str, tuple[pd.DataFrame, dict[str, dict[str, list[str]]]]] = {}
    feature_protocol: dict[str, object] = {"surface": surface_columns, "representations": {}}
    for representation, directory in (
        ("capped", config.capped_features),
        ("unbounded", config.unbounded_features),
    ):
        e1, groups = load_feature_groups(directory)
        merged = _merge_feature_sources(e1, surface)
        columns = build_model_feature_columns(groups, surface_columns)
        representation_data[representation] = (merged, columns)
        feature_protocol["representations"][representation] = {
            "feature_dir": str(directory.resolve()),
            "feature_counts_by_dimension": {
                dimension: {name: len(values) for name, values in models.items()}
                for dimension, models in columns.items()
            },
        }

    models: dict[str, pd.DataFrame] = {"mean_loqo": mean_loqo_predictions(labels)}
    selections: list[pd.DataFrame] = []
    structure_frame, structure_columns_by_dimension = representation_data["capped"]
    structure_map = {
        dimension: structure_columns_by_dimension[dimension]["structure"] for dimension in DIMS
    }
    prediction, selected = nested_loqo_ridge_predictions(structure_frame, labels, structure_map)
    models["structure"] = prediction
    selected.insert(0, "model", "structure")
    selections.append(selected)

    for representation, (frame, columns_by_dimension) in representation_data.items():
        for feature_set in ("global", "rubric", "rubric_structure", "all"):
            name = f"{feature_set}_{representation}"
            mapping = {
                dimension: columns_by_dimension[dimension][feature_set] for dimension in DIMS
            }
            prediction, selected = nested_loqo_ridge_predictions(frame, labels, mapping)
            models[name] = prediction
            selected.insert(0, "model", name)
            selections.append(selected)

    pd.concat(selections, ignore_index=True).to_csv(
        output / "selected_hyperparameters.csv", index=False, float_format="%.8f"
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
    bootstrap = question_bootstrap(
        details_by_model, n_resamples=config.bootstrap_resamples, seed=config.seed
    )
    bootstrap.to_csv(output / "bootstrap_ci.csv", index=False, float_format="%.8f")
    comparisons: list[tuple[str, str]] = []
    for representation in ("capped", "unbounded"):
        comparisons.extend(
            [
                (f"rubric_{representation}", f"global_{representation}"),
                (f"rubric_{representation}", "structure"),
                (f"rubric_structure_{representation}", "structure"),
                (f"all_{representation}", f"rubric_{representation}"),
                (f"all_{representation}", "structure"),
            ]
        )
    comparisons.extend(
        [
            ("global_unbounded", "global_capped"),
            ("rubric_unbounded", "rubric_capped"),
            ("all_unbounded", "all_capped"),
        ]
    )
    paired = paired_question_bootstrap(
        details_by_model,
        comparisons,
        n_resamples=config.bootstrap_resamples,
        seed=config.seed,
    )
    paired.to_csv(output / "paired_bootstrap.csv", index=False, float_format="%.8f")
    _write_report(output / "e1_4_conclusions.md", metrics, paired)
    feature_protocol["ridge_alphas"] = list(RIDGE_ALPHAS)
    (output / "feature_protocol.json").write_text(
        json.dumps(feature_protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    protocol = {
        "name": "AEOLLM-2 E1.4 nested LOQO Ridge",
        "outer_split": "Leave-One-Question-Out",
        "inner_selection": "up to 5-fold GroupKFold by question, minimum MAE",
        "ridge_alphas": list(RIDGE_ALPHAS),
        "score_clip": [0.0, 10.0],
        "bootstrap_unit": "question",
        "bootstrap_resamples": config.bootstrap_resamples,
        "seed": config.seed,
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
        },
    }
    (output / "protocol.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    status = {
        "status": "complete",
        "models": len(models),
        "documents": len(labels),
        "questions": len(question_ids),
        "gpu_used": False,
    }
    (output / "run_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run E1.4 nested LOQO Ridge ablations")
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
        "--capped-features", type=Path, default=root / "outputs/e1/features/qwen3-0.6b-capped"
    )
    parser.add_argument(
        "--unbounded-features",
        type=Path,
        default=root / "outputs/e1/features/qwen3-0.6b-unbounded",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/e1/e1_4")
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = E14Config(
        labels=args.labels,
        rubric_dir=args.rubric_dir,
        surface_features=args.surface_features,
        capped_features=args.capped_features,
        unbounded_features=args.unbounded_features,
        output_dir=args.output_dir,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    metrics = run_e1_4(config)
    print(metrics[["model", "spearman", "kendall", "accuracy", "mae"]].to_string(index=False))
    return 0
