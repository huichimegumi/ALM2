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

from .data import build_manifest_and_features, sha256_file
from .metrics import (
    DIMS,
    KEY_COLUMNS,
    PredictionValidationError,
    evaluate_predictions,
    load_dimension_weights,
    normalize_labels,
    normalize_predictions,
    to_submission_frame,
    validate_predictions,
    weighted_total,
)
from .modeling import (
    METADATA_COLUMNS,
    loqo_calibration_predictions,
    loqo_ridge_predictions,
    mean_loqo_predictions,
    random_kfold_ridge_predictions,
)
from .statistics import paired_question_bootstrap, question_bootstrap


@dataclass(frozen=True)
class E0Config:
    labels: Path
    report_root: Path
    rubric_dir: Path
    mapping: Path
    legacy_output_dir: Path
    odat_prediction: Path
    output_dir: Path
    bootstrap_resamples: int = 5000
    seed: int = 20260721


def _read_prediction_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return normalize_predictions(frame)


def _diagnostic_errors(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    weights: dict[int, dict[str, float]],
) -> dict[str, float]:
    merged = predictions.merge(labels, on=KEY_COLUMNS, suffixes=("_pred", "_gt"))
    pred_matrix = merged[[f"{dim}_pred" for dim in DIMS]].to_numpy(dtype=float)
    gold_matrix = merged[[f"{dim}_gt" for dim in DIMS]].to_numpy(dtype=float)
    result: dict[str, float] = {}
    for index, dim in enumerate(DIMS):
        error = pred_matrix[:, index] - gold_matrix[:, index]
        result[f"mae_{dim}"] = float(np.mean(np.abs(error)))
        result[f"rmse_{dim}"] = float(np.sqrt(np.mean(error**2)))
    predicted_totals = []
    gold_totals = []
    for question_id, sub in merged.groupby("questionId", sort=True):
        predicted_totals.extend(
            weighted_total(sub[[f"{dim}_pred" for dim in DIMS]].to_numpy(dtype=float), weights[int(question_id)])
        )
        gold_totals.extend(
            weighted_total(sub[[f"{dim}_gt" for dim in DIMS]].to_numpy(dtype=float), weights[int(question_id)])
        )
    total_error = np.asarray(predicted_totals) - np.asarray(gold_totals)
    result["mae"] = float(np.mean(np.abs(total_error)))
    result["rmse"] = float(np.sqrt(np.mean(total_error**2)))
    return result


def _evaluate_model(
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
    submission = to_submission_frame(clean)
    submission.to_csv(
        prediction_dir / f"{name}.tsv",
        sep="\t",
        index=False,
        encoding="utf-8",
        float_format="%.8f",
    )
    details = details.copy()
    details.insert(0, "model", name)
    return record, details


def _index_legacy_predictions(
    directory: Path,
    labels: pd.DataFrame,
    weights: dict[int, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    expected_keys = set(map(tuple, labels[KEY_COLUMNS].itertuples(index=False, name=None)))
    for path in sorted(directory.glob("baseline_*.txt")):
        record: dict[str, object] = {"file": str(path.resolve()), "name": path.stem.replace("baseline_", "")}
        try:
            raw = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
            pred = normalize_predictions(raw)
            keys = set(map(tuple, pred[KEY_COLUMNS].itertuples(index=False, name=None)))
            values = pred[DIMS].to_numpy(dtype=float)
            record.update(
                {
                    "rows": int(len(pred)),
                    "unique_keys": int(len(keys)),
                    "questions": int(pred["questionId"].nunique()),
                    "missing_keys": int(len(expected_keys - keys)),
                    "extra_keys": int(len(keys - expected_keys)),
                    "duplicate_rows": int(pred.duplicated(KEY_COLUMNS, keep=False).sum()),
                    "missing_dimension_values": int(np.isnan(values).sum()),
                    "out_of_range_values": int(((values < 0.0) | (values > 10.0)).sum()),
                }
            )
            try:
                valid = validate_predictions(pred, labels)
                record["strictly_eligible"] = True
                summary, _ = evaluate_predictions(valid, labels, weights)
                metric_rows.append(
                    {
                        "name": record["name"],
                        "file": record["file"],
                        **summary.iloc[0].to_dict(),
                        **_diagnostic_errors(valid, labels, weights),
                    }
                )
            except PredictionValidationError as exc:
                record["strictly_eligible"] = False
                record["validation_error"] = str(exc)
        except Exception as exc:  # keep the audit comprehensive even for malformed legacy files
            record["strictly_eligible"] = False
            record["validation_error"] = f"{type(exc).__name__}: {exc}"
        index_rows.append(record)
    return pd.DataFrame(index_rows), pd.DataFrame(metric_rows)


def _write_integrity_report(
    path: Path,
    labels: pd.DataFrame,
    manifest: pd.DataFrame,
    errors: list[str],
    legacy_index: pd.DataFrame,
) -> None:
    counts = labels.groupby("questionId").size().to_dict()
    table_documents = int(manifest["has_table"].sum()) if len(manifest) else 0
    eligible = int(legacy_index.get("strictly_eligible", pd.Series(dtype=bool)).fillna(False).sum())
    lines = [
        "# E0 integrity report",
        "",
        f"- Labels: {len(labels)}",
        f"- Questions: {labels['questionId'].nunique()}",
        f"- Documents per question: `{counts}`",
        f"- Manifest documents: {len(manifest)}",
        f"- Documents containing Word tables: {table_documents}",
        f"- Strictly eligible legacy prediction files: {eligible}/{len(legacy_index)}",
        f"- Integrity errors: {len(errors)}",
        "",
        "## Errors",
        "",
    ]
    lines.extend([f"- {error}" for error in errors] or ["- None"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _check_official_parity(
    odat_path: Path,
    summary: dict[str, object],
    legacy_output_dir: Path,
) -> dict[str, object]:
    model_name = odat_path.stem.replace("baseline_", "")
    expected_path = legacy_output_dir / "eval_results" / model_name / "metrics_summary.csv"
    result: dict[str, object] = {
        "model": model_name,
        "expected_metrics_path": str(expected_path.resolve()),
        "metrics": {},
    }
    if not expected_path.exists():
        result.update({"status": "missing_reference", "max_absolute_difference": None})
        return result
    expected = pd.read_csv(expected_path).iloc[0]
    differences = {}
    for metric in ("spearman", "kendall", "accuracy"):
        observed_value = float(summary[metric])
        expected_value = float(expected[metric])
        differences[metric] = {
            "observed": observed_value,
            "official_legacy_reference": expected_value,
            "absolute_difference": abs(observed_value - expected_value),
        }
    max_difference = max(item["absolute_difference"] for item in differences.values())
    result.update(
        {
            "status": "pass" if max_difference <= 1e-6 else "fail",
            "max_absolute_difference": max_difference,
            "metrics": differences,
        }
    )
    return result


def _write_summary_report(
    path: Path,
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    integrity_errors: list[str],
) -> None:
    ranked = metrics.sort_values(["accuracy", "spearman"], ascending=False)
    display_columns = ["model", "accuracy", "spearman", "kendall", "mae", "rmse"]
    indexed = metrics.set_index("model")
    surface = indexed.loc["surface_ridge_loqo"]
    metadata = indexed.loc["metadata_ridge_loqo"]
    mixed = indexed.loc["surface_metadata_ridge_loqo"]
    odat = indexed.loc["odat_raw"]
    affine = indexed.loc["odat_affine_loqo"]
    mae_reduction = 100.0 * (float(odat["mae"]) - float(affine["mae"])) / float(odat["mae"])
    lines = [
        "# AEOLLM-2 E0 conclusions",
        "",
        "All learned results are out-of-fold. The primary protocol is leave-one-question-out;",
        "`random_split_surface_ridge` is a deliberately leaky diagnostic and is not a valid final result.",
        "",
        f"Integrity gate: {'PASS' if not integrity_errors else 'FAIL'} ({len(integrity_errors)} errors).",
        "",
        "## Observed findings",
        "",
        f"- Surface-only LOQO reaches Accuracy {surface['accuracy']:.4f}, Spearman {surface['spearman']:.4f}, and Kendall {surface['kendall']:.4f}; superficial document properties contain signal but do not explain ODAT.",
        f"- Generator/prompt metadata alone reaches Accuracy {metadata['accuracy']:.4f}. This is evidence of source/prompt confounding and must remain a diagnostic rather than a final evaluator.",
        f"- Adding metadata to surface features changes Accuracy from {surface['accuracy']:.4f} to {mixed['accuracy']:.4f}; see the paired question bootstrap below for uncertainty.",
        f"- Affine LOQO calibration reduces ODAT weighted-total MAE from {odat['mae']:.4f} to {affine['mae']:.4f} ({mae_reduction:.1f}% reduction), while its three official ranking metrics are unchanged at displayed precision. ODAT therefore has a large scale bias that simple calibration can fix, but calibration does not improve its ordering.",
        f"- Random document splitting changes surface-model MAE from {surface['mae']:.4f} (LOQO) to {indexed.loc['random_split_surface_ridge', 'mae']:.4f}. Its rank metrics are not directly interpretable because documents within a question are scored by different fold models; LOQO remains the sole primary protocol.",
        "",
        "## Main results",
        "",
        ranked[display_columns].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Paired question bootstrap",
        "",
        paired.to_markdown(index=False, floatfmt=".4f") if len(paired) else "No comparisons available.",
        "",
        "## Interpretation rules",
        "",
        "- Prefer LOQO results over random document splits.",
        "- Treat metadata-only performance as generator/prompt confounding, not evaluation ability.",
        "- Calibration improving MAE without rank metrics indicates scale bias rather than ranking improvement.",
        "- With only 10 independent questions, use the question-bootstrap intervals rather than pair-level p-values.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_e0(config: E0Config) -> pd.DataFrame:
    output = config.output_dir
    prediction_dir = output / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    raw_labels = pd.read_csv(config.labels)
    labels = normalize_labels(raw_labels)
    weights = load_dimension_weights(config.rubric_dir, sorted(labels["questionId"].unique().tolist()))
    manifest, features, integrity_errors = build_manifest_and_features(
        raw_labels, config.report_root, config.rubric_dir, config.mapping
    )
    manifest.to_csv(output / "data_manifest.csv", index=False, encoding="utf-8")
    features.to_csv(output / "surface_features.csv", index=False, encoding="utf-8")

    legacy_index, legacy_metrics = _index_legacy_predictions(config.legacy_output_dir, labels, weights)
    legacy_index.to_csv(output / "legacy_predictions_index.csv", index=False, encoding="utf-8")
    legacy_metrics.to_csv(output / "legacy_recomputed_metrics.csv", index=False, encoding="utf-8")
    _write_integrity_report(output / "integrity_report.md", labels, manifest, integrity_errors, legacy_index)

    protocol = {
        "name": "AEOLLM-2 E0",
        "primary_split": "Leave-One-Question-Out",
        "inner_selection": (
            "GroupKFold by question; maximize pairwise accuracy, then Spearman, "
            "then minimize MAE"
        ),
        "primary_metric": "Pairwise Accuracy",
        "secondary_metrics": ["Spearman", "Kendall"],
        "official_metrics": ["Pairwise Accuracy", "Spearman", "Kendall"],
        "official_aggregation": {
            "spearman": "macro mean over questions",
            "kendall": "macro mean over questions",
            "accuracy": "sum correct pairs / sum all pairs",
        },
        "score_range": [0.0, 10.0],
        "bootstrap_unit": "question",
        "bootstrap_resamples": config.bootstrap_resamples,
        "seed": config.seed,
        "gpu_required": False,
        "paths": {key: str(value) for key, value in asdict(config).items() if isinstance(value, Path)},
        "input_hashes": {
            "labels": sha256_file(config.labels),
            "mapping": sha256_file(config.mapping),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (output / "protocol.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    numeric_columns = [
        column
        for column in features.columns
        if column not in {*KEY_COLUMNS, "model_name", *METADATA_COLUMNS, "prompt_position"}
        and pd.api.types.is_numeric_dtype(features[column])
    ]
    models: dict[str, pd.DataFrame] = {"mean_loqo": mean_loqo_predictions(labels)}
    hyperparameters: list[pd.DataFrame] = []

    surface, selected = loqo_ridge_predictions(
        features, labels, numeric_columns=numeric_columns
    )
    models["surface_ridge_loqo"] = surface
    selected.insert(0, "model", "surface_ridge_loqo")
    hyperparameters.append(selected)

    metadata, selected = loqo_ridge_predictions(
        features,
        labels,
        numeric_columns=[],
        categorical_columns=METADATA_COLUMNS,
    )
    models["metadata_ridge_loqo"] = metadata
    selected.insert(0, "model", "metadata_ridge_loqo")
    hyperparameters.append(selected)

    mixed, selected = loqo_ridge_predictions(
        features,
        labels,
        numeric_columns=numeric_columns,
        categorical_columns=METADATA_COLUMNS,
    )
    models["surface_metadata_ridge_loqo"] = mixed
    selected.insert(0, "model", "surface_metadata_ridge_loqo")
    hyperparameters.append(selected)
    models["random_split_surface_ridge"] = random_kfold_ridge_predictions(
        features, labels, numeric_columns, random_state=config.seed
    )

    raw_odat = validate_predictions(_read_prediction_file(config.odat_prediction), labels)
    models["odat_raw"] = raw_odat
    for method in ("affine", "isotonic", "multioutput_ridge"):
        models[f"odat_{method}_loqo"] = loqo_calibration_predictions(raw_odat, labels, method)

    pd.concat(hyperparameters, ignore_index=True).to_csv(
        output / "selected_hyperparameters.csv", index=False, encoding="utf-8"
    )
    metric_records: list[dict[str, object]] = []
    details_by_model: dict[str, pd.DataFrame] = {}
    detail_frames: list[pd.DataFrame] = []
    for name, predictions in models.items():
        record, details = _evaluate_model(name, predictions, labels, weights, prediction_dir)
        metric_records.append(record)
        detail_frames.append(details)
        details_by_model[name] = details.drop(columns="model")
    metrics = pd.DataFrame(metric_records).sort_values("model").reset_index(drop=True)
    odat_summary = next(record for record in metric_records if record["model"] == "odat_raw")
    parity = _check_official_parity(config.odat_prediction, odat_summary, config.legacy_output_dir)
    (output / "official_metric_parity.json").write_text(
        json.dumps(parity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if parity["status"] == "fail":
        raise RuntimeError(
            f"official metric parity failed: max difference={parity['max_absolute_difference']}"
        )
    metrics.to_csv(output / "model_metrics.csv", index=False, encoding="utf-8", float_format="%.8f")
    split_diagnostic = metrics[
        metrics["model"].isin(["surface_ridge_loqo", "random_split_surface_ridge"])
    ][["model", "mae", "rmse"]].copy()
    split_diagnostic["rank_metrics_comparable"] = False
    split_diagnostic["note"] = [
        "primary question-held-out protocol" if name == "surface_ridge_loqo" else
        "error-only diagnostic; OOF rank scores mix fold-specific calibrations"
        for name in split_diagnostic["model"]
    ]
    split_diagnostic.to_csv(output / "split_diagnostic.csv", index=False, encoding="utf-8")
    pd.concat(detail_frames, ignore_index=True).to_csv(
        output / "per_question_metrics.csv", index=False, encoding="utf-8", float_format="%.8f"
    )

    bootstrap = question_bootstrap(
        details_by_model, n_resamples=config.bootstrap_resamples, seed=config.seed
    )
    bootstrap.to_csv(output / "bootstrap_ci.csv", index=False, encoding="utf-8", float_format="%.8f")
    comparisons = [
        ("surface_ridge_loqo", "mean_loqo"),
        ("metadata_ridge_loqo", "mean_loqo"),
        ("surface_metadata_ridge_loqo", "surface_ridge_loqo"),
        ("odat_raw", "surface_ridge_loqo"),
        ("odat_raw", "metadata_ridge_loqo"),
        ("odat_affine_loqo", "odat_raw"),
        ("odat_isotonic_loqo", "odat_raw"),
        ("odat_multioutput_ridge_loqo", "odat_raw"),
    ]
    paired = paired_question_bootstrap(
        details_by_model,
        comparisons,
        n_resamples=config.bootstrap_resamples,
        seed=config.seed,
    )
    paired.to_csv(output / "paired_bootstrap.csv", index=False, encoding="utf-8", float_format="%.8f")
    _write_summary_report(output / "e0_conclusions.md", metrics, paired, integrity_errors)
    (output / "run_status.json").write_text(
        json.dumps(
            {
                "status": "complete" if not integrity_errors else "integrity_failed",
                "integrity_errors": integrity_errors,
                "models_evaluated": sorted(models),
                "gpu_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if integrity_errors:
        raise RuntimeError(f"E0 integrity gate failed with {len(integrity_errors)} errors")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run the CPU-only AEOLLM-2 E0 experiment")
    parser.add_argument(
        "--labels",
        type=Path,
        default=root / "data/official/hf-aeollm/aeollm-2-train/train_deepresearch.csv",
    )
    parser.add_argument(
        "--report-root", type=Path, default=root / "data/incoming/google-drive/train"
    )
    parser.add_argument(
        "--rubric-dir",
        type=Path,
        default=root / "data/official/hf-aeollm/aeollm-2-train/rubric_dataset",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=root / "legacy/aeollm2_train_code/prompts/mapping_key_Readability.xlsx",
    )
    parser.add_argument(
        "--legacy-output-dir",
        type=Path,
        default=root / "legacy/aeollm2_train_code/outputs",
    )
    parser.add_argument(
        "--odat-prediction",
        type=Path,
        default=root
        / "legacy/aeollm2_train_code/outputs/baseline_onedim_decimal_thinking-off_deepseek-v4-pro.txt",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/e0_accuracy")
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = E0Config(
        labels=args.labels,
        report_root=args.report_root,
        rubric_dir=args.rubric_dir,
        mapping=args.mapping,
        legacy_output_dir=args.legacy_output_dir,
        odat_prediction=args.odat_prediction,
        output_dir=args.output_dir,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    metrics = run_e0(config)
    print(metrics[["model", "accuracy", "spearman", "kendall", "mae"]].to_string(index=False))
    print(f"\nE0 outputs: {config.output_dir.resolve()}")
    return 0
