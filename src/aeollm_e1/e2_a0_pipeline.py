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
from aeollm_e0.metrics import DIMS, KEY_COLUMNS, load_dimension_weights, normalize_labels
from aeollm_e0.statistics import paired_question_bootstrap, question_bootstrap

from .diagonal_interaction import (
    ALIGNMENT_DIMS,
    DiagonalTrainingConfig,
    InteractionCorpus,
    config_dict,
    load_interaction_corpus,
    train_interaction_fold_ensemble,
)
from .e1_5_pipeline import _evaluate
from .e1_6_pipeline import _alignment_paired_bootstrap
from .ridge_scoring import (
    RIDGE_ALPHAS,
    fit_grouped_ridge_fold,
    load_feature_groups,
    numeric_surface_columns,
)

ALIGNMENT_OUTPUT_INDICES = [DIMS.index(dimension) for dimension in ALIGNMENT_DIMS]


@dataclass(frozen=True)
class E2A0Config:
    labels: Path
    rubric_dir: Path
    surface_features: Path
    unbounded_features: Path
    base_cache: Path
    query_variant_root: Path
    output_dir: Path
    device: str
    seeds: tuple[int, ...] = (20260721, 20260722, 20260723)
    mismatch_count: int = 5
    bootstrap_resamples: int = 5000
    bootstrap_seed: int = 20260721
    overwrite_checkpoints: bool = False


def _merge_baseline_features(
    feature_frame: pd.DataFrame,
    surface: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    merged = feature_frame.merge(
        surface, on=KEY_COLUMNS, validate="one_to_one", suffixes=("", "_surface")
    ).merge(labels, on=KEY_COLUMNS, validate="one_to_one")
    if len(merged) != len(labels):
        raise ValueError("baseline feature and label keys do not match exactly")
    return merged.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _validate_corpus_keys(corpus: InteractionCorpus, labels: pd.DataFrame) -> None:
    corpus_keys = {
        (question_id, document_id)
        for question_id, chunks in corpus.chunks.items()
        for document_id in chunks.document_ids
    }
    label_keys = set(labels[KEY_COLUMNS].itertuples(index=False, name=None))
    if corpus_keys != label_keys:
        raise ValueError(
            f"embedding and label keys differ: missing={len(label_keys-corpus_keys)}, "
            f"extra={len(corpus_keys-label_keys)}"
        )


def _arrays_by_question(
    corpus: InteractionCorpus,
    keyed_values: pd.DataFrame,
    value_columns: list[str],
) -> dict[int, np.ndarray]:
    indexed = keyed_values.set_index(KEY_COLUMNS)
    result: dict[int, np.ndarray] = {}
    for question_id, chunks in corpus.chunks.items():
        keys = [(question_id, document_id) for document_id in chunks.document_ids]
        values = indexed.loc[keys, value_columns].to_numpy(dtype=np.float32)
        if values.shape != (len(keys), len(value_columns)) or not np.isfinite(values).all():
            raise ValueError(f"invalid aligned values for question {question_id}")
        result[question_id] = values
    return result


def _checkpoint_manifest(config: E2A0Config, training: DiagonalTrainingConfig) -> dict[str, object]:
    return {
        "labels_sha256": sha256_file(config.labels),
        "surface_sha256": sha256_file(config.surface_features),
        "seeds": list(config.seeds),
        "mismatch_count": config.mismatch_count,
        "training": config_dict(training),
        "base_cache": str(config.base_cache.resolve()),
        "query_variant_root": str(config.query_variant_root.resolve()),
    }


def _prepare_checkpoints(
    directory: Path,
    expected: dict[str, object],
    *,
    overwrite: bool,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "checkpoint_manifest.json"
    if overwrite:
        for path in directory.iterdir():
            if path.is_file():
                path.unlink()
    elif manifest_path.exists():
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if observed != expected:
            raise ValueError(
                "checkpoint protocol differs from this run; use --overwrite-checkpoints"
            )
    elif any(directory.iterdir()):
        raise ValueError("checkpoint directory is non-empty but has no protocol manifest")
    manifest_path.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _fit_or_load_baseline_fold(
    data: pd.DataFrame,
    columns: list[str],
    held_out: int,
    checkpoint_dir: Path,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    path = checkpoint_dir / f"baseline_q{held_out}.npz"
    train_mask = data["questionId"].to_numpy(dtype=int) != held_out
    test_mask = ~train_mask
    if path.exists():
        saved = np.load(path)
        return saved["train_prediction"], saved["test_prediction"], pd.DataFrame(
            {
                "held_out_question": [held_out] * len(DIMS),
                "dimension": DIMS,
                "alpha": saved["alpha"],
                "inner_mae": saved["inner_mae"],
            }
        )
    train_prediction = np.empty((int(train_mask.sum()), len(DIMS)), dtype=np.float32)
    test_prediction = np.empty((int(test_mask.sum()), len(DIMS)), dtype=np.float32)
    alphas: list[float] = []
    inner_mae: list[float] = []
    groups = data.loc[train_mask, "questionId"].to_numpy(dtype=int)
    for dimension_index, dimension in enumerate(DIMS):
        train_pred, test_pred, alpha, mae = fit_grouped_ridge_fold(
            data.loc[train_mask, columns],
            data.loc[test_mask, columns],
            data.loc[train_mask, dimension].to_numpy(dtype=float),
            groups,
            alphas=RIDGE_ALPHAS,
        )
        train_prediction[:, dimension_index] = train_pred
        test_prediction[:, dimension_index] = test_pred
        alphas.append(alpha)
        inner_mae.append(mae)
    np.savez_compressed(
        path,
        train_prediction=train_prediction,
        test_prediction=test_prediction,
        alpha=np.asarray(alphas),
        inner_mae=np.asarray(inner_mae),
    )
    selections = pd.DataFrame(
        {
            "held_out_question": [held_out] * len(DIMS),
            "dimension": DIMS,
            "alpha": alphas,
            "inner_mae": inner_mae,
        }
    )
    return train_prediction, test_prediction, selections


def _model_specs(mismatch_count: int) -> list[tuple[str, str, bool]]:
    specs = [
        ("fixed_matched_hybrid", "matched", False),
        ("diagonal_matched_hybrid", "matched", True),
        ("diagonal_generic_hybrid", "generic", True),
    ]
    specs.extend(
        (f"diagonal_mismatch_shift{index}_hybrid", f"mismatch_shift{index}", True)
        for index in range(1, mismatch_count + 1)
    )
    return specs


def _mean_predictions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    ordered = [frame.sort_values(KEY_COLUMNS).reset_index(drop=True) for frame in frames]
    keys = ordered[0][KEY_COLUMNS]
    result = keys.copy()
    for frame in ordered[1:]:
        if not frame[KEY_COLUMNS].equals(keys):
            raise ValueError("mismatch prediction keys do not align")
    for dimension in DIMS:
        result[dimension] = np.mean(
            np.stack([frame[dimension].to_numpy(dtype=float) for frame in ordered]), axis=0
        )
    return result


def _write_report(
    path: Path,
    metrics: pd.DataFrame,
    alignment_paired: pd.DataFrame,
) -> None:
    ranked = metrics.sort_values("spearman", ascending=False)
    primary = alignment_paired[alignment_paired["metric"] == "alignment_spearman"]
    indexed = primary.set_index(["candidate", "reference"])
    learned = indexed.loc[("diagonal_matched_hybrid", "fixed_matched_hybrid")]
    incremental = indexed.loc[("diagonal_matched_hybrid", "ridge_global_structure")]
    generic = indexed.loc[("diagonal_matched_hybrid", "diagonal_generic_hybrid")]
    mismatch = indexed.loc[
        ("diagonal_matched_hybrid", "diagonal_mismatched_ensemble_hybrid")
    ]
    passed = bool(
        float(learned["mean_delta"]) > 0
        and float(learned["probability_delta_gt_zero"]) >= 0.90
        and int(learned["positive_questions"]) >= 7
    )
    lines = [
        "# E2-A0 minimal learned diagonal interaction",
        "",
        "All models use frozen Qwen embeddings and unbounded chunks. The interaction",
        "branch changes only comprehensiveness and instruction following; insight and",
        "readability remain the outer-fold global+structure Ridge predictions.",
        "",
        "## Main results",
        "",
        ranked[
            [
                "model",
                "spearman",
                "kendall",
                "accuracy",
                "spearman_comprehensiveness",
                "spearman_instruction_following",
                "mae",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Pre-registered alignment comparisons",
        "",
        primary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Decision",
        "",
        f"- Learned diagonal versus fixed cosine: {float(learned['mean_delta']):+.4f} "
        f"alignment Spearman, P(delta>0)={float(learned['probability_delta_gt_zero']):.4f}, "
        f"{int(learned['positive_questions'])}/10 positive questions.",
        f"- Learned diagonal versus global+structure: {float(incremental['mean_delta']):+.4f}.",
        f"- Matched versus generic learned interaction: {float(generic['mean_delta']):+.4f}.",
        f"- Matched versus mismatched learned ensemble: {float(mismatch['mean_delta']):+.4f}.",
        f"- E2-A0 learned-interaction gate: {'PASS' if passed else 'FAIL'}.",
        "",
        "The gate is deliberately based on matched learned interaction versus the same",
        "fixed-cosine architecture, not on the official aggregate alone.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_e2_a0(config: E2A0Config, training: DiagonalTrainingConfig) -> pd.DataFrame:
    started = time.perf_counter()
    output = config.output_dir
    prediction_dir = output / "predictions"
    checkpoint_dir = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    _prepare_checkpoints(
        checkpoint_dir,
        _checkpoint_manifest(config, training),
        overwrite=config.overwrite_checkpoints,
    )
    labels = normalize_labels(pd.read_csv(config.labels)).sort_values(KEY_COLUMNS).reset_index(drop=True)
    question_ids = sorted(int(value) for value in labels["questionId"].unique())
    weights = load_dimension_weights(config.rubric_dir, question_ids)
    feature_frame, feature_groups = load_feature_groups(config.unbounded_features)
    surface = pd.read_csv(config.surface_features)
    surface_columns = numeric_surface_columns(surface)
    baseline_columns = [*feature_groups["global"], *surface_columns]
    data = _merge_baseline_features(feature_frame, surface, labels)

    corpus, mismatch_maps = load_interaction_corpus(
        config.base_cache,
        config.query_variant_root,
        mismatch_count=config.mismatch_count,
        device=config.device,
    )
    _validate_corpus_keys(corpus, labels)
    targets_by_question = _arrays_by_question(corpus, labels, list(ALIGNMENT_DIMS))
    specs = _model_specs(config.mismatch_count)
    models = {
        "ridge_global_structure": data[KEY_COLUMNS].copy(),
        **{name: data[KEY_COLUMNS].copy() for name, _, _ in specs},
    }
    for frame in models.values():
        frame[DIMS] = np.nan
    diagnostics: list[dict[str, object]] = []
    baseline_selections: list[pd.DataFrame] = []

    for held_out in question_ids:
        train_mask = data["questionId"].to_numpy(dtype=int) != held_out
        test_mask = ~train_mask
        base_train, base_test, selections = _fit_or_load_baseline_fold(
            data, baseline_columns, held_out, checkpoint_dir
        )
        baseline_selections.append(selections)
        models["ridge_global_structure"].loc[test_mask, DIMS] = base_test
        fold_values = data[KEY_COLUMNS].copy()
        fold_values[DIMS] = np.nan
        fold_values.loc[train_mask, DIMS] = base_train
        fold_values.loc[test_mask, DIMS] = base_test
        baseline_by_question = _arrays_by_question(
            corpus, fold_values, list(ALIGNMENT_DIMS)
        )
        train_questions = [question_id for question_id in question_ids if question_id != held_out]
        for model_name, view_name, learn_diagonal in specs:
            checkpoint = checkpoint_dir / f"{model_name}_q{held_out}.npz"
            diagnostic_path = checkpoint_dir / f"{model_name}_q{held_out}.json"
            if checkpoint.exists() and diagnostic_path.exists():
                alignment_prediction = np.load(checkpoint)["prediction"]
                fold_diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            else:
                alignment_prediction, fold_diagnostics = train_interaction_fold_ensemble(
                    corpus,
                    corpus.rubric_views[view_name],
                    train_questions,
                    held_out,
                    targets_by_question,
                    baseline_by_question,
                    config=training,
                    seeds=config.seeds,
                    learn_diagonal=learn_diagonal,
                )
                np.savez_compressed(checkpoint, prediction=alignment_prediction)
                diagnostic_path.write_text(
                    json.dumps(fold_diagnostics, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            hybrid = base_test.copy()
            hybrid[:, ALIGNMENT_OUTPUT_INDICES] = np.clip(alignment_prediction, 0.0, 10.0)
            models[model_name].loc[test_mask, DIMS] = hybrid
            for row in fold_diagnostics:
                diagnostics.append(
                    {
                        "model": model_name,
                        "rubric_view": view_name,
                        "held_out_question": held_out,
                        **row,
                    }
                )

    mismatch_models = [
        models[f"diagonal_mismatch_shift{index}_hybrid"]
        for index in range(1, config.mismatch_count + 1)
    ]
    models["diagonal_mismatched_ensemble_hybrid"] = _mean_predictions(mismatch_models)
    for name, frame in models.items():
        if not np.isfinite(frame[DIMS].to_numpy(dtype=float)).all():
            raise ValueError(f"model has incomplete predictions: {name}")

    pd.concat(baseline_selections, ignore_index=True).to_csv(
        output / "baseline_selected_hyperparameters.csv", index=False, float_format="%.8f"
    )
    pd.DataFrame(diagnostics).to_csv(
        output / "training_diagnostics.csv", index=False, float_format="%.8f"
    )
    metric_records: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    details_by_model: dict[str, pd.DataFrame] = {}
    for name, prediction in models.items():
        record, details = _evaluate(name, prediction, labels, weights, prediction_dir)
        metric_records.append(record)
        detail_frames.append(details)
        details_by_model[name] = details.drop(columns="model")
    metrics = pd.DataFrame(metric_records).sort_values("model").reset_index(drop=True)
    metrics.to_csv(output / "model_metrics.csv", index=False, float_format="%.8f")
    pd.concat(detail_frames, ignore_index=True).to_csv(
        output / "per_question_metrics.csv", index=False, float_format="%.8f"
    )
    question_bootstrap(
        details_by_model,
        n_resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
    ).to_csv(output / "bootstrap_ci.csv", index=False, float_format="%.8f")
    comparisons = [
        ("diagonal_matched_hybrid", "fixed_matched_hybrid"),
        ("diagonal_matched_hybrid", "ridge_global_structure"),
        ("fixed_matched_hybrid", "ridge_global_structure"),
        ("diagonal_matched_hybrid", "diagonal_generic_hybrid"),
        ("diagonal_matched_hybrid", "diagonal_mismatched_ensemble_hybrid"),
    ]
    paired_question_bootstrap(
        details_by_model,
        comparisons,
        n_resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
    ).to_csv(output / "paired_bootstrap.csv", index=False, float_format="%.8f")
    alignment_paired = _alignment_paired_bootstrap(
        details_by_model,
        comparisons,
        n_resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
    )
    alignment_paired.to_csv(
        output / "alignment_paired_bootstrap.csv", index=False, float_format="%.8f"
    )
    _write_report(output / "e2_a0_conclusions.md", metrics, alignment_paired)

    elapsed = time.perf_counter() - started
    protocol = {
        "name": "AEOLLM-2 E2-A0 minimal learned diagonal interaction",
        "outer_split": "Leave-One-Question-Out",
        "model_selection": "none; architecture, epochs, and optimization fixed",
        "encoder": "frozen Qwen/Qwen3-Embedding-0.6B",
        "chunks": "unbounded E1.1 structure-preserving chunks",
        "baseline": "global embedding + surface features nested Ridge",
        "alignment_dimensions": list(ALIGNMENT_DIMS),
        "untouched_dimensions": [dimension for dimension in DIMS if dimension not in ALIGNMENT_DIMS],
        "interaction": "cosine + sqrt(1024) * dot(r * learned_diagonal, h)",
        "pooling": ["mean", "max", "top10pct_mean", "logmeanexp_t005"],
        "criterion_aggregation": "fixed official criterion weights",
        "training": config_dict(training),
        "seeds": list(config.seeds),
        "mismatch_maps": [
            {str(key): int(value) for key, value in sorted(mapping.items())}
            for mapping in mismatch_maps
        ],
        "bootstrap_unit": "question",
        "bootstrap_resamples": config.bootstrap_resamples,
        "bootstrap_seed": config.bootstrap_seed,
        "device": config.device,
        "gpu_required": config.device.startswith("cuda:"),
        "baseline_feature_count": len(baseline_columns),
        "paths": {
            key: str(value.resolve())
            for key, value in asdict(config).items()
            if isinstance(value, Path)
        },
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
        "questions": len(question_ids),
        "elapsed_seconds": elapsed,
        "device": config.device,
        "gpu_used": config.device.startswith("cuda:"),
    }
    (output / "run_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run E2-A0 learned diagonal interaction")
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
        "--base-cache", type=Path, default=root / "outputs/e1/embeddings/qwen3-0.6b-unbounded"
    )
    parser.add_argument(
        "--query-variant-root",
        type=Path,
        default=root / "outputs/e1/embeddings/qwen3-0.6b-query-variants",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/e2/e2_a0")
    parser.add_argument("--device", required=True, help="cpu or an explicit cuda:N")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--mismatch-count", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--overwrite-checkpoints", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = E2A0Config(
        labels=args.labels,
        rubric_dir=args.rubric_dir,
        surface_features=args.surface_features,
        unbounded_features=args.unbounded_features,
        base_cache=args.base_cache,
        query_variant_root=args.query_variant_root,
        output_dir=args.output_dir,
        device=args.device,
        mismatch_count=args.mismatch_count,
        bootstrap_resamples=args.bootstrap_resamples,
        overwrite_checkpoints=args.overwrite_checkpoints,
    )
    training = DiagonalTrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    if config.device.startswith("cuda:"):
        torch.set_float32_matmul_precision("high")
    metrics = run_e2_a0(config, training)
    print(
        metrics[
            [
                "model",
                "spearman",
                "kendall",
                "accuracy",
                "spearman_comprehensiveness",
                "spearman_instruction_following",
                "mae",
            ]
        ].sort_values("spearman", ascending=False).to_string(index=False)
    )
    return 0
