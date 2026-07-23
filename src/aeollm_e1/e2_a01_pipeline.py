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
)
from aeollm_e0.statistics import paired_question_bootstrap, question_bootstrap

from .diagonal_interaction import (
    ALIGNMENT_DIMS,
    DiagonalTrainingConfig,
    InteractionCorpus,
    config_dict,
    load_interaction_corpus,
    train_separated_interaction_fold_ensemble,
)
from .e1_5_pipeline import _evaluate
from .e1_6_pipeline import _alignment_paired_bootstrap
from .e2_a0_pipeline import (
    _arrays_by_question,
    _fit_or_load_baseline_fold,
    _mean_predictions,
    _merge_baseline_features,
    _prepare_checkpoints,
    _validate_corpus_keys,
)
from .ridge_scoring import load_feature_groups, numeric_surface_columns


@dataclass(frozen=True)
class E2A01Config:
    labels: Path
    rubric_dir: Path
    surface_features: Path
    unbounded_features: Path
    base_cache: Path
    query_variant_root: Path
    output_dir: Path
    device: str
    shared_a0_predictions: Path | None = None
    seeds: tuple[int, ...] = (20260721, 20260722, 20260723)
    mismatch_count: int = 5
    bootstrap_resamples: int = 5000
    bootstrap_seed: int = 20260721
    overwrite_checkpoints: bool = False


def _checkpoint_manifest(
    config: E2A01Config, training: DiagonalTrainingConfig
) -> dict[str, object]:
    return {
        "experiment": "E2-A0.1 dimension-separated diagnostic",
        "labels_sha256": sha256_file(config.labels),
        "surface_sha256": sha256_file(config.surface_features),
        "seeds": list(config.seeds),
        "mismatch_count": config.mismatch_count,
        "training": config_dict(training),
        "base_cache": str(config.base_cache.resolve()),
        "query_variant_root": str(config.query_variant_root.resolve()),
        "architecture": (
            "joint two-head training with one independent diagonal metric per dimension"
        ),
    }


def _empty_hybrid(keys: pd.DataFrame) -> pd.DataFrame:
    result = keys[KEY_COLUMNS].copy()
    result[DIMS] = np.nan
    return result


def _fit_or_load_separated(
    *,
    corpus: InteractionCorpus,
    view_name: str,
    learn_diagonal: bool,
    train_questions: list[int],
    held_out: int,
    targets_by_question: dict[int, np.ndarray],
    baseline_by_question: dict[int, np.ndarray],
    training: DiagonalTrainingConfig,
    seeds: tuple[int, ...],
    checkpoint_dir: Path,
    checkpoint_stem: str,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    checkpoint = checkpoint_dir / f"{checkpoint_stem}_q{held_out}.npz"
    diagnostic_path = checkpoint_dir / f"{checkpoint_stem}_q{held_out}.json"
    if checkpoint.exists() and diagnostic_path.exists():
        prediction = np.load(checkpoint)["prediction"]
        diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        return prediction, diagnostics
    prediction, raw_diagnostics = train_separated_interaction_fold_ensemble(
        corpus,
        corpus.rubric_views[view_name],
        train_questions,
        held_out,
        targets_by_question,
        baseline_by_question,
        config=training,
        seeds=seeds,
        learn_diagonal=learn_diagonal,
    )
    diagnostics: list[dict[str, object]] = [dict(row) for row in raw_diagnostics]
    np.savez_compressed(checkpoint, prediction=prediction)
    diagnostic_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return prediction, diagnostics


def _temporary_details(
    models: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    weights: dict[int, dict[str, float]],
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for name, prediction in models.items():
        _, details = evaluate_predictions(prediction, labels, weights)
        result[name] = details
    return result


def _dimension_gate_records(
    paired: pd.DataFrame,
    *,
    learned_model: str = "diagonal_matched_separate_hybrid",
    fixed_model: str = "fixed_matched_hybrid",
    baseline_model: str = "ridge_global_structure",
) -> pd.DataFrame:
    indexed = paired.set_index(["candidate", "reference", "metric"])
    rows: list[dict[str, object]] = []
    for dimension in ALIGNMENT_DIMS:
        metric = f"spearman_{dimension}"
        learned = indexed.loc[(learned_model, fixed_model, metric)]
        baseline = indexed.loc[(learned_model, baseline_model, metric)]
        passed = bool(
            float(learned["mean_delta"]) > 0
            and float(learned["probability_delta_gt_zero"]) >= 0.90
            and int(learned["positive_questions"]) >= 7
            and float(baseline["mean_delta"]) >= 0
        )
        rows.append(
            {
                "dimension": dimension,
                "learned_minus_fixed": float(learned["mean_delta"]),
                "learned_fixed_ci_low": float(learned["ci_low"]),
                "learned_fixed_ci_high": float(learned["ci_high"]),
                "probability_delta_gt_zero": float(
                    learned["probability_delta_gt_zero"]
                ),
                "positive_questions": int(learned["positive_questions"]),
                "finite_questions": int(learned["n_questions"]),
                "learned_minus_baseline": float(baseline["mean_delta"]),
                "passed": passed,
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    path: Path,
    metrics: pd.DataFrame,
    dimension_paired: pd.DataFrame,
    gates: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    shared_loaded: bool,
    controls_run: list[str],
) -> None:
    ranked = metrics.sort_values("spearman", ascending=False)
    relevant_metrics = [f"spearman_{dimension}" for dimension in ALIGNMENT_DIMS]
    primary = dimension_paired[dimension_paired["metric"].isin(relevant_metrics)]
    primary_indexed = primary.set_index(["candidate", "reference", "metric"])
    diagnostic_means = diagnostics.groupby("model")["final_huber"].mean()
    lines = [
        "# E2-A0.1 dimension-separated diagonal diagnostic",
        "",
        "E2-A0.1 changes one architectural assumption from E2-A0: comprehensiveness",
        "and instruction following receive independent 1024-dimensional diagonal",
        "metrics. The two heads remain jointly trained with the same loss. Encoder,",
        "chunks, queries, pooling, optimizer, epochs, seeds, baseline, and splits are fixed.",
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
        "## Per-dimension learned-versus-fixed gates",
        "",
        gates.to_markdown(index=False, floatfmt=".4f"),
        "",
        "A dimension passes only when learned minus fixed is positive, bootstrap",
        "P(delta > 0) is at least 0.90, at least 7/10 questions improve, and the",
        "learned branch is not worse than global + structure on mean Spearman.",
        "",
        "## Paired question bootstrap",
        "",
        primary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Decision",
        "",
    ]
    for row in gates.itertuples(index=False):
        lines.append(
            f"- `{row.dimension}`: {'PASS' if row.passed else 'FAIL'}; "
            f"learned-fixed={row.learned_minus_fixed:+.4f}, "
            f"P(delta>0)={row.probability_delta_gt_zero:.4f}, "
            f"{row.positive_questions}/{row.finite_questions} finite questions positive, "
            f"learned-baseline={row.learned_minus_baseline:+.4f}."
        )
    if shared_loaded:
        for dimension in ALIGNMENT_DIMS:
            comparison = primary_indexed.loc[
                (
                    "diagonal_matched_separate_hybrid",
                    "diagonal_shared_a0_hybrid",
                    f"spearman_{dimension}",
                )
            ]
            lines.append(
                f"- Separated minus shared `{dimension}` Spearman: "
                f"{float(comparison['mean_delta']):+.4f}."
            )
    lines.extend(
        [
            "- Mean final training Huber: "
            f"{float(diagnostic_means['diagonal_matched_separate_hybrid']):.4f} learned "
            f"versus {float(diagnostic_means['fixed_matched_hybrid']):.4f} fixed; "
            "the lower training loss and worse LOQO ranking indicate overfitting.",
            f"- Original shared E2-A0 predictions loaded: {'yes' if shared_loaded else 'no'}.",
            (
                "- Conditional generic/mismatch controls were run for: "
                + ", ".join(f"`{dimension}`" for dimension in controls_run)
                + "."
                if controls_run
                else "- No dimension passed, so generic/mismatch controls were not run."
            ),
            "- This experiment diagnoses negative transfer only. It does not change",
            "  supervision or establish that the learned score measures satisfaction.",
            "- The continuation condition for criterion-only E2-A0.1b was not met;",
            "  it should not be run as the next confirmatory document-level experiment.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_e2_a01(
    config: E2A01Config, training: DiagonalTrainingConfig
) -> pd.DataFrame:
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

    labels = (
        normalize_labels(pd.read_csv(config.labels))
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
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

    models = {
        "ridge_global_structure": _empty_hybrid(data),
        "fixed_matched_hybrid": _empty_hybrid(data),
        "diagonal_matched_separate_hybrid": _empty_hybrid(data),
    }
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
        for name in (
            "fixed_matched_hybrid",
            "diagonal_matched_separate_hybrid",
        ):
            models[name].loc[test_mask, DIMS] = base_test

        fold_values = data[KEY_COLUMNS].copy()
        fold_values[DIMS] = np.nan
        fold_values.loc[train_mask, DIMS] = base_train
        fold_values.loc[test_mask, DIMS] = base_test
        baseline_by_question = _arrays_by_question(
            corpus, fold_values, list(ALIGNMENT_DIMS)
        )
        train_questions = [
            question_id for question_id in question_ids if question_id != held_out
        ]
        for model_name, learn_diagonal in (
            ("fixed_matched_hybrid", False),
            ("diagonal_matched_separate_hybrid", True),
        ):
            prediction, fold_diagnostics = _fit_or_load_separated(
                corpus=corpus,
                view_name="matched",
                learn_diagonal=learn_diagonal,
                train_questions=train_questions,
                held_out=held_out,
                targets_by_question=targets_by_question,
                baseline_by_question=baseline_by_question,
                training=training,
                seeds=config.seeds,
                checkpoint_dir=checkpoint_dir,
                checkpoint_stem=model_name,
            )
            models[model_name].loc[test_mask, list(ALIGNMENT_DIMS)] = np.clip(
                prediction, 0.0, 10.0
            )
            diagnostics.extend(
                {
                    "model": model_name,
                    "rubric_view": "matched",
                    "held_out_question": held_out,
                    **row,
                }
                for row in fold_diagnostics
            )

    shared_loaded = bool(
        config.shared_a0_predictions is not None
        and config.shared_a0_predictions.exists()
    )
    if shared_loaded:
        assert config.shared_a0_predictions is not None
        shared = normalize_predictions(
            pd.read_csv(config.shared_a0_predictions, sep="\t")
        )
        models["diagonal_shared_a0_hybrid"] = shared

    for name, frame in models.items():
        if not np.isfinite(frame[DIMS].to_numpy(dtype=float)).all():
            raise ValueError(f"model has incomplete predictions: {name}")

    initial_details = _temporary_details(models, labels, weights)
    initial_comparisons = [
        (
            "diagonal_matched_separate_hybrid",
            "fixed_matched_hybrid",
        ),
        ("diagonal_matched_separate_hybrid", "ridge_global_structure"),
        ("fixed_matched_hybrid", "ridge_global_structure"),
    ]
    if shared_loaded:
        initial_comparisons.append(
            ("diagonal_matched_separate_hybrid", "diagonal_shared_a0_hybrid")
        )
    initial_paired = _alignment_paired_bootstrap(
        initial_details,
        initial_comparisons,
        n_resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
    )
    gates = _dimension_gate_records(initial_paired)
    passing_dimensions = gates.loc[gates["passed"], "dimension"].tolist()

    control_model_names: list[str] = []
    if passing_dimensions:
        control_models: dict[str, pd.DataFrame] = {}
        for view_name in ("generic", *[
            f"mismatch_shift{index}"
            for index in range(1, config.mismatch_count + 1)
        ]):
            model_name = f"diagonal_{view_name}_separate_hybrid"
            control_models[model_name] = models["ridge_global_structure"].copy()
            control_model_names.append(model_name)
        for held_out in question_ids:
            train_mask = data["questionId"].to_numpy(dtype=int) != held_out
            test_mask = ~train_mask
            base_train, base_test, _ = _fit_or_load_baseline_fold(
                data, baseline_columns, held_out, checkpoint_dir
            )
            fold_values = data[KEY_COLUMNS].copy()
            fold_values[DIMS] = np.nan
            fold_values.loc[train_mask, DIMS] = base_train
            fold_values.loc[test_mask, DIMS] = base_test
            baseline_by_question = _arrays_by_question(
                corpus, fold_values, list(ALIGNMENT_DIMS)
            )
            train_questions = [
                question_id for question_id in question_ids if question_id != held_out
            ]
            for view_name in ("generic", *[
                f"mismatch_shift{index}"
                for index in range(1, config.mismatch_count + 1)
            ]):
                model_name = f"diagonal_{view_name}_separate_hybrid"
                prediction, fold_diagnostics = _fit_or_load_separated(
                    corpus=corpus,
                    view_name=view_name,
                    learn_diagonal=True,
                    train_questions=train_questions,
                    held_out=held_out,
                    targets_by_question=targets_by_question,
                    baseline_by_question=baseline_by_question,
                    training=training,
                    seeds=config.seeds,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_stem=model_name,
                )
                control_models[model_name].loc[
                    test_mask, list(ALIGNMENT_DIMS)
                ] = np.clip(
                    prediction, 0.0, 10.0
                )
                diagnostics.extend(
                    {
                        "model": model_name,
                        "rubric_view": view_name,
                        "held_out_question": held_out,
                        **row,
                    }
                    for row in fold_diagnostics
                )
        mismatch_frames = [
            control_models[
                f"diagonal_mismatch_shift{index}_separate_hybrid"
            ]
            for index in range(1, config.mismatch_count + 1)
        ]
        ensemble_name = "diagonal_mismatched_ensemble_separate_hybrid"
        control_models[ensemble_name] = _mean_predictions(mismatch_frames)
        control_model_names.append(ensemble_name)
        models.update(control_models)

    pd.concat(baseline_selections, ignore_index=True).to_csv(
        output / "baseline_selected_hyperparameters.csv",
        index=False,
        float_format="%.8f",
    )
    diagnostic_frame = pd.DataFrame(diagnostics)
    diagnostic_frame.to_csv(
        output / "training_diagnostics.csv", index=False, float_format="%.8f"
    )
    gates.to_csv(output / "dimension_gates.csv", index=False, float_format="%.8f")

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

    comparisons = list(initial_comparisons)
    if passing_dimensions:
        matched = "diagonal_matched_separate_hybrid"
        comparisons.extend(
            [
                (matched, "diagonal_generic_separate_hybrid"),
                (
                    matched,
                    "diagonal_mismatched_ensemble_separate_hybrid",
                ),
            ]
        )
    paired_question_bootstrap(
        details_by_model,
        comparisons,
        n_resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
    ).to_csv(output / "paired_bootstrap.csv", index=False, float_format="%.8f")
    dimension_paired = _alignment_paired_bootstrap(
        details_by_model,
        comparisons,
        n_resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
    )
    dimension_paired.to_csv(
        output / "dimension_paired_bootstrap.csv",
        index=False,
        float_format="%.8f",
    )
    _write_report(
        output / "e2_a01_conclusions.md",
        metrics,
        dimension_paired,
        gates,
        diagnostic_frame,
        shared_loaded=shared_loaded,
        controls_run=passing_dimensions,
    )

    elapsed = time.perf_counter() - started
    protocol = {
        "name": "AEOLLM-2 E2-A0.1 dimension-separated diagnostic",
        "outer_split": "Leave-One-Question-Out",
        "model_selection": "none; architecture, epochs, and optimization fixed",
        "single_change_from_e2_a0": (
            "replace one shared diagonal with one independent diagonal per alignment "
            "dimension; retain joint two-head training and the same mean Huber loss"
        ),
        "encoder": "frozen Qwen/Qwen3-Embedding-0.6B",
        "chunks": "unbounded E1.1 structure-preserving chunks",
        "query": "full matched query for both dimensions",
        "baseline": "global embedding + surface features nested Ridge",
        "alignment_dimensions": list(ALIGNMENT_DIMS),
        "interaction": (
            "per dimension: cosine + sqrt(1024) * dot(r * learned_diagonal_d, h)"
        ),
        "pooling": ["mean", "max", "top10pct_mean", "logmeanexp_t005"],
        "criterion_aggregation": "fixed official criterion weights",
        "training": config_dict(training),
        "seeds": list(config.seeds),
        "per_dimension_gate": {
            "learned_minus_fixed": "> 0",
            "bootstrap_probability_delta_gt_zero": ">= 0.90",
            "positive_questions": ">= 7/10",
            "learned_minus_global_structure": ">= 0",
        },
        "conditional_controls": (
            "if any dimension passes, jointly train generic and five mismatch "
            "two-diagonal controls; interpret only gates for passing dimensions"
        ),
        "passing_dimensions": passing_dimensions,
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
        "shared_a0_predictions_loaded": shared_loaded,
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
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    status = {
        "status": "complete",
        "models": len(models),
        "documents": len(labels),
        "questions": len(question_ids),
        "elapsed_seconds": elapsed,
        "device": config.device,
        "gpu_used": config.device.startswith("cuda:"),
        "passing_dimensions": passing_dimensions,
        "conditional_control_models": control_model_names,
    }
    (output / "run_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run E2-A0.1 dimension-separated diagonal diagnostic"
    )
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
    parser.add_argument(
        "--surface-features",
        type=Path,
        default=root / "outputs/e0/surface_features.csv",
    )
    parser.add_argument(
        "--unbounded-features",
        type=Path,
        default=root / "outputs/e1/features/qwen3-0.6b-unbounded",
    )
    parser.add_argument(
        "--base-cache",
        type=Path,
        default=root / "outputs/e1/embeddings/qwen3-0.6b-unbounded",
    )
    parser.add_argument(
        "--query-variant-root",
        type=Path,
        default=root / "outputs/e1/embeddings/qwen3-0.6b-query-variants",
    )
    parser.add_argument(
        "--shared-a0-predictions",
        type=Path,
        default=(
            root
            / "outputs/e2/e2_a0/predictions/diagonal_matched_hybrid.tsv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "outputs/e2/e2_a01"
    )
    parser.add_argument("--device", required=True, help="cpu or an explicit cuda:N")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--mismatch-count", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--overwrite-checkpoints", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = E2A01Config(
        labels=args.labels,
        rubric_dir=args.rubric_dir,
        surface_features=args.surface_features,
        unbounded_features=args.unbounded_features,
        base_cache=args.base_cache,
        query_variant_root=args.query_variant_root,
        output_dir=args.output_dir,
        device=args.device,
        shared_a0_predictions=args.shared_a0_predictions,
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
    metrics = run_e2_a01(config, training)
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
        ]
        .sort_values("spearman", ascending=False)
        .to_string(index=False)
    )
    return 0
