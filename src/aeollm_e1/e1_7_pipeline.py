from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from aeollm_e0.data import sha256_file
from aeollm_e0.metrics import (
    DIMS,
    KEY_COLUMNS,
    load_dimension_weights,
    normalize_labels,
    pairwise_accuracy,
)
from aeollm_e0.statistics import paired_question_bootstrap, question_bootstrap

from .e1_4_pipeline import _evaluate
from .e1_6_pipeline import _alignment_paired_bootstrap
from .ridge_scoring import (
    RIDGE_ALPHAS,
    _ridge_pipeline,
    build_model_feature_columns,
    load_feature_groups,
    numeric_surface_columns,
)

ALIGNMENT_DIMS = ("comprehensiveness", "instruction_following")
QUERY_SOURCES = ("matched_full", "criterion_only")


@dataclass(frozen=True)
class E17Config:
    labels: Path
    rubric_dir: Path
    surface_features: Path
    matched_features: Path
    criterion_only_features: Path
    output_dir: Path
    bootstrap_resamples: int = 5000
    seed: int = 20260721


def _merge_sources(e1: pd.DataFrame, surface: pd.DataFrame) -> pd.DataFrame:
    merged = e1.merge(surface, on=KEY_COLUMNS, validate="one_to_one", suffixes=("", "_surface"))
    if len(merged) != len(e1) or len(merged) != len(surface):
        raise ValueError("E1 and surface feature keys do not match exactly")
    return merged.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _validate_query_frames(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("at least one query feature frame is required")
    ordered: pd.DataFrame | None = None
    for query, frame in frames.items():
        if frame.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"duplicate feature keys for query {query}")
        keys = frame.sort_values(KEY_COLUMNS).reset_index(drop=True)[KEY_COLUMNS]
        if ordered is None:
            ordered = keys
        elif not keys.equals(ordered):
            raise ValueError(f"query feature keys are not aligned: {query}")
    assert ordered is not None
    return ordered


def _inner_candidate_scores(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    alphas: Sequence[float],
) -> list[tuple[float, float, float, float]]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("inner grouped validation needs at least two questions")
    splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
    scores: list[tuple[float, float, float, float]] = []
    for alpha in alphas:
        out_of_fold = np.full(len(y), np.nan, dtype=float)
        for train_index, validation_index in splitter.split(x, y, groups):
            model = _ridge_pipeline(float(alpha)).fit(x.iloc[train_index], y[train_index])
            out_of_fold[validation_index] = np.asarray(
                model.predict(x.iloc[validation_index]), dtype=float
            )
        correct = 0
        total = 0
        correlations: list[float] = []
        for question_id in unique_groups:
            mask = groups == question_id
            _, group_correct, group_total = pairwise_accuracy(y[mask], out_of_fold[mask])
            correct += group_correct
            total += group_total
            if np.ptp(y[mask]) > 0 and np.ptp(out_of_fold[mask]) > 0:
                correlations.append(float(spearmanr(y[mask], out_of_fold[mask])[0]))
        accuracy = float(correct / total) if total else float("-inf")
        spearman = float(np.mean(correlations)) if correlations else float("-inf")
        mae = float(np.mean(np.abs(out_of_fold - y)))
        scores.append((float(alpha), accuracy, spearman, mae))
    return scores


def nested_loqo_query_ridge_predictions(
    frames_by_query: Mapping[str, pd.DataFrame],
    labels: pd.DataFrame,
    candidate_queries_by_dimension: Mapping[str, Sequence[str]],
    feature_columns_by_query_dimension: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    alphas: Sequence[float] = RIDGE_ALPHAS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Jointly select query representation and Ridge alpha without outer-fold leakage."""
    if not alphas or any(float(alpha) <= 0 for alpha in alphas):
        raise ValueError("all Ridge alphas must be positive")
    keys = _validate_query_frames(frames_by_query)
    truth = labels.sort_values(KEY_COLUMNS).reset_index(drop=True)
    if not truth[KEY_COLUMNS].equals(keys):
        raise ValueError("feature and label keys do not match exactly")
    missing_dimensions = sorted(set(DIMS) - set(candidate_queries_by_dimension))
    if missing_dimensions:
        raise ValueError(f"missing query candidates for dimensions: {missing_dimensions}")

    frames = {
        query: frame.sort_values(KEY_COLUMNS).reset_index(drop=True)
        for query, frame in frames_by_query.items()
    }
    result = keys.copy()
    for dimension in DIMS:
        result[dimension] = np.nan
    selections: list[dict[str, object]] = []
    candidate_records: list[dict[str, object]] = []

    for held_out in sorted(truth["questionId"].unique()):
        train_mask = truth["questionId"].to_numpy(dtype=int) != int(held_out)
        test_mask = ~train_mask
        groups = truth.loc[train_mask, "questionId"].to_numpy(dtype=int)
        for dimension in DIMS:
            queries = list(candidate_queries_by_dimension[dimension])
            if not queries:
                raise ValueError(f"no query candidates for {dimension}")
            y_train = truth.loc[train_mask, dimension].to_numpy(dtype=float)
            ranked_candidates: list[
                tuple[float, float, float, int, int, str, float, int]
            ] = []
            for query_order, query in enumerate(queries):
                if query not in frames:
                    raise ValueError(f"unknown query candidate: {query}")
                try:
                    columns = list(feature_columns_by_query_dimension[query][dimension])
                except KeyError as error:
                    raise ValueError(f"missing features for {query}/{dimension}") from error
                missing = sorted(set(columns) - set(frames[query].columns))
                if missing:
                    raise ValueError(f"missing columns for {query}/{dimension}: {missing[:10]}")
                x_train = frames[query].loc[train_mask, columns]
                for alpha_order, (
                    alpha,
                    inner_accuracy,
                    inner_spearman,
                    inner_mae,
                ) in enumerate(
                    _inner_candidate_scores(x_train, y_train, groups, alphas=alphas)
                ):
                    ranked_candidates.append(
                        (
                            inner_accuracy,
                            inner_spearman,
                            -inner_mae,
                            -query_order,
                            -alpha_order,
                            query,
                            alpha,
                            len(columns),
                        )
                    )
                    candidate_records.append(
                        {
                            "held_out_question": int(held_out),
                            "dimension": dimension,
                            "query_source": query,
                            "alpha": alpha,
                            "inner_accuracy": inner_accuracy,
                            "inner_spearman": inner_spearman,
                            "inner_mae": inner_mae,
                            "feature_count": len(columns),
                        }
                    )
            (
                inner_accuracy,
                inner_spearman,
                negative_inner_mae,
                _,
                _,
                selected_query,
                alpha,
                feature_count,
            ) = max(ranked_candidates)
            inner_mae = -negative_inner_mae
            columns = list(feature_columns_by_query_dimension[selected_query][dimension])
            model = _ridge_pipeline(alpha).fit(
                frames[selected_query].loc[train_mask, columns], y_train
            )
            result.loc[test_mask, dimension] = np.clip(
                model.predict(frames[selected_query].loc[test_mask, columns]), 0.0, 10.0
            )
            selections.append(
                {
                    "held_out_question": int(held_out),
                    "dimension": dimension,
                    "query_source": selected_query,
                    "alpha": alpha,
                    "inner_accuracy": inner_accuracy,
                    "inner_spearman": inner_spearman,
                    "inner_mae": inner_mae,
                    "feature_count": feature_count,
                    "candidate_queries": "|".join(queries),
                    "train_documents": int(train_mask.sum()),
                    "test_documents": int(test_mask.sum()),
                }
            )
    if not np.isfinite(result[DIMS].to_numpy(dtype=float)).all():
        raise ValueError("nested LOQO produced non-finite predictions")
    return result, pd.DataFrame(selections), pd.DataFrame(candidate_records)


def _query_policy(
    comprehensiveness: Sequence[str],
    instruction_following: Sequence[str],
) -> dict[str, list[str]]:
    return {
        "comprehensiveness": list(comprehensiveness),
        "insight": ["matched_full"],
        "instruction_following": list(instruction_following),
        "readability": ["matched_full"],
    }


def _write_report(
    path: Path,
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    alignment_paired: pd.DataFrame,
    selections: pd.DataFrame,
) -> None:
    ranked = metrics.sort_values(["accuracy", "spearman"], ascending=False)
    primary = paired[paired["metric"] == "accuracy"]
    indexed = primary.set_index(["candidate", "reference"])
    diagnostics = alignment_paired[
        alignment_paired["metric"].isin(
            ["alignment_accuracy", "alignment_spearman"]
        )
    ]
    nested_vs_base = indexed.loc[("nested_query_selective", "global_structure")]
    fixed_vs_base = indexed.loc[("fixed_dimension_selective", "global_structure")]
    nested_vs_fixed = indexed.loc[
        ("nested_query_selective", "fixed_dimension_selective")
    ]
    nested_rows = selections[selections["model"] == "nested_query_selective"]
    query_counts = (
        nested_rows[nested_rows["dimension"].isin(ALIGNMENT_DIMS)]
        .groupby(["dimension", "query_source"], sort=True)
        .size()
        .rename("outer_folds_selected")
        .reset_index()
    )
    pass_gate = (
        float(nested_vs_base["mean_delta"]) > 0
        and float(nested_vs_base["probability_delta_gt_zero"]) >= 0.90
        and int(nested_vs_base["net_correct_pairs"]) > 0
    )
    lines = [
        "# E1.7 selective query Ridge",
        "",
        "E1.7 keeps the rich E1.3 criterion–chunk cosine summaries and asks whether",
        "rubric evidence should be routed only to comprehensiveness and instruction",
        "following. Insight and readability always use global + structure features.",
        "",
        "## Main results",
        "",
        ranked[
            [
                "model",
                "accuracy",
                "spearman",
                "kendall",
                "accuracy_comprehensiveness",
                "accuracy_instruction_following",
                "mae",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Primary official-accuracy comparisons",
        "",
        primary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Dimension-mechanism diagnostics",
        "",
        diagnostics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Nested query choices",
        "",
        query_counts.to_markdown(index=False) if len(query_counts) else "No choices.",
        "",
        "## Decision",
        "",
        f"- Nested selective minus global + structure official accuracy: "
        f"{float(nested_vs_base['mean_delta']):+.4f} "
        f"(95% CI [{float(nested_vs_base['ci_low']):.4f}, "
        f"{float(nested_vs_base['ci_high']):.4f}], "
        f"P(delta > 0)={float(nested_vs_base['probability_delta_gt_zero']):.4f}, "
        f"{int(nested_vs_base['net_correct_pairs'])} net correct pairs).",
        f"- Fixed dimension policy minus global + structure: "
        f"{float(fixed_vs_base['mean_delta']):+.4f}; nested minus fixed: "
        f"{float(nested_vs_fixed['mean_delta']):+.4f}.",
        f"- E1.7 official-accuracy gate: **{'PASS' if pass_gate else 'FAIL'}**.",
        *(
            [
                "- This pass supports selective fixed-representation rubric routing; it does not",
                "  establish criterion satisfaction or justify a higher-capacity interaction model.",
            ]
            if pass_gate
            else [
                "- This fail means query routing did not generalize reliably under the",
                "  ten-question outer LOQO protocol.",
            ]
        ),
        "- The fixed dimension policy is an exploratory, researcher-informed analysis",
        "  motivated by E1.6; its stronger total score is not fresh confirmatory evidence.",
        "- The nested policy records its fold-level query choices above.",
        "  Query-policy differences should be interpreted through paired outer-fold",
        "  accuracy rather than by selecting the highest observed point estimate.",
        "",
        "## Leakage controls",
        "",
        "- Query source and Ridge alpha are selected jointly inside each outer training set.",
        "- Prospective inner validation maximizes grouped pairwise accuracy, then",
        "  Spearman, then minimizes MAE. A retrospective note means saved predictions",
        "  retain their historical selection objective.",
        "- The held-out question is never used for query selection, scaling, or fitting.",
        "- Query candidates are ordered `matched_full`, then `criterion_only`; that order",
        "  is used only as a deterministic exact-tie break.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_e1_7(config: E17Config) -> pd.DataFrame:
    output = config.output_dir
    prediction_dir = output / "predictions"
    checkpoint_dir = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    selection_marker = checkpoint_dir / "selection_objective.json"
    selection_protocol = {
        "primary": "grouped_pairwise_accuracy",
        "tie_breaks": ["grouped_spearman", "mae", "declared_candidate_order"],
    }
    if any(checkpoint_dir.glob("*.csv")) and not selection_marker.exists():
        raise ValueError(
            "existing E1.7 checkpoints use the historical selection protocol; "
            "run the accuracy-first protocol with a new --output-dir"
        )
    if selection_marker.exists():
        observed = json.loads(selection_marker.read_text(encoding="utf-8"))
        if observed != selection_protocol:
            raise ValueError("E1.7 checkpoint selection protocol does not match")
    else:
        selection_marker.write_text(
            json.dumps(selection_protocol, indent=2) + "\n", encoding="utf-8"
        )

    labels = normalize_labels(pd.read_csv(config.labels))
    question_ids = sorted(int(value) for value in labels["questionId"].unique())
    weights = load_dimension_weights(config.rubric_dir, question_ids)
    surface = pd.read_csv(config.surface_features)
    surface_columns = numeric_surface_columns(surface)
    if not surface_columns:
        raise ValueError("no numeric E0 surface features")

    frames: dict[str, pd.DataFrame] = {}
    columns: dict[str, dict[str, dict[str, list[str]]]] = {}
    for query, directory in (
        ("matched_full", config.matched_features),
        ("criterion_only", config.criterion_only_features),
    ):
        e1, groups = load_feature_groups(directory)
        frames[query] = _merge_sources(e1, surface)
        columns[query] = build_model_feature_columns(groups, surface_columns)
    _validate_query_frames(frames)

    feature_maps = {
        query: {
            dimension: columns[query][dimension][
                "global_structure"
                if dimension not in ALIGNMENT_DIMS
                else "all"
            ]
            for dimension in DIMS
        }
        for query in QUERY_SOURCES
    }
    baseline_maps = {
        "matched_full": {
            dimension: columns["matched_full"][dimension]["global_structure"]
            for dimension in DIMS
        },
        "criterion_only": {
            dimension: columns["criterion_only"][dimension]["global_structure"]
            for dimension in DIMS
        },
    }

    policies = {
        "global_structure": (
            _query_policy(["matched_full"], ["matched_full"]),
            baseline_maps,
        ),
        "matched_full_selective": (
            _query_policy(["matched_full"], ["matched_full"]),
            feature_maps,
        ),
        "criterion_only_selective": (
            _query_policy(["criterion_only"], ["criterion_only"]),
            feature_maps,
        ),
        "fixed_dimension_selective": (
            _query_policy(["criterion_only"], ["matched_full"]),
            feature_maps,
        ),
        "nested_query_selective": (
            _query_policy(QUERY_SOURCES, QUERY_SOURCES),
            feature_maps,
        ),
    }

    models: dict[str, pd.DataFrame] = {}
    selection_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    for name, (policy, model_feature_maps) in policies.items():
        prediction_checkpoint = checkpoint_dir / f"{name}.csv"
        selection_checkpoint = checkpoint_dir / f"{name}_selected.csv"
        candidate_checkpoint = checkpoint_dir / f"{name}_inner_candidates.csv"
        if (
            prediction_checkpoint.exists()
            and selection_checkpoint.exists()
            and candidate_checkpoint.exists()
        ):
            prediction = pd.read_csv(prediction_checkpoint)
            selections = pd.read_csv(selection_checkpoint)
            candidates = pd.read_csv(candidate_checkpoint)
        else:
            prediction, selections, candidates = nested_loqo_query_ridge_predictions(
                frames,
                labels,
                policy,
                model_feature_maps,
            )
            prediction.to_csv(prediction_checkpoint, index=False, float_format="%.8f")
            selections.to_csv(selection_checkpoint, index=False, float_format="%.8f")
            candidates.to_csv(candidate_checkpoint, index=False, float_format="%.8f")
        models[name] = prediction
        selections.insert(0, "model", name)
        candidates.insert(0, "model", name)
        selection_frames.append(selections)
        candidate_frames.append(candidates)

    all_selections = pd.concat(selection_frames, ignore_index=True)
    all_selections.to_csv(
        output / "selected_query_and_hyperparameters.csv", index=False, float_format="%.8f"
    )
    pd.concat(candidate_frames, ignore_index=True).to_csv(
        output / "inner_candidate_scores.csv", index=False, float_format="%.8f"
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
    question_bootstrap(
        details_by_model, n_resamples=config.bootstrap_resamples, seed=config.seed
    ).to_csv(output / "bootstrap_ci.csv", index=False, float_format="%.8f")

    comparisons = [
        ("matched_full_selective", "global_structure"),
        ("criterion_only_selective", "global_structure"),
        ("fixed_dimension_selective", "global_structure"),
        ("nested_query_selective", "global_structure"),
        ("fixed_dimension_selective", "matched_full_selective"),
        ("nested_query_selective", "fixed_dimension_selective"),
        ("nested_query_selective", "matched_full_selective"),
    ]
    paired = paired_question_bootstrap(
        details_by_model,
        comparisons,
        n_resamples=config.bootstrap_resamples,
        seed=config.seed,
    )
    paired.to_csv(output / "paired_bootstrap.csv", index=False, float_format="%.8f")
    alignment_paired = _alignment_paired_bootstrap(
        details_by_model,
        comparisons,
        n_resamples=config.bootstrap_resamples,
        seed=config.seed,
    )
    alignment_paired.to_csv(
        output / "alignment_paired_bootstrap.csv", index=False, float_format="%.8f"
    )
    _write_report(
        output / "e1_7_conclusions.md",
        metrics,
        paired,
        alignment_paired,
        all_selections,
    )

    protocol = {
        "name": "AEOLLM-2 E1.7 selective query Ridge",
        "outer_split": "Leave-One-Question-Out",
        "inner_selection": (
            "joint query-source and Ridge-alpha selection using up to 5-fold "
            "GroupKFold by question; maximize pairwise accuracy, then Spearman, "
            "then minimize MAE"
        ),
        "query_candidates": list(QUERY_SOURCES),
        "primary_metric": "official weighted-total pairwise accuracy",
        "secondary_metrics": ["Spearman", "Kendall"],
        "query_candidate_tie_break_order": list(QUERY_SOURCES),
        "ridge_alphas": list(RIDGE_ALPHAS),
        "alignment_dimensions": list(ALIGNMENT_DIMS),
        "non_alignment_policy": "global + structure only",
        "fixed_dimension_policy": {
            "comprehensiveness": "criterion_only",
            "instruction_following": "matched_full",
            "status": "exploratory and researcher-informed by E1.6",
        },
        "primary_gate": {
            "comparison": "nested_query_selective > global_structure",
            "metric": "official weighted-total pairwise accuracy",
            "mean_delta": "> 0",
            "bootstrap_probability_delta_gt_zero": ">= 0.90",
            "net_correct_pairs": "> 0",
        },
        "bootstrap_unit": "question",
        "bootstrap_resamples": config.bootstrap_resamples,
        "seed": config.seed,
        "gpu_required": False,
        "gpu_note": "scikit-learn Ridge is CPU-only; this dataset is too small for transfer to help",
        "paths": {
            key: str(value.resolve())
            for key, value in asdict(config).items()
            if isinstance(value, Path)
        },
        "input_hashes": {
            "labels": sha256_file(config.labels),
            "surface_features": sha256_file(config.surface_features),
            "matched_manifest": sha256_file(config.matched_features / "feature_manifest.json"),
            "criterion_only_manifest": sha256_file(
                config.criterion_only_features / "feature_manifest.json"
            ),
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
    parser = argparse.ArgumentParser(description="Run E1.7 selective query Ridge")
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
        "--surface-features", type=Path, default=root / "outputs/e0/surface_features.csv"
    )
    parser.add_argument(
        "--matched-features",
        type=Path,
        default=root / "outputs/e1/features/qwen3-0.6b-unbounded",
    )
    parser.add_argument(
        "--criterion-only-features",
        type=Path,
        default=root / "outputs/e1/e1_6/control_features/criterion_only",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "outputs/e1/e1_7_accuracy"
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = E17Config(
        labels=args.labels,
        rubric_dir=args.rubric_dir,
        surface_features=args.surface_features,
        matched_features=args.matched_features,
        criterion_only_features=args.criterion_only_features,
        output_dir=args.output_dir,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    metrics = run_e1_7(config)
    print(
        metrics.sort_values(["accuracy", "spearman"], ascending=False).to_json(
            orient="records", indent=2
        )
    )
    return 0
