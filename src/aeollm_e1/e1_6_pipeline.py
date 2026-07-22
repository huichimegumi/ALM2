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
from aeollm_e0.metrics import DIMS, KEY_COLUMNS, load_dimension_weights, normalize_labels
from aeollm_e0.statistics import paired_question_bootstrap, question_bootstrap

from .cosine_features import build_cosine_features
from .e1_4_pipeline import _evaluate
from .ridge_scoring import (
    RIDGE_ALPHAS,
    build_model_feature_columns,
    load_feature_groups,
    nested_loqo_ridge_predictions,
    numeric_surface_columns,
)

QUERY_CONTROL_VARIANTS = (
    "criterion_only",
    "prompt_only",
    "generic_dimension",
    "matched_full_no_instruction",
)
FACTORIAL_FEATURE_SETS = (
    "structure",
    "global",
    "rubric",
    "global_structure",
    "global_rubric",
    "rubric_structure",
    "all",
)
CONTROL_FEATURE_SETS = ("rubric", "all")


@dataclass(frozen=True)
class E16Config:
    labels: Path
    rubric_dir: Path
    surface_features: Path
    base_cache: Path
    matched_features: Path
    query_variant_root: Path
    output_dir: Path
    mismatch_count: int = 5
    bootstrap_resamples: int = 5000
    seed: int = 20260721
    overwrite_control_features: bool = False


def cyclic_derangements(question_ids: list[int], count: int) -> list[dict[int, int]]:
    if len(question_ids) < 2:
        raise ValueError("mismatched-rubric controls need at least two questions")
    if count <= 0 or count >= len(question_ids):
        raise ValueError("mismatch_count must be between 1 and n_questions - 1")
    ordered = sorted(question_ids)
    return [
        {question_id: ordered[(index + shift) % len(ordered)] for index, question_id in enumerate(ordered)}
        for shift in range(1, count + 1)
    ]


def _ensure_control_features(
    config: E16Config,
    question_ids: list[int],
) -> tuple[dict[str, Path], list[dict[int, int]]]:
    feature_root = config.output_dir / "control_features"
    paths: dict[str, Path] = {"matched_full": config.matched_features}
    identity = {question_id: question_id for question_id in question_ids}
    for variant in QUERY_CONTROL_VARIANTS:
        path = feature_root / variant
        manifest_path = path / "feature_manifest.json"
        if config.overwrite_control_features or not manifest_path.exists():
            build_cosine_features(
                config.base_cache,
                path,
                criterion_cache_dir=config.query_variant_root / variant,
                rubric_question_map=identity,
                overwrite=config.overwrite_control_features,
            )
        paths[variant] = path
    mappings = cyclic_derangements(question_ids, config.mismatch_count)
    for index, mapping in enumerate(mappings, start=1):
        name = f"mismatch_shift{index}"
        path = feature_root / name
        manifest_path = path / "feature_manifest.json"
        if config.overwrite_control_features or not manifest_path.exists():
            build_cosine_features(
                config.base_cache,
                path,
                rubric_question_map=mapping,
                overwrite=config.overwrite_control_features,
            )
        paths[name] = path
    return paths, mappings


def _merge_sources(e1: pd.DataFrame, surface: pd.DataFrame) -> pd.DataFrame:
    merged = e1.merge(surface, on=KEY_COLUMNS, validate="one_to_one", suffixes=("", "_surface"))
    if len(merged) != len(e1) or len(merged) != len(surface):
        raise ValueError("E1 and surface feature keys do not match exactly")
    return merged


def _alignment_paired_bootstrap(
    details_by_model: dict[str, pd.DataFrame],
    comparisons: list[tuple[str, str]],
    *,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    metric_columns = {
        "alignment_spearman": (
            "spearman_comprehensiveness",
            "spearman_instruction_following",
        ),
        "spearman_comprehensiveness": ("spearman_comprehensiveness",),
        "spearman_instruction_following": ("spearman_instruction_following",),
    }
    for candidate, reference in comparisons:
        left = details_by_model[candidate].set_index("reportId")
        right = details_by_model[reference].set_index("reportId")
        common = left.index.intersection(right.index)
        for metric, columns in metric_columns.items():
            left_values = left.loc[common, list(columns)].mean(axis=1).to_numpy(dtype=float)
            right_values = right.loc[common, list(columns)].mean(axis=1).to_numpy(dtype=float)
            deltas = left_values - right_values
            finite = np.isfinite(deltas)
            deltas = deltas[finite]
            if not len(deltas):
                continue
            samples = np.empty(n_resamples, dtype=float)
            for position in range(n_resamples):
                selected = rng.integers(0, len(deltas), size=len(deltas))
                samples[position] = float(deltas[selected].mean())
            rows.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    "mean_delta": float(deltas.mean()),
                    "ci_low": float(np.quantile(samples, 0.025)),
                    "ci_high": float(np.quantile(samples, 0.975)),
                    "probability_delta_gt_zero": float(np.mean(samples > 0)),
                    "positive_questions": int(np.sum(deltas > 0)),
                    "tied_questions": int(np.sum(deltas == 0)),
                    "n_questions": int(len(deltas)),
                }
            )
    return pd.DataFrame(rows)


def _mean_predictions(predictions: list[pd.DataFrame]) -> pd.DataFrame:
    if not predictions:
        raise ValueError("cannot average an empty prediction list")
    ordered = [frame.sort_values(KEY_COLUMNS).reset_index(drop=True) for frame in predictions]
    keys = ordered[0][KEY_COLUMNS]
    for frame in ordered[1:]:
        if not frame[KEY_COLUMNS].equals(keys):
            raise ValueError("mismatch prediction keys are not aligned")
    result = keys.copy()
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
    indexed = metrics.set_index("model")
    align = alignment_paired[alignment_paired["metric"] == "alignment_spearman"]
    align_indexed = align.set_index(["candidate", "reference"])
    incremental = align_indexed.loc[("matched_full_all", "matched_full_global_structure")]
    mismatch = align_indexed.loc[("matched_full_rubric", "mismatched_ensemble_rubric")]
    generic = align_indexed.loc[("matched_full_rubric", "generic_dimension_rubric")]
    total_delta = float(indexed.loc["matched_full_all", "spearman"]) - float(
        indexed.loc["matched_full_global_structure", "spearman"]
    )
    lines = [
        "# E1.6 rubric attribution controls",
        "",
        "All learned predictions use outer leave-one-question-out Ridge. Query controls",
        "change only the rubric query text; mismatch controls use five fixed cyclic",
        "derangements and never pair a report with its own question rubric.",
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
        align.to_markdown(index=False, floatfmt=".4f") if len(align) else "No comparisons.",
        "",
        "## Decision",
        "",
        f"- Matched rubric adds {float(incremental['mean_delta']):+.4f} alignment Spearman over",
        "  `global+structure` "
        f"(95% question-bootstrap CI [{float(incremental['ci_low']):.4f}, "
        f"{float(incremental['ci_high']):.4f}]; "
        f"{int(incremental['positive_questions'])}/10 questions positive).",
        f"- The corresponding official weighted-total Spearman delta is {total_delta:+.4f};",
        "  rubric gains on comprehensiveness and instruction following are offset by the",
        "  dimensions for which criterion retrieval is not the appropriate inductive bias.",
        f"- Matched rubric beats the mismatched ensemble by {float(mismatch['mean_delta']):+.4f}",
        f"  alignment Spearman and generic dimensions by {float(generic['mean_delta']):+.4f}.",
        "- `criterion_only_all` has the best official total, while the full query is stronger",
        "  than criterion-only for instruction following. Prompt context is therefore useful",
        "  selectively rather than uniformly across all four heads.",
        "- E1.6 supports proceeding to the minimal E2-A0 learned interaction, focused on",
        "  comprehensiveness and instruction following. It does not support feeding the same",
        "  rubric branch indiscriminately to all four output heads.",
        "",
        "## Interpretation rules",
        "",
        "- `matched_full_all > matched_full_global_structure` tests incremental rubric value.",
        "- Matched versus generic or mismatched tests criterion-specific conditioning.",
        "- `alignment_spearman` is the mean of comprehensiveness and instruction-following",
        "  Spearman within each question, followed by a question-level macro average.",
        "- `mismatched_ensemble_*` averages predictions from five independently retrained",
        "  mismatch controls and is therefore a conservative negative control.",
        "- The held-out question is never used for scaling, alpha selection, or fitting.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_e1_6(config: E16Config) -> pd.DataFrame:
    output = config.output_dir
    prediction_dir = output / "predictions"
    checkpoint_dir = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    labels = normalize_labels(pd.read_csv(config.labels))
    question_ids = sorted(int(value) for value in labels["questionId"].unique())
    weights = load_dimension_weights(config.rubric_dir, question_ids)
    surface = pd.read_csv(config.surface_features)
    surface_columns = numeric_surface_columns(surface)
    feature_paths, mismatch_maps = _ensure_control_features(config, question_ids)

    feature_data: dict[str, tuple[pd.DataFrame, dict[str, dict[str, list[str]]]]] = {}
    for name, directory in feature_paths.items():
        e1, groups = load_feature_groups(directory)
        frame = _merge_sources(e1, surface)
        feature_data[name] = (frame, build_model_feature_columns(groups, surface_columns))

    models: dict[str, pd.DataFrame] = {}
    selection_frames: list[pd.DataFrame] = []

    def fit(name: str, source: str, feature_set: str) -> None:
        prediction_checkpoint = checkpoint_dir / f"{name}.csv"
        selection_checkpoint = checkpoint_dir / f"{name}_selected.csv"
        if prediction_checkpoint.exists() and selection_checkpoint.exists():
            models[name] = pd.read_csv(prediction_checkpoint)
            selection_frames.append(pd.read_csv(selection_checkpoint))
            return
        frame, mappings = feature_data[source]
        mapping = {dimension: mappings[dimension][feature_set] for dimension in DIMS}
        prediction, selections = nested_loqo_ridge_predictions(frame, labels, mapping)
        models[name] = prediction
        selections.insert(0, "model", name)
        selections.insert(1, "query_source", source)
        selections.insert(2, "feature_set", feature_set)
        selection_frames.append(selections)
        prediction.to_csv(prediction_checkpoint, index=False, float_format="%.8f")
        selections.to_csv(selection_checkpoint, index=False, float_format="%.8f")

    for feature_set in FACTORIAL_FEATURE_SETS:
        fit(f"matched_full_{feature_set}", "matched_full", feature_set)
    for variant in QUERY_CONTROL_VARIANTS:
        for feature_set in CONTROL_FEATURE_SETS:
            fit(f"{variant}_{feature_set}", variant, feature_set)
    mismatch_predictions: dict[str, list[pd.DataFrame]] = {name: [] for name in CONTROL_FEATURE_SETS}
    for index in range(1, config.mismatch_count + 1):
        source = f"mismatch_shift{index}"
        for feature_set in CONTROL_FEATURE_SETS:
            name = f"{source}_{feature_set}"
            fit(name, source, feature_set)
            mismatch_predictions[feature_set].append(models[name])
    for feature_set, frames in mismatch_predictions.items():
        models[f"mismatched_ensemble_{feature_set}"] = _mean_predictions(frames)

    pd.concat(selection_frames, ignore_index=True).to_csv(
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
    question_bootstrap(
        details_by_model, n_resamples=config.bootstrap_resamples, seed=config.seed
    ).to_csv(output / "bootstrap_ci.csv", index=False, float_format="%.8f")

    comparisons = [
        ("matched_full_all", "matched_full_global_structure"),
        ("matched_full_global_rubric", "matched_full_global"),
        ("matched_full_rubric", "criterion_only_rubric"),
        ("matched_full_rubric", "prompt_only_rubric"),
        ("matched_full_rubric", "generic_dimension_rubric"),
        ("matched_full_rubric", "matched_full_no_instruction_rubric"),
        ("matched_full_rubric", "mismatched_ensemble_rubric"),
        ("matched_full_all", "generic_dimension_all"),
        ("matched_full_all", "mismatched_ensemble_all"),
    ]
    paired_question_bootstrap(
        details_by_model,
        comparisons,
        n_resamples=config.bootstrap_resamples,
        seed=config.seed,
    ).to_csv(output / "paired_bootstrap.csv", index=False, float_format="%.8f")
    alignment_paired = _alignment_paired_bootstrap(
        details_by_model,
        comparisons,
        n_resamples=config.bootstrap_resamples,
        seed=config.seed,
    )
    alignment_paired.to_csv(
        output / "alignment_paired_bootstrap.csv", index=False, float_format="%.8f"
    )
    _write_report(output / "e1_6_conclusions.md", metrics, alignment_paired)

    protocol = {
        "name": "AEOLLM-2 E1.6 rubric attribution controls",
        "outer_split": "Leave-One-Question-Out",
        "inner_selection": "up to 5-fold GroupKFold by question, minimum MAE",
        "ridge_alphas": list(RIDGE_ALPHAS),
        "query_variants": list(QUERY_CONTROL_VARIANTS),
        "factorial_feature_sets": list(FACTORIAL_FEATURE_SETS),
        "mismatch_strategy": "five fixed cyclic question derangements",
        "mismatch_maps": [
            {str(key): int(value) for key, value in sorted(mapping.items())}
            for mapping in mismatch_maps
        ],
        "alignment_primary": "mean question-level Spearman over comprehensiveness and instruction_following",
        "bootstrap_unit": "question",
        "bootstrap_resamples": config.bootstrap_resamples,
        "seed": config.seed,
        "gpu_required": False,
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
        "mismatch_controls": config.mismatch_count,
        "gpu_used": False,
    }
    (output / "run_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run E1.6 rubric attribution controls")
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
        "--base-cache", type=Path, default=root / "outputs/e1/embeddings/qwen3-0.6b-unbounded"
    )
    parser.add_argument(
        "--matched-features", type=Path, default=root / "outputs/e1/features/qwen3-0.6b-unbounded"
    )
    parser.add_argument(
        "--query-variant-root",
        type=Path,
        default=root / "outputs/e1/embeddings/qwen3-0.6b-query-variants",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/e1/e1_6")
    parser.add_argument("--mismatch-count", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--overwrite-control-features", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = E16Config(
        labels=args.labels,
        rubric_dir=args.rubric_dir,
        surface_features=args.surface_features,
        base_cache=args.base_cache,
        matched_features=args.matched_features,
        query_variant_root=args.query_variant_root,
        output_dir=args.output_dir,
        mismatch_count=args.mismatch_count,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
        overwrite_control_features=args.overwrite_control_features,
    )
    metrics = run_e1_6(config)
    print(metrics.sort_values("spearman", ascending=False).to_json(orient="records", indent=2))
    return 0
