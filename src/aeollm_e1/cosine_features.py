from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from aeollm_e0.metrics import DIMS, KEY_COLUMNS

PRIMARY_SIMILARITY_STATS = (
    "sim_max",
    "sim_mean",
    "sim_std",
    "sim_q50",
    "sim_q75",
    "sim_q90",
    "sim_q95",
    "sim_top10pct_mean",
    "sim_top25pct_mean",
    "sim_logmeanexp_t005",
)
DIAGNOSTIC_SIMILARITY_STATS = ("sim_top3_mean", "sim_top5_mean")
ALL_SIMILARITY_STATS = (*PRIMARY_SIMILARITY_STATS, *DIAGNOSTIC_SIMILARITY_STATS)
CRITERION_AGGREGATIONS = ("wmean", "min", "std")


@dataclass(frozen=True)
class CacheBundle:
    directory: Path
    manifest: dict[str, object]
    chunk_embeddings: np.ndarray
    criterion_embeddings: np.ndarray
    chunk_index: pd.DataFrame
    criterion_index: pd.DataFrame


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty JSONL index: {path}")
    return pd.DataFrame(rows)


def _check_rows(frame: pd.DataFrame, expected: int, name: str) -> None:
    rows = frame["embedding_row"].to_numpy(dtype=int)
    if len(frame) != expected:
        raise ValueError(f"{name} index has {len(frame)} rows but array has {expected}")
    if not np.array_equal(rows, np.arange(expected)):
        raise ValueError(f"{name} embedding_row must be contiguous and array-aligned")


def _check_normalized(matrix: np.ndarray, name: str) -> None:
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError(f"{name} embeddings must be a non-empty matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} embeddings contain NaN or infinity")
    norms = np.linalg.norm(matrix.astype(np.float32), axis=1)
    if float(np.max(np.abs(norms - 1.0))) > 5e-3:
        raise ValueError(f"{name} embeddings are not L2-normalized")


def load_cache(directory: Path) -> CacheBundle:
    manifest = json.loads((directory / "embedding_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or int(manifest.get("truncated_inputs", -1)) != 0:
        raise ValueError("embedding cache is incomplete or contains truncated inputs")
    chunks = np.load(directory / "chunk_embeddings.npy", mmap_mode="r")
    criteria = np.load(directory / "criterion_embeddings.npy", mmap_mode="r")
    chunk_index = _read_jsonl(directory / "chunk_index.jsonl")
    criterion_index = _read_jsonl(directory / "criterion_index.jsonl")
    _check_rows(chunk_index, len(chunks), "chunk")
    _check_rows(criterion_index, len(criteria), "criterion")
    _check_normalized(chunks, "chunk")
    _check_normalized(criteria, "criterion")
    if chunks.shape[1] != criteria.shape[1]:
        raise ValueError("chunk and criterion embedding dimensions differ")
    if chunk_index.duplicated(["question_id", "document_id", "chunk_id"]).any():
        raise ValueError("duplicate chunk keys")
    if criterion_index.duplicated(["question_id", "dimension", "criterion_index"]).any():
        raise ValueError("duplicate criterion keys")
    if set(criterion_index["dimension"]) != set(DIMS):
        raise ValueError("criterion index does not contain exactly the four dimensions")
    return CacheBundle(directory, manifest, chunks, criteria, chunk_index, criterion_index)


def _top_fraction_mean(values: np.ndarray, fraction: float) -> float:
    count = max(1, math.ceil(len(values) * fraction))
    return float(np.mean(np.partition(values, len(values) - count)[-count:]))


def _top_k_mean(values: np.ndarray, k: int) -> float:
    count = min(k, len(values))
    return float(np.mean(np.partition(values, len(values) - count)[-count:]))


def similarity_statistics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("similarities must be a finite, non-empty vector")
    temperature = 0.05
    scaled = values / temperature
    stable_max = float(np.max(scaled))
    normalized_lme = temperature * (
        stable_max + math.log(float(np.mean(np.exp(scaled - stable_max))))
    )
    return {
        "sim_max": float(np.max(values)),
        "sim_mean": float(np.mean(values)),
        "sim_std": float(np.std(values, ddof=0)),
        "sim_q50": float(np.quantile(values, 0.50)),
        "sim_q75": float(np.quantile(values, 0.75)),
        "sim_q90": float(np.quantile(values, 0.90)),
        "sim_q95": float(np.quantile(values, 0.95)),
        "sim_top10pct_mean": _top_fraction_mean(values, 0.10),
        "sim_top25pct_mean": _top_fraction_mean(values, 0.25),
        "sim_logmeanexp_t005": float(normalized_lme),
        "sim_top3_mean": _top_k_mean(values, 3),
        "sim_top5_mean": _top_k_mean(values, 5),
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0:
        weights = np.ones_like(weights)
        total = float(len(weights))
    return float(np.sum(values * weights) / total)


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mean = _weighted_mean(values, weights)
    total = float(np.sum(weights))
    if total <= 0:
        weights = np.ones_like(weights)
        total = float(len(weights))
    return float(np.sqrt(np.sum(weights * (values - mean) ** 2) / total))


def _normalized_centroid(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights))
    if total <= 0:
        weights = np.ones_like(weights)
        total = float(len(weights))
    centroid = np.sum(matrix.astype(np.float32) * (weights / total)[:, None], axis=0)
    norm = float(np.linalg.norm(centroid))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("cannot normalize a zero or non-finite centroid")
    return centroid / norm


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_cosine_features(
    cache_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    cache = load_cache(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    document_path = output_dir / "document_features.csv"
    criterion_path = output_dir / "criterion_chunk_features.csv"
    manifest_path = output_dir / "feature_manifest.json"
    existing = [path for path in (document_path, criterion_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"feature outputs already exist: {[str(path) for path in existing]}")

    criterion_by_question = {
        int(question_id): group.sort_values("embedding_row").reset_index(drop=True)
        for question_id, group in cache.criterion_index.groupby("question_id", sort=True)
    }
    criterion_rows: list[dict[str, object]] = []
    document_rows: list[dict[str, object]] = []
    global_columns = [f"global_{index:04d}" for index in range(cache.chunk_embeddings.shape[1])]

    grouped_chunks = cache.chunk_index.groupby(["question_id", "document_id"], sort=True)
    for (question_id_raw, document_id_raw), chunks in grouped_chunks:
        question_id = int(question_id_raw)
        document_id = str(document_id_raw)
        chunks = chunks.sort_values("chunk_id").reset_index(drop=True)
        chunk_rows = chunks["embedding_row"].to_numpy(dtype=int)
        chunk_matrix = np.asarray(cache.chunk_embeddings[chunk_rows], dtype=np.float32)
        chunk_weights = chunks["structural_token_count"].to_numpy(dtype=np.float32)
        chunk_weights = np.maximum(chunk_weights, 1.0)
        global_embedding = _normalized_centroid(chunk_matrix, chunk_weights)

        criteria = criterion_by_question.get(question_id)
        if criteria is None or criteria.empty:
            raise ValueError(f"no criteria for question {question_id}")
        criterion_embedding_rows = criteria["embedding_row"].to_numpy(dtype=int)
        criterion_matrix = np.asarray(
            cache.criterion_embeddings[criterion_embedding_rows], dtype=np.float32
        )
        similarities = criterion_matrix @ chunk_matrix.T

        document_row: dict[str, object] = {
            "questionId": question_id,
            "answerId": document_id,
            "chunk_count": int(len(chunks)),
            "log1p_chunk_count": float(np.log1p(len(chunks))),
            "structural_token_count": int(chunks["structural_token_count"].sum()),
            "mean_chunk_tokens": float(chunks["structural_token_count"].mean()),
            "std_chunk_tokens": float(chunks["structural_token_count"].std(ddof=0)),
        }
        type_counts = chunks["type"].value_counts()
        for chunk_type in ("heading", "paragraph", "list", "table", "mixed"):
            document_row[f"chunk_fraction_{chunk_type}"] = float(
                type_counts.get(chunk_type, 0) / len(chunks)
            )
        document_row.update(zip(global_columns, global_embedding.tolist()))

        for criterion_position, criterion in criteria.iterrows():
            stats = similarity_statistics(similarities[criterion_position])
            top_local = int(np.argmax(similarities[criterion_position]))
            top_chunk = chunks.iloc[top_local]
            criterion_rows.append(
                {
                    "questionId": question_id,
                    "answerId": document_id,
                    "dimension": str(criterion["dimension"]),
                    "criterion_index": int(criterion["criterion_index"]),
                    "criterion": str(criterion["criterion"]),
                    "criterion_weight": float(criterion["weight"]),
                    "chunk_count": int(len(chunks)),
                    "top_chunk_id": int(top_chunk["chunk_id"]),
                    "top_chunk_type": str(top_chunk["type"]),
                    "top_chunk_embedding_row": int(top_chunk["embedding_row"]),
                    **stats,
                }
            )

        document_criterion_rows = criterion_rows[-len(criteria) :]
        criterion_frame = pd.DataFrame(document_criterion_rows)
        for dimension in DIMS:
            dim_values = criterion_frame[criterion_frame["dimension"] == dimension]
            dim_criteria = criteria[criteria["dimension"] == dimension]
            if dim_values.empty or len(dim_values) != len(dim_criteria):
                raise ValueError(f"criterion aggregation mismatch for question {question_id}, {dimension}")
            weights = dim_values["criterion_weight"].to_numpy(dtype=np.float32)
            for stat in ALL_SIMILARITY_STATS:
                values = dim_values[stat].to_numpy(dtype=np.float32)
                prefix = f"rubric_{dimension}_{stat}"
                document_row[f"{prefix}_wmean"] = _weighted_mean(values, weights)
                document_row[f"{prefix}_min"] = float(np.min(values))
                document_row[f"{prefix}_std"] = _weighted_std(values, weights)
            dim_embedding_rows = dim_criteria["embedding_row"].to_numpy(dtype=int)
            dim_matrix = np.asarray(
                cache.criterion_embeddings[dim_embedding_rows], dtype=np.float32
            )
            rubric_centroid = _normalized_centroid(dim_matrix, weights)
            document_row[f"rubric_{dimension}_global_cosine"] = float(
                rubric_centroid @ global_embedding
            )
            document_row[f"rubric_{dimension}_criterion_count"] = int(len(dim_values))
        document_rows.append(document_row)

    document_features = pd.DataFrame(document_rows).sort_values(KEY_COLUMNS).reset_index(drop=True)
    criterion_features = pd.DataFrame(criterion_rows).sort_values(
        [*KEY_COLUMNS, "dimension", "criterion_index"]
    ).reset_index(drop=True)
    if document_features.duplicated(KEY_COLUMNS).any():
        raise ValueError("duplicate document feature keys")
    numeric = document_features.drop(columns=KEY_COLUMNS).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("document features contain NaN or infinity")
    document_features.to_csv(document_path, index=False, float_format="%.8g")
    criterion_features.to_csv(criterion_path, index=False, float_format="%.8g")

    primary_rubric_columns = [
        f"rubric_{dimension}_{stat}_{aggregation}"
        for dimension in DIMS
        for stat in PRIMARY_SIMILARITY_STATS
        for aggregation in CRITERION_AGGREGATIONS
    ] + [f"rubric_{dimension}_global_cosine" for dimension in DIMS]
    diagnostic_rubric_columns = [
        f"rubric_{dimension}_{stat}_{aggregation}"
        for dimension in DIMS
        for stat in DIAGNOSTIC_SIMILARITY_STATS
        for aggregation in CRITERION_AGGREGATIONS
    ]
    structure_columns = [
        "chunk_count",
        "log1p_chunk_count",
        "structural_token_count",
        "mean_chunk_tokens",
        "std_chunk_tokens",
        "chunk_fraction_heading",
        "chunk_fraction_paragraph",
        "chunk_fraction_list",
        "chunk_fraction_table",
        "chunk_fraction_mixed",
    ]
    manifest: dict[str, object] = {
        "status": "complete",
        "source_cache": str(cache_dir.resolve()),
        "source_embedding_manifest": cache.manifest,
        "documents": int(len(document_features)),
        "questions": int(document_features["questionId"].nunique()),
        "criterion_document_rows": int(len(criterion_features)),
        "embedding_dimension": int(cache.chunk_embeddings.shape[1]),
        "feature_columns": int(len(document_features.columns) - len(KEY_COLUMNS)),
        "feature_groups": {
            "global": global_columns,
            "rubric_primary": primary_rubric_columns,
            "rubric_fixed_topk_diagnostic": diagnostic_rubric_columns,
            "structure_diagnostic": structure_columns,
            "criterion_count_diagnostic": [
                f"rubric_{dimension}_criterion_count" for dimension in DIMS
            ],
        },
        "definitions": {
            "global": "L2-normalized structural-token-weighted mean of chunk embeddings",
            "cosine": "dot product of L2-normalized criterion and chunk embeddings",
            "top_fraction": "ceil(fraction * chunk_count), minimum one chunk",
            "logmeanexp": "0.05 * log(mean(exp(similarity / 0.05)))",
            "criterion_aggregation": "official criterion-weighted mean, minimum, and weighted std",
        },
        "gpu_used": False,
    }
    _write_json(manifest_path, manifest)
    return manifest
