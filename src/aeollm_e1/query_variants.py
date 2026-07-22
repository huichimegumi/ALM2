from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .embedding import (
    DIMENSION_ORDER,
    EmbeddingConfig,
    TextEmbedder,
    _normalized,
    _sha256_text,
    criterion_text,
    detailed_query,
    load_chunk_records,
    sha256_path,
)

QUERY_VARIANTS = (
    "criterion_only",
    "prompt_only",
    "generic_dimension",
    "matched_full_no_instruction",
)

GENERIC_DIMENSION_TEXT = {
    "comprehensiveness": "Evaluate the report's comprehensiveness and coverage of the requested material.",
    "insight": "Evaluate the report's insight, analytical depth, and quality of reasoning.",
    "instruction_following": "Evaluate whether the report follows the task instructions and requirements.",
    "readability": "Evaluate the report's readability, organization, and clarity.",
}


def render_query_variant(
    variant: str,
    *,
    prompt: str,
    dimension: str,
    criterion: dict[str, object],
    query_instruction: str,
) -> str:
    name = str(criterion.get("criterion", "")).strip()
    explanation = str(criterion.get("explanation", "")).strip()
    if not name or not explanation:
        raise ValueError("each criterion needs non-empty criterion and explanation fields")
    if variant == "criterion_only":
        raw = f"Evaluation criterion: {name}\nExplanation: {explanation}"
        return detailed_query(query_instruction, raw)
    if variant == "prompt_only":
        return detailed_query(query_instruction, f"Task: {prompt.strip()}")
    if variant == "generic_dimension":
        return detailed_query(query_instruction, GENERIC_DIMENSION_TEXT[dimension])
    if variant == "matched_full_no_instruction":
        return criterion_text(prompt, criterion)
    raise ValueError(f"unknown query variant: {variant}")


def collect_variant_queries(
    records: Sequence[dict[str, object]],
    variant: str,
    config: EmbeddingConfig,
) -> tuple[list[str], list[dict[str, object]]]:
    if variant not in QUERY_VARIANTS:
        raise ValueError(f"variant must be one of {QUERY_VARIANTS}")
    texts: list[str] = []
    index: list[dict[str, object]] = []
    seen_questions: dict[int, str] = {}
    for record in records:
        question_id = int(record["question_id"])
        rubric_hash = str(record.get("rubric_sha256", ""))
        if question_id in seen_questions:
            if seen_questions[question_id] != rubric_hash:
                raise ValueError(f"question {question_id} has inconsistent rubrics")
            continue
        seen_questions[question_id] = rubric_hash
        prompt = str(record.get("prompt", ""))
        rubric = record["rubric"]  # type: ignore[index]
        criterions = rubric["criterions"]  # type: ignore[index]
        for dimension in DIMENSION_ORDER:
            for position, criterion in enumerate(criterions.get(dimension, [])):  # type: ignore[union-attr]
                query = render_query_variant(
                    variant,
                    prompt=prompt,
                    dimension=dimension,
                    criterion=criterion,
                    query_instruction=config.query_instruction,
                )
                row = len(texts)
                texts.append(query)
                index.append(
                    {
                        "embedding_row": row,
                        "question_id": question_id,
                        "dimension": dimension,
                        "criterion_index": position,
                        "criterion": str(criterion["criterion"]),
                        "weight": float(criterion["weight"]),
                        "query_variant": variant,
                        "query_sha256": _sha256_text(query),
                    }
                )
    return texts, index


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_query_variant_caches(
    input_path: Path,
    output_root: Path,
    embedder: TextEmbedder,
    config: EmbeddingConfig,
    *,
    variants: Sequence[str] = QUERY_VARIANTS,
    overwrite: bool = False,
    runtime_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    config.validate()
    records = load_chunk_records(input_path)
    summaries: dict[str, object] = {}
    for variant in variants:
        if variant not in QUERY_VARIANTS:
            raise ValueError(f"unknown query variant: {variant}")
        output_dir = output_root / variant
        embedding_path = output_dir / "criterion_embeddings.npy"
        index_path = output_dir / "criterion_index.jsonl"
        manifest_path = output_dir / "variant_manifest.json"
        existing = [path for path in (embedding_path, index_path, manifest_path) if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(f"query variant outputs already exist: {[str(path) for path in existing]}")
        output_dir.mkdir(parents=True, exist_ok=True)
        texts, index = collect_variant_queries(records, variant, config)
        lengths = embedder.token_lengths(texts)
        too_long = [position for position, length in enumerate(lengths) if length > config.max_length]
        if too_long:
            raise ValueError(
                f"refusing silent truncation for {variant}: {len(too_long)} queries exceed "
                f"max_length={config.max_length}; max_query_tokens={max(lengths)}"
            )
        embeddings = embedder.encode(texts, is_query=True)
        if embeddings.shape != (len(texts), embedder.dimension) or not _normalized(embeddings):
            raise ValueError(f"invalid query embeddings for {variant}: {embeddings.shape}")
        dtype = np.float16 if config.output_dtype == "float16" else np.float32
        np.save(embedding_path, embeddings.astype(dtype))
        _write_jsonl(index_path, index)
        manifest = {
            "status": "complete",
            "variant": variant,
            "input_path": str(input_path.resolve()),
            "input_sha256": sha256_path(input_path),
            "model_name": embedder.model_name,
            "model_revision": embedder.model_revision,
            "embedding_dimension": embedder.dimension,
            "normalized": True,
            "output_dtype": config.output_dtype,
            "questions": len({int(row["question_id"]) for row in index}),
            "criteria": len(index),
            "unique_query_texts": len(set(texts)),
            "max_query_model_tokens": max(lengths),
            "truncated_inputs": 0,
            "config": asdict(config),
            "runtime": runtime_metadata or {},
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summaries[variant] = manifest
    root_manifest = {
        "status": "complete",
        "variants": list(variants),
        "model_name": embedder.model_name,
        "model_revision": embedder.model_revision,
        "runtime": runtime_metadata or {},
        "variant_summaries": summaries,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "query_variant_manifest.json").write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return root_manifest
