from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aeollm_e0.metrics import DIMS
from aeollm_e1.cosine_features import build_cosine_features, similarity_statistics


def _unit(values: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_cache(path: Path) -> None:
    path.mkdir()
    chunks = _unit([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
    criteria = _unit([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
    np.save(path / "chunk_embeddings.npy", chunks)
    np.save(path / "criterion_embeddings.npy", criteria)
    _write_jsonl(
        path / "chunk_index.jsonl",
        [
            {"embedding_row": 0, "question_id": 1, "document_id": "A", "chunk_id": 0, "type": "heading", "structural_token_count": 1},
            {"embedding_row": 1, "question_id": 1, "document_id": "A", "chunk_id": 1, "type": "paragraph", "structural_token_count": 3},
            {"embedding_row": 2, "question_id": 1, "document_id": "B", "chunk_id": 0, "type": "paragraph", "structural_token_count": 2},
            {"embedding_row": 3, "question_id": 1, "document_id": "B", "chunk_id": 1, "type": "table", "structural_token_count": 2},
        ],
    )
    criterion_rows = []
    for row, dimension in enumerate(DIMS):
        criterion_rows.append(
            {"embedding_row": row, "question_id": 1, "dimension": dimension, "criterion_index": 0, "criterion": dimension, "weight": 1.0}
        )
    _write_jsonl(path / "criterion_index.jsonl", criterion_rows)
    (path / "embedding_manifest.json").write_text(
        json.dumps({"status": "complete", "truncated_inputs": 0, "model_name": "fake"}),
        encoding="utf-8",
    )


def test_similarity_statistics_are_count_normalized() -> None:
    short = similarity_statistics(np.array([0.1, 0.9], dtype=np.float32))
    repeated = similarity_statistics(np.array([0.1, 0.9, 0.1, 0.9], dtype=np.float32))
    assert short["sim_mean"] == pytest.approx(repeated["sim_mean"])
    assert short["sim_logmeanexp_t005"] == pytest.approx(repeated["sim_logmeanexp_t005"])
    assert short["sim_top25pct_mean"] == pytest.approx(0.9)


def test_build_features_produces_fixed_document_columns_and_long_audit_table(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    make_cache(cache)
    output = tmp_path / "features"
    manifest = build_cosine_features(cache, output)
    documents = pd.read_csv(output / "document_features.csv")
    criteria = pd.read_csv(output / "criterion_chunk_features.csv")
    assert len(documents) == 2
    assert len(criteria) == 8
    assert manifest["documents"] == 2
    assert manifest["criterion_document_rows"] == 8
    assert len(manifest["feature_groups"]["global"]) == 3
    assert documents["rubric_comprehensiveness_sim_max_wmean"].iloc[0] == pytest.approx(1.0)
    assert set(criteria["top_chunk_id"]) <= {0, 1}
    assert np.isfinite(documents.drop(columns=["questionId", "answerId"]).to_numpy()).all()


def test_existing_outputs_require_explicit_overwrite(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    make_cache(cache)
    output = tmp_path / "features"
    build_cosine_features(cache, output)
    with pytest.raises(FileExistsError):
        build_cosine_features(cache, output)
