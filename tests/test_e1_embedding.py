from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from aeollm_e1.embedding import (
    EmbeddingConfig,
    build_embedding_cache,
    collect_embedding_inputs,
    criterion_text,
    detailed_query,
    parse_cuda_index,
)


class FakeEmbedder:
    model_name = "fake/frozen"
    model_revision = "test-revision"
    dimension = 8

    def token_lengths(self, texts: list[str]) -> list[int]:
        return [len(text.split()) + 2 for text in texts]

    def encode(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        rows = []
        for text in texts:
            digest = hashlib.sha256((str(is_query) + text).encode()).digest()[: self.dimension]
            vector = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) + 1.0
            rows.append(vector / np.linalg.norm(vector))
        return np.stack(rows)


def example_records() -> list[dict[str, object]]:
    criterion = {"criterion": "Coverage", "explanation": "Covers all requested topics", "weight": 1.0}
    criterions = {
        "comprehensiveness": [criterion],
        "insight": [criterion],
        "instruction_following": [criterion],
        "readability": [criterion],
    }
    base = {
        "question_id": 1,
        "prompt": "Research task",
        "rubric_sha256": "rubric-hash",
        "rubric": {"criterions": criterions},
    }
    return [
        {
            **base,
            "document_id": "Doc_1",
            "chunks": [
                {"chunk_id": 0, "type": "heading", "text": "Title", "token_count": 1, "source_block_ids": [0]},
                {"chunk_id": 1, "type": "paragraph", "text": "Body text", "token_count": 2, "source_block_ids": [1]},
            ],
        },
        {
            **base,
            "document_id": "Doc_2",
            "chunks": [
                {"chunk_id": 0, "type": "paragraph", "text": "Other body", "token_count": 2, "source_block_ids": [0]}
            ],
        },
    ]


def test_criteria_are_deduplicated_per_question_and_include_task_context() -> None:
    config = EmbeddingConfig()
    chunks, chunk_index, criteria, criterion_index = collect_embedding_inputs(example_records(), config)
    assert len(chunks) == len(chunk_index) == 3
    assert len(criteria) == len(criterion_index) == 4
    assert criteria[0].startswith("Instruct:")
    assert "Task: Research task" in criteria[0]
    assert [row["dimension"] for row in criterion_index] == [
        "comprehensiveness",
        "insight",
        "instruction_following",
        "readability",
    ]


def test_build_cache_writes_normalized_arrays_and_auditable_indices(tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in example_records()), encoding="utf-8"
    )
    output = tmp_path / "cache"
    manifest = build_embedding_cache(
        input_path, output, FakeEmbedder(), EmbeddingConfig(max_length=100), runtime_metadata={"device": "cpu"}
    )
    chunks = np.load(output / "chunk_embeddings.npy")
    criteria = np.load(output / "criterion_embeddings.npy")
    assert chunks.shape == (3, 8)
    assert criteria.shape == (4, 8)
    assert np.allclose(np.linalg.norm(chunks, axis=1), 1.0)
    assert manifest["truncated_inputs"] == 0
    assert manifest["runtime"] == {"device": "cpu"}
    assert len((output / "chunk_index.jsonl").read_text().splitlines()) == 3
    assert len((output / "criterion_index.jsonl").read_text().splitlines()) == 4


def test_cache_refuses_silent_truncation(tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    input_path.write_text(json.dumps(example_records()[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_chunk_tokens=.*worst_chunks"):
        build_embedding_cache(input_path, tmp_path / "cache", FakeEmbedder(), EmbeddingConfig(max_length=2))


def test_query_format_and_device_selection_are_explicit() -> None:
    assert detailed_query("retrieve passages", "criterion") == "Instruct: retrieve passages\nQuery:criterion"
    assert "Evaluation criterion: Name" in criterion_text(
        "Prompt", {"criterion": "Name", "explanation": "Details"}
    )
    assert parse_cuda_index("cpu") is None
    assert parse_cuda_index("cuda:3") == 3
    with pytest.raises(ValueError, match="explicit cuda:N"):
        parse_cuda_index("cuda")
