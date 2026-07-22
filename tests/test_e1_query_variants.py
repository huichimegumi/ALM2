from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from aeollm_e1.e1_6_pipeline import cyclic_derangements
from aeollm_e1.embedding import EmbeddingConfig
from aeollm_e1.query_variants import (
    QUERY_VARIANTS,
    build_query_variant_caches,
    collect_variant_queries,
    render_query_variant,
)


class FakeEmbedder:
    model_name = "fake/frozen"
    model_revision = "revision"
    dimension = 4

    def token_lengths(self, texts: list[str]) -> list[int]:
        return [len(text.split()) for text in texts]

    def encode(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        rows = []
        for text in texts:
            digest = hashlib.sha256((str(is_query) + text).encode()).digest()[: self.dimension]
            vector = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) + 1
            rows.append(vector / np.linalg.norm(vector))
        return np.stack(rows)


def records() -> list[dict[str, object]]:
    criterion = {"criterion": "Coverage", "explanation": "Cover every topic", "weight": 1}
    criterions = {
        dimension: [criterion]
        for dimension in (
            "comprehensiveness",
            "insight",
            "instruction_following",
            "readability",
        )
    }
    base = {
        "question_id": 1,
        "prompt": "Research the topic",
        "rubric_sha256": "same",
        "rubric": {"criterions": criterions},
    }
    chunk = {"chunk_id": 0, "type": "paragraph", "text": "Body", "token_count": 1}
    return [
        {**base, "document_id": "A", "chunks": [chunk]},
        {**base, "document_id": "B", "chunks": [chunk]},
    ]


def test_query_variants_are_distinct_and_deduplicated_by_question() -> None:
    config = EmbeddingConfig()
    rendered = {
        variant: render_query_variant(
            variant,
            prompt="Prompt",
            dimension="comprehensiveness",
            criterion={"criterion": "Name", "explanation": "Details"},
            query_instruction=config.query_instruction,
        )
        for variant in QUERY_VARIANTS
    }
    assert len(set(rendered.values())) == len(QUERY_VARIANTS)
    assert not rendered["matched_full_no_instruction"].startswith("Instruct:")
    texts, index = collect_variant_queries(records(), "prompt_only", config)
    assert len(texts) == len(index) == 4
    assert len(set(texts)) == 1


def test_variant_builder_writes_criterion_only_caches(tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records()), encoding="utf-8"
    )
    manifest = build_query_variant_caches(
        input_path,
        tmp_path / "variants",
        FakeEmbedder(),
        EmbeddingConfig(max_length=100),
        variants=("criterion_only", "generic_dimension"),
    )
    assert manifest["variants"] == ["criterion_only", "generic_dimension"]
    assert np.load(tmp_path / "variants/criterion_only/criterion_embeddings.npy").shape == (4, 4)
    assert not (tmp_path / "variants/criterion_only/chunk_embeddings.npy").exists()


def test_cyclic_derangements_are_balanced_and_never_self_match() -> None:
    question_ids = [1, 2, 3, 4, 5, 6]
    mappings = cyclic_derangements(question_ids, 3)
    assert len(mappings) == 3
    for mapping in mappings:
        assert set(mapping) == set(question_ids)
        assert set(mapping.values()) == set(question_ids)
        assert all(target != source for target, source in mapping.items())
