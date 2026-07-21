from __future__ import annotations

from pathlib import Path

import pytest

from aeollm_e0.data import extract_docx_features, load_mapping

ROOT = Path(__file__).resolve().parents[1]


def test_mapping_reader_recovers_generation_metadata() -> None:
    path = ROOT / "legacy/aeollm2_train_code/prompts/mapping_key_Readability.xlsx"
    if not path.exists():
        pytest.skip("legacy mapping is not present")
    mapping = load_mapping(path)
    row = mapping[(mapping["questionId"] == 1) & (mapping["answerId"] == "Doc_001")].iloc[0]
    assert int(row["model_id"]) == 3
    assert int(row["prompt_variant"]) == 1


def test_docx_surface_features_include_tables_and_text() -> None:
    path = ROOT / "data/incoming/google-drive/train/Report1/Doc_001.docx"
    if not path.exists():
        pytest.skip("training DOCX files are not present")
    features = extract_docx_features(path)
    assert features["character_count"] > 0
    assert features["paragraph_count"] > 0
    assert features["table_count"] >= 0
