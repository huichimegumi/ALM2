from __future__ import annotations

import numpy as np
import pytest
import torch

from aeollm_e1.diagonal_interaction import (
    CriterionChunkInteraction,
    DiagonalTrainingConfig,
    DimensionSeparatedCriterionChunkInteraction,
    InteractionCorpus,
    QuestionChunks,
    RubricTensor,
    aggregate_criterion_evidence,
    cyclic_rubric_maps,
    pool_criterion_chunk_scores,
    train_interaction_fold_ensemble,
    train_separated_interaction_fold_ensemble,
)


def _unit(values: list[list[float]]) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float32)
    return torch.nn.functional.normalize(tensor, p=2, dim=1)


def test_zero_diagonal_is_exactly_fixed_cosine() -> None:
    criteria = _unit([[1, 0, 0], [0, 1, 0]])
    chunks = _unit([[1, 1, 0], [0, 1, 1]])
    fixed = CriterionChunkInteraction(3, learn_diagonal=False)
    learned = CriterionChunkInteraction(3, learn_diagonal=True)
    assert torch.equal(learned.diagonal, torch.zeros(3))
    assert torch.allclose(fixed.score_pairs(criteria, chunks), learned.score_pairs(criteria, chunks))


def test_pooling_and_official_weight_aggregation_are_deterministic() -> None:
    scores = torch.tensor([[0.1, 0.2, 0.9], [0.4, 0.5, 0.6], [0.0, 0.2, 0.4]])
    pooled = pool_criterion_chunk_scores(scores, logmeanexp_temperature=0.05)
    assert pooled.shape == (3, 4)
    assert pooled[0, 0].item() == pytest.approx(0.4)
    assert pooled[0, 1].item() == pytest.approx(0.9)
    assert pooled[0, 2].item() == pytest.approx(0.9)
    aggregated = aggregate_criterion_evidence(
        pooled,
        torch.tensor([0, 0, 1]),
        torch.tensor([1.0, 3.0, 2.0]),
    )
    assert aggregated.shape == (2, 4)
    assert torch.allclose(aggregated[0], (pooled[0] + 3 * pooled[1]) / 4)
    assert torch.allclose(aggregated[1], pooled[2])


def test_joint_separated_model_uses_dimension_specific_diagonals() -> None:
    criteria = _unit([[1, 0, 0], [1, 0, 0]])
    chunks = _unit([[1, 0, 0], [0, 1, 0]])
    dimensions = torch.tensor([0, 1])
    model = DimensionSeparatedCriterionChunkInteraction(3)
    fixed = criteria @ chunks.T
    assert torch.allclose(model.score_pairs(criteria, chunks, dimensions), fixed)
    with torch.no_grad():
        model.diagonal[0, 0] = 1.0
    scores = model.score_pairs(criteria, chunks, dimensions)
    assert scores[0, 0] > fixed[0, 0]
    assert torch.equal(scores[1], fixed[1])


def _tiny_corpus() -> InteractionCorpus:
    chunks = {}
    rubrics = {}
    for question_id in (1, 2):
        chunks[question_id] = QuestionChunks(
            question_id=question_id,
            document_ids=(f"Q{question_id}A", f"Q{question_id}B"),
            embeddings=_unit([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]),
            spans=((0, 2), (2, 4)),
        )
        rubrics[question_id] = RubricTensor(
            target_question_id=question_id,
            source_question_id=question_id,
            embeddings=_unit([[1, 0, 0], [0, 1, 0]]),
            dimension_indices=torch.tensor([0, 1]),
            weights=torch.tensor([1.0, 1.0]),
        )
    return InteractionCorpus(chunks=chunks, rubric_views={"matched": rubrics}, embedding_dimension=3)


def test_tiny_fold_training_returns_only_held_out_documents() -> None:
    corpus = _tiny_corpus()
    targets = {
        1: np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        2: np.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    }
    baseline = {
        1: np.asarray([[2.5, 3.5], [3.5, 4.5]], dtype=np.float32),
        2: np.asarray([[3.5, 4.5], [4.5, 5.5]], dtype=np.float32),
    }
    prediction, diagnostics = train_interaction_fold_ensemble(
        corpus,
        corpus.rubric_views["matched"],
        train_questions=(1,),
        test_question=2,
        targets_by_question=targets,
        baseline_by_question=baseline,
        config=DiagonalTrainingConfig(epochs=2),
        seeds=(7, 8),
        learn_diagonal=True,
    )
    assert prediction.shape == (2, 2)
    assert np.isfinite(prediction).all()
    assert len(diagnostics) == 2
    assert all(row["trainable_parameters"] == 13 for row in diagnostics)

    fixed_prediction, fixed_diagnostics = train_interaction_fold_ensemble(
        corpus,
        corpus.rubric_views["matched"],
        train_questions=(1,),
        test_question=2,
        targets_by_question=targets,
        baseline_by_question=baseline,
        config=DiagonalTrainingConfig(epochs=2),
        seeds=(7,),
        learn_diagonal=False,
    )
    assert fixed_prediction.shape == (2, 2)
    assert fixed_diagnostics[0]["trainable_parameters"] == 10


def test_joint_separated_training_preserves_two_head_output() -> None:
    corpus = _tiny_corpus()
    targets = {
        1: np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        2: np.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    }
    baseline = {
        1: np.asarray([[2.5, 3.5], [3.5, 4.5]], dtype=np.float32),
        2: np.asarray([[3.5, 4.5], [4.5, 5.5]], dtype=np.float32),
    }
    prediction, diagnostics = train_separated_interaction_fold_ensemble(
        corpus,
        corpus.rubric_views["matched"],
        train_questions=(1,),
        test_question=2,
        targets_by_question=targets,
        baseline_by_question=baseline,
        config=DiagonalTrainingConfig(epochs=2),
        seeds=(7,),
        learn_diagonal=True,
    )
    assert prediction.shape == (2, 2)
    assert diagnostics[0]["trainable_parameters"] == 16
    assert "diagonal_l2_norm_comprehensiveness" in diagnostics[0]
    assert "diagonal_l2_norm_instruction_following" in diagnostics[0]
    separated_fixed, _ = train_separated_interaction_fold_ensemble(
        corpus,
        corpus.rubric_views["matched"],
        train_questions=(1,),
        test_question=2,
        targets_by_question=targets,
        baseline_by_question=baseline,
        config=DiagonalTrainingConfig(epochs=2),
        seeds=(7,),
        learn_diagonal=False,
    )
    original_fixed, _ = train_interaction_fold_ensemble(
        corpus,
        corpus.rubric_views["matched"],
        train_questions=(1,),
        test_question=2,
        targets_by_question=targets,
        baseline_by_question=baseline,
        config=DiagonalTrainingConfig(epochs=2),
        seeds=(7,),
        learn_diagonal=False,
    )
    assert np.array_equal(separated_fixed, original_fixed)


def test_cyclic_rubric_maps_never_self_match() -> None:
    maps = cyclic_rubric_maps([1, 2, 3, 4], 2)
    assert len(maps) == 2
    for mapping in maps:
        assert set(mapping.values()) == {1, 2, 3, 4}
        assert all(target != source for target, source in mapping.items())
