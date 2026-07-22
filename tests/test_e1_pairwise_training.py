from __future__ import annotations

import numpy as np
import pytest
import torch

from aeollm_e1.pairwise_training import (
    MLPTrainingConfig,
    build_within_question_pairs,
    fit_fold_preprocessor,
    pairwise_logistic_loss,
    train_fold_ensemble,
)


def test_pairs_are_only_within_question_and_ignore_near_ties() -> None:
    questions = np.array([1, 1, 2, 2])
    targets = np.array(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 2.05, 1.0, 5.0], [9.0, 8.0, 7.0, 6.0], [8.0, 7.0, 6.0, 5.0]],
        dtype=np.float32,
    )
    pairs = build_within_question_pairs(
        questions, targets, tie_threshold=0.1, max_pair_weight=2.0
    )
    assert pairs.count == 7
    assert all(questions[left] == questions[right] for left, right in zip(pairs.left, pairs.right))
    assert float(pairs.weight.mean()) == pytest.approx(1.0)


def test_pairwise_loss_rewards_correct_ordering() -> None:
    questions = np.array([1, 1])
    targets = np.array([[2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    pairs = build_within_question_pairs(
        questions, targets, tie_threshold=0.1, max_pair_weight=2.0
    )
    correct = torch.tensor([[2.0] * 4, [1.0] * 4])
    reversed_order = torch.tensor([[1.0] * 4, [2.0] * 4])
    assert pairwise_logistic_loss(correct, pairs, temperature=1.0) < pairwise_logistic_loss(
        reversed_order, pairs, temperature=1.0
    )


def test_fold_training_returns_seed_ensemble_predictions() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(12, 6)).astype(np.float32)
    targets = np.stack([features[:, 0] + index for index in range(4)], axis=1).astype(np.float32)
    questions = np.repeat(np.arange(3), 4)
    config = MLPTrainingConfig(epochs=3, hidden_size=8, bottleneck_size=4, dropout=0.0)
    prediction, diagnostics = train_fold_ensemble(
        features[:8],
        features[8:],
        targets[:8],
        questions[:8],
        config=config,
        seeds=(1, 2),
        use_pairwise=True,
    )
    assert prediction.shape == (4, 4)
    assert len(diagnostics) == 2
    assert all(row["pair_count"] > 0 for row in diagnostics)
    assert np.isfinite(prediction).all()


def test_high_dimensional_preprocessing_fits_pca_on_training_rows_only() -> None:
    rng = np.random.default_rng(11)
    train = rng.normal(size=(20, 40))
    config = MLPTrainingConfig(pca_threshold=30, pca_components=8)
    preprocessor = fit_fold_preprocessor(train, config, seed=3)
    assert preprocessor.pca is not None
    assert preprocessor.transform(train).shape == (20, 8)
