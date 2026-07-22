from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MLPTrainingConfig:
    hidden_size: int = 64
    bottleneck_size: int = 16
    dropout: float = 0.20
    epochs: int = 150
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    huber_beta: float = 1.0
    pair_lambda: float = 0.5
    pair_temperature: float = 1.0
    tie_threshold: float = 0.1
    max_pair_weight: float = 2.0
    pca_threshold: int = 128
    pca_components: int = 64
    gradient_clip: float = 5.0

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.bottleneck_size <= 0:
            raise ValueError("hidden sizes must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.epochs <= 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimization configuration")
        if self.huber_beta <= 0 or self.pair_temperature <= 0:
            raise ValueError("loss scales must be positive")
        if self.pair_lambda < 0 or self.tie_threshold < 0 or self.max_pair_weight <= 0:
            raise ValueError("invalid pairwise configuration")
        if self.pca_threshold <= 0 or self.pca_components <= 0:
            raise ValueError("PCA settings must be positive")


class FourHeadMLP(nn.Module):
    def __init__(self, input_size: int, config: MLPTrainingConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.bottleneck_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.bottleneck_size, 4),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


@dataclass(frozen=True)
class PairTensors:
    left: torch.Tensor
    right: torch.Tensor
    dimension: torch.Tensor
    sign: torch.Tensor
    weight: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.left.numel())


def build_within_question_pairs(
    question_ids: np.ndarray,
    targets: np.ndarray,
    *,
    tie_threshold: float,
    max_pair_weight: float,
) -> PairTensors:
    question_ids = np.asarray(question_ids)
    targets = np.asarray(targets, dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] != 4 or len(targets) != len(question_ids):
        raise ValueError("targets must have shape [documents, 4]")
    left: list[int] = []
    right: list[int] = []
    dimensions: list[int] = []
    signs: list[float] = []
    weights: list[float] = []
    for question_id in np.unique(question_ids):
        indices = np.flatnonzero(question_ids == question_id)
        for local_left in range(len(indices)):
            for local_right in range(local_left + 1, len(indices)):
                first = int(indices[local_left])
                second = int(indices[local_right])
                gaps = targets[first] - targets[second]
                for dimension, gap in enumerate(gaps):
                    absolute_gap = float(abs(gap))
                    if absolute_gap <= tie_threshold:
                        continue
                    left.append(first)
                    right.append(second)
                    dimensions.append(dimension)
                    signs.append(1.0 if gap > 0 else -1.0)
                    weights.append(min(absolute_gap, max_pair_weight))
    if not left:
        raise ValueError("no non-tied within-question pairs")
    weight_array = np.asarray(weights, dtype=np.float32)
    weight_array /= float(np.mean(weight_array))
    return PairTensors(
        left=torch.tensor(left, dtype=torch.long),
        right=torch.tensor(right, dtype=torch.long),
        dimension=torch.tensor(dimensions, dtype=torch.long),
        sign=torch.tensor(signs, dtype=torch.float32),
        weight=torch.tensor(weight_array, dtype=torch.float32),
    )


def pairwise_logistic_loss(
    predictions: torch.Tensor,
    pairs: PairTensors,
    *,
    temperature: float,
) -> torch.Tensor:
    left_scores = predictions[pairs.left, pairs.dimension]
    right_scores = predictions[pairs.right, pairs.dimension]
    signed_difference = pairs.sign * (left_scores - right_scores) / temperature
    return torch.mean(pairs.weight * functional.softplus(-signed_difference))


@dataclass
class FoldPreprocessor:
    scaler: StandardScaler
    pca: PCA | None

    def transform(self, values: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(values)
        transformed = self.pca.transform(scaled) if self.pca is not None else scaled
        return np.asarray(transformed, dtype=np.float32)


def fit_fold_preprocessor(
    train_values: np.ndarray,
    config: MLPTrainingConfig,
    *,
    seed: int,
) -> FoldPreprocessor:
    train_values = np.asarray(train_values, dtype=np.float64)
    if train_values.ndim != 2 or not np.isfinite(train_values).all():
        raise ValueError("training features must be a finite matrix")
    scaler = StandardScaler().fit(train_values)
    scaled = scaler.transform(train_values)
    pca: PCA | None = None
    if scaled.shape[1] >= config.pca_threshold:
        components = min(config.pca_components, scaled.shape[0] - 1, scaled.shape[1])
        pca = PCA(n_components=components, svd_solver="randomized", random_state=seed).fit(scaled)
    return FoldPreprocessor(scaler=scaler, pca=pca)


def _standardize_targets(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(targets, axis=0, dtype=np.float64)
    scale = np.std(targets, axis=0, dtype=np.float64)
    scale = np.where(scale < 1e-6, 1.0, scale)
    standardized = (targets - mean) / scale
    return standardized.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)


def train_fold_ensemble(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_targets: np.ndarray,
    train_question_ids: np.ndarray,
    *,
    config: MLPTrainingConfig,
    seeds: Sequence[int],
    use_pairwise: bool,
) -> tuple[np.ndarray, list[dict[str, float | int | bool]]]:
    config.validate()
    if not seeds:
        raise ValueError("at least one seed is required")
    train_features = np.asarray(train_features, dtype=np.float32)
    test_features = np.asarray(test_features, dtype=np.float32)
    train_targets = np.asarray(train_targets, dtype=np.float32)
    standardized_targets, target_mean, target_scale = _standardize_targets(train_targets)
    pairs = build_within_question_pairs(
        train_question_ids,
        train_targets,
        tie_threshold=config.tie_threshold,
        max_pair_weight=config.max_pair_weight,
    )
    x_train = torch.from_numpy(train_features)
    x_test = torch.from_numpy(test_features)
    y_train = torch.from_numpy(standardized_targets)
    seed_predictions: list[np.ndarray] = []
    diagnostics: list[dict[str, float | int | bool]] = []

    for seed in seeds:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        model = FourHeadMLP(train_features.shape[1], config)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        final_huber = float("nan")
        final_pair = float("nan")
        model.train()
        for _ in range(config.epochs):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_train)
            huber = functional.smooth_l1_loss(
                prediction, y_train, beta=config.huber_beta, reduction="mean"
            )
            pair = pairwise_logistic_loss(
                prediction, pairs, temperature=config.pair_temperature
            )
            loss = huber + (config.pair_lambda * pair if use_pairwise else 0.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            final_huber = float(huber.detach())
            final_pair = float(pair.detach())
        model.eval()
        with torch.inference_mode():
            standardized_prediction = model(x_test).numpy()
        prediction = standardized_prediction * target_scale + target_mean
        seed_predictions.append(prediction)
        diagnostics.append(
            {
                "seed": int(seed),
                "use_pairwise": bool(use_pairwise),
                "train_documents": int(len(train_features)),
                "test_documents": int(len(test_features)),
                "input_features": int(train_features.shape[1]),
                "pair_count": pairs.count,
                "final_huber": final_huber,
                "final_pairwise": final_pair,
            }
        )
    return np.mean(seed_predictions, axis=0), diagnostics


def config_dict(config: MLPTrainingConfig) -> dict[str, object]:
    return asdict(config)
