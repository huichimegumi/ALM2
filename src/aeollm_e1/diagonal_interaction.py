from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from aeollm_e0.metrics import DIMS

from .cosine_features import CacheBundle, load_cache, load_cache_with_variant

ALIGNMENT_DIMS = ("comprehensiveness", "instruction_following")
ALIGNMENT_DIM_INDICES = {dimension: index for index, dimension in enumerate(ALIGNMENT_DIMS)}
POOL_NAMES = ("mean", "max", "top10pct_mean", "logmeanexp_t005")


@dataclass(frozen=True)
class DiagonalTrainingConfig:
    epochs: int = 60
    learning_rate: float = 1e-2
    weight_decay: float = 1e-2
    diagonal_l2: float = 1e-3
    huber_beta: float = 0.5
    gradient_clip: float = 5.0
    logmeanexp_temperature: float = 0.05

    def validate(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")
        if self.weight_decay < 0 or self.diagonal_l2 < 0:
            raise ValueError("regularization strengths must be non-negative")
        if self.huber_beta <= 0 or self.gradient_clip <= 0:
            raise ValueError("loss and clipping scales must be positive")
        if self.logmeanexp_temperature <= 0:
            raise ValueError("logmeanexp_temperature must be positive")


@dataclass(frozen=True)
class QuestionChunks:
    question_id: int
    document_ids: tuple[str, ...]
    embeddings: torch.Tensor
    spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class RubricTensor:
    target_question_id: int
    source_question_id: int
    embeddings: torch.Tensor
    dimension_indices: torch.Tensor
    weights: torch.Tensor


@dataclass(frozen=True)
class InteractionCorpus:
    chunks: Mapping[int, QuestionChunks]
    rubric_views: Mapping[str, Mapping[int, RubricTensor]]
    embedding_dimension: int


def cyclic_rubric_maps(question_ids: Sequence[int], count: int) -> list[dict[int, int]]:
    ordered = sorted(int(value) for value in question_ids)
    if len(ordered) < 2 or count <= 0 or count >= len(ordered):
        raise ValueError("count must be between 1 and n_questions - 1")
    return [
        {
            question_id: ordered[(position + shift) % len(ordered)]
            for position, question_id in enumerate(ordered)
        }
        for shift in range(1, count + 1)
    ]


def _question_chunks(cache: CacheBundle, device: torch.device) -> dict[int, QuestionChunks]:
    result: dict[int, QuestionChunks] = {}
    for question_id_raw, question_rows in cache.chunk_index.groupby("question_id", sort=True):
        question_id = int(question_id_raw)
        document_ids: list[str] = []
        row_indices: list[int] = []
        spans: list[tuple[int, int]] = []
        for document_id_raw, document_rows in question_rows.groupby("document_id", sort=True):
            document_rows = document_rows.sort_values("chunk_id")
            start = len(row_indices)
            row_indices.extend(document_rows["embedding_row"].to_numpy(dtype=int).tolist())
            end = len(row_indices)
            if end <= start:
                raise ValueError(f"document has no chunks: {question_id}/{document_id_raw}")
            document_ids.append(str(document_id_raw))
            spans.append((start, end))
        embeddings = torch.as_tensor(
            np.asarray(cache.chunk_embeddings[row_indices], dtype=np.float32), device=device
        )
        result[question_id] = QuestionChunks(
            question_id=question_id,
            document_ids=tuple(document_ids),
            embeddings=embeddings,
            spans=tuple(spans),
        )
    return result


def _rubric_view(
    cache: CacheBundle,
    target_questions: Sequence[int],
    question_map: Mapping[int, int],
    device: torch.device,
) -> dict[int, RubricTensor]:
    grouped = {
        int(question_id): rows.sort_values("embedding_row")
        for question_id, rows in cache.criterion_index.groupby("question_id", sort=True)
    }
    result: dict[int, RubricTensor] = {}
    for target_question in target_questions:
        source_question = int(question_map[int(target_question)])
        rows = grouped[source_question]
        rows = rows[rows["dimension"].isin(ALIGNMENT_DIMS)].copy()
        if rows.empty or set(rows["dimension"]) != set(ALIGNMENT_DIMS):
            raise ValueError(f"source rubric {source_question} lacks alignment dimensions")
        embedding_rows = rows["embedding_row"].to_numpy(dtype=int)
        dimensions = np.asarray(
            [ALIGNMENT_DIM_INDICES[str(value)] for value in rows["dimension"]], dtype=np.int64
        )
        weights = rows["weight"].to_numpy(dtype=np.float32)
        result[int(target_question)] = RubricTensor(
            target_question_id=int(target_question),
            source_question_id=source_question,
            embeddings=torch.as_tensor(
                np.asarray(cache.criterion_embeddings[embedding_rows], dtype=np.float32),
                device=device,
            ),
            dimension_indices=torch.as_tensor(dimensions, device=device),
            weights=torch.as_tensor(weights, device=device),
        )
    return result


def load_interaction_corpus(
    base_cache_dir: Path,
    query_variant_root: Path,
    *,
    mismatch_count: int,
    device: str,
) -> tuple[InteractionCorpus, list[dict[int, int]]]:
    torch_device = torch.device(device)
    base = load_cache(base_cache_dir)
    generic = load_cache_with_variant(base_cache_dir, query_variant_root / "generic_dimension")
    questions = sorted(int(value) for value in base.chunk_index["question_id"].unique())
    identity = {question_id: question_id for question_id in questions}
    mismatch_maps = cyclic_rubric_maps(questions, mismatch_count)
    views: dict[str, Mapping[int, RubricTensor]] = {
        "matched": _rubric_view(base, questions, identity, torch_device),
        "generic": _rubric_view(generic, questions, identity, torch_device),
    }
    for index, mapping in enumerate(mismatch_maps, start=1):
        views[f"mismatch_shift{index}"] = _rubric_view(base, questions, mapping, torch_device)
    corpus = InteractionCorpus(
        chunks=_question_chunks(base, torch_device),
        rubric_views=views,
        embedding_dimension=int(base.chunk_embeddings.shape[1]),
    )
    return corpus, mismatch_maps


def pool_criterion_chunk_scores(
    scores: torch.Tensor,
    *,
    logmeanexp_temperature: float,
) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("scores must have shape [criteria, non-empty chunks]")
    chunk_count = int(scores.shape[1])
    top_count = max(1, math.ceil(chunk_count * 0.10))
    mean = scores.mean(dim=1)
    maximum = scores.max(dim=1).values
    top_mean = scores.topk(top_count, dim=1).values.mean(dim=1)
    temperature = float(logmeanexp_temperature)
    logmeanexp = temperature * (
        torch.logsumexp(scores / temperature, dim=1) - math.log(chunk_count)
    )
    return torch.stack((mean, maximum, top_mean, logmeanexp), dim=1)


def aggregate_criterion_evidence(
    pooled: torch.Tensor,
    dimension_indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for dimension in range(len(ALIGNMENT_DIMS)):
        mask = dimension_indices == dimension
        if not bool(mask.any()):
            raise ValueError(f"no criteria for alignment dimension {dimension}")
        selected_weights = weights[mask]
        total = selected_weights.sum()
        if float(total.detach().cpu()) <= 0:
            selected_weights = torch.ones_like(selected_weights)
            total = selected_weights.sum()
        rows.append((pooled[mask] * selected_weights[:, None]).sum(dim=0) / total)
    return torch.stack(rows, dim=0)


class CriterionChunkInteraction(nn.Module):
    def __init__(self, embedding_dimension: int, *, learn_diagonal: bool) -> None:
        super().__init__()
        self.embedding_dimension = int(embedding_dimension)
        self.learn_diagonal = bool(learn_diagonal)
        if learn_diagonal:
            self.diagonal = nn.Parameter(torch.zeros(embedding_dimension))
        else:
            self.register_parameter("diagonal", None)
        self.head_weight = nn.Parameter(torch.empty(len(ALIGNMENT_DIMS), len(POOL_NAMES)))
        self.head_bias = nn.Parameter(torch.zeros(len(ALIGNMENT_DIMS)))
        nn.init.normal_(self.head_weight, mean=0.0, std=0.02)

    def score_pairs(self, criteria: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        cosine = criteria @ chunks.T
        if self.diagonal is None:
            return cosine
        diagonal = math.sqrt(self.embedding_dimension) * (
            (criteria * self.diagonal[None, :]) @ chunks.T
        )
        return cosine + diagonal

    def evidence(
        self,
        chunks: QuestionChunks,
        rubric: RubricTensor,
        *,
        logmeanexp_temperature: float,
    ) -> torch.Tensor:
        all_scores = self.score_pairs(rubric.embeddings, chunks.embeddings)
        documents: list[torch.Tensor] = []
        for start, end in chunks.spans:
            pooled = pool_criterion_chunk_scores(
                all_scores[:, start:end],
                logmeanexp_temperature=logmeanexp_temperature,
            )
            documents.append(
                aggregate_criterion_evidence(
                    pooled, rubric.dimension_indices, rubric.weights
                )
            )
        return torch.stack(documents, dim=0)

    def forward_from_evidence(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.ndim != 3 or evidence.shape[1:] != (
            len(ALIGNMENT_DIMS),
            len(POOL_NAMES),
        ):
            raise ValueError("evidence must have shape [documents, 2, 4]")
        return (evidence * self.head_weight[None, :, :]).sum(dim=2) + self.head_bias


class DimensionSeparatedCriterionChunkInteraction(nn.Module):
    """Joint two-head model with one diagonal metric per alignment dimension."""

    def __init__(self, embedding_dimension: int) -> None:
        super().__init__()
        self.embedding_dimension = int(embedding_dimension)
        self.diagonal = nn.Parameter(
            torch.zeros(len(ALIGNMENT_DIMS), embedding_dimension)
        )
        self.head_weight = nn.Parameter(
            torch.empty(len(ALIGNMENT_DIMS), len(POOL_NAMES))
        )
        self.head_bias = nn.Parameter(torch.zeros(len(ALIGNMENT_DIMS)))
        nn.init.normal_(self.head_weight, mean=0.0, std=0.02)

    def score_pairs(
        self,
        criteria: torch.Tensor,
        chunks: torch.Tensor,
        dimension_indices: torch.Tensor,
    ) -> torch.Tensor:
        cosine = criteria @ chunks.T
        selected_diagonals = self.diagonal[dimension_indices]
        diagonal = math.sqrt(self.embedding_dimension) * (
            (criteria * selected_diagonals) @ chunks.T
        )
        return cosine + diagonal

    def evidence(
        self,
        chunks: QuestionChunks,
        rubric: RubricTensor,
        *,
        logmeanexp_temperature: float,
    ) -> torch.Tensor:
        all_scores = self.score_pairs(
            rubric.embeddings,
            chunks.embeddings,
            rubric.dimension_indices,
        )
        documents: list[torch.Tensor] = []
        for start, end in chunks.spans:
            pooled = pool_criterion_chunk_scores(
                all_scores[:, start:end],
                logmeanexp_temperature=logmeanexp_temperature,
            )
            documents.append(
                aggregate_criterion_evidence(
                    pooled, rubric.dimension_indices, rubric.weights
                )
            )
        return torch.stack(documents, dim=0)

    def forward_from_evidence(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.ndim != 3 or evidence.shape[1:] != (
            len(ALIGNMENT_DIMS),
            len(POOL_NAMES),
        ):
            raise ValueError("evidence must have shape [documents, 2, 4]")
        return (evidence * self.head_weight[None, :, :]).sum(dim=2) + self.head_bias


def precompute_fixed_evidence(
    corpus: InteractionCorpus,
    rubric_view: Mapping[int, RubricTensor],
    *,
    temperature: float,
) -> dict[int, torch.Tensor]:
    model = CriterionChunkInteraction(corpus.embedding_dimension, learn_diagonal=False)
    device = next(iter(corpus.chunks.values())).embeddings.device
    model = model.to(device)
    result: dict[int, torch.Tensor] = {}
    # The evidence is constant, but it must remain a normal tensor because the
    # trainable linear head saves it for backward. inference_mode tensors cannot
    # participate in that autograd graph; no_grad avoids building the cosine graph
    # while preserving compatibility with head training.
    with torch.no_grad():
        for question_id, chunks in corpus.chunks.items():
            result[question_id] = model.evidence(
                chunks, rubric_view[question_id], logmeanexp_temperature=temperature
            ).detach()
    return result


def train_interaction_fold_ensemble(
    corpus: InteractionCorpus,
    rubric_view: Mapping[int, RubricTensor],
    train_questions: Sequence[int],
    test_question: int,
    targets_by_question: Mapping[int, np.ndarray],
    baseline_by_question: Mapping[int, np.ndarray],
    *,
    config: DiagonalTrainingConfig,
    seeds: Sequence[int],
    learn_diagonal: bool,
) -> tuple[np.ndarray, list[dict[str, float | int | bool]]]:
    config.validate()
    if not seeds:
        raise ValueError("at least one seed is required")
    device = next(iter(corpus.chunks.values())).embeddings.device
    train_questions = tuple(sorted(int(value) for value in train_questions))
    y_train = np.concatenate([targets_by_question[q] for q in train_questions], axis=0)
    mean = y_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = y_train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    target_tensors = {
        q: torch.as_tensor((targets_by_question[q] - mean) / scale, device=device)
        for q in train_questions
    }
    baseline_tensors = {
        q: torch.as_tensor((baseline_by_question[q] - mean) / scale, device=device)
        for q in (*train_questions, int(test_question))
    }
    fixed_evidence = (
        precompute_fixed_evidence(
            corpus, rubric_view, temperature=config.logmeanexp_temperature
        )
        if not learn_diagonal
        else None
    )
    seed_predictions: list[np.ndarray] = []
    diagnostics: list[dict[str, float | int | bool]] = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        np.random.seed(int(seed))
        model = CriterionChunkInteraction(
            corpus.embedding_dimension, learn_diagonal=learn_diagonal
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        final_huber = float("nan")
        final_regularization = float("nan")
        model.train()
        for _ in range(config.epochs):
            optimizer.zero_grad(set_to_none=True)
            predictions: list[torch.Tensor] = []
            targets: list[torch.Tensor] = []
            for question_id in train_questions:
                evidence = (
                    fixed_evidence[question_id]
                    if fixed_evidence is not None
                    else model.evidence(
                        corpus.chunks[question_id],
                        rubric_view[question_id],
                        logmeanexp_temperature=config.logmeanexp_temperature,
                    )
                )
                correction = model.forward_from_evidence(evidence)
                predictions.append(baseline_tensors[question_id] + correction)
                targets.append(target_tensors[question_id])
            prediction = torch.cat(predictions, dim=0)
            target = torch.cat(targets, dim=0)
            huber = functional.smooth_l1_loss(
                prediction, target, beta=config.huber_beta, reduction="mean"
            )
            regularization = (
                config.diagonal_l2 * model.diagonal.square().mean()
                if model.diagonal is not None
                else torch.zeros((), device=device)
            )
            loss = huber + regularization
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            final_huber = float(huber.detach().cpu())
            final_regularization = float(regularization.detach().cpu())
        model.eval()
        with torch.inference_mode():
            evidence = (
                fixed_evidence[int(test_question)]
                if fixed_evidence is not None
                else model.evidence(
                    corpus.chunks[int(test_question)],
                    rubric_view[int(test_question)],
                    logmeanexp_temperature=config.logmeanexp_temperature,
                )
            )
            standardized = baseline_tensors[int(test_question)] + model.forward_from_evidence(
                evidence
            )
            prediction = standardized.cpu().numpy() * scale + mean
        seed_predictions.append(prediction)
        diagnostics.append(
            {
                "seed": int(seed),
                "learn_diagonal": bool(learn_diagonal),
                "train_documents": int(sum(len(corpus.chunks[q].document_ids) for q in train_questions)),
                "test_documents": int(len(corpus.chunks[int(test_question)].document_ids)),
                "trainable_parameters": int(sum(p.numel() for p in model.parameters())),
                "final_huber": final_huber,
                "final_diagonal_regularization": final_regularization,
                "diagonal_l2_norm": (
                    float(torch.linalg.vector_norm(model.diagonal).detach().cpu())
                    if model.diagonal is not None
                    else 0.0
                ),
                "head_l2_norm": float(
                    torch.linalg.vector_norm(model.head_weight).detach().cpu()
                ),
            }
        )
    return np.mean(seed_predictions, axis=0), diagnostics


def train_separated_interaction_fold_ensemble(
    corpus: InteractionCorpus,
    rubric_view: Mapping[int, RubricTensor],
    train_questions: Sequence[int],
    test_question: int,
    targets_by_question: Mapping[int, np.ndarray],
    baseline_by_question: Mapping[int, np.ndarray],
    *,
    config: DiagonalTrainingConfig,
    seeds: Sequence[int],
    learn_diagonal: bool,
) -> tuple[np.ndarray, list[dict[str, float | int | bool]]]:
    """Train jointly while replacing only the shared diagonal with two diagonals."""
    config.validate()
    if not seeds:
        raise ValueError("at least one seed is required")
    device = next(iter(corpus.chunks.values())).embeddings.device
    train_questions = tuple(sorted(int(value) for value in train_questions))
    y_train = np.concatenate([targets_by_question[q] for q in train_questions], axis=0)
    mean = y_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = y_train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    target_tensors = {
        q: torch.as_tensor((targets_by_question[q] - mean) / scale, device=device)
        for q in train_questions
    }
    baseline_tensors = {
        q: torch.as_tensor((baseline_by_question[q] - mean) / scale, device=device)
        for q in (*train_questions, int(test_question))
    }
    fixed_evidence = (
        precompute_fixed_evidence(
            corpus, rubric_view, temperature=config.logmeanexp_temperature
        )
        if not learn_diagonal
        else None
    )
    seed_predictions: list[np.ndarray] = []
    diagnostics: list[dict[str, float | int | bool]] = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        np.random.seed(int(seed))
        model: CriterionChunkInteraction | DimensionSeparatedCriterionChunkInteraction
        if learn_diagonal:
            model = DimensionSeparatedCriterionChunkInteraction(
                corpus.embedding_dimension
            ).to(device)
        else:
            model = CriterionChunkInteraction(
                corpus.embedding_dimension, learn_diagonal=False
            ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        final_huber = float("nan")
        final_regularization = float("nan")
        model.train()
        for _ in range(config.epochs):
            optimizer.zero_grad(set_to_none=True)
            predictions: list[torch.Tensor] = []
            targets: list[torch.Tensor] = []
            for question_id in train_questions:
                evidence = (
                    fixed_evidence[question_id]
                    if fixed_evidence is not None
                    else model.evidence(
                        corpus.chunks[question_id],
                        rubric_view[question_id],
                        logmeanexp_temperature=config.logmeanexp_temperature,
                    )
                )
                correction = model.forward_from_evidence(evidence)
                predictions.append(baseline_tensors[question_id] + correction)
                targets.append(target_tensors[question_id])
            prediction = torch.cat(predictions, dim=0)
            target = torch.cat(targets, dim=0)
            huber = functional.smooth_l1_loss(
                prediction, target, beta=config.huber_beta, reduction="mean"
            )
            regularization = (
                config.diagonal_l2 * model.diagonal.square().mean()
                if learn_diagonal
                else torch.zeros((), device=device)
            )
            loss = huber + regularization
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            final_huber = float(huber.detach().cpu())
            final_regularization = float(regularization.detach().cpu())
        model.eval()
        with torch.inference_mode():
            evidence = (
                fixed_evidence[int(test_question)]
                if fixed_evidence is not None
                else model.evidence(
                    corpus.chunks[int(test_question)],
                    rubric_view[int(test_question)],
                    logmeanexp_temperature=config.logmeanexp_temperature,
                )
            )
            standardized = baseline_tensors[int(test_question)] + model.forward_from_evidence(
                evidence
            )
            prediction = standardized.cpu().numpy() * scale + mean
        seed_predictions.append(prediction)
        diagonal_norms = (
            torch.linalg.vector_norm(model.diagonal, dim=1).detach().cpu().tolist()
            if learn_diagonal
            else [0.0] * len(ALIGNMENT_DIMS)
        )
        diagnostics.append(
            {
                "seed": int(seed),
                "learn_diagonal": bool(learn_diagonal),
                "train_documents": int(
                    sum(len(corpus.chunks[q].document_ids) for q in train_questions)
                ),
                "test_documents": int(
                    len(corpus.chunks[int(test_question)].document_ids)
                ),
                "trainable_parameters": int(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
                "final_huber": final_huber,
                "final_diagonal_regularization": final_regularization,
                "diagonal_l2_norm_comprehensiveness": float(diagonal_norms[0]),
                "diagonal_l2_norm_instruction_following": float(diagonal_norms[1]),
                "head_l2_norm": float(
                    torch.linalg.vector_norm(model.head_weight).detach().cpu()
                ),
            }
        )
    return np.mean(seed_predictions, axis=0), diagnostics


def config_dict(config: DiagonalTrainingConfig) -> dict[str, object]:
    return asdict(config)
