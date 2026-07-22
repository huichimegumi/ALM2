"""AEOLLM-2 E1 frozen-representation experiments."""

from .chunking import ChunkingConfig, chunk_docx
from .cosine_features import build_cosine_features
from .embedding import EmbeddingConfig, build_embedding_cache
from .ridge_scoring import nested_loqo_ridge_predictions
from .pairwise_training import MLPTrainingConfig, train_fold_ensemble

__all__ = [
    "ChunkingConfig",
    "EmbeddingConfig",
    "build_cosine_features",
    "build_embedding_cache",
    "chunk_docx",
    "nested_loqo_ridge_predictions",
    "MLPTrainingConfig",
    "train_fold_ensemble",
]
