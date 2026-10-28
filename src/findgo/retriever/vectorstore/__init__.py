"""向量库抽象层。"""

from .base import (
    VectorHit,
    VectorStore,
    VectorStoreError,
    l2_normalize,
    normalize_scores,
    sanitize_metadata,
)
from .factory import BACKENDS, create_vector_store

__all__ = [
    "BACKENDS",
    "VectorHit",
    "VectorStore",
    "VectorStoreError",
    "create_vector_store",
    "l2_normalize",
    "normalize_scores",
    "sanitize_metadata",
]
