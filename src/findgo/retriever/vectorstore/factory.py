"""向量库工厂：配置 -> 具体实现。

上层永远通过这里拿实例，绝不直接 new ChromaVectorStore / QdrantVectorStore。
"""

from __future__ import annotations

import logging

from ..config import Config
from .base import VectorHit, VectorStore, VectorStoreError
from .chroma_store import ChromaVectorStore
from .qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)

BACKENDS = ("chromadb", "qdrant")

_ALIASES = {
    "chroma": "chromadb",
    "chromadb": "chromadb",
    "qdrant": "qdrant",
    "qdrant-local": "qdrant",
}


def create_vector_store(cfg: Config, backend: str | None = None) -> VectorStore:
    """按配置创建向量库。

    Args:
        cfg: 全局配置
        backend: 可选覆盖（界面切换后端时用），不传则读 config.vector_store.backend
    """
    name = _ALIASES.get((backend or cfg.vector_store.backend or "").lower())
    if name is None:
        raise VectorStoreError(
            f"未知向量库后端: {backend or cfg.vector_store.backend!r}，"
            f"可选: {', '.join(BACKENDS)}"
        )

    vs = cfg.vector_store
    dim = int(cfg.embedding.embedding_dim)

    if name == "chromadb":
        return ChromaVectorStore(
            collection=vs.collection,
            embedding_dim=dim,
            path=cfg.resolve(vs.chromadb.path),
        )

    return QdrantVectorStore(
        collection=vs.collection,
        embedding_dim=dim,
        mode=vs.qdrant.mode,
        path=cfg.resolve(vs.qdrant.path),
        url=vs.qdrant.url,
        api_key=vs.qdrant.api_key,
        timeout=vs.qdrant.timeout,
    )


__all__ = [
    "BACKENDS",
    "VectorHit",
    "VectorStore",
    "VectorStoreError",
    "ChromaVectorStore",
    "QdrantVectorStore",
    "create_vector_store",
]
