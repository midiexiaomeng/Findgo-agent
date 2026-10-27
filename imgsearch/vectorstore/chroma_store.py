"""ChromaDB 实现。

Chroma 的定位是"零配置本地库"：PersistentClient 直接落盘，不需要服务进程。
本类只做接口适配，不向上层暴露任何 chromadb 概念。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .base import VectorHit, VectorStore, VectorStoreError, sanitize_metadata

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    backend_name = "chromadb"

    def __init__(
        self,
        collection: str = "images",
        embedding_dim: int = 512,
        path: str | Path = "./data/chromadb",
    ):
        super().__init__(collection=collection, embedding_dim=embedding_dim)
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._col = None
        self._connect()

    # ── 初始化 ──────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as e:  # pragma: no cover
            raise VectorStoreError(
                "未安装 chromadb，请执行: pip install chromadb"
            ) from e

        try:
            self._client = chromadb.PersistentClient(
                path=str(self.path),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
        except TypeError:
            # 老版本 Settings 没有 allow_reset 参数
            self._client = chromadb.PersistentClient(
                path=str(self.path),
                settings=Settings(anonymized_telemetry=False),
            )

        # get_or_create 一步到位，避免 get 抛异常再 create 的丑陋写法
        self._col = self._client.get_or_create_collection(
            name=self.collection,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,  # 我们自己算向量，不让 Chroma 偷偷调用模型
        )

        actual_dim = None
        try:
            # chromadb 0.5+ 可以拿到 collection 的维度信息
            meta = getattr(self._col, "metadata", None) or {}
            actual_dim = meta.get("embedding_dim")
        except Exception:
            actual_dim = None
        if actual_dim and int(actual_dim) != self.embedding_dim:
            logger.warning(
                "collection %s 记录维度 %s 与当前配置 %s 不一致，检索可能失败；"
                "换模型后请重建索引",
                self.collection,
                actual_dim,
                self.embedding_dim,
            )

        logger.info(
            "ChromaDB 就绪: %s | collection=%s | 条目=%d",
            self.path,
            self.collection,
            self._col.count(),
        )

    # ── 写 ──────────────────────────────────────────────────────────────────

    def add(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        arr = self._check_batch(ids, embeddings, metadatas)
        metas = [sanitize_metadata(m or {}) for m in metadatas]

        self._col.upsert(
            ids=list(ids),
            embeddings=arr,
            metadatas=metas,
        )

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        self._col.delete(ids=list(ids))

    def reset(self) -> None:
        try:
            self._client.delete_collection(self.collection)
        except Exception:
            pass
        self._col = self._client.get_or_create_collection(
            name=self.collection,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        logger.info("已清空 collection: %s", self.collection)

    # ── 读 ──────────────────────────────────────────────────────────────────

    def query(
        self,
        embedding: np.ndarray,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        vec = self._check_query(embedding)
        if top_k <= 0:
            return []
        total = self.count()
        if total == 0:
            return []

        try:
            res = self._col.query(
                query_embeddings=vec.reshape(1, -1),
                n_results=min(top_k, total),
                where=where or None,
                include=["metadatas", "distances"],
            )
        except Exception as e:
            raise VectorStoreError(f"ChromaDB 检索失败: {e}") from e

        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]

        hits: list[VectorHit] = []
        for i, _id in enumerate(ids):
            distance = float(dists[i]) if i < len(dists) else 1.0
            meta = dict(metas[i]) if i < len(metas) and metas[i] else {}
            hits.append(
                VectorHit(
                    id=str(_id),
                    # cosine distance -> cosine similarity
                    score=1.0 - distance,
                    metadata=meta,
                )
            )
        return hits

    def get(
        self,
        ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[VectorHit]:
        kwargs: dict[str, Any] = {"include": ["metadatas"]}
        if ids:
            kwargs["ids"] = list(ids)
        if limit:
            kwargs["limit"] = limit
        try:
            res = self._col.get(**kwargs)
        except Exception as e:
            raise VectorStoreError(f"ChromaDB 读取失败: {e}") from e

        out: list[VectorHit] = []
        got_ids = res.get("ids") or []
        metas = res.get("metadatas") or []
        for i, _id in enumerate(got_ids):
            meta = dict(metas[i]) if i < len(metas) and metas[i] else {}
            out.append(VectorHit(id=str(_id), score=1.0, metadata=meta))
        return out

    def count(self) -> int:
        return int(self._col.count())

    def get_all_ids(self) -> list[str]:
        """分页拉取，避免一次性把十万条 metadata 拉进内存。"""
        ids: list[str] = []
        offset = 0
        page = 5000
        while True:
            res = self._col.get(include=[], limit=page, offset=offset)
            batch = res.get("ids") or []
            if not batch:
                break
            ids.extend(str(i) for i in batch)
            if len(batch) < page:
                break
            offset += page
        return ids

    def close(self) -> None:
        # PersistentClient 没有显式 close，置空引用即可释放
        self._col = None
        self._client = None
