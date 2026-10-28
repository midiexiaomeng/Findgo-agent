"""Qdrant 实现。

两种模式：
- ``local``  —— 嵌入式文件模式，数据落在 data/qdrant 目录，无需启动服务，
                真正做到"解压即用"。
- ``server`` —— 连接 qdrant 服务（docker 或远端），适合多进程/多人共享。

注意：嵌入式模式下同一路径同时只能被一个进程打开，这是 Qdrant 的文件锁
机制决定的，不是 bug。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 项目自带的依赖目录（部分环境没有写权限安装到 site-packages）
_VENDOR = Path(__file__).resolve().parents[4] / "_vendor"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

import logging  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from .base import VectorHit, VectorStore, VectorStoreError, sanitize_metadata  # noqa: E402

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStore):
    backend_name = "qdrant"

    def __init__(
        self,
        collection: str = "images",
        embedding_dim: int = 512,
        mode: str = "local",
        path: str | Path = "./data/qdrant",
        url: str = "http://localhost:6333",
        api_key: str = "",
        timeout: int = 30,
    ):
        super().__init__(collection=collection, embedding_dim=embedding_dim)
        self.mode = (mode or "local").lower()
        self.path = Path(path)
        self.url = url
        self.api_key = api_key or None
        self.timeout = timeout
        self._client = None
        self._models = None
        self._connect()

    # ── 初始化 ──────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qm
        except ImportError as e:  # pragma: no cover
            raise VectorStoreError("未安装 qdrant-client，请执行: pip install qdrant-client") from e

        self._models = qm

        if self.mode == "local":
            self.path.mkdir(parents=True, exist_ok=True)
            try:
                self._client = QdrantClient(path=str(self.path), timeout=self.timeout)
            except RuntimeError as e:
                raise VectorStoreError(
                    f"无法打开本地 Qdrant 存储 {self.path}：{e}\n"
                    "通常是另一个进程正在占用该目录（例如 Web 界面还开着）。"
                    "请先关闭占用进程，或改用 qdrant 的 server 模式。"
                ) from e
        else:
            self._client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=self.timeout,
            )

        self._ensure_collection()
        logger.info(
            "Qdrant 就绪: mode=%s | %s | collection=%s | 条目=%d",
            self.mode,
            self.path if self.mode == "local" else self.url,
            self.collection,
            self.count(),
        )

    def _exists(self) -> bool:
        try:
            if hasattr(self._client, "collection_exists"):
                return bool(self._client.collection_exists(self.collection))
            names = {c.name for c in self._client.get_collections().collections}
            return self.collection in names
        except Exception:
            return False

    def _ensure_collection(self) -> None:
        qm = self._models
        if self._exists():
            info = self._client.get_collection(self.collection)
            try:
                size = info.config.params.vectors.size
                if int(size) != self.embedding_dim:
                    logger.warning(
                        "Qdrant collection %s 维度为 %s，当前配置为 %s；换模型后请重建索引",
                        self.collection,
                        size,
                        self.embedding_dim,
                    )
            except Exception:
                pass
            return

        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(
                size=self.embedding_dim,
                distance=qm.Distance.COSINE,
            ),
        )
        logger.info("创建 Qdrant collection: %s (dim=%d)", self.collection, self.embedding_dim)

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

        points = [
            self._models.PointStruct(
                id=str(ids[i]),
                vector=arr[i].tolist(),
                payload=metas[i],
            )
            for i in range(len(ids))
        ]
        self._client.upsert(collection_name=self.collection, points=points)

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        self._client.delete(
            collection_name=self.collection,
            points_selector=self._models.PointIdsList(points=[str(i) for i in ids]),
        )

    def reset(self) -> None:
        """清空 collection。

        注意：本地模式下 qdrant-client 的 delete_collection 并不会清理
        data/qdrant/collection/<name> 目录，同名重建会把旧数据"复活"。
        因此这里用"删除全部 points"的方式清空，collection 配置保留。
        """
        qm = self._models
        try:
            self._client.delete(
                collection_name=self.collection,
                points_selector=qm.FilterSelector(filter=qm.Filter()),
            )
        except Exception as e:
            logger.warning("批量清空失败，回退到重建 collection: %s", e)
            try:
                self._client.delete_collection(self.collection)
            except Exception:
                pass
            self._ensure_collection()
        logger.info("已清空 collection: %s", self.collection)

    # ── 读 ──────────────────────────────────────────────────────────────────

    def query(
        self,
        embedding: np.ndarray,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        vec = self._check_query(embedding)
        if top_k <= 0 or self.count() == 0:
            return []

        qfilter = self._build_filter(where)
        points = self._query_points(vec, top_k, qfilter)

        return [
            VectorHit(
                id=str(p.id),
                score=float(p.score),
                metadata=dict(p.payload or {}),
            )
            for p in points
        ]

    def _query_points(self, vec: np.ndarray, top_k: int, qfilter: Any):
        """兼容新旧两代 qdrant-client。"""
        if hasattr(self._client, "query_points"):
            res = self._client.query_points(
                collection_name=self.collection,
                query=vec.tolist(),
                limit=top_k,
                query_filter=qfilter,
                with_payload=True,
                with_vectors=False,
            )
            return list(res.points)

        return list(
            self._client.search(  # type: ignore[attr-defined]
                collection_name=self.collection,
                query_vector=vec.tolist(),
                limit=top_k,
                query_filter=qfilter,
                with_payload=True,
                with_vectors=False,
            )
        )

    def _build_filter(self, where: dict[str, Any] | None):
        if not where:
            return None
        qm = self._models
        must = []
        for key, value in where.items():
            if isinstance(value, dict):
                # 支持 {"$in": [...]} / {"$gte": ...} 这类简单算子
                for op, operand in value.items():
                    if op in ("$in", "in"):
                        must.append(
                            qm.FieldCondition(key=str(key), match=qm.MatchAny(any=list(operand)))
                        )
                    elif op in ("$gte", ">="):
                        must.append(qm.FieldCondition(key=str(key), range=qm.Range(gte=operand)))
                    elif op in ("$lte", "<="):
                        must.append(qm.FieldCondition(key=str(key), range=qm.Range(lte=operand)))
                    else:
                        must.append(
                            qm.FieldCondition(key=str(key), match=qm.MatchValue(value=operand))
                        )
            else:
                must.append(qm.FieldCondition(key=str(key), match=qm.MatchValue(value=value)))
        return qm.Filter(must=must) if must else None

    def get(
        self,
        ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[VectorHit]:
        if ids:
            points = self._client.retrieve(
                collection_name=self.collection,
                ids=[str(i) for i in ids],
                with_payload=True,
                with_vectors=False,
            )
            return [
                VectorHit(id=str(p.id), score=1.0, metadata=dict(p.payload or {})) for p in points
            ]

        out: list[VectorHit] = []
        offset = None
        remaining = limit or 10000
        while remaining > 0:
            batch, offset = self._client.scroll(
                collection_name=self.collection,
                limit=min(remaining, 1000),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in batch:
                out.append(VectorHit(id=str(p.id), score=1.0, metadata=dict(p.payload or {})))
            remaining -= len(batch)
            if not batch or offset is None:
                break
        return out

    def count(self) -> int:
        try:
            return int(self._client.count(self.collection, exact=True).count)
        except Exception:
            return 0

    def update_payload(self, ids, payload: dict) -> None:
        if not ids:
            return
        self._client.set_payload(
            collection_name=self.collection,
            payload=dict(payload),
            points=[str(i) for i in ids],
        )

    def get_vectors(self, ids=None, limit: int | None = None) -> list[tuple[str, np.ndarray, dict]]:
        if ids:
            points = self._client.retrieve(
                collection_name=self.collection,
                ids=[str(i) for i in ids],
                with_payload=True,
                with_vectors=True,
            )
            return [
                (
                    str(p.id),
                    np.asarray(p.vector, dtype=np.float32),
                    dict(p.payload or {}),
                )
                for p in points
            ]

        out: list[tuple[str, np.ndarray, dict]] = []
        offset = None
        remaining = int(limit or 100000)

        while remaining > 0:
            batch, offset = self._client.scroll(
                collection_name=self.collection,
                limit=min(1000, remaining),
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in batch:
                out.append(
                    (
                        str(point.id),
                        np.asarray(point.vector, dtype=np.float32),
                        dict(point.payload or {}),
                    )
                )
            remaining -= len(batch)
            if not batch or offset is None:
                break
        return out

    def get_all_ids(self) -> list[str]:
        ids: list[str] = []
        offset = None
        while True:
            batch, offset = self._client.scroll(
                collection_name=self.collection,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.extend(str(p.id) for p in batch)
            if not batch or offset is None:
                break
        return ids

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
