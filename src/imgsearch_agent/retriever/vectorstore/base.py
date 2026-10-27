"""向量库抽象层。

上层（索引器、检索器）只依赖这里的 :class:`VectorStore` 接口，
不 import 任何 chromadb / qdrant 的具体代码。新增一种后端只需要：

1. 继承 ``VectorStore``
2. 实现 7 个方法
3. 在 ``factory.py`` 注册

接口约定
--------
- 向量一律是 L2 归一化后的 float32，距离度量统一为余弦。
- ``score`` 表示余弦相似度（越大越相似，范围约 [-1, 1]）。
- metadata 里保存图片的全部可检索字段（path / filename / caption ...）。
- ``add`` 是幂等 upsert：同 id 重复写入会覆盖，不会报错。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass
class VectorHit:
    """一条检索结果。"""

    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> str:
        return str(self.metadata.get("path", ""))

    @property
    def filename(self) -> str:
        return str(self.metadata.get("filename", ""))

    @property
    def caption(self) -> str:
        return str(self.metadata.get("caption", "") or "")


class VectorStoreError(RuntimeError):
    """向量库层的统一异常，便于上层给出可读提示。"""


class VectorStore(abc.ABC):
    """向量库统一接口。"""

    #: 后端标识，用于日志与界面展示
    backend_name: str = "base"

    def __init__(self, collection: str, embedding_dim: int):
        if not collection or len(collection) < 3:
            # chromadb 1.x 要求名字 3-512 字符，这里提前校验，两个后端保持一致
            raise ValueError(f"collection 名称至少 3 个字符，当前: {collection!r}")
        self.collection = collection
        self.embedding_dim = embedding_dim

    # ── 写 ──────────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def add(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        """写入或覆盖一批记录（upsert 语义）。"""

    @abc.abstractmethod
    def delete(self, ids: Sequence[str]) -> None:
        """按 id 删除。不存在的 id 应被静默忽略。"""

    @abc.abstractmethod
    def reset(self) -> None:
        """清空整个 collection，但保留 collection 本身。"""

    # ── 读 ──────────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def query(
        self,
        embedding: np.ndarray,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """ANN 检索，返回按相似度降序的 top_k 条。"""

    @abc.abstractmethod
    def get(
        self,
        ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[VectorHit]:
        """按 id 取记录；ids 为空时返回前 limit 条（用于遍历）。"""

    def get_vectors(
        self, limit: int | None = None
    ) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
        """返回 (id, 向量, metadata)，用于相似度分析。默认后端不支持时抛错。"""
        raise NotImplementedError(f"{self.backend_name} 不支持读取向量")

    @abc.abstractmethod
    def count(self) -> int:
        """记录总数。"""

    @abc.abstractmethod
    def get_all_ids(self) -> list[str]:
        """全部 id，用于增量索引时做差集。"""

    @abc.abstractmethod
    def close(self) -> None:
        """释放连接/文件锁。"""

    # ── 通用 ────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "collection": self.collection,
            "count": self.count(),
            "embedding_dim": self.embedding_dim,
        }

    def __len__(self) -> int:
        return self.count()

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ── 参数校验 ────────────────────────────────────────────────────────────

    def _check_batch(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        metadatas: Sequence[dict[str, Any]] | None,
    ) -> np.ndarray:
        if len(ids) != len(embeddings):
            raise VectorStoreError(
                f"ids 数量({len(ids)}) 与 embeddings 数量({len(embeddings)}) 不一致"
            )
        if metadatas is not None and len(metadatas) != len(ids):
            raise VectorStoreError(
                f"metadatas 数量({len(metadatas)}) 与 ids 数量({len(ids)}) 不一致"
            )
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.embedding_dim:
            raise VectorStoreError(
                f"向量维度不匹配：期望 {self.embedding_dim}，实际 {arr.shape[1]}。"
                "如果换过嵌入模型，需要重建索引。"
            )
        return arr

    def _check_query(self, embedding: np.ndarray) -> np.ndarray:
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.embedding_dim:
            raise VectorStoreError(
                f"查询向量维度不匹配：期望 {self.embedding_dim}，实际 {arr.shape[0]}"
            )
        return arr


# ── 工具函数 ──────────────────────────────────────────────────────────────────


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """按行做 L2 归一化。"""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr / max(norm, eps)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, eps)


def sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """把 metadata 收敛成两个后端都接受的标量类型。

    Chroma 不接受 None 和嵌套 dict；Qdrant 虽然接受，但统一处理更省心。
    """
    clean: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            clean[str(key)] = value
        elif isinstance(value, (list, tuple)):
            clean[str(key)] = ", ".join(str(v) for v in value)
        else:
            clean[str(key)] = str(value)
    return clean


def normalize_scores(scores: Iterable[float]) -> list[float]:
    """把一组余弦相似度映射到 [0,1]，用于和 LLM 分数融合。

    CLIP 的图文相似度实际分布很窄（常见 0.2~0.45），直接和 0~1 的 LLM 分
    相加会让 LLM 分数完全主导。这里做 min-max，并处理全等/单条的情况。
    """
    values = [float(s) for s in scores]
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-6:
        # 全部接近：退化为把余弦相似度截断到 [0,1]
        return [min(max(v, 0.0), 1.0) for v in values]
    return [(v - lo) / (hi - lo) for v in values]
