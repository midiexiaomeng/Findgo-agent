"""长期向量记忆：跨会话的稳定偏好，存 Qdrant。

和 agent/memory.py 的分工：
- Memory：会话消息、任务链、结构化事实、摘要（sqlite）
- LongTermMemory：跨会话事实（Qdrant），只做参考资料，不覆盖结构化事实

写入前由 LLM 判断是否属于长期偏好，闲聊和一次性任务不入库。
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass

logger = logging.getLogger("imgsearch_agent.long_term")


@dataclass
class LongTermConfig:
    collection: str = "memories"
    top_k: int = 5
    similarity_threshold: float = 0.25
    enabled: bool = True


class LongTermMemory:
    def __init__(self, config: LongTermConfig | None = None, store=None):
        self.config = config or LongTermConfig()
        self._store = store

    @property
    def store(self):
        if self._store is None:
            from .retriever.config import Config
            from .retriever.vectorstore.qdrant_store import QdrantVectorStore

            cfg = Config.from_yaml()
            self._store = QdrantVectorStore(
                collection=self.config.collection,
                embedding_dim=int(cfg.embedding.embedding_dim),
                mode=cfg.vector_store.qdrant.mode,
                path=cfg.resolve(cfg.vector_store.qdrant.path),
            )
        return self._store

    def add(
        self,
        user_id: str,
        session_id: str,
        text: str,
        vector,
        tags: list[str] | None = None,
    ) -> str:
        memory_id = f"mem-{uuid.uuid4().hex[:16]}"
        self.store.add(
            [memory_id],
            vector.reshape(1, -1),
            [
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "text": text,
                    "tags": ",".join(tags or []),
                    "created_at": time.time(),
                }
            ],
        )
        logger.info("长期记忆写入 user=%s session=%s", user_id, session_id)
        return memory_id

    def search(self, user_id: str, vector, top_k: int | None = None) -> list[dict]:
        k = top_k or self.config.top_k
        hits = self.store.query(vector, top_k=max(k * 3, 10))
        results = []
        for hit in hits:
            meta = hit.metadata
            if str(meta.get("user_id", "")) != user_id:
                continue
            if float(hit.score) < self.config.similarity_threshold:
                continue
            results.append(
                {
                    "id": hit.id,
                    "text": meta.get("text", ""),
                    "tags": meta.get("tags", ""),
                    "score": round(float(hit.score), 4),
                }
            )
            if len(results) >= k:
                break
        return results

    def delete(self, memory_id: str) -> None:
        self.store.delete([memory_id])
        logger.info("长期记忆删除 %s", memory_id)

    def clear(self, user_id: str) -> int:
        all_ids = self.store.get_all_ids()
        if not all_ids:
            return 0
        records = self.store.get(ids=all_ids)
        targets = [
            r.id for r in records if str(r.metadata.get("user_id", "")) == user_id
        ]
        if targets:
            self.store.delete(targets)
        logger.info("清除用户 %s 的长期记忆 %d 条", user_id, len(targets))
        return len(targets)

    def list_for_user(self, user_id: str) -> list[dict]:
        all_ids = self.store.get_all_ids()
        if not all_ids:
            return []
        records = self.store.get(ids=all_ids)
        return [
            {
                "id": r.id,
                "text": r.metadata.get("text", ""),
                "tags": r.metadata.get("tags", ""),
                "session_id": r.metadata.get("session_id", ""),
                "created_at": r.metadata.get("created_at", 0),
            }
            for r in records
            if str(r.metadata.get("user_id", "")) == user_id
        ]
