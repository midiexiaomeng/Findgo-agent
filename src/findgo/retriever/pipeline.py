"""组装层：把编码器、向量库、重排器、索引器、检索器拼成一个对象。

界面和 CLI 只跟 Pipeline 打交道。切换向量库后端也在这里完成，
上层代码不需要知道自己用的是 Chroma 还是 Qdrant。
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from .captioner import Captioner
from .clip_encoder import ClipEncoder
from .config import Config
from .indexer import Indexer, IndexStats
from .llm.client import LLMClient
from .llm.reranker import LLMReranker
from .runtime import describe_device, setup_logging
from .searcher import Searcher, SearchResponse
from .vectorstore import create_vector_store
from .vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, cfg: Config | None = None, backend: str | None = None):
        self.cfg = cfg or Config.from_yaml()
        setup_logging(self.cfg.app.log_level)

        self._lock = threading.RLock()
        self._health_cache: tuple[float, dict] = (0.0, {})
        self.encoder = ClipEncoder(self.cfg)
        self.captioner = Captioner(self.cfg)
        self.reranker = LLMReranker(self.cfg)
        self.store: VectorStore = create_vector_store(self.cfg, backend)
        self._bind()

    def _bind(self) -> None:
        self.indexer = Indexer(self.cfg, self.encoder, self.store, self.captioner)
        self.searcher = Searcher(self.cfg, self.encoder, self.store, self.reranker, self.captioner)

    # ── 检索 ────────────────────────────────────────────────────────────────

    def search_text(self, query: str, **kwargs: Any) -> SearchResponse:
        return self.searcher.search_text(query, **kwargs)

    def search_image(self, image_path: str | Path, **kwargs: Any) -> SearchResponse:
        return self.searcher.search_image(image_path, **kwargs)

    def index_folder(self, folder: str | Path, **kwargs: Any) -> IndexStats:
        with self._lock:
            return self.indexer.index_folder(folder, **kwargs)

    # ── 后端切换 ────────────────────────────────────────────────────────────

    def switch_backend(self, backend: str) -> str:
        """切换向量库。

        顺序是「先建新连接 → 自检 → 再关旧连接」，任何一步失败都保留原后端，
        避免出现旧库已关闭、新库又不可用的空档。

        注意：两个库的数据是各自独立的，切换后需要重新索引。
        """
        with self._lock:
            if backend == self.store.backend_name:
                return f"当前已经是 {backend}"

            old = self.store

            # 1. 创建新连接（旧连接保持不动）
            try:
                new_store = create_vector_store(self.cfg, backend)
            except Exception as e:
                logger.warning("切换 %s 失败，保留原后端: %s", backend, e)
                raise RuntimeError(f"目标后端 {backend} 初始化失败，仍使用 {old.backend_name}：{e}") from e

            # 2. 写入 + 查询 + 删除，确认真能读写
            try:
                self._probe_backend(new_store)
            except Exception as e:
                try:
                    new_store.close()
                except Exception:
                    pass
                logger.warning("目标后端 %s 自检失败，保留原后端: %s", backend, e)
                raise RuntimeError(f"目标后端 {backend} 自检失败，仍使用 {old.backend_name}：{e}") from e

            # 3. 自检通过，关掉旧连接再切换
            try:
                old.close()
            except Exception:
                pass

            self.store = new_store
            self._bind()
            self.cfg.vector_store.backend = backend
            logger.info("向量库已切换: %s", backend)
            return (
                f"已切换到 {backend}（collection={self.store.collection}，"
                f"现有 {self.store.count()} 条）。两个后端的数据不共享，需要重新索引。"
            )

    @staticmethod
    def _probe_backend(store) -> None:
        """写入一条探测记录，确认能查能删。"""
        import uuid

        import numpy as np

        probe_id = f"__probe__{uuid.uuid4().hex}"
        dim = int(getattr(store, "embedding_dim", 512) or 512)
        vector = np.random.rand(dim).astype(np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-8)

        store.add(
            [probe_id],
            vector.reshape(1, -1),
            [{"path": "__probe__", "filename": "__probe__"}],
        )

        hits = store.query(vector, top_k=1)
        if not hits:
            raise RuntimeError("写入成功但查询不到记录")

        store.delete([probe_id])
        leftover = [h for h in store.query(vector, top_k=1) if h.id == probe_id]
        if leftover:
            raise RuntimeError("删除后仍能查到探测记录")

    # ── 状态 ────────────────────────────────────────────────────────────────

    def warmup(self, load_clip: bool = True) -> None:
        """预热：提前把 CLIP 加载进显存，第一次搜索就不用等。"""
        if load_clip:
            self.encoder.load()

    def health(self, ttl: float = 30.0) -> dict:
        """给界面顶部状态栏用的体检结果。带 TTL，避免每次搜索都探测 LLM。"""
        now = time.time()
        cached_at, cached = self._health_cache
        if cached and now - cached_at < ttl:
            return cached

        ok, llm_msg = (False, "未检查")
        try:
            client = LLMClient(self.cfg)
            ok, llm_msg = client.is_available()
            if not self.cfg.rerank.enabled:
                ok, llm_msg = False, "配置中已关闭重排"
        except Exception as e:
            llm_msg = f"检查失败: {e}"

        result = {
            "device": describe_device(self.encoder.device)
            if self.encoder.loaded
            else (describe_device(_resolve_device(self.cfg))),
            "encoder_loaded": self.encoder.loaded,
            "llm_ok": ok,
            "llm_message": llm_msg,
            "caption_enabled": self.cfg.caption.enabled,
            "rerank_enabled": self.cfg.rerank.enabled,
        }
        self._health_cache = (now, result)
        return result

    def stats(self) -> dict:
        return self.searcher.stats()

    def close(self) -> None:
        with self._lock:
            try:
                self.store.close()
            except Exception:
                pass
            try:
                self.captioner.close()
            except Exception:
                pass
            try:
                self.encoder.unload()
            except Exception:
                pass

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _resolve_device(cfg: Config) -> str:
    from .runtime import pick_device

    try:
        return pick_device(cfg.device)
    except Exception:
        return "cpu"


def build_pipeline(config_path: str | Path | None = None, backend: str | None = None) -> Pipeline:
    cfg = Config.from_yaml(config_path)
    return Pipeline(cfg, backend=backend)
