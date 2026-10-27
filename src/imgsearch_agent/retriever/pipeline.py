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
        self.searcher = Searcher(
            self.cfg, self.encoder, self.store, self.reranker, self.captioner
        )

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

        注意：两个库的数据是各自独立的，切换后需要重新索引。
        这里只换连接，不搬数据；搬数据应该由用户显式决定。
        """
        with self._lock:
            if backend == self.store.backend_name:
                return f"当前已经是 {backend}"

            old = self.store
            try:
                old.close()
            except Exception:
                pass

            self.store = create_vector_store(self.cfg, backend)
            self._bind()
            self.cfg.vector_store.backend = backend
            logger.info("向量库已切换: %s", backend)
            return (
                f"已切换到 {backend}（collection={self.store.collection}，"
                f"现有 {self.store.count()} 条）。两个后端的数据不共享，需要重新索引。"
            )

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
            "device": describe_device(self.encoder.device) if self.encoder.loaded else (
                describe_device(_resolve_device(self.cfg))
            ),
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
