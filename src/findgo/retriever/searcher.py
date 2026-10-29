"""检索器：把三级链路串起来。

以文搜图：
    文本 -> CLIP 文本向量 -> ANN 召回 Top-K -> LLM 结合描述精排

以图搜图：
    查询图 -> CLIP 图像向量 -> ANN 召回 Top-K
           -> BLIP 描述查询图 -> 用描述作为语义查询交给 LLM 精排

为什么以图搜图也要先生成描述：LLM 读不了图（Qwen2.5-7B 是纯文本模型），
必须把查询图翻译成一段文字，它才有东西可以和候选描述比对。这一步用的是
同一个 BLIP，所以额外开销只有一张图。
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .captioner import Captioner
from .clip_encoder import ClipEncoder
from .config import Config
from .indexer import Indexer
from .llm.reranker import LLMReranker, RerankReport
from .runtime import human_time
from .vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    rank: int
    id: str
    path: str
    filename: str
    caption: str
    vec_score: float
    llm_score: float | None
    score: float
    exists: bool
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "rank": self.rank,
            "id": self.id,
            "path": self.path,
            "filename": self.filename,
            "caption": self.caption,
            "vec_score": round(float(self.vec_score), 4),
            "llm_score": None if self.llm_score is None else round(float(self.llm_score), 4),
            "score": round(float(self.score), 4),
            "exists": self.exists,
        }


@dataclass
class SearchResponse:
    query: str
    mode: str
    results: list[SearchResult]
    report: RerankReport
    query_caption: str = ""
    recall_count: int = 0
    elapsed: float = 0.0
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "mode": self.mode,
            "query_caption": self.query_caption,
            "recall_count": self.recall_count,
            "elapsed": round(self.elapsed, 2),
            "note": self.note,
            "rerank": {
                "used": self.report.used,
                "llm_scored": self.report.llm_scored,
                "candidates": self.report.candidates,
                "errors": self.report.errors,
            },
            "results": [r.as_dict() for r in self.results],
        }

    def status_line(self) -> str:
        parts = [
            f"召回 {self.recall_count} 条",
            f"返回 {len(self.results)} 张",
            human_time(self.elapsed),
        ]
        if self.report.used:
            parts.insert(1, f"LLM 精排 {self.report.llm_scored}/{self.report.candidates}")
        elif self.report.candidates:
            parts.insert(1, "纯向量排序")
        return " · ".join(parts)


class Searcher:
    def __init__(
        self,
        cfg: Config,
        encoder: ClipEncoder,
        store: VectorStore,
        reranker: LLMReranker,
        captioner: Captioner,
    ):
        self.cfg = cfg
        self.encoder = encoder
        self.store = store
        self.reranker = reranker
        self.captioner = captioner

    # ── 以文搜图 ────────────────────────────────────────────────────────────

    def search_text(
        self,
        query: str,
        top_k: int | None = None,
        recall_k: int | None = None,
        use_rerank: bool = True,
        min_score: float | None = None,
        scope: str | None = None,
        all_results: bool = False,
        should_stop=None,
        exclude_paths: set[str] | None = None,
        constraints: dict | None = None,
    ) -> SearchResponse:
        query = (query or "").strip()
        started = time.time()
        if not query and not constraints:
            return SearchResponse(
                query="",
                mode="text",
                results=[],
                report=RerankReport(),
                note="查询为空",
            )

        logger.info("以文搜图: %s", query)
        top_k = int(top_k or self.cfg.search.top_k)
        recall_k = int(recall_k or self.cfg.search.recall_k)
        min_score = self.cfg.search.min_score if min_score is None else float(min_score)

        if all_results:
            limit = int(getattr(self.cfg.search, "max_all", 1000))
            top_k = max(top_k, min(self.store.count(), limit))

        # 要得多就得召回得多，否则后面没有候选可排
        recall_k = max(recall_k, min(top_k * 2, 5000))

        query_vec = self.encoder.encode_text(query)
        hits = self._recall(query_vec, recall_k, min_score, scope, exclude_paths)

        candidates, report = self.reranker.rerank(
            query,
            hits,
            top_k=top_k,
            use_llm=use_rerank,
            should_stop=should_stop,
            constraints=constraints,
        )

        return SearchResponse(
            query=query,
            mode="text",
            results=self._to_results(candidates),
            report=report,
            recall_count=len(hits),
            elapsed=time.time() - started,
            note="; ".join(report.errors[:3]),
        )

    # ── 以图搜图 ────────────────────────────────────────────────────────────

    def search_image(
        self,
        image_path: str | Path,
        top_k: int | None = None,
        recall_k: int | None = None,
        use_rerank: bool = True,
        min_score: float | None = None,
        exclude_self: bool = True,
        scope: str | None = None,
    ) -> SearchResponse:
        image_path = str(image_path)
        started = time.time()
        top_k = int(top_k or self.cfg.search.top_k)
        recall_k = int(recall_k or self.cfg.search.recall_k)
        min_score = self.cfg.search.min_score if min_score is None else float(min_score)

        logger.info("以图搜图: %s", image_path)
        query_vec = self.encoder.encode_image(image_path)

        # 多取一条，避免排除自身后不够 top_k
        extra = 1 if exclude_self else 0
        hits = self._recall(query_vec, recall_k + extra, min_score, scope)

        if exclude_self:
            try:
                self_abs = str(Path(image_path).expanduser().resolve()).lower()
            except OSError:
                self_abs = image_path.lower()
            hits = [h for h in hits if str(h.metadata.get("path", "")).lower() != self_abs]

        # 查询图的语义描述：LLM 重排的唯一依据
        query_caption = ""
        notes: list[str] = []
        can_rerank = use_rerank
        if use_rerank:
            if self.cfg.caption.enabled:
                try:
                    query_caption = self.captioner.caption_image(image_path)
                except Exception as e:
                    notes.append(f"查询图描述生成失败: {e}")
            else:
                notes.append("未启用 BLIP，无法生成查询图描述")

            if not query_caption:
                can_rerank = False
                notes.append("没有查询图描述，跳过 LLM 重排（纯向量结果）")

        candidates, report = self.reranker.rerank(
            query_caption, hits, top_k=top_k, use_llm=can_rerank
        )

        return SearchResponse(
            query=image_path,
            mode="image",
            query_caption=query_caption,
            results=self._to_results(candidates),
            report=report,
            recall_count=len(hits),
            elapsed=time.time() - started,
            note="; ".join(notes + report.errors[:2]),
        )

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _recall(
        self,
        query_vec,
        recall_k: int,
        min_score: float,
        scope: str | None = None,
        exclude_paths: set[str] | None = None,
    ):
        # 指定目录时多召回一些，过滤后再截断；软删除记录也占名额，留出余量
        limit = recall_k * 3 if scope else recall_k
        limit = int(limit * 1.5) + 20
        if exclude_paths:
            limit += len(exclude_paths)

        # 硬上限：排除列表很长时不要无限拉大查询，
        # 多出来的部分靠二次检索补齐，而不是一次拉几千条进内存
        search_cfg = getattr(getattr(self, "cfg", None), "search", None)
        cap = int(getattr(search_cfg, "max_recall_limit", 800))
        limit = min(limit, max(cap, recall_k))

        hits = self.store.query(query_vec, top_k=max(1, limit))
        logger.info(
            "召回: query_top=%d 实际返回=%d 排除=%d limit=%d",
            limit,
            len(hits),
            len(exclude_paths or ()),
            limit,
        )

        # 已暂存待删的图片不再参与检索
        hits = [h for h in hits if not h.metadata.get("is_staged")]

        # 上一轮已经返回过的图片，本轮排除
        if exclude_paths:
            normalized = {str(p).lower() for p in exclude_paths}
            hits = [h for h in hits if str(h.path).lower() not in normalized]

        if scope:
            try:
                scope_norm = str(Path(scope).expanduser().resolve()).lower()
            except OSError:
                scope_norm = str(scope).lower()
            hits = [
                h for h in hits if str(h.metadata.get("path", "")).lower().startswith(scope_norm)
            ]
            hits = hits[:recall_k]

        if min_score and min_score > 0:
            hits = [h for h in hits if float(h.score) >= min_score]
        logger.info("召回过滤后: %d 条（scope=%s）", len(hits), scope or "全库")
        return hits

    @staticmethod
    def _to_results(candidates: Sequence) -> list[SearchResult]:
        results: list[SearchResult] = []
        for rank, cand in enumerate(candidates, start=1):
            meta = cand.metadata or {}
            path = str(meta.get("path", ""))
            results.append(
                SearchResult(
                    rank=rank,
                    id=cand.id,
                    path=path,
                    filename=str(meta.get("filename", "")),
                    caption=str(meta.get("caption", "") or ""),
                    vec_score=float(cand.vec_score),
                    llm_score=cand.llm_score,
                    score=float(cand.final_score),
                    exists=bool(path) and os.path.exists(path),
                    metadata=meta,
                )
            )
        return results

    # ── 统计 ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        store_stats = self.store.stats()
        return {
            "images": store_stats["count"],
            "backend": store_stats["backend"],
            "collection": store_stats["collection"],
            "dim": store_stats["embedding_dim"],
            "device": self.encoder.device,
            "encoder_loaded": self.encoder.loaded,
            "llm": self.reranker.client.describe_backend(),
            "rerank_enabled": bool(self.cfg.rerank.enabled),
            "caption": self.captioner.stats(),
        }

    def reindex_paths(self, paths: Sequence[str]) -> int:
        """给界面上的"单张补索引"用。"""
        indexer = Indexer(self.cfg, self.encoder, self.store, self.captioner)
        stats = indexer.index_paths(paths)
        return stats.indexed
