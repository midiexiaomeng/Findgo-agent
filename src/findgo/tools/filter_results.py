"""结果集二次过滤：在已有检索结果上做后置筛选，不碰向量库。"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .base_tool import BaseTool

logger = logging.getLogger("findgo.filter_results")


class FilterResultsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: list[str] = Field(..., min_length=1, description="上一轮结果的图片路径")
    exclude_staged: bool = Field(True, description="排除已移入暂存区的图片")
    min_similarity: float = Field(
        0.0, ge=0.0, le=1.0, description="最低相似度，需要同时提供 scores"
    )
    must_not_keywords: list[str] = Field(
        default_factory=list, description="描述里含这些词的图片会被排除"
    )
    scores: dict[str, float] = Field(
        default_factory=dict, description="路径到分数的映射，用于 min_similarity"
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict, description="结构化约束：core_query / must / must_not / should"
    )
    top_k: int = Field(0, ge=0, le=1000, description="最多返回几张，0 表示不限")


class FilterResultsTool(BaseTool):
    name = "filter_results"
    description = "对已有检索结果做二次过滤：排除暂存图、按相似度阈值、按描述关键词排除"
    permission = "read"
    args_schema = FilterResultsArgs

    def __init__(self, pipeline=None):
        self._pipeline = pipeline

    @property
    def pipeline(self):
        if self._pipeline is None:
            from ..retriever import build_pipeline

            self._pipeline = build_pipeline()
        return self._pipeline

    def run(
        self,
        paths: list[str],
        exclude_staged: bool = True,
        min_similarity: float = 0.0,
        must_not_keywords: list[str] | None = None,
        scores: dict[str, float] | None = None,
        constraints: dict | None = None,
        top_k: int = 0,
    ) -> dict[str, Any]:
        from ..agent.constraints import filter_by_constraints
        from ..retriever.indexer import Indexer

        keywords = [k.strip().lower() for k in (must_not_keywords or []) if k.strip()]
        scores = scores or {}
        constraints = constraints or {}

        # 结构化约束里的 must_not 优先于关键词排除
        must = [str(x) for x in (constraints.get("must") or []) if str(x).strip()]
        must_not = [str(x) for x in (constraints.get("must_not") or []) if str(x).strip()]

        ids = [Indexer.image_id(p) for p in paths]
        hits = self.pipeline.store.get(ids=ids)
        meta = {str(h.path).lower(): h for h in hits}

        kept: list[dict] = []
        dropped: list[dict] = []

        for path in paths:
            hit = meta.get(str(path).lower())
            caption = str(hit.metadata.get("caption", "") if hit else "")
            reason = ""

            if exclude_staged and hit and hit.metadata.get("is_staged"):
                reason = "已暂存"
            elif scores and path in scores and scores[path] < min_similarity:
                reason = f"相似度 {scores[path]:.3f} 低于 {min_similarity}"
            elif keywords and any(k in caption.lower() for k in keywords):
                reason = "描述命中排除词"

            if reason:
                dropped.append({"path": path, "reason": reason})
            else:
                kept.append(
                    {
                        "path": path,
                        "caption": caption,
                        "score": scores.get(path),
                    }
                )

        # 再走一遍结构化约束
        if must or must_not:
            kept, blocked = filter_by_constraints(kept, must_not, must)
            dropped.extend(
                {"path": b["path"], "reason": b.get("_drop_reason", "约束不满足")} for b in blocked
            )

        # 按原顺序保留，不完全重排
        order = {p.lower(): i for i, p in enumerate(paths)}
        kept.sort(key=lambda x: order.get(str(x["path"]).lower(), 9999))

        if top_k > 0:
            kept = kept[:top_k]

        images = [
            {
                "rank": i + 1,
                "path": item["path"],
                "caption": item["caption"],
                "score": float(item.get("score") or 0.0),
                "reason": "在上轮结果中命中筛选条件",
            }
            for i, item in enumerate(kept)
        ]

        confidence = sum(float(i.get("score") or 0.0) for i in kept) / len(kept) if kept else 0.0

        return {
            "kept": kept,
            "dropped": dropped,
            "count": len(kept),
            "images": images,
            "confidence": round(confidence, 3),
            "summary": f"保留 {len(kept)} 张，过滤掉 {len(dropped)} 张",
        }
