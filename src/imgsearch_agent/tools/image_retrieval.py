"""图像检索工具：把 retriever 的多模态检索封装成标准 Tool。

Agent 只看到 name / description / args_schema / run，
不关心底层是 CLIP + 向量库 + 重排。
"""

from __future__ import annotations

import logging

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..utils.path_guard import PathGuard, PathSecurityError
from ..utils.schemas import RetrievalResult, RetrievedImage
from .base_tool import BaseTool

logger = logging.getLogger("imgsearch_agent.image_retrieval")


class ImageRetrievalArgs(BaseModel):
    query: str = Field(..., description="图片内容描述")
    top_k: int = Field(8, description="返回数量")
    path: str = Field("", description="限定检索的文件夹路径，空表示全库")


class ImageRetrievalTool(BaseTool):
    name = "image_retrieval"
    description = "用自然语言描述搜索本地图片库，返回图片路径、描述与相似度"
    args_schema = ImageRetrievalArgs

    def __init__(self, pipeline=None, guard: PathGuard | None = None):
        self._pipeline = pipeline
        self._indexed: set[str] = set()
        self.guard = guard or PathGuard()

    @property
    def pipeline(self):
        if self._pipeline is None:
            # 延迟加载：只有真正执行检索时才初始化模型与向量库
            from ..retriever import build_pipeline

            self._pipeline = build_pipeline()
        return self._pipeline

    def _ensure_indexed(self, folder: Path) -> None:
        """用户指定新目录时自动建索引；已索引过的目录直接跳过。"""
        key = str(folder.resolve()).lower()
        if key in self._indexed:
            return
        try:
            from ..retriever.indexer import Indexer

            pipeline = self.pipeline
            Indexer(
                pipeline.cfg, pipeline.encoder, pipeline.store, pipeline.captioner
            ).index_folder(folder, recursive=True, skip_existing=True)
        except Exception as e:
            logger.warning("自动索引 %s 失败: %s", folder, e)
        self._indexed.add(key)

    def run(self, query: str, top_k: int = 8, path: str = "") -> dict[str, Any]:
        scope = ""
        if path:
            try:
                folder = self.guard.check(path)
            except PathSecurityError as e:
                # 越界路径直接拒绝，不做任何索引动作
                logger.warning("路径校验拒绝: %s", e)
                return RetrievalResult(
                    query=query,
                    images=[],
                    confidence=0.0,
                    summary=f"路径不安全，已拒绝：{e}",
                ).model_dump()
            if folder.is_dir():
                self._ensure_indexed(folder)
                scope = str(folder)

        resp = self.pipeline.searcher.search_text(
            query, top_k=top_k, scope=scope or None
        )
        images = [
            RetrievedImage(
                rank=r.rank,
                path=r.path,
                caption=r.caption or "",
                score=round(float(r.score), 3),
            )
            for r in resp.results
        ]
        confidence = (
            sum(i.score for i in images) / len(images) if images else 0.0
        )
        result = RetrievalResult(
            query=query,
            images=images,
            confidence=round(confidence, 3),
            summary=resp.status_line(),
        )
        return result.model_dump()
