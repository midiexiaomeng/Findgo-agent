"""图像检索工具：把 retriever 的多模态检索封装成标准 Tool。

Agent 只看到 name / description / args_schema / run，
不关心底层是 CLIP + 向量库 + 重排。
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..utils.path_guard import PathGuard, PathSecurityError
from ..utils.schemas import RetrievalResult, RetrievedImage
from .base_tool import BaseTool

logger = logging.getLogger("imgsearch_agent.image_retrieval")


class ImageRetrievalArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="图片内容描述")
    top_k: int = Field(8, ge=1, le=1000, description="返回数量，最大 1000")
    folder_path: str = Field("", description="限定检索的文件夹路径，空表示全库")
    all_results: bool = Field(False, description="返回全部匹配结果，忽略 top_k")
    path: str = Field("", description="（兼容）同 folder_path")


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

    def _ensure_indexed(self, folder) -> None:
        """用户指定新目录时自动建索引；已索引过的目录直接跳过。"""
        key = str(folder.resolve()).lower()
        if key in self._indexed:
            return

        logger.info("新目录，开始增量索引: %s", folder)
        last = {"pct": -1}

        def progress(done: int, total: int, name: str) -> None:
            pct = int(done * 100 / max(total, 1))
            if pct >= last["pct"] + 10 or done == total:
                last["pct"] = pct
                logger.info("索引进度 %d/%d (%d%%) %s", done, total, pct, name)

        try:
            stats = self.pipeline.index_folder(
                folder,
                recursive=True,
                skip_existing=True,
                progress=progress,
            )
            logger.info("索引完成: %s", stats.summary())
        except Exception as e:
            logger.warning("自动索引 %s 失败: %s", folder, e)
        self._indexed.add(key)

    def run(
        self,
        query: str,
        top_k: int = 8,
        folder_path: str = "",
        path: str = "",
        all_results: bool = False,
        should_stop=None,
    ) -> dict[str, Any]:
        target = folder_path or path
        scope = ""
        if target:
            try:
                folder = self.guard.check(target, "read")
            except PathSecurityError as e:
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
            query,
            top_k=top_k,
            scope=scope or None,
            all_results=all_results,
            should_stop=should_stop,
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
