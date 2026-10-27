"""以图搜图工具。"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..utils.path_guard import PathGuard, PathSecurityError
from ..utils.schemas import RetrievalResult, RetrievedImage
from .base_tool import BaseTool

logger = logging.getLogger("imgsearch_agent.image_search_by_image")


class ImageSearchByImageArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_path: str = Field(..., min_length=1, description="查询图片的本地路径")
    top_k: int = Field(8, ge=1, le=50, description="返回数量")
    folder_path: str = Field("", description="限定检索的文件夹路径，空表示全库")


class ImageSearchByImageTool(BaseTool):
    name = "image_search_by_image"
    description = "以图搜图：给定一张图片，找出本地图库中视觉相似的图片"
    args_schema = ImageSearchByImageArgs

    def __init__(self, pipeline=None, guard: PathGuard | None = None):
        self._pipeline = pipeline
        self.guard = guard or PathGuard()

    @property
    def pipeline(self):
        if self._pipeline is None:
            from ..retriever import build_pipeline

            self._pipeline = build_pipeline()
        return self._pipeline

    def run(
        self, image_path: str, top_k: int = 8, folder_path: str = ""
    ) -> dict[str, Any]:
        try:
            image = self.guard.check(image_path)
        except PathSecurityError as e:
            return RetrievalResult(
                query=image_path,
                images=[],
                confidence=0.0,
                summary=f"路径不安全，已拒绝：{e}",
            ).model_dump()

        if not image.is_file():
            return RetrievalResult(
                query=image_path,
                images=[],
                confidence=0.0,
                summary=f"{image} 不是文件",
            ).model_dump()

        scope = ""
        if folder_path:
            try:
                scope = str(self.guard.check(folder_path))
            except PathSecurityError as e:
                return RetrievalResult(
                    query=image_path,
                    images=[],
                    confidence=0.0,
                    summary=f"路径不安全，已拒绝：{e}",
                ).model_dump()

        resp = self.pipeline.searcher.search_image(
            str(image), top_k=top_k, scope=scope or None
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
        confidence = sum(i.score for i in images) / len(images) if images else 0.0
        return RetrievalResult(
            query=image_path,
            images=images,
            confidence=round(confidence, 3),
            summary=resp.status_line(),
        ).model_dump()
