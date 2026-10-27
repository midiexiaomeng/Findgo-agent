"""图像检索工具：把 retriever 的多模态检索封装成标准 Tool。

Agent 只看到 name / description / args_schema / run，
不关心底层是 CLIP + 向量库 + 重排。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..utils.schemas import RetrievalResult, RetrievedImage
from .base_tool import BaseTool


class ImageRetrievalArgs(BaseModel):
    query: str = Field(..., description="图片内容描述")
    top_k: int = Field(8, description="返回数量")


class ImageRetrievalTool(BaseTool):
    name = "image_retrieval"
    description = "用自然语言描述搜索本地图片库，返回图片路径、描述与相似度"
    args_schema = ImageRetrievalArgs

    def __init__(self, pipeline=None):
        self._pipeline = pipeline

    @property
    def pipeline(self):
        if self._pipeline is None:
            # 延迟加载：只有真正执行检索时才初始化模型与向量库
            from ..retriever import build_pipeline

            self._pipeline = build_pipeline()
        return self._pipeline

    def run(self, query: str, top_k: int = 8) -> dict[str, Any]:
        resp = self.pipeline.search_text(query, top_k=top_k)
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
