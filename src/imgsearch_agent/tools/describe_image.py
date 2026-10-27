"""单图描述工具：用 BLIP 给一张本地图片生成中文描述。"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..utils.path_guard import PathGuard, PathSecurityError
from .base_tool import BaseTool

logger = logging.getLogger("imgsearch_agent.describe_image")


class DescribeImageArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_path: str = Field(..., min_length=1, description="本地图片路径")


class DescribeImageTool(BaseTool):
    name = "describe_image"
    description = "给一张本地图片生成中文描述，用于回答「这张图是什么」"
    args_schema = DescribeImageArgs

    def __init__(self, pipeline=None, guard: PathGuard | None = None):
        self._pipeline = pipeline
        self.guard = guard or PathGuard()

    @property
    def pipeline(self):
        if self._pipeline is None:
            from ..retriever import build_pipeline

            self._pipeline = build_pipeline()
        return self._pipeline

    def run(self, image_path: str) -> dict[str, Any]:
        try:
            image = self.guard.check(image_path, "read")
        except PathSecurityError as e:
            return {
                "path": image_path,
                "caption": "",
                "summary": f"路径不安全，已拒绝：{e}",
            }

        if not image.is_file():
            return {"path": str(image), "caption": "", "summary": f"{image} 不是文件"}

        try:
            caption = self.pipeline.captioner.caption_image(str(image))
        except Exception as e:
            logger.warning("生成描述失败: %s", e)
            return {
                "path": str(image),
                "caption": "",
                "summary": f"生成描述失败：{e}",
            }

        return {
            "path": str(image),
            "caption": caption,
            "summary": caption or "未能生成描述",
        }
