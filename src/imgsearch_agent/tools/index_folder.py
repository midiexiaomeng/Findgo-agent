"""目录索引工具：对指定文件夹增量构建图像索引。"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from ..utils.path_guard import PathGuard, PathSecurityError
from .base_tool import BaseTool

logger = logging.getLogger("imgsearch_agent.index_folder")


class IndexFolderArgs(BaseModel):
    folder_path: str = Field(..., description="要建立索引的文件夹路径")
    rebuild: bool = Field(False, description="True 全量重建，False 增量更新")


class IndexFolderTool(BaseTool):
    name = "index_folder"
    description = "对指定文件夹构建图像索引；默认增量，已索引过的图片自动跳过"
    args_schema = IndexFolderArgs

    def __init__(self, pipeline=None, guard: PathGuard | None = None):
        self._pipeline = pipeline
        self.guard = guard or PathGuard()

    @property
    def pipeline(self):
        if self._pipeline is None:
            from ..retriever import build_pipeline

            self._pipeline = build_pipeline()
        return self._pipeline

    def run(self, folder_path: str, rebuild: bool = False) -> dict[str, Any]:
        try:
            folder = self.guard.check(folder_path)
        except PathSecurityError as e:
            return {
                "count": 0,
                "total": 0,
                "failed": 0,
                "summary": f"路径不安全，已拒绝：{e}",
                "error": str(e),
            }

        if not folder.is_dir():
            return {
                "count": 0,
                "total": 0,
                "failed": 0,
                "summary": f"{folder} 不是文件夹",
            }

        stats = self.pipeline.index_folder(
            folder,
            recursive=True,
            reset=rebuild,
            skip_existing=not rebuild,
        )
        return {
            "count": stats.indexed,
            "total": stats.total,
            "failed": stats.failed,
            "summary": stats.summary(),
        }
