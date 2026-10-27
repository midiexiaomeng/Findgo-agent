"""目录扫描工具：扫描文件夹里的图片并返回统计。

这是项目里第二个 Agent 工具，用来演示「新增能力只需继承 BaseTool
并在编排器注册」的扩展方式。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..utils.image_scan import scan_images
from ..utils.path_guard import PathGuard, PathSecurityError
from .base_tool import BaseTool

logger = logging.getLogger("imgsearch_agent.folder_scan")


class FolderScanArgs(BaseModel):
    path: str = Field(..., description="要扫描的文件夹路径")
    recursive: bool = Field(True, description="是否包含子目录")


class FolderScanTool(BaseTool):
    name = "folder_scan"
    description = "扫描本地文件夹，统计其中有多少张图片并列出示例路径"
    args_schema = FolderScanArgs

    def __init__(self, guard: PathGuard | None = None):
        self.guard = guard or PathGuard()

    def run(self, path: str, recursive: bool = True) -> dict[str, Any]:
        try:
            folder = self.guard.check(path, "read")
        except PathSecurityError as e:
            logger.warning("路径校验拒绝: %s", e)
            return {
                "count": 0,
                "images": [],
                "summary": f"路径不安全，已拒绝：{e}",
                "error": str(e),
            }

        if not folder.is_dir():
            return {
                "count": 0,
                "images": [],
                "summary": f"{folder} 不是文件夹",
            }

        images = scan_images(folder, recursive=recursive)
        return {
            "count": len(images),
            "images": [str(p) for p in images[:30]],
            "summary": f"{folder} 下共找到 {len(images)} 张图片",
        }
