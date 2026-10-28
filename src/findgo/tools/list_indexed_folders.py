"""已索引目录清单工具。

从向量库读回所有记录的 path，按父目录聚合，
用于回答「我现在索引了哪些文件夹、各有多少张」。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .base_tool import BaseTool

logger = logging.getLogger("findgo.list_indexed_folders")


class ListIndexedFoldersArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(20000, ge=1, le=200000, description="最多扫描多少条记录")


class ListIndexedFoldersTool(BaseTool):
    name = "list_indexed_folders"
    description = "列出当前索引包含的文件夹，以及每个文件夹里的图片数量"
    args_schema = ListIndexedFoldersArgs

    def __init__(self, pipeline=None):
        self._pipeline = pipeline

    @property
    def pipeline(self):
        if self._pipeline is None:
            from ..retriever import build_pipeline

            self._pipeline = build_pipeline()
        return self._pipeline

    def run(self, limit: int = 20000) -> dict[str, Any]:
        hits = self.pipeline.store.get(limit=int(limit))

        folders: dict[str, int] = {}
        for hit in hits:
            path = hit.path
            if not path:
                continue
            folder = str(Path(path).parent)
            folders[folder] = folders.get(folder, 0) + 1

        items = [
            {"folder": folder, "count": count}
            for folder, count in sorted(folders.items(), key=lambda kv: (-kv[1], kv[0].lower()))
        ]
        return {
            "folders": items,
            "count": len(items),
            "total_images": len(hits),
            "summary": f"共 {len(items)} 个文件夹，{len(hits)} 张图片",
        }
