"""索引内图片数量统计：逻辑视图，不含磁盘校验。

与 folder_scan 的区别：
- folder_scan：扫磁盘实际文件
- count_indexed_images：查向量库里有多少条记录，排除已暂存的
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..utils.path_guard import PathSecurityError, PathGuard
from .base_tool import BaseTool


class CountIndexedArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folder_path: str = Field("", description="只统计这个目录，留空统计全库")


class CountIndexedImagesTool(BaseTool):
    name = "count_indexed_images"
    description = "统计向量索引内的图片数量，可对比磁盘实际数量排查不一致"
    permission = "read"
    args_schema = CountIndexedArgs

    def __init__(self, pipeline=None, guard: PathGuard | None = None):
        self._pipeline = pipeline
        self.guard = guard or PathGuard()

    @property
    def pipeline(self):
        if self._pipeline is None:
            from ..retriever import build_pipeline

            self._pipeline = build_pipeline()
        return self._pipeline

    def run(self, folder_path: str = "") -> dict[str, Any]:
        hits = self.pipeline.store.get(limit=200000)

        if folder_path:
            try:
                scope = str(self.guard.check(folder_path, "read")).lower()
            except PathSecurityError as e:
                return {
                    "total": 0,
                    "effective": 0,
                    "staged": 0,
                    "summary": f"路径不安全，已拒绝：{e}",
                }
            hits = [
                h for h in hits if str(h.path).lower().startswith(scope)
            ]

        staged = sum(1 for h in hits if h.metadata.get("is_staged"))
        missing = sum(1 for h in hits if h.path and not _exists(h.path))

        return {
            "total": len(hits),
            "staged": staged,
            "effective": len(hits) - staged,
            "missing_files": missing,
            "summary": (
                f"索引内 {len(hits)} 条，已暂存 {staged} 条，"
                f"有效 {len(hits) - staged} 条"
                + (f"，原图缺失 {missing} 条" if missing else "")
            ),
        }


def _exists(path: str) -> bool:
    from pathlib import Path

    try:
        return Path(path).exists()
    except OSError:
        return False
