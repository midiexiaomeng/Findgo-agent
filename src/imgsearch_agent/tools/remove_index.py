"""索引删除工具：按目录移除向量库里的记录，不动原始图片。

默认 dry_run=True，只统计会删多少条，确认后再改成 false 真正删除。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..utils.path_guard import PathGuard, PathSecurityError
from .base_tool import BaseTool

logger = logging.getLogger("imgsearch_agent.remove_index")


class RemoveIndexArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folder_path: str = Field("", description="要移除索引的文件夹；留空时需开启 missing_only")
    dry_run: bool = Field(True, description="True 只统计不删除，False 真正删除")
    missing_only: bool = Field(
        False, description="只清理原图已不存在的孤儿向量"
    )


class RemoveIndexTool(BaseTool):
    name = "remove_index"
    description = "移除某个文件夹在向量库里的索引记录（不删除原图），默认只试运行"
    args_schema = RemoveIndexArgs

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
        self,
        folder_path: str = "",
        dry_run: bool = True,
        missing_only: bool = False,
    ) -> dict[str, Any]:
        hits = self.pipeline.store.get(limit=200000)

        if missing_only:
            targets = [
                hit
                for hit in hits
                if hit.path and not Path(hit.path).exists()
            ]
            label = "原图已不存在的孤儿向量"
        else:
            if not folder_path:
                return {
                    "count": 0,
                    "deleted": False,
                    "summary": "需要指定 folder_path，或开启 missing_only 清理孤儿向量",
                }
            try:
                folder = self.guard.check(folder_path, "read")
            except PathSecurityError as e:
                return {
                    "count": 0,
                    "deleted": False,
                    "summary": f"路径不安全，已拒绝：{e}",
                }

            prefix = str(folder).lower()
            targets = [
                hit
                for hit in hits
                if str(hit.path).lower().startswith(prefix)
            ]
            label = f"{folder} 下的索引"

        if not targets:
            return {
                "count": 0,
                "deleted": False,
                "summary": f"{label}：没有需要处理的内容",
            }

        if dry_run:
            return {
                "count": len(targets),
                "deleted": False,
                "dry_run": True,
                "summary": (
                    f"{label}：将删除 {len(targets)} 条（dry_run，未真正删除）。"
                    "确认后用 dry_run=False 执行。"
                ),
            }

        ids = [hit.id for hit in targets]
        try:
            self.pipeline.store.delete(ids)
        except Exception as e:
            logger.warning("删除索引失败: %s", e)
            return {"count": 0, "deleted": False, "summary": f"删除失败：{e}"}

        remainder = self.pipeline.store.count()
        return {
            "count": len(ids),
            "deleted": True,
            "remaining": remainder,
            "summary": f"已删除 {len(ids)} 条索引，库内剩余 {remainder} 条（原图未动）",
        }
