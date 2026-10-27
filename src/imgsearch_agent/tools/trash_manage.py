"""暂存区管理工具：暂存 / 查看 / 恢复 / 永久删除 / 清理过期。

Agent 的“删除”永远只是 stage，不会直接删用户原图。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..utils.sandbox import Sandbox, SandboxError
from .base_tool import BaseTool

logger = logging.getLogger("imgsearch_agent.trash_manage")


class TrashManageArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = Field(
        ...,
        pattern="^(stage|list|restore|purge|cleanup)$",
        description="stage 暂存 / list 查看 / restore 恢复 / purge 永久删除 / cleanup 清理",
    )
    paths: list[str] = Field(default_factory=list, description="stage 时要暂存的图片路径")
    task_id: str = Field("", description="任务 ID，留空自动生成")
    ids: list[str] = Field(default_factory=list, description="指定要恢复或删除的记录 ID")
    max_days: int = Field(0, ge=0, le=365, description="cleanup 超时天数，0 表示用设置里的值")
    max_total_gb: float = Field(0.0, ge=0.0, le=1000, description="cleanup 容量上限 GB，0 表示用设置里的值")


class TrashManageTool(BaseTool):
    name = "trash_manage"
    permission = "file_write"
    description = (
        "管理暂存区：stage 把图片移入暂存区（不删原图）/ list 查看 / "
        "restore 恢复 / purge 永久删除 / cleanup 按超时与容量清理"
    )
    args_schema = TrashManageArgs

    def __init__(self, pipeline=None, guard: Sandbox | None = None, staging=None):
        self._pipeline = pipeline
        self.guard = guard or Sandbox()
        self._staging = staging

        try:
            from ..config import AgentConfig

            cfg = AgentConfig()
            self.max_days = cfg.trash_max_days
            self.max_total_gb = cfg.trash_max_gb
        except Exception:
            self.max_days = 14
            self.max_total_gb = 10.0

    def permission_for(self, args: dict) -> str:
        # 永久删除单独提升为高危，其余暂存/恢复属于文件写
        return "danger_delete" if str(args.get("action")) == "purge" else "file_write"

    @property
    def pipeline(self):
        if self._pipeline is None:
            from ..retriever import build_pipeline

            self._pipeline = build_pipeline()
        return self._pipeline

    @property
    def staging(self):
        if self._staging is None:
            from ..retriever.runtime import PROJECT_ROOT
            from ..trash import TrashStaging

            self._staging = TrashStaging(
                Path(PROJECT_ROOT) / "data" / "trash_staging"
            )
        return self._staging

    def run(
        self,
        action: str,
        paths: list[str] | None = None,
        task_id: str = "",
        ids: list[str] | None = None,
        max_days: int = 0,
        max_total_gb: float = 0.0,
        allow_danger: bool = False,
    ) -> dict[str, Any]:
        action = (action or "").lower()

        if action == "purge" and not allow_danger:
            # Agent 不能自主永久删除，必须由界面确认后调用
            return {
                "error": "need_confirm",
                "need_user_confirm": True,
                "summary": "永久删除需要人工确认，请在设置面板的暂存区操作",
            }

        if action == "stage":
            return self._stage(paths or [], task_id)

        if action == "list":
            items = self.staging.list_staged(task_id or None)
            stats = self.staging.stats()
            return {
                "items": items,
                "count": len(items),
                "total_bytes": stats["total_bytes"],
                "summary": (
                    f"暂存区有 {len(items)} 张，"
                    f"占用 {stats['total_bytes'] / 1024**2:.1f} MB"
                ),
            }

        if action in ("restore", "purge"):
            if not task_id and not ids:
                return {
                    "error": "need_target",
                    "summary": "恢复或永久删除必须指定 task_id 或 ids，避免误操作",
                }
            if action == "restore":
                result = self.staging.restore(task_id or None, ids or None)
                self._mark(result.get("paths", []), staged=False)
                return {
                    **result,
                    "summary": (
                        f"已恢复 {result['restored']} 张"
                        + (f"，{result['conflicts']} 张同名已备份" if result["conflicts"] else "")
                    ),
                }
            result = self.staging.purge(task_id or None, ids or None)
            self._drop(result.get("paths", []))
            return {
                **result,
                "summary": (
                    f"已永久删除 {result['purged']} 张，"
                    f"释放 {result['freed_bytes'] / 1024**2:.1f} MB"
                ),
            }

        if action == "cleanup":
            days = max_days or self.max_days
            gb = max_total_gb or self.max_total_gb
            result = self.staging.cleanup(days, gb)
            return {
                **result,
                "summary": (
                    f"清理 {result['total_purged']} 张，"
                    f"释放 {result['freed_bytes'] / 1024**2:.1f} MB"
                ),
            }

        return {"error": "unknown_action", "summary": f"不支持的操作：{action}"}

    def _mark(self, paths: list[str], staged: bool) -> None:
        """同步向量库软删除标记。失败不影响文件操作。"""
        if not paths:
            return
        try:
            from ..retriever.indexer import Indexer

            ids = [Indexer.image_id(p) for p in paths]
            self.pipeline.store.update_payload(ids, {"is_staged": staged})
        except Exception as e:
            logger.warning("更新向量标记失败: %s", e)

    def _drop(self, paths: list[str]) -> None:
        """永久删除后，把对应向量从库里移除。"""
        if not paths:
            return
        try:
            from ..retriever.indexer import Indexer

            ids = [Indexer.image_id(p) for p in paths]
            self.pipeline.store.delete(ids)
        except Exception as e:
            logger.warning("删除向量失败: %s", e)

    def _stage(self, paths: list[str], task_id: str) -> dict[str, Any]:
        if not paths:
            return {"staged": 0, "summary": "没有提供要暂存的图片路径"}

        safe: list[str] = []
        for raw in paths:
            try:
                # 只读校验：素材区允许读，写操作由暂存区（workspace）承担
                safe.append(str(self.guard.check(raw, "read")))
            except SandboxError as e:
                return {"staged": 0, "summary": f"路径不安全，已拒绝：{e}"}

        tid = task_id or f"trash_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = self.staging.stage(safe, tid)
        self._mark(safe, staged=True)
        return {
            **result,
            "summary": (
                f"已暂存 {result['staged']} 张到 {tid}"
                + (f"，跳过 {result['skipped']} 张" if result["skipped"] else "")
                + (f"，失败 {len(result['failed'])} 张" if result["failed"] else "")
            ),
        }
