"""暂存区只读预览：从 trash_manage 拆出来的查看能力。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .base_tool import BaseTool


class PreviewStagedArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field("", description="只看某个任务的暂存记录，留空看全部")
    limit: int = Field(100, ge=1, le=1000, description="最多返回多少条")


class PreviewStagedTool(BaseTool):
    name = "preview_staged"
    description = "只读查看暂存区里待删除的图片清单，没有任何修改能力"
    permission = "read"
    args_schema = PreviewStagedArgs

    def __init__(self, staging=None):
        self._staging = staging

    @property
    def staging(self):
        if self._staging is None:
            from pathlib import Path

            from ..config import AgentConfig
            from ..retriever.runtime import PROJECT_ROOT
            from ..trash import TrashStaging

            raw = AgentConfig().trash_path
            root = Path(raw).expanduser()
            if not root.is_absolute():
                root = Path(PROJECT_ROOT) / root
            self._staging = TrashStaging(root)
        return self._staging

    def run(self, task_id: str = "", limit: int = 100) -> dict[str, Any]:
        items = self.staging.list_staged(task_id or None)
        total_bytes = sum(int(i["size"]) for i in items)

        return {
            "items": items[:limit],
            "count": len(items),
            "returned": min(len(items), limit),
            "total_bytes": total_bytes,
            "summary": (
                f"暂存区有 {len(items)} 张待删除图片，" f"占用 {total_bytes / 1024**2:.1f} MB"
            ),
        }
