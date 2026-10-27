"""权限分组与高危操作拦截测试。"""

from __future__ import annotations

from imgsearch_agent.agent.executor import ToolExecutor
from imgsearch_agent.tools.base_tool import BaseTool
from imgsearch_agent.tools import TrashManageTool, IndexFolderTool, ExportResultsTool
from imgsearch_agent.trash import TrashStaging
from imgsearch_agent.utils.sandbox import Sandbox


class _DangerTool(BaseTool):
    name = "danger"
    description = "deletes"
    permission = "danger_delete"

    def run(self, **kwargs):
        return {"ok": True}


def test_executor_blocks_danger_without_confirmation():
    executor = ToolExecutor()
    result, _ = executor.execute(_DangerTool(), {})
    assert result["error"] == "need_confirm"
    assert result["need_user_confirm"] is True


def test_executor_allows_danger_when_confirmed():
    executor = ToolExecutor()
    result, _ = executor.execute(_DangerTool(), {}, allow_danger=True)
    assert result.get("ok") is True


def test_permissions_assigned():
    assert IndexFolderTool.permission == "index_write"
    assert ExportResultsTool.permission == "file_write"
    assert TrashManageTool.permission == "file_write"


def test_trash_permission_by_action():
    tool = TrashManageTool(staging=TrashStaging("data/trash_staging"))
    assert tool.permission_for({"action": "stage"}) == "file_write"
    assert tool.permission_for({"action": "purge"}) == "danger_delete"


def test_purge_requires_explicit_flag(tmp_path):
    import shutil

    tool = TrashManageTool(
        guard=Sandbox(image_roots=[str(tmp_path)], workspace_roots=[str(tmp_path)]),
        staging=TrashStaging(tmp_path / "trash"),
    )
    result = tool.run("purge", task_id="t1")
    assert result["need_user_confirm"] is True
    assert "人工确认" in result["summary"]
