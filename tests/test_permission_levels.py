"""权限等级与命令菜单测试。"""

from __future__ import annotations

import pytest
from findgo.agent.executor import ToolExecutor
from findgo.tools.base_tool import BaseTool
from pydantic import BaseModel


class _Args(BaseModel):
    pass


class _ReadTool(BaseTool):
    name = "read_tool"
    description = "只读"
    args_schema = _Args
    permission = "read"

    def run(self) -> dict:
        return {"ok": True}


class _WriteTool(BaseTool):
    name = "write_tool"
    description = "写文件"
    args_schema = _Args
    permission = "file_write"

    def run(self) -> dict:
        return {"ok": True}


class _DangerTool(BaseTool):
    name = "danger_tool"
    description = "删原图"
    args_schema = _Args
    permission = "danger_delete"

    def run(self) -> dict:
        return {"ok": True}


@pytest.fixture()
def executor():
    ex = ToolExecutor()
    yield ex
    ex._pool.shutdown(wait=False)


def test_readonly_allows_read(executor):
    result, _ = executor.execute(_ReadTool(), {}, access_level="readonly")
    assert result.get("ok") is True


def test_readonly_blocks_write(executor):
    result, _ = executor.execute(_WriteTool(), {}, access_level="readonly")
    assert result["error"] == "readonly_blocked"


def test_readonly_blocks_danger(executor):
    result, _ = executor.execute(_DangerTool(), {}, access_level="readonly")
    assert result["error"] == "need_confirm"


def test_ask_requires_confirm_for_write(executor):
    result, _ = executor.execute(_WriteTool(), {}, access_level="ask")
    assert result["error"] == "need_confirm"
    assert result["need_user_confirm"] is True


def test_ask_allows_read(executor):
    result, _ = executor.execute(_ReadTool(), {}, access_level="ask")
    assert result.get("ok") is True


def test_auto_allows_write(executor):
    result, _ = executor.execute(_WriteTool(), {}, access_level="auto")
    assert result.get("ok") is True


def test_auto_blocks_danger(executor):
    result, _ = executor.execute(_DangerTool(), {}, access_level="auto")
    assert result["error"] == "need_confirm"


def test_full_allows_write(executor):
    result, _ = executor.execute(_WriteTool(), {}, access_level="full")
    assert result.get("ok") is True


def test_full_still_blocks_danger(executor):
    """完整权限也不自动删原图，这是安全底线。"""
    result, _ = executor.execute(_DangerTool(), {}, access_level="full")
    assert result["error"] == "need_confirm"


def test_danger_allowed_with_explicit_flag(executor):
    result, _ = executor.execute(_DangerTool(), {}, access_level="full", allow_danger=True)
    assert result.get("ok") is True


def test_unknown_level_behaves_like_auto(executor):
    result, _ = executor.execute(_WriteTool(), {}, access_level="whatever")
    assert result.get("ok") is True


# ── 命令与权限按钮 ────────────────────────────────────────────────────────


def test_command_filter():
    pytest.importorskip("PyQt5")
    from findgo.ui.command_panel import match_commands

    assert len(match_commands("")) == 6
    assert [c["name"] for c in match_commands("/d")] == ["/diary", "/office-documents"]
    assert [c["name"] for c in match_commands("/of")] == ["/office-documents"]


def test_permission_cycle():
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    from findgo.ui.command_panel import PermissionButton

    btn = PermissionButton("ask")
    assert btn.key() == "ask"
    assert btn.text() == "？"

    btn.next()
    assert btn.key() == "readonly"
    assert btn.text() == "[|]"

    btn.next()
    assert btn.key() == "full"
    assert btn.text() == ">_"

    btn.next()
    assert btn.key() == "auto"
    assert btn.text() == "//"

    btn.next()
    assert btn.key() == "ask"
