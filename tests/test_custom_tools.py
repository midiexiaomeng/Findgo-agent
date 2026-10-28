"""自定义全局工具库测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from findgo.agent import custom_tools
from findgo.utils.sandbox import Sandbox

SPEC = {
    "name": "count_jpg",
    "description": "统计目录下 jpg 数量",
    "tags": ["read-only"],
    "timeout": 30,
    "args_schema_code": (
        "class Args(BaseModel):\n"
        '    folder: str = Field(..., description="目录")\n'
    ),
    "run_func_code": (
        "def run(self, args):\n"
        '    path = sandbox_check_path(args["folder"], "read")\n'
        '    return {"count": len(list(Path(path).glob("*.jpg"))), "summary": "ok"}\n'
    ),
    "safety_note": "只读",
}


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    target = tmp_path / "custom_tools.yaml"
    monkeypatch.setattr(custom_tools, "config_path", lambda: target)
    return target


def test_save_and_list(temp_store):
    custom_tools.save_custom_tool(SPEC)
    items = custom_tools.list_custom_tools()
    assert len(items) == 1
    assert items[0]["name"] == "count_jpg"
    assert temp_store.exists()


def test_save_overwrites_same_name(temp_store):
    custom_tools.save_custom_tool(SPEC)
    updated = dict(SPEC, description="新描述")
    custom_tools.save_custom_tool(updated)
    items = custom_tools.list_custom_tools()
    assert len(items) == 1
    assert items[0]["description"] == "新描述"


def test_delete(temp_store):
    custom_tools.save_custom_tool(SPEC)
    assert custom_tools.delete_custom_tool("count_jpg") is True
    assert custom_tools.list_custom_tools() == []
    assert custom_tools.delete_custom_tool("count_jpg") is False


def test_load_and_run(temp_store, tmp_path):
    custom_tools.save_custom_tool(SPEC)
    for i in range(2):
        (tmp_path / f"{i}.jpg").write_bytes(b"x")

    sandbox = Sandbox(image_roots=[str(tmp_path)], workspace_roots=[str(tmp_path)])
    tools = custom_tools.load_custom_tools(sandbox)

    assert "count_jpg" in tools
    result = tools["count_jpg"].run(folder=str(tmp_path))
    assert result["count"] == 2


def test_load_skips_broken_tool(temp_store):
    bad = dict(SPEC, name="bad_tool", run_func_code="def run(self, args):\n    return eval('1')\n")
    custom_tools.save_custom_tool(SPEC)
    custom_tools.save_custom_tool(bad)

    tools = custom_tools.load_custom_tools()
    assert "count_jpg" in tools
    assert "bad_tool" not in tools


def test_empty_file_returns_empty(temp_store):
    assert custom_tools.list_custom_tools() == []
    assert custom_tools.load_custom_tools() == {}
