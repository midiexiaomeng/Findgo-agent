"""动态工具加载与安全闸门测试。"""

from __future__ import annotations

import pytest

from imgsearch_agent.agent.dynamic import (
    DynamicToolError,
    DynamicToolSpec,
    build_dynamic_tool,
    scan_source,
)
from imgsearch_agent.utils.sandbox import Sandbox


def make_spec(run_code: str | None = None, args_code: str | None = None):
    return DynamicToolSpec(
        name="count_images",
        description="统计目录下的图片数量",
        tags=["read-only"],
        timeout=30,
        args_schema_code=args_code
        or (
            "class Args(BaseModel):\n"
            '    folder: str = Field(..., description="目录")\n'
        ),
        run_func_code=run_code
        or (
            "def run(self, args):\n"
            '    path = sandbox_check_path(args["folder"], "read")\n'
            '    files = [p for p in Path(path).glob("*.jpg")]\n'
            '    return {"count": len(files), "summary": "ok"}\n'
        ),
        safety_note="只读，不修改文件",
    )


def test_build_and_run(tmp_path):
    for i in range(3):
        (tmp_path / f"{i}.jpg").write_bytes(b"x")

    sandbox = Sandbox(image_roots=[str(tmp_path)], workspace_roots=[str(tmp_path)])
    tool = build_dynamic_tool(make_spec(), sandbox)

    assert tool.name == "count_images"
    assert tool.permission == "read"
    assert tool.timeout == 30

    result = tool.run(folder=str(tmp_path))
    assert result["count"] == 3


def test_blocked_keyword_rejected():
    spec = make_spec(run_code="def run(self, args):\n    return eval('1')\n")
    with pytest.raises(DynamicToolError) as e:
        build_dynamic_tool(spec)
    assert "禁止" in str(e.value)


def test_blocked_subprocess_rejected():
    spec = make_spec(
        run_code="import subprocess\ndef run(self, args):\n    return 1\n"
    )
    with pytest.raises(DynamicToolError):
        build_dynamic_tool(spec)


def test_sandbox_blocks_outside_path(tmp_path):
    spec = make_spec()
    sandbox = Sandbox(image_roots=[str(tmp_path)], workspace_roots=[str(tmp_path)])
    tool = build_dynamic_tool(spec, sandbox)

    with pytest.raises(Exception):
        tool.run(folder="C:/Windows")


def test_missing_args_model_rejected():
    spec = make_spec(args_code="x = 1\n")
    with pytest.raises(DynamicToolError) as e:
        build_dynamic_tool(spec)
    assert "Args" in str(e.value)


def test_missing_run_function_rejected():
    spec = make_spec(run_code="y = 2\n")
    with pytest.raises(DynamicToolError) as e:
        build_dynamic_tool(spec)
    assert "run" in str(e.value)


def test_scan_source_passes_clean_code():
    scan_source("def run(self, args):\n    return {'ok': True}\n")
