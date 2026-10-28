"""动态工具路径逃逸测试。"""

from __future__ import annotations

import pytest

from imgsearch_agent.agent.dynamic import DynamicToolSpec, build_dynamic_tool
from imgsearch_agent.utils.sandbox import Sandbox, SandboxError


def make_spec(run_code: str) -> DynamicToolSpec:
    return DynamicToolSpec(
        name="escape_test",
        description="测试路径逃逸",
        tags=["read-only"],
        timeout=30,
        args_schema_code="class Args(BaseModel):\n    pass\n",
        run_func_code=run_code,
        safety_note="测试",
    )


def test_read_text_blocked_outside_root(tmp_path):
    code = (
        "def run(self, args):\n"
        "    return {'v': Path('C:/Windows/win.ini').read_text(errors='ignore')}\n"
    )
    tool = build_dynamic_tool(make_spec(code), Sandbox(image_roots=[str(tmp_path)]))
    with pytest.raises(SandboxError):
        tool.run()


def test_write_text_blocked_in_image_root(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    code = (
        "def run(self, args):\n"
        "    p = Path(r'" + str(images.resolve()).replace('\\', '\\\\') + "')\n"
        "    (p / 'x.txt').write_text('x')\n"
        "    return {'ok': True}\n"
    )
    tool = build_dynamic_tool(
        make_spec(code),
        Sandbox(image_roots=[str(images)], workspace_roots=[str(tmp_path / "ws")]),
    )
    with pytest.raises(SandboxError):
        tool.run()


def test_read_inside_root_allowed(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    target = images / "a.jpg"
    target.write_bytes(b"x")

    code = (
        "def run(self, args):\n"
        "    return {'ok': Path(r'" + str(target).replace('\\', '\\\\') + "').exists()}\n"
    )
    tool = build_dynamic_tool(make_spec(code), Sandbox(image_roots=[str(images)]))
    assert tool.run()["ok"] is True


def test_unlink_blocked(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    target = images / "a.jpg"
    target.write_bytes(b"x")

    code = (
        "def run(self, args):\n"
        "    Path(r'" + str(target).replace('\\', '\\\\') + "').unlink()\n"
        "    return {'ok': True}\n"
    )
    tool = build_dynamic_tool(
        make_spec(code),
        Sandbox(image_roots=[str(images)], workspace_roots=[str(tmp_path / "ws")]),
    )
    with pytest.raises(SandboxError):
        tool.run()
    assert target.exists()
