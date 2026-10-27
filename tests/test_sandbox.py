"""分域沙箱测试：读、写、删三种模式的边界。"""

from __future__ import annotations

import pytest

from imgsearch_agent.utils.sandbox import Sandbox, SandboxError


def make(tmp_path):
    images = tmp_path / "images"
    workspace = tmp_path / "data"
    images.mkdir()
    workspace.mkdir()
    return Sandbox(
        image_roots=[str(images)],
        workspace_roots=[str(workspace)],
    ), images, workspace


def test_read_image_root_allowed(tmp_path):
    sandbox, images, _ = make(tmp_path)
    assert sandbox.check(images / "a.jpg", "read") == (images / "a.jpg").resolve()


def test_write_image_root_blocked(tmp_path):
    sandbox, images, _ = make(tmp_path)
    with pytest.raises(SandboxError) as e:
        sandbox.check(images / "new.jpg", "write")
    assert "workspace" in str(e.value)


def test_delete_image_root_blocked(tmp_path):
    sandbox, images, _ = make(tmp_path)
    with pytest.raises(SandboxError):
        sandbox.check(images / "a.jpg", "delete")


def test_write_workspace_allowed(tmp_path):
    sandbox, _, workspace = make(tmp_path)
    assert sandbox.check(workspace / "export.csv", "write")


def test_delete_workspace_allowed(tmp_path):
    sandbox, _, workspace = make(tmp_path)
    assert sandbox.check(workspace / "old.json", "delete")


def test_system_directory_blocked_all_modes(tmp_path):
    sandbox, _, _ = make(tmp_path)
    for mode in ("read", "write", "delete"):
        with pytest.raises(SandboxError):
            sandbox.check("C:/Windows/System32", mode)


def test_outside_all_roots_blocked(tmp_path):
    sandbox, _, _ = make(tmp_path)
    with pytest.raises(SandboxError):
        sandbox.check(tmp_path / "elsewhere" / "x.txt", "read")


def test_unknown_mode_rejected(tmp_path):
    sandbox, _, _ = make(tmp_path)
    with pytest.raises(SandboxError):
        sandbox.check(str(tmp_path), "execute")


def test_is_safe_helper(tmp_path):
    sandbox, images, workspace = make(tmp_path)
    assert sandbox.is_safe(images / "a.jpg", "read") is True
    assert sandbox.is_safe(images / "a.jpg", "write") is False
    assert sandbox.is_safe(workspace / "x", "write") is True


def test_describe_lists_roots(tmp_path):
    sandbox, _, _ = make(tmp_path)
    text = sandbox.describe()
    assert "只读目录" in text
    assert "可写目录" in text

def test_symlink_rejected(tmp_path):
    sandbox, images, _ = make(tmp_path)
    target = images / "a.txt"
    target.write_text("x", encoding="utf-8")
    link = images / "link.txt"

    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("当前环境不支持创建符号链接")

    with pytest.raises(SandboxError):
        sandbox.check(link, "read")
