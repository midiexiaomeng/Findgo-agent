"""路径安全与目录扫描测试。"""

from __future__ import annotations

import pytest
from PIL import Image

from findgo.tools.folder_scan import FolderScanTool
from findgo.tools.image_retrieval import ImageRetrievalTool
from findgo.utils.path_guard import PathGuard, PathSecurityError


def make_guard(root) -> PathGuard:
    return PathGuard(allowed_roots=[str(root)])


def test_allows_inside_root(tmp_path):
    sub = tmp_path / "photos"
    sub.mkdir()
    guard = make_guard(tmp_path)
    assert guard.check(sub) == sub.resolve()


def test_blocks_traversal_outside_root(tmp_path):
    guard = make_guard(tmp_path)
    with pytest.raises(PathSecurityError):
        guard.check(str(tmp_path / ".." / ".." / "whatever"))


def test_blocks_relative_traversal(tmp_path):
    guard = make_guard(tmp_path)
    with pytest.raises(PathSecurityError):
        guard.check("../../etc/passwd")


def test_blocks_system_directory():
    guard = PathGuard()
    with pytest.raises(PathSecurityError):
        guard.check("C:/Windows/System32")


def test_is_safe_helper(tmp_path):
    guard = make_guard(tmp_path)
    assert guard.is_safe(str(tmp_path)) is True
    assert guard.is_safe("C:/Windows") is False


def test_folder_scan_counts_images(tmp_path):
    for i in range(3):
        Image.new("RGB", (8, 8), (i * 40, 0, 0)).save(tmp_path / f"{i}.png")
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")

    tool = FolderScanTool(make_guard(tmp_path))
    result = tool.run(str(tmp_path))
    assert result["count"] == 3
    assert "3 张图片" in result["summary"]


def test_folder_scan_rejects_unsafe_path(tmp_path):
    tool = FolderScanTool(make_guard(tmp_path))
    result = tool.run("C:/Windows")
    assert result["count"] == 0
    assert "不安全" in result["summary"]


def test_folder_scan_non_directory(tmp_path):
    file_path = tmp_path / "a.txt"
    file_path.write_text("x", encoding="utf-8")
    tool = FolderScanTool(make_guard(tmp_path))
    result = tool.run(str(file_path))
    assert result["count"] == 0


class _FakeResp:
    results = []

    @staticmethod
    def status_line():
        return "0"


class _FakeSearcher:
    def search_text(self, query, top_k=8, scope=None):
        return _FakeResp()


class _FakePipeline:
    def __init__(self):
        self.searcher = _FakeSearcher()


def test_image_retrieval_rejects_unsafe_path(tmp_path):
    tool = ImageRetrievalTool(_FakePipeline(), guard=make_guard(tmp_path))
    result = tool.run("猫", path="C:/Windows")
    assert result["images"] == []
    assert "不安全" in result["summary"]
