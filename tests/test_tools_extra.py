"""新增工具测试：已索引目录清单、单图描述。"""

from __future__ import annotations

from findgo.tools import DescribeImageTool, ListIndexedFoldersTool
from findgo.utils.path_guard import PathGuard
from PIL import Image


class _Hit:
    def __init__(self, path: str):
        self.path = path


class _FolderStore:
    def get(self, limit=None):
        return [
            _Hit(r"D:\photos\a\1.jpg"),
            _Hit(r"D:\photos\a\2.jpg"),
            _Hit(r"D:\photos\b\3.jpg"),
        ]


class _FolderPipeline:
    store = _FolderStore()


def test_list_indexed_folders():
    tool = ListIndexedFoldersTool(_FolderPipeline())
    result = tool.run()
    assert result["count"] == 2
    assert result["total_images"] == 3
    assert result["folders"][0]["count"] == 2


class _FakeCaptioner:
    def caption_image(self, path: str) -> str:
        return "一只橘猫趴在窗台上"


class _DescribePipeline:
    captioner = _FakeCaptioner()


def test_describe_image(tmp_path):
    image = tmp_path / "cat.jpg"
    Image.new("RGB", (8, 8), (200, 120, 60)).save(image)

    tool = DescribeImageTool(_DescribePipeline(), PathGuard(allowed_roots=[str(tmp_path)]))
    result = tool.run(str(image))
    assert result["caption"] == "一只橘猫趴在窗台上"
    assert "橘猫" in result["summary"]


def test_describe_image_unsafe(tmp_path):
    tool = DescribeImageTool(_DescribePipeline(), PathGuard(allowed_roots=[str(tmp_path)]))
    result = tool.run("C:/Windows/win.ini")
    assert result["caption"] == ""
    assert "不安全" in result["summary"]


def test_describe_image_missing(tmp_path):
    tool = DescribeImageTool(_DescribePipeline(), PathGuard(allowed_roots=[str(tmp_path)]))
    result = tool.run(str(tmp_path / "nope.jpg"))
    assert result["caption"] == ""
    assert "不是文件" in result["summary"]
