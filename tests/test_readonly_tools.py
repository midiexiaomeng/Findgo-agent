"""新增只读工具测试：结果过滤、暂存预览、索引计数。"""

from __future__ import annotations

from pathlib import Path

from findgo.tools import (
    CountIndexedImagesTool,
    FilterResultsTool,
    PreviewStagedTool,
)
from findgo.trash import TrashStaging
from findgo.utils.sandbox import Sandbox


class _Hit:
    def __init__(self, path: str, caption: str = "", staged: bool = False):
        self.path = path
        self.metadata = {"path": path, "caption": caption, "is_staged": staged}


class _Store:
    def __init__(self, hits):
        self.hits = hits

    def get(self, ids=None, limit=None):
        if ids:
            wanted = {str(i).lower() for i in ids}
            return [h for h in self.hits]
        return self.hits


class _Pipeline:
    def __init__(self, hits):
        self.store = _Store(hits)


def test_filter_excludes_staged():
    hits = [
        _Hit("D:/a/1.jpg", "a cat"),
        _Hit("D:/a/2.jpg", "a cat", staged=True),
    ]
    tool = FilterResultsTool(_Pipeline(hits))
    result = tool.run(paths=["D:/a/1.jpg", "D:/a/2.jpg"])
    assert result["count"] == 1
    assert result["kept"][0]["path"] == "D:/a/1.jpg"
    assert result["dropped"][0]["reason"] == "已暂存"


def test_filter_by_keyword():
    hits = [
        _Hit("D:/a/1.jpg", "a cat on the sofa"),
        _Hit("D:/a/2.jpg", "a cat and a dog"),
    ]
    tool = FilterResultsTool(_Pipeline(hits))
    result = tool.run(paths=["D:/a/1.jpg", "D:/a/2.jpg"], must_not_keywords=["dog"])
    assert result["count"] == 1


def test_filter_by_score():
    hits = [_Hit("D:/a/1.jpg"), _Hit("D:/a/2.jpg")]
    tool = FilterResultsTool(_Pipeline(hits))
    result = tool.run(
        paths=["D:/a/1.jpg", "D:/a/2.jpg"],
        min_similarity=0.8,
        scores={"D:/a/1.jpg": 0.9, "D:/a/2.jpg": 0.5},
    )
    assert result["count"] == 1


def test_preview_staged_reads_manifest(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for i in range(2):
        (images / f"{i}.jpg").write_bytes(b"x" * 64 + bytes([i]))

    staging = TrashStaging(tmp_path / "trash")
    staging.stage([str(images / f"{i}.jpg") for i in range(2)], "t1")

    tool = PreviewStagedTool(staging=staging)
    result = tool.run()
    assert result["count"] == 2
    assert result["total_bytes"] == 130


def test_preview_staged_empty(tmp_path):
    tool = PreviewStagedTool(staging=TrashStaging(tmp_path / "trash"))
    result = tool.run()
    assert result["count"] == 0
    assert "0 张" in result["summary"]


def test_count_indexed_excludes_staged(tmp_path):
    hits = [
        _Hit(str(tmp_path / "1.jpg")),
        _Hit(str(tmp_path / "2.jpg")),
        _Hit(str(tmp_path / "3.jpg"), staged=True),
    ]
    tool = CountIndexedImagesTool(
        _Pipeline(hits), Sandbox(image_roots=[str(tmp_path)])
    )
    result = tool.run()
    assert result["total"] == 3
    assert result["staged"] == 1
    assert result["effective"] == 2


def test_count_indexed_by_folder(tmp_path):
    hits = [
        _Hit(str(tmp_path / "a" / "1.jpg")),
        _Hit(str(tmp_path / "b" / "2.jpg")),
    ]
    tool = CountIndexedImagesTool(
        _Pipeline(hits), Sandbox(image_roots=[str(tmp_path)])
    )
    result = tool.run(str(tmp_path / "a"))
    assert result["total"] == 1
