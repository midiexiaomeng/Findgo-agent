"""参数校验与索引 mtime 缓存测试。"""

from __future__ import annotations

import pytest

from findgo.tools import ImageRetrievalTool, IndexFolderTool
from findgo.utils.exceptions import ValidationError
from findgo.utils.path_guard import PathGuard


class _FakeIndexStats:
    total = 1
    indexed = 1
    failed = 0

    def summary(self) -> str:
        return "扫描 1 张，索引 1 张"


class _FakePipeline:
    def __init__(self):
        self.index_calls = 0

    def index_folder(self, folder, **kwargs):
        self.index_calls += 1
        return _FakeIndexStats()


def test_reject_top_k_out_of_range():
    tool = ImageRetrievalTool(_FakePipeline())
    with pytest.raises(ValidationError):
        tool.validate({"query": "猫", "top_k": 1001})


def test_reject_empty_query():
    tool = ImageRetrievalTool(_FakePipeline())
    with pytest.raises(ValidationError):
        tool.validate({"query": ""})


def test_reject_unknown_field():
    tool = ImageRetrievalTool(_FakePipeline())
    with pytest.raises(ValidationError):
        tool.validate({"query": "猫", "unexpected": 1})


def test_accept_valid_args():
    tool = ImageRetrievalTool(_FakePipeline())
    payload = tool.validate({"query": "猫", "top_k": 2, "folder_path": "D:/a"})
    assert payload["query"] == "猫"
    assert payload["top_k"] == 2


def test_index_folder_mtime_cache(tmp_path, monkeypatch):
    cache_file = tmp_path.parent / "index_mtime.json"
    monkeypatch.setattr(
        IndexFolderTool, "_cache_file", staticmethod(lambda: cache_file)
    )

    pipeline = _FakePipeline()
    tool = IndexFolderTool(pipeline, PathGuard(allowed_roots=[str(tmp_path)]))

    first = tool.run(str(tmp_path))
    assert first["cached"] is False
    assert pipeline.index_calls == 1

    second = tool.run(str(tmp_path))
    assert second["cached"] is True
    assert pipeline.index_calls == 1

    third = tool.run(str(tmp_path), rebuild=True)
    assert third["cached"] is False
    assert pipeline.index_calls == 2
