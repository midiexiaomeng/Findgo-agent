"""Agent 工具层测试：只验证参数校验与 Pipeline 调用，不碰真实模型。"""

from __future__ import annotations

import pytest
from PIL import Image

from imgsearch_agent.tools import (
    GetIndexStatsTool,
    ImageRetrievalByImageTool,
    IndexFolderTool,
    SwitchVectorBackendTool,
)
from imgsearch_agent.utils.path_guard import PathGuard


class FakeIndexStats:
    total = 3
    indexed = 3
    failed = 0

    def summary(self) -> str:
        return "扫描 3 张，索引 3 张，跳过 0 张"


class FakeStore:
    backend_name = "qdrant"
    collection = "images"

    def count(self) -> int:
        return 1867


class FakePipeline:
    def __init__(self):
        self.store = FakeStore()
        self.index_calls: list[tuple[str, bool]] = []
        self.switch_calls: list[str] = []

    def stats(self) -> dict:
        return {
            "images": 1867,
            "backend": "qdrant",
            "collection": "images",
            "device": "cpu",
            "encoder_loaded": False,
        }

    def index_folder(self, folder, reset=False, skip_existing=True, **kw):
        self.index_calls.append((str(folder), reset))
        return FakeIndexStats()

    def switch_backend(self, name: str) -> str:
        self.switch_calls.append(name)
        self.store.backend_name = name
        return f"已切换到 {name}"


def make_guard(root) -> PathGuard:
    return PathGuard(allowed_roots=[str(root)])


def test_get_index_stats():
    tool = GetIndexStatsTool(FakePipeline())
    result = tool.run()
    assert result["count"] == 1867
    assert result["backend"] == "qdrant"
    assert "1867" in result["summary"]


def test_index_folder_incremental(tmp_path):
    pipeline = FakePipeline()
    tool = IndexFolderTool(pipeline, make_guard(tmp_path))
    result = tool.run(str(tmp_path))
    assert result["count"] == 3
    assert pipeline.index_calls == [(str(tmp_path.resolve()), False)]


def test_index_folder_rebuild(tmp_path):
    pipeline = FakePipeline()
    tool = IndexFolderTool(pipeline, make_guard(tmp_path))
    tool.run(str(tmp_path), rebuild=True)
    assert pipeline.index_calls == [(str(tmp_path.resolve()), True)]


def test_index_folder_rejects_unsafe(tmp_path):
    pipeline = FakePipeline()
    tool = IndexFolderTool(pipeline, make_guard(tmp_path))
    result = tool.run("C:/Windows")
    assert result["count"] == 0
    assert "不安全" in result["summary"]
    assert pipeline.index_calls == []


def test_switch_backend_valid():
    pipeline = FakePipeline()
    tool = SwitchVectorBackendTool(pipeline)
    result = tool.run("chroma")
    assert result["backend"] == "chroma"
    assert pipeline.switch_calls == ["chroma"]


def test_switch_backend_invalid():
    pipeline = FakePipeline()
    tool = SwitchVectorBackendTool(pipeline)
    result = tool.run("faiss")
    assert result["error"] == "unsupported backend"
    assert pipeline.switch_calls == []


def test_image_retrieval_by_image_missing_file(tmp_path):
    tool = ImageRetrievalByImageTool(FakePipeline(), make_guard(tmp_path))
    result = tool.run(str(tmp_path / "none.jpg"))
    assert result["images"] == []
    assert "不是文件" in result["summary"]


def test_image_retrieval_by_image_unsafe_path(tmp_path):
    tool = ImageRetrievalByImageTool(FakePipeline(), make_guard(tmp_path))
    result = tool.run("C:/Windows/System32/calc.exe")
    assert result["images"] == []
    assert "不安全" in result["summary"]
