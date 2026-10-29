"""建库与检索分离测试：场景路由、实时检索、不隐式建库。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from findgo.tools.image_retrieval import ImageRetrievalTool


class _Hit:
    def __init__(self, path):
        self.path = path
        self.metadata = {"path": path}


class _Store:
    def __init__(self, paths=()):
        self.paths = list(paths)

    def get(self, limit=None):
        return [_Hit(p) for p in self.paths]


class _Encoder:
    def __init__(self, dim=8):
        self.dim = dim
        self.calls = 0

    def encode_text(self, text):
        self.calls += 1
        v = np.zeros(self.dim, dtype=np.float32)
        v[0] = 1.0
        return v

    def encode_images(self, paths):
        self.calls += 1
        out = np.zeros((len(paths), self.dim), dtype=np.float32)
        for i in range(len(paths)):
            out[i, 0] = 1.0
            out[i, 1] = i * 0.01
        return out


class _Pipeline:
    def __init__(self, indexed_paths=()):
        self.store = _Store(indexed_paths)
        self.encoder = _Encoder()
        self.index_called = 0
        self.searcher = self

    def search_text(self, query, **kwargs):
        class _Resp:
            results = []
            report = type(
                "R", (), {"used": False, "llm_scored": 0, "candidates": 0, "errors": []}
            )()

            def status_line(self):
                return "召回 0 条"

        return _Resp()

    def index_folder(self, folder, **kwargs):
        self.index_called += 1

        class _Stats:
            locked = False

            def summary(self):
                return "索引完成"

        return _Stats()


def test_is_indexed_detects_parent(tmp_path):
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    sample = str(deep / "1.jpg")

    tool = ImageRetrievalTool(_Pipeline([sample]))
    assert tool._is_indexed(tmp_path) is True


def test_is_indexed_false_for_unrelated(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    sample = str(tmp_path / "x" / "1.jpg")

    tool = ImageRetrievalTool(_Pipeline([sample]))
    assert tool._is_indexed(other) is False


def test_live_search_rejects_large_folder(tmp_path, monkeypatch):
    images = tmp_path / "imgs"
    images.mkdir()
    for i in range(ImageRetrievalTool.LIVE_LIMIT + 5):
        (images / f"{i}.jpg").write_bytes(b"x")

    tool = ImageRetrievalTool(_Pipeline())
    assert tool._live_search(images, "猫", 5) is None


def test_live_search_returns_results(tmp_path, monkeypatch):
    images = tmp_path / "imgs"
    images.mkdir()
    for i in range(3):
        (images / f"{i}.jpg").write_bytes(b"x")

    monkeypatch.setattr(
        "findgo.utils.image_scan.scan_images",
        lambda folder, recursive=True: sorted(images.glob("*.jpg")),
    )

    tool = ImageRetrievalTool(_Pipeline())
    result = tool._live_search(images, "猫", 2)

    assert result is not None
    assert len(result["images"]) == 2
    assert "实时检索" in result["summary"]


def test_run_does_not_auto_index_by_default(tmp_path, monkeypatch):
    """未索引的大目录不应该在检索链路里偷偷跑索引。"""
    images = tmp_path / "imgs"
    images.mkdir()

    tool = ImageRetrievalTool(_Pipeline())
    monkeypatch.setattr(tool, "_live_search", lambda *a, **k: None)

    class _Guard:
        def check(self, path, mode):
            return Path(images)

    tool.guard = _Guard()

    result = tool.run("猫", folder_path=str(images))

    assert tool.pipeline.index_called == 0
    assert "还没建索引" in result["summary"]


def test_run_auto_index_when_explicitly_requested(tmp_path, monkeypatch):
    images = tmp_path / "imgs"
    images.mkdir()

    tool = ImageRetrievalTool(_Pipeline())
    monkeypatch.setattr(tool, "_is_indexed", lambda folder: False)

    class _Guard:
        def check(self, path, mode):
            return Path(images)

    tool.guard = _Guard()

    tool.run("猫", folder_path=str(images), auto_index=True)
    assert tool.pipeline.index_called == 1


def test_run_skips_live_when_already_indexed(tmp_path, monkeypatch):
    images = tmp_path / "imgs"
    images.mkdir()

    tool = ImageRetrievalTool(_Pipeline())
    monkeypatch.setattr(tool, "_is_indexed", lambda folder: True)
    called = []
    monkeypatch.setattr(tool, "_live_search", lambda *a, **k: called.append(1))

    class _Guard:
        def check(self, path, mode):
            return Path(images)

    tool.guard = _Guard()

    tool.run("猫", folder_path=str(images))
    assert called == []


def test_planner_prompt_separates_index_and_search():
    import inspect

    from findgo.agent.planner import Planner

    src = inspect.getsource(Planner.plan_tasks)
    assert "建库与检索是两件事" in src
    assert "不要同时带检索" in src
