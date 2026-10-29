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


def test_small_folder_uses_live_search(tmp_path, monkeypatch):
    """小目录走实时检索，不建库。"""
    images = tmp_path / "imgs"
    images.mkdir()
    for i in range(3):
        (images / f"{i}.jpg").write_bytes(b"x")

    tool = ImageRetrievalTool(_Pipeline())
    called = []
    monkeypatch.setattr(tool, "_is_indexed", lambda folder: False)
    monkeypatch.setattr(
        "findgo.utils.image_scan.scan_images",
        lambda folder, recursive=True: sorted(images.glob("*.jpg")),
    )
    monkeypatch.setattr(
        tool,
        "_live_search",
        lambda *a, **k: called.append(1) or {"images": [], "summary": "实时"},
    )

    class _Guard:
        def check(self, path, mode):
            return Path(images)

    tool.guard = _Guard()

    tool.run("猫", folder_path=str(images))

    assert called == [1]
    assert tool.pipeline.index_called == 0


def test_large_folder_auto_indexes(tmp_path, monkeypatch):
    """大目录由工具自己判断并建库，不需要用户手动。"""
    images = tmp_path / "imgs"
    images.mkdir()

    tool = ImageRetrievalTool(_Pipeline())
    monkeypatch.setattr(tool, "_is_indexed", lambda folder: False)
    monkeypatch.setattr(
        "findgo.utils.image_scan.scan_images",
        lambda folder, recursive=True: [Path(f"{i}.jpg") for i in range(500)],
    )

    class _Guard:
        def check(self, path, mode):
            return Path(images)

    tool.guard = _Guard()

    tool.run("猫", folder_path=str(images))

    assert tool.pipeline.index_called == 1


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


def test_index_can_stop(monkeypatch):
    """should_stop 在批次间生效。"""
    from findgo.retriever.indexer import Indexer, IndexStats

    idx = Indexer.__new__(Indexer)
    idx.cfg = type("C", (), {"caption": type("CC", (), {"unload_after_index": False})()})()
    idx.captioner = type("Cap", (), {"unload": lambda self: None})()

    calls = {"n": 0}

    class _Store:
        def get_all_ids(self):
            return set()

    idx.store = _Store()
    idx._index_batch = lambda batch, with_caption=None: calls.__setitem__("n", calls["n"] + 1) or {
        "indexed": len(batch),
        "failed": 0,
        "captioned": 0,
        "errors": [],
    }

    stats = IndexStats()
    pending = [Path(f"{i}.jpg") for i in range(100)]
    bs = 10
    done = 0
    stop_after = 2

    for start in range(0, len(pending), bs):
        if calls["n"] >= stop_after:
            stats.cancelled = True
            break
        batch = pending[start : start + bs]
        r = idx._index_batch(batch)
        stats.indexed += r["indexed"]
        done += len(batch)

    assert stats.cancelled is True
    assert calls["n"] == stop_after
    assert "已中断" in stats.summary()


def test_index_stats_cancelled_flag():
    from findgo.retriever.indexer import IndexStats

    s = IndexStats(total=10, indexed=3, cancelled=True)
    assert s.as_dict()["cancelled"] is True
    assert "已中断" in s.summary()


def test_pause_hook_exists():
    """索引器必须接受 should_pause 参数。"""
    import inspect

    from findgo.retriever.indexer import Indexer

    params = inspect.signature(Indexer.index_folder).parameters
    assert "should_pause" in params
    assert "should_stop" in params


def test_planner_prompt_separates_index_and_search():
    import inspect

    from findgo.agent.planner import Planner

    src = inspect.getsource(Planner.plan_tasks)
    assert "检索任务永远不生成 index 任务" in src
    assert "一条指令只能生成一类任务" in src
