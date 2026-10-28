"""检索结果跨轮排除测试。"""

from __future__ import annotations

from findgo.retriever.searcher import Searcher


class _Hit:
    def __init__(self, path: str):
        self.path = path
        self.metadata = {"path": path}


class _Store:
    def __init__(self, paths):
        self.paths = paths

    def query(self, vector, top_k=10):
        return [_Hit(p) for p in self.paths]


def make_searcher(paths):
    searcher = Searcher.__new__(Searcher)
    searcher.store = _Store(paths)
    return searcher


def test_recall_excludes_previous_paths():
    searcher = make_searcher(["D:/a/1.jpg", "D:/a/2.jpg", "D:/a/3.jpg"])
    hits = Searcher._recall(
        searcher, None, 10, 0.0, None, {"d:/a/1.jpg", "d:/a/2.jpg"}
    )
    assert len(hits) == 1
    assert hits[0].path == "D:/a/3.jpg"


def test_recall_without_exclude_returns_all():
    searcher = make_searcher(["D:/a/1.jpg", "D:/a/2.jpg"])
    hits = Searcher._recall(searcher, None, 10, 0.0, None, None)
    assert len(hits) == 2


def test_recall_case_insensitive():
    searcher = make_searcher(["D:/A/1.JPG", "D:/a/2.jpg"])
    hits = Searcher._recall(searcher, None, 10, 0.0, None, {"d:/a/1.jpg"})
    assert len(hits) == 1
