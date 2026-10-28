"""检索结果相似去重测试。"""

from __future__ import annotations

import numpy as np

from findgo.retriever.indexer import Indexer
from findgo.tools import ImageRetrievalTool
from findgo.utils.schemas import RetrievedImage


class _Store:
    def __init__(self, vectors):
        self.vectors = vectors

    def get_vectors(self, ids=None, limit=None):
        items = self.vectors.items()
        if ids:
            wanted = set(ids)
            items = [(k, v) for k, v in items if k in wanted]
        return [(k, np.asarray(v, dtype=np.float32), {}) for k, v in items]


class _Pipeline:
    def __init__(self, vectors):
        self.store = _Store(vectors)


def make_tool(vectors):
    tool = ImageRetrievalTool(_Pipeline(vectors))
    return tool


def make_images(paths):
    return [
        RetrievedImage(rank=i + 1, path=p, caption="", score=0.9 - i * 0.01)
        for i, p in enumerate(paths)
    ]


def test_dedupe_removes_near_duplicates():
    a, b, c = "D:/a/1.jpg", "D:/a/2.jpg", "D:/a/3.jpg"
    vectors = {
        Indexer.image_id(a): [1.0, 0.0, 0.0],
        Indexer.image_id(b): [0.999, 0.01, 0.0],   # 与 a 几乎一致
        Indexer.image_id(c): [0.0, 1.0, 0.0],
    }
    tool = make_tool(vectors)
    kept = tool._dedupe(make_images([a, b, c]), 0.95)
    assert len(kept) == 2
    assert kept[0].path == a


def test_dedupe_keeps_distinct_images():
    a, b = "D:/a/1.jpg", "D:/a/2.jpg"
    vectors = {
        Indexer.image_id(a): [1.0, 0.0],
        Indexer.image_id(b): [0.0, 1.0],
    }
    tool = make_tool(vectors)
    kept = tool._dedupe(make_images([a, b]), 0.95)
    assert len(kept) == 2


def test_dedupe_threshold_controls():
    a, b = "D:/a/1.jpg", "D:/a/2.jpg"
    vectors = {
        Indexer.image_id(a): [1.0, 0.0],
        Indexer.image_id(b): [0.9, 0.435],
    }
    tool = make_tool(vectors)
    assert len(tool._dedupe(make_images([a, b]), 0.8)) == 1
    assert len(tool._dedupe(make_images([a, b]), 0.99)) == 2


def test_dedupe_handles_missing_vector():
    a, b = "D:/a/1.jpg", "D:/a/2.jpg"
    tool = make_tool({Indexer.image_id(a): [1.0, 0.0]})
    kept = tool._dedupe(make_images([a, b]), 0.95)
    assert len(kept) == 2
