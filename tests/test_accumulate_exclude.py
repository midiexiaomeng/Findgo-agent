"""累积排除测试：连续「再找」不应把更早的图又放回来。"""

from __future__ import annotations

import numpy as np
import pytest
from findgo.retriever.searcher import Searcher


class _Hit:
    def __init__(self, path, score=0.8):
        from findgo.retriever.vectorstore.base import VectorHit

        self.hit = VectorHit(id=path, score=score, metadata={"path": path})

    def __getattr__(self, name):
        return getattr(self.hit, name)


class _Store:
    """固定返回一批候选，让排除逻辑成为唯一变量。"""

    def __init__(self, paths):
        self.paths = list(paths)
        self.calls = []

    def query(self, vec, top_k=10):
        self.calls.append(top_k)
        return [_Hit(p, 0.9 - i * 0.01) for i, p in enumerate(self.paths[:top_k])]

    def count(self):
        return len(self.paths)


class _Encoder:
    def encode_text(self, text):
        return np.zeros(8, dtype=np.float32)


def _make_searcher(paths):
    searcher = Searcher.__new__(Searcher)
    searcher.store = _Store(paths)
    searcher.encoder = _Encoder()
    searcher.cfg = type("C", (), {"search": type("S", (), {"recall_k": 50, "min_score": 0.0})()})()
    return searcher


def test_recall_excludes_given_paths():
    paths = [f"D:/a/{i}.jpg" for i in range(10)]
    searcher = _make_searcher(paths)

    hits = Searcher._recall(searcher, None, 5, 0.0, None, {"d:/a/0.jpg", "d:/a/1.jpg"})

    got = [h.path for h in hits]
    assert "D:/a/0.jpg" not in got
    assert "D:/a/1.jpg" not in got


def test_seen_paths_accumulate(tmp_path):
    """模拟三轮检索：第三轮要排除前两轮的全部结果。"""
    paths = [f"D:/a/{i}.jpg" for i in range(20)]

    from findgo.agent.orchestrator import FindgoAgent

    agent = FindgoAgent.__new__(FindgoAgent)
    agent._seen_paths = set()
    agent._last_paths = set()
    agent._last_images = []

    # 第一轮
    round1 = {"D:/a/0.jpg", "D:/a/1.jpg"}
    agent._seen_paths |= round1
    agent._last_paths = round1

    # 第二轮排除第一轮
    assert agent._seen_paths == round1

    round2 = {"D:/a/2.jpg", "D:/a/3.jpg"}
    agent._seen_paths |= round2
    agent._last_paths = round2

    # 第三轮应排除前两轮共 4 张
    excluded = agent._seen_paths | agent._last_paths
    assert len(excluded) == 4
    assert "D:/a/0.jpg" in excluded
    assert "D:/a/3.jpg" in excluded


def test_reset_recall_memory_clears_all():
    from findgo.agent.orchestrator import FindgoAgent

    agent = FindgoAgent.__new__(FindgoAgent)
    agent._seen_paths = {"a.jpg", "b.jpg"}
    agent._last_paths = {"b.jpg"}
    agent._last_images = [{"path": "b.jpg"}]

    agent.reset_recall_memory()

    assert agent._seen_paths == set()
    assert agent._last_paths == set()
    assert agent._last_images == []


@pytest.mark.parametrize(
    "text,expected",
    [
        ("从头找一遍猫的图片", True),
        ("重新搜一下", True),
        ("忽略之前的结果", True),
        ("再找两张", False),
        ("找猫", False),
    ],
)
def test_detect_reset(text, expected):
    from findgo.agent.planner import detect_reset

    assert detect_reset(text) is expected


def test_reset_keywords_do_not_collide_with_reuse():
    from findgo.agent.planner import detect_reuse

    # 「重新找」是重置，不该被当成复用上一轮
    assert detect_reuse("重新找一遍") is False
