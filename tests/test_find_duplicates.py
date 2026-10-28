"""重复图片检测测试。"""

from __future__ import annotations

import numpy as np

from findgo.tools import FindDuplicatesTool
from findgo.utils.sandbox import Sandbox


class _Store:
    def __init__(self, rows):
        self.rows = rows

    def get_vectors(self, limit=None):
        return self.rows


def _pipeline(rows):
    class P:
        store = _Store(rows)

    return P()


def test_finds_duplicate_pair():
    rows = [
        ("1", np.array([1.0, 0.0, 0.0], dtype=np.float32), {"path": "D:/a/1.jpg"}),
        ("2", np.array([1.0, 0.0, 0.0], dtype=np.float32), {"path": "D:/a/2.jpg"}),
        ("3", np.array([0.0, 1.0, 0.0], dtype=np.float32), {"path": "D:/a/3.jpg"}),
    ]
    tool = FindDuplicatesTool(_pipeline(rows))
    result = tool.run(threshold=0.95)
    assert result["count"] == 1
    assert result["groups"][0]["count"] == 2
    assert "1.jpg" in result["groups"][0]["paths"][0]


def test_no_duplicates():
    rows = [
        ("1", np.array([1.0, 0.0], dtype=np.float32), {"path": "D:/a/1.jpg"}),
        ("2", np.array([0.0, 1.0], dtype=np.float32), {"path": "D:/a/2.jpg"}),
    ]
    tool = FindDuplicatesTool(_pipeline(rows))
    result = tool.run(threshold=0.95)
    assert result["count"] == 0


def test_empty_index():
    tool = FindDuplicatesTool(_pipeline([]))
    result = tool.run()
    assert result["count"] == 0
    assert "索引为空" in result["summary"]


def test_threshold_controls_grouping():
    rows = [
        ("1", np.array([1.0, 0.0], dtype=np.float32), {"path": "D:/a/1.jpg"}),
        ("2", np.array([0.9, 0.435], dtype=np.float32), {"path": "D:/a/2.jpg"}),
    ]
    loose = FindDuplicatesTool(_pipeline(rows)).run(threshold=0.8)
    strict = FindDuplicatesTool(_pipeline(rows)).run(threshold=0.99)
    assert loose["count"] == 1
    assert strict["count"] == 0
