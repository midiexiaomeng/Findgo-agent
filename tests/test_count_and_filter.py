"""元数据前置过滤与计数任务测试。"""

from __future__ import annotations

import numpy as np
import pytest

# ── 向量库过滤条件 ────────────────────────────────────────────────────────


def test_qdrant_filter_supports_ne():
    """$ne 应该生成 must_not 条件，而不是 must。"""
    from findgo.retriever.vectorstore.qdrant_store import QdrantVectorStore

    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store._models = __import__("qdrant_client.models", fromlist=["models"])

    qfilter = store._build_filter({"is_staged": {"$ne": True}})

    assert qfilter is not None
    assert qfilter.must_not
    assert not qfilter.must


def test_qdrant_filter_supports_must():
    from findgo.retriever.vectorstore.qdrant_store import QdrantVectorStore

    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store._models = __import__("qdrant_client.models", fromlist=["models"])

    qfilter = store._build_filter({"backend": "qdrant"})
    assert qfilter.must
    assert not qfilter.must_not


def test_qdrant_filter_empty_returns_none():
    from findgo.retriever.vectorstore.qdrant_store import QdrantVectorStore

    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store._models = __import__("qdrant_client.models", fromlist=["models"])

    assert store._build_filter(None) is None
    assert store._build_filter({}) is None


def test_searcher_passes_staged_filter():
    """召回时应该把 is_staged 交给向量库，而不是拉回来再丢。"""
    import inspect

    from findgo.retriever.searcher import Searcher

    src = inspect.getsource(Searcher._recall)
    assert "is_staged" in src
    assert "$ne" in src


# ── 计数模式 ──────────────────────────────────────────────────────────────


class _Pipe:
    def __init__(self):
        self.calls = []
        self.searcher = self

    def search_text(self, query, **kwargs):
        self.calls.append(kwargs)

        class _R:
            def __init__(self, path):
                self.rank = 1
                self.path = path
                self.caption = "cat"
                self.score = 0.8
                self.reason = ""

        class _Resp:
            results = [_R("a.jpg"), _R("b.jpg"), _R("c.jpg")]
            report = type(
                "Rep", (), {"used": False, "llm_scored": 0, "candidates": 0, "errors": []}
            )()

            def status_line(self):
                return "返回 3 张"

        return _Resp()


class _Store:
    def get_vectors(self, ids=None):
        return []


def _tool():
    from findgo.tools.image_retrieval import ImageRetrievalTool

    pipe = _Pipe()
    pipe.store = _Store()
    tool = ImageRetrievalTool(pipe)
    return tool, pipe


def test_count_only_returns_number():
    tool, _ = _tool()
    result = tool.run("猫", count_only=True)

    assert result["count"] == 3
    assert result["images"] == []
    assert "共匹配 3 张" in result["summary"]


def test_count_only_skips_dedupe_output():
    tool, _ = _tool()
    result = tool.run("猫", count_only=True, dedupe=True)
    assert isinstance(result["count"], int)


def test_normal_mode_returns_images():
    tool, _ = _tool()
    result = tool.run("猫")
    assert len(result["images"]) == 3


def test_count_only_in_args_schema():
    from findgo.tools.image_retrieval import ImageRetrievalArgs

    args = ImageRetrievalArgs(query="猫", count_only=True)
    assert args.count_only is True
    assert ImageRetrievalArgs(query="猫").count_only is False


# ── 任务分支 ──────────────────────────────────────────────────────────────


def test_orchestrator_handles_count_action():
    import inspect

    from findgo.agent.orchestrator import FindgoAgent

    src = inspect.getsource(FindgoAgent.run_tasks)
    assert 'task.action == "count"' in src
    assert "count_only" in src


def test_planner_prompt_mentions_count():
    import inspect

    from findgo.agent.planner import Planner

    src = inspect.getsource(Planner.plan_tasks)
    assert "action 设为 count" in src
    assert "拆成两个任务" in src
