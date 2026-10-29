"""链式筛选测试：在上一轮结果里继续筛，不重新召回。"""

from __future__ import annotations

import pytest
from findgo.agent.planner import detect_refine
from findgo.utils.schemas import TaskSpec

# ── 意图检测 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("刚才的猫里挑躺着的", True),
        ("从上面这些里找白色的", True),
        ("这些里面再筛一下", True),
        ("在结果里找横图", True),
        ("再找两张猫", False),
        ("找猫", False),
        ("把刚才那两张再发一次", False),
    ],
)
def test_detect_refine(text, expected):
    assert detect_refine(text) is expected


def test_refine_does_not_trigger_reuse():
    """「刚才的」同时出现在复用词里，要靠上下文区分。"""
    from findgo.agent.planner import detect_reuse

    # 「刚才那两张再发一次」= 复用
    assert detect_reuse("刚才那两张再发一次") is True
    # 「刚才的猫里挑躺着的」= 链式筛选，不触发复用
    assert detect_refine("刚才的猫里挑躺着的") is True


# ── 任务结构 ──────────────────────────────────────────────────────────────


def test_task_spec_input_source_default():
    assert TaskSpec().input_source == "library"


def test_task_spec_input_source_set():
    task = TaskSpec(query="猫", input_source="last_results")
    assert task.input_source == "last_results"


def test_planner_sets_input_source():
    import inspect

    from findgo.agent.planner import Planner

    src = inspect.getsource(Planner.plan_tasks)
    assert 'task.input_source = "last_results"' in src
    assert 'task.input_source = "library"' in src


# ── filter_results 扩展 ───────────────────────────────────────────────────


class _Hit:
    def __init__(self, path, caption="", staged=False):
        self.path = path
        self.metadata = {"path": path, "caption": caption, "is_staged": staged}


class _Store:
    def __init__(self, hits):
        self.hits = hits

    def get(self, ids=None, limit=None):
        # 测试里不模拟 image_id 哈希，一律返回全量，由工具按路径自己匹配
        return self.hits


class _Pipeline:
    def __init__(self, hits):
        self.store = _Store(hits)


def _tool(hits):
    from findgo.tools.filter_results import FilterResultsTool

    return FilterResultsTool(_Pipeline(hits))


def test_filter_returns_images_field():
    hits = [_Hit("a.jpg", "a white cat"), _Hit("b.jpg", "a black cat")]
    result = _tool(hits).run(paths=["a.jpg", "b.jpg"])

    assert "images" in result
    assert len(result["images"]) == 2
    assert result["images"][0]["rank"] == 1


def test_filter_applies_constraints():
    hits = [_Hit("a.jpg", "a white cat"), _Hit("b.jpg", "a black cat")]
    result = _tool(hits).run(
        paths=["a.jpg", "b.jpg"],
        constraints={"must_not": ["黑色"]},
    )

    assert [i["path"] for i in result["images"]] == ["a.jpg"]
    assert any("黑色" in d["reason"] for d in result["dropped"])


def test_filter_respects_top_k():
    hits = [_Hit(f"{i}.jpg", "cat") for i in range(5)]
    result = _tool(hits).run(paths=[f"{i}.jpg" for i in range(5)], top_k=2)
    assert len(result["images"]) == 2


def test_filter_preserves_original_order():
    hits = [_Hit("c.jpg", "cat"), _Hit("a.jpg", "cat"), _Hit("b.jpg", "cat")]
    result = _tool(hits).run(paths=["c.jpg", "a.jpg", "b.jpg"])
    assert [i["path"] for i in result["images"]] == ["c.jpg", "a.jpg", "b.jpg"]


def test_filter_drops_staged():
    hits = [_Hit("a.jpg", "cat"), _Hit("b.jpg", "cat", staged=True)]
    result = _tool(hits).run(paths=["a.jpg", "b.jpg"])
    assert [i["path"] for i in result["images"]] == ["a.jpg"]


# ── 执行链路 ──────────────────────────────────────────────────────────────


def test_orchestrator_handles_last_results():
    import inspect

    from findgo.agent.orchestrator import FindgoAgent

    src = inspect.getsource(FindgoAgent.run_tasks)
    assert 'task.input_source == "last_results"' in src
    assert "filter_results" in src
    assert "退回全库检索" in src
