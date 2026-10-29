"""Planner 加固测试：任务白名单、core_query 清洗、路径沙箱、JSON 重试。"""

from __future__ import annotations

import pytest
from findgo.agent.constraints import clean_core_query
from findgo.agent.path_extractor import PathExtractor
from findgo.agent.planner import has_index_intent

# ── core_query 清洗 ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expect_core",
    [
        ("找两张猫的图片", "猫"),
        ("两张白色的猫", "白色的猫"),
        ("5 张海边日落", "海边日落"),
        ("猫，最好躺着的", "猫"),
        ("两张", ""),
        ("图片", ""),
        ("", ""),
    ],
)
def test_clean_core_query(raw, expect_core):
    core, _, _ = clean_core_query(raw)
    assert core == expect_core


def test_clean_extracts_negative():
    core, neg, _ = clean_core_query("找两张猫的图片，不要黑色的")
    assert core == "猫"
    assert "黑色" in neg


def test_clean_extracts_preference():
    _, _, pref = clean_core_query("猫，最好躺着的")
    assert "躺着" in pref


def test_clean_keeps_subject():
    core, _, _ = clean_core_query("白色的猫")
    assert "猫" in core


# ── 建库意图 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("把 D:\\Photos 建索引", True),
        ("这个目录加入库", True),
        ("重新建库", True),
        ("在 D:\\Photos 找猫", False),
        ("找两张猫的图片", False),
        ("这张图入库了吗", True),
    ],
)
def test_has_index_intent(text, expected):
    assert has_index_intent(text) is expected


# ── 路径沙箱 ──────────────────────────────────────────────────────────────


class _DenySandbox:
    def check(self, path, mode):
        raise PermissionError(f"不在白名单: {path}")


def test_path_extractor_rejects_blocked_candidates():
    extractor = PathExtractor(sandbox=_DenySandbox())
    result = extractor.resolve_with_llm("在 D:\\Secret 找图", ["D:\\Secret"], client=None)
    assert result == ""


def test_path_extractor_allows_without_sandbox():
    extractor = PathExtractor(sandbox=None)
    assert extractor._allowed("D:\\any") is True


def test_path_extractor_allowed_with_no_block():
    class _Allow:
        def check(self, path, mode):
            return path

    extractor = PathExtractor(sandbox=_Allow())
    assert extractor._allowed("D:\\ok") is True


# ── 提示词约束 ────────────────────────────────────────────────────────────


def test_prompt_has_hard_rules():
    import inspect

    from findgo.agent.planner import Planner

    src = inspect.getsource(Planner.plan_tasks)
    assert "硬性铁则" in src
    assert "检索任务永远不生成 index 任务" in src
    assert "一条指令只能生成一类任务" in src
    assert "严禁把数字、否定词、数量词写进 core_query" in src


def test_prompt_has_json_lock():
    import inspect

    from findgo.agent.planner import Planner

    src = inspect.getsource(Planner.plan_tasks)
    assert "只返回纯 JSON" in src
    assert "不新增字段" in src


def test_extra_context_marked_reference_only():
    import inspect

    from findgo.agent.planner import Planner

    src = inspect.getsource(Planner.plan_tasks)
    assert "仅作上下文参考" in src
    assert "不得据此新增任务" in src


def test_has_retry_logic():
    import inspect

    from findgo.agent.planner import Planner

    src = inspect.getsource(Planner.plan_tasks)
    assert "for attempt in range(2)" in src
    assert "降级为正则拆解" in src


# ── 任务类型白名单 ────────────────────────────────────────────────────────


def test_index_action_downgraded_without_intent():
    """用户没要求建库时，模型给的 index 任务应降回 retrieve。"""
    import inspect

    from findgo.agent.planner import Planner

    src = inspect.getsource(Planner.plan_tasks)
    assert 'task.action == "index" and not index_mode' in src
    assert 'task.action = "retrieve"' in src


def test_index_action_set_with_intent():
    import inspect

    from findgo.agent.planner import Planner

    src = inspect.getsource(Planner.plan_tasks)
    assert 'task.action = "index"' in src
