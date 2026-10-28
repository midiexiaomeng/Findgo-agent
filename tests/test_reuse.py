"""复用上一轮结果的意图识别测试。"""

from __future__ import annotations

from imgsearch_agent.agent.planner import detect_reuse
from imgsearch_agent.utils.schemas import TaskSpec


def test_detect_reuse_positive():
    for text in (
        "把刚才那两张再发一次",
        "上次找的图再来一遍",
        "重复一下刚才的结果",
        "还是那两个",
    ):
        assert detect_reuse(text) is True, text


def test_detect_reuse_negative():
    for text in ("找两张猫的图片", "帮我找海边的日落", "索引这个目录"):
        assert detect_reuse(text) is False, text


def test_task_spec_default_not_reuse():
    assert TaskSpec().reuse_last is False
