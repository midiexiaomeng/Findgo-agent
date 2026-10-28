"""Critic 测试：置信判定、空结果、查询改写。"""

from __future__ import annotations

from findgo.agent.critic import Critic


def test_check_passes_above_threshold():
    critique = Critic(threshold=0.25).check({"images": [{"path": "a.jpg"}], "confidence": 0.5})
    assert critique.passed is True
    assert critique.confidence == 0.5


def test_check_fails_below_threshold():
    critique = Critic(threshold=0.25).check({"images": [{"path": "a.jpg"}], "confidence": 0.1})
    assert critique.passed is False
    assert "低于阈值" in critique.reason


def test_check_fails_on_empty_results():
    critique = Critic().check({"images": [], "confidence": 0.0})
    assert critique.passed is False
    assert "没有检索到" in critique.reason


def test_rewrite_without_client_returns_original():
    assert Critic().rewrite("猫", "低置信") == "猫"


def test_rewrite_with_client():
    class FakeClient:
        def chat(self, messages, json_mode=False, max_tokens=512):
            return "一只橘猫在窗台上"

    assert Critic(FakeClient()).rewrite("猫", "低置信") == "一只橘猫在窗台上"


def test_rewrite_failure_falls_back():
    class Boom:
        def chat(self, *args, **kwargs):
            raise RuntimeError("boom")

    assert Critic(Boom()).rewrite("猫", "低置信") == "猫"
