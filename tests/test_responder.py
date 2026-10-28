"""Responder 测试：LLM 合成与模板兜底。"""

from __future__ import annotations

from findgo.agent.responder import Responder


def test_fallback_without_client():
    text = Responder().compose("猫", {"images": [{"path": "a.jpg"}], "summary": "召回 50 条"})
    assert "1 张" in text


def test_llm_compose():
    class FakeClient:
        def chat(self, messages, json_mode=False, max_tokens=512):
            return "找到了猫的图片。"

    text = Responder(FakeClient(), {}).compose("猫", {"images": [{"path": "a.jpg"}], "summary": ""})
    assert text == "找到了猫的图片。"


def test_empty_results_fallback():
    text = Responder().compose("x", {"images": [], "summary": ""})
    assert "没有找到" in text


def test_llm_failure_falls_back():
    class Boom:
        def chat(self, *args, **kwargs):
            raise RuntimeError("boom")

    text = Responder(Boom(), {}).compose("猫", {"images": [{"path": "a.jpg"}], "summary": ""})
    assert "1 张" in text
