"""Agent 编排逻辑测试：不依赖真实模型，用假客户端验证决策、执行、降级。"""

from __future__ import annotations

import pytest

from imgsearch.agent import ImageSearchAgent


class FakeResult:
    def __init__(self, filename="a.jpg"):
        self.rank = 1
        self.filename = filename
        self.path = "D:/" + filename
        self.score = 0.9
        self.caption = "a cat"


class FakeResp:
    def __init__(self):
        self.results = [FakeResult()]

    def status_line(self):
        return "召回 50 条 · 返回 1 张"


class FakePipeline:
    def search_text(self, query, top_k=6):
        return FakeResp()

    def search_image(self, path, top_k=6):
        return FakeResp()

    def stats(self):
        return {"images": 10, "backend": "qdrant", "rerank_enabled": True}

    def index_folder(self, path):
        class S:
            def summary(self):
                return "索引 10 张"

        return S()


class FakeClient:
    def __init__(self, decision='{"tool":"search_images","args":{"query":"猫"}}'):
        self.decision = decision
        self.n = 0

    def chat(self, messages, json_mode=False, max_tokens=None):
        self.n += 1
        if self.n == 1:
            return self.decision
        return "找到一张猫的图片。"


def test_search_via_tool():
    agent = ImageSearchAgent(FakePipeline(), FakeClient())
    r = agent.run("帮我找猫")
    assert r.tool == "search_images"
    assert len(r.results) == 1
    assert "猫" in r.reply


def test_chitchat_without_tool():
    agent = ImageSearchAgent(FakePipeline(), FakeClient('{"tool":"none","reply":"你好"}'))
    r = agent.run("你好")
    assert r.tool == ""
    assert r.reply == "你好"


def test_stats_tool():
    agent = ImageSearchAgent(FakePipeline(), FakeClient('{"tool":"get_stats","args":{}}'))
    r = agent.run("库里有多少张图")
    assert r.tool == "get_stats"


def test_decision_failure_falls_back():
    class BadClient:
        def __init__(self):
            self.n = 0

        def chat(self, *args, **kwargs):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("boom")
            return "ok"

    agent = ImageSearchAgent(FakePipeline(), BadClient())
    r = agent.run("猫")
    assert r.tool == "search_images"


def test_unknown_tool_handled():
    agent = ImageSearchAgent(FakePipeline(), FakeClient('{"tool":"nope","args":{}}'))
    r = agent.run("随便")
    assert "无法处理" in r.reply
