"""编排器测试：工具调用、直接回答、低置信重试。"""

from __future__ import annotations

from imgsearch_agent.agent import ImageSearchAgent
from imgsearch_agent.config import AgentConfig


class FakeResult:
    def __init__(self, score=0.9):
        self.rank = 1
        self.path = "D:/cat.jpg"
        self.caption = "a cat"
        self.score = score


class FakeResp:
    def __init__(self, score=0.9):
        self.results = [FakeResult(score)]

    def status_line(self):
        return "召回 50 条 · 返回 1 张"


class FakePipeline:
    def __init__(self, score=0.9):
        self.score = score
        self.calls = 0

    def search_text(self, query, top_k=8):
        self.calls += 1
        return FakeResp(self.score)


class FakeClient:
    def __init__(self, action=None):
        self.action = action or (
            '{"action_type":"tool_call","tool_name":"image_retrieval",'
            '"tool_args":{"query":"猫"}}'
        )
        self.calls: list[str] = []

    def chat(self, messages, json_mode=False, max_tokens=512):
        content = messages[-1]["content"]
        self.calls.append(content)
        if "原始查询" in content:
            return "橘猫"
        if "检索结果" in content:
            return "找到了一张猫的图片。"
        if "可用工具" in content:
            return self.action
        return "好的。"


def make_agent(client, pipeline, **overrides):
    cfg = AgentConfig()
    cfg.memory_persist = False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return ImageSearchAgent(config=cfg, client=client, pipeline=pipeline)


def test_tool_call_flow():
    pipeline = FakePipeline()
    agent = make_agent(FakeClient(), pipeline)
    result = agent.run("找猫")

    assert result.tool == "image_retrieval"
    assert len(result.results) == 1
    assert "猫" in result.reply
    assert pipeline.calls == 1
    assert result.retries == 0


def test_answer_without_tool():
    client = FakeClient('{"action_type":"answer","reply":"你好，我是检索助手"}')
    agent = make_agent(client, FakePipeline())
    result = agent.run("你好")

    assert result.tool == ""
    assert result.reply == "你好，我是检索助手"


def test_low_confidence_triggers_retry():
    client = FakeClient()
    pipeline = FakePipeline(score=0.1)
    agent = make_agent(client, pipeline, max_retries=1)

    result = agent.run("找猫")

    assert result.retries >= 1
    assert pipeline.calls >= 2
    # 改写查询被实际使用
    assert any("原始查询" in c for c in client.calls)


def test_unknown_tool_handled():
    client = FakeClient(
        '{"action_type":"tool_call","tool_name":"nonexistent","tool_args":{}}'
    )
    agent = make_agent(client, FakePipeline())
    result = agent.run("随便")
    assert "没有找到可用的工具" in result.reply
