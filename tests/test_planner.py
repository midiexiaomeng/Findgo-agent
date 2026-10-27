"""Planner 单元测试：验证动作解析、异常降级。"""

from __future__ import annotations

from imgsearch_agent.agent.planner import Planner


class FakeClient:
    def __init__(self, reply: str):
        self.reply = reply

    def chat(self, messages, json_mode=False, max_tokens=512):
        return self.reply


def make_planner(reply: str) -> Planner:
    return Planner(
        FakeClient(reply),
        {"system": "system", "react_template": "工具：{tools}\n查询：{query}"},
        max_iterations=5,
    )


def test_plan_tool_call():
    planner = make_planner(
        '{"action_type":"tool_call","tool_name":"image_retrieval",'
        '"tool_args":{"query":"猫"},"thought":"需要检索"}'
    )
    action = planner.plan("找猫", "image_retrieval — 搜索")
    assert action.action_type == "tool_call"
    assert action.tool_name == "image_retrieval"
    assert action.tool_args["query"] == "猫"


def test_plan_answer():
    planner = make_planner('{"action_type":"answer","reply":"你好"}')
    action = planner.plan("你好", "tools")
    assert action.action_type == "answer"
    assert action.reply == "你好"


def test_plan_json_with_fence():
    planner = make_planner('```json\n{"action_type":"answer","reply":"ok"}\n```')
    action = planner.plan("x", "y")
    assert action.action_type == "answer"
    assert action.reply == "ok"


def test_plan_invalid_json_falls_back():
    planner = make_planner("这不是 JSON")
    action = planner.plan("随便", "tools")
    assert action.action_type == "answer"
    assert "理解" in action.reply


def test_plan_llm_error_falls_back():
    class Boom:
        def chat(self, *args, **kwargs):
            raise RuntimeError("network down")

    planner = Planner(Boom(), {}, max_iterations=5)
    action = planner.plan("x", "y")
    assert action.action_type == "answer"
    assert "无法完成" in action.reply
