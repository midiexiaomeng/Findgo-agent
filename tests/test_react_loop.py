"""ReAct 循环与补救策略测试。"""

from __future__ import annotations

import pytest
from findgo.agent.critic import Critic, Critique
from findgo.agent.planner import needs_exploration

# ── 探索意图 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("帮我整理这个文件夹里的图片，按内容分类", True),
        ("把这些图分组", True),
        ("逐个标注", True),
        ("有哪些类型的图片", True),
        ("找两张猫的图片", False),
        ("再发一次", False),
        ("导出结果", False),
    ],
)
def test_needs_exploration(text, expected):
    assert needs_exploration(text) is expected


# ── 补救策略 ──────────────────────────────────────────────────────────────


def test_strategy_passed():
    assert Critic().choose_strategy(Critique(True, 0.5, "ok")) == "none"


def test_strategy_expand_k_for_count_gap():
    critique = Critique(False, 0.5, "数量不够", missing_count=3)
    assert Critic().choose_strategy(critique) == "expand_k"


def test_strategy_relax_must_for_violation():
    critique = Critique(False, 0.5, "约束不中", violated=["白色"])
    assert Critic().choose_strategy(critique) == "relax_must"


def test_strategy_rewrite_for_low_confidence():
    critique = Critique(False, 0.1, "置信低")
    assert Critic().choose_strategy(critique) == "rewrite"


def test_strategy_none_when_no_issue():
    critique = Critique(False, 0.9, "未知")
    assert Critic().choose_strategy(critique) == "none"


def test_count_gap_takes_priority_over_violation():
    """数量不足和约束不中都存在时，先扩量。"""
    critique = Critique(False, 0.5, "x", missing_count=2, violated=["白色"])
    assert Critic().choose_strategy(critique) == "expand_k"


# ── 循环实现 ──────────────────────────────────────────────────────────────


class _Action:
    def __init__(self, action_type, tool_name=None, tool_args=None, reply="", thought=""):
        self.action_type = action_type
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
        self.reply = reply
        self.thought = thought


class _FakePlanner:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    def plan(self, query, tools_desc, history=(), extra_context="", steps=()):
        self.calls.append({"steps": len(steps)})
        if self.actions:
            return self.actions.pop(0)
        return _Action("answer", reply="默认回答")


class _FakeTool:
    name = "image_retrieval"
    permission = "read"
    timeout = 60

    def __init__(self):
        self.calls = 0

    def signature(self):
        return "image_retrieval(query, top_k)"

    def run(self, **kwargs):
        self.calls += 1
        return {
            "images": [{"path": f"img{self.calls}.jpg", "caption": "cat"}],
            "confidence": 0.6,
            "summary": f"找到 {self.calls} 张",
        }

    def permission_for(self, args):
        return "read"

    def validate(self, args):
        return {"query": args.get("query", "猫"), "top_k": args.get("top_k", 8)}


def _agent(actions):
    from findgo.agent.orchestrator import FindgoAgent

    agent = FindgoAgent.__new__(FindgoAgent)
    agent.planner = _FakePlanner(actions)
    tool = _FakeTool()
    agent.tools = {"image_retrieval": tool}

    class _Mem:
        session_id = "s1"

        def add(self, *a, **k):
            pass

        def recent(self, n=10):
            return []

        def add_trace(self, *a, **k):
            pass

        def get_facts(self):
            return {}

    class _Trace:
        def info(self, *a, **k):
            pass

        def error(self, *a, **k):
            pass

    class _Critic:
        max_retries = 0

        def check(self, result, task=None):
            return Critique(True, 0.6, "ok")

        def choose_strategy(self, c):
            return "none"

    from findgo.agent.executor import ToolExecutor

    agent.memory = _Mem()
    agent.trace = _Trace()
    agent.critic = _Critic()
    agent.executor = ToolExecutor()
    agent.client = None
    agent.config = type(
        "C",
        (),
        {
            "top_k": 8,
            "memory_persist": False,
            "memory_max_sessions": 20,
        },
    )()
    agent._memory_context = lambda: ""
    return agent, tool


def test_react_stops_on_answer():
    agent, tool = _agent(
        [
            _Action("tool_call", "image_retrieval", {"query": "猫"}, thought="先找猫"),
            _Action("answer", reply="找到了。"),
        ]
    )
    result = agent.run_react("找猫", max_steps=5)

    assert "找到了" in result.reply
    assert tool.calls == 1
    assert len(result.results) == 1
    agent.executor._pool.shutdown(wait=False)


def test_react_passes_previous_steps():
    agent, _ = _agent(
        [
            _Action("tool_call", "image_retrieval", {"query": "猫"}, thought="第一步"),
            _Action("answer", reply="完成"),
        ]
    )
    agent.run_react("找猫", max_steps=5)

    # 第二次调用时应该已经能看到第一步
    assert agent.planner.calls[1]["steps"] == 1
    agent.executor._pool.shutdown(wait=False)


def test_react_breaks_on_repeated_action():
    """同一工具同一参数做两次，直接停。"""
    same = {"query": "猫"}
    agent, tool = _agent(
        [
            _Action("tool_call", "image_retrieval", same, thought="A"),
            _Action("tool_call", "image_retrieval", same, thought="B"),
            _Action("tool_call", "image_retrieval", same, thought="C"),
        ]
    )
    result = agent.run_react("找猫", max_steps=5)

    assert tool.calls == 1
    assert result.reply
    agent.executor._pool.shutdown(wait=False)


def test_react_stops_on_should_stop():
    agent, tool = _agent([_Action("tool_call", "image_retrieval", {"query": "猫"})])
    result = agent.run_react("找猫", max_steps=5, should_stop=lambda: True)

    assert tool.calls == 0
    assert result.reply
    agent.executor._pool.shutdown(wait=False)


def test_react_handles_unknown_tool():
    agent, tool = _agent(
        [
            _Action("tool_call", "nonexistent", {}),
            _Action("answer", reply="换个方式"),
        ]
    )
    result = agent.run_react("做点什么", max_steps=5)

    assert "换个方式" in result.reply
    assert tool.calls == 0
    agent.executor._pool.shutdown(wait=False)


def test_react_hits_max_steps():
    actions = [_Action("tool_call", "image_retrieval", {"query": f"q{i}"}) for i in range(5)]
    agent, tool = _agent(actions)
    result = agent.run_react("找猫", max_steps=3)

    assert tool.calls == 3
    assert result.reply
    agent.executor._pool.shutdown(wait=False)


# ── 观察压缩 ──────────────────────────────────────────────────────────────


def test_compress_images_summary():
    from findgo.agent.orchestrator import FindgoAgent

    agent = FindgoAgent.__new__(FindgoAgent)
    text = agent._compress_observation({"images": [{"path": f"D:/a/{i}.jpg"} for i in range(12)]})
    assert "返回 12 张图" in text
    assert "等 12 张" in text


def test_compress_text_summary():
    from findgo.agent.orchestrator import FindgoAgent

    agent = FindgoAgent.__new__(FindgoAgent)
    text = agent._compress_observation({"summary": "扫描完成，共 30 张"})
    assert "扫描完成" in text


def test_compress_error():
    from findgo.agent.orchestrator import FindgoAgent

    agent = FindgoAgent.__new__(FindgoAgent)
    text = agent._compress_observation({"error": "timeout", "summary": "超时了"})
    assert "超时了" in text


def test_compress_empty():
    from findgo.agent.orchestrator import FindgoAgent

    agent = FindgoAgent.__new__(FindgoAgent)
    assert agent._compress_observation({}) == "无返回内容"


# ── 路由 ──────────────────────────────────────────────────────────────────


def test_agent_runner_routes_to_react():
    import inspect

    from findgo.ui.agent_runner import AgentRunnerMixin

    src = inspect.getsource(AgentRunnerMixin._run_turn)
    assert "needs_exploration" in src
    assert "run_react" in src


def test_planner_accepts_steps():
    import inspect

    from findgo.agent.planner import Planner

    params = inspect.signature(Planner.plan).parameters
    assert "steps" in params
