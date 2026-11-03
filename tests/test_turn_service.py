"""TurnService：不依赖 Qt 的完整一轮执行。

覆盖命令、任务链、ReAct、动态工具待审核、联网补足、以图搜图六条分支，
以及「一轮只写一对 user/assistant」这条记忆契约。
"""

from __future__ import annotations

from types import SimpleNamespace

from findgo.agent import turn_service as turn_service_module
from findgo.agent.orchestrator import AgentResult
from findgo.agent.turn_service import TurnService
from findgo.tools.base_tool import BaseTool
from findgo.utils.schemas import TaskSpec


class FakeResp:
    def __init__(self):
        self.results = [SimpleNamespace(rank=1, path="D:/a.jpg", caption="a cat", score=0.9)]

    @staticmethod
    def status_line() -> str:
        return "召回 50 条 · 返回 1 张"


class FakeSearcher:
    def search_text(self, query, **kwargs):
        return FakeResp()


class FakePipeline:
    def __init__(self):
        self.searcher = FakeSearcher()


class FakeClient:
    def chat(self, messages, json_mode=False, max_tokens=512):
        return "好的。"


def make_agent():
    from findgo.agent import FindgoAgent
    from findgo.config import AgentConfig

    cfg = AgentConfig()
    cfg.memory_persist = False
    return FindgoAgent(config=cfg, client=FakeClient(), pipeline=FakePipeline())


class FakeWebTool:
    name = "web_image_search"

    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def run(self, query, top_k):
        self.calls.append((query, top_k))
        return {"summary": "下载 20 张"}


class FakeImageTool(BaseTool):
    name = "image_search_by_image"
    permission = "read"
    timeout = 5
    args_schema = None

    def __init__(self):
        self.calls: list[dict] = []

    def run(self, image_path, top_k=12):
        self.calls.append({"image_path": image_path, "top_k": top_k})
        return {
            "images": [{"path": "D:/similar.jpg"}],
            "summary": "找到 1 张",
            "confidence": 0.9,
        }


# ── 任务链 ────────────────────────────────────────────────────────────────


def test_run_executes_tasks_and_writes_one_exchange(monkeypatch):
    agent = make_agent()
    monkeypatch.setattr(agent, "plan_tasks", lambda text: [TaskSpec(id=1, query="猫")])

    outcome = TurnService(agent).run("找猫")

    assert outcome.mode == "tasks"
    assert outcome.reply
    assert len(outcome.images) == 1

    roles = [m["role"] for m in agent.memory.recent(n=10)]
    assert roles.count("user") == 1
    assert roles.count("assistant") == 1


def test_default_scope_fills_missing_task_path(monkeypatch):
    agent = make_agent()
    monkeypatch.setattr(agent, "plan_tasks", lambda text: [TaskSpec(id=1, query="猫")])

    outcome = TurnService(agent).run("找猫", default_scope="D:/photos")

    assert outcome.tasks[0].path == "D:/photos"


def test_dynamic_task_waits_for_review(monkeypatch):
    agent = make_agent()
    task = TaskSpec(id=1, query="统计", is_dynamic=True, dynamic_spec={"name": "count_width"})
    monkeypatch.setattr(agent, "plan_tasks", lambda text: [task])

    outcome = TurnService(agent).run("统计一下")

    assert outcome.mode == "tasks"
    assert "审核" in outcome.reply
    assert outcome.results == []
    assert outcome.tasks == [task]


def test_progress_reports_task_list(monkeypatch):
    agent = make_agent()
    tasks = [TaskSpec(id=1, query="猫")]
    monkeypatch.setattr(agent, "plan_tasks", lambda text: tasks)
    seen = []

    TurnService(agent, on_progress=seen.append).run("找猫")

    assert seen == [tasks]


# ── ReAct ─────────────────────────────────────────────────────────────────


def test_react_branch_uses_run_react(monkeypatch):
    agent = make_agent()
    monkeypatch.setattr(turn_service_module, "needs_exploration", lambda text: True)

    def fake_react(text, should_stop=None, access_level="auto", on_step=None):
        if on_step is not None:
            on_step({"index": 1, "thought": "t", "tool": "x", "action": "x", "observation": "o"})
        return AgentResult(reply="探索完成", results=[{"path": "D:/x.jpg"}])

    monkeypatch.setattr(agent, "run_react", fake_react)
    steps, phases = [], []

    outcome = TurnService(agent, on_step=steps.append, on_phase=phases.append).run("把图片分类")

    assert outcome.mode == "react"
    assert outcome.reply == "探索完成"
    assert outcome.images == [{"path": "D:/x.jpg"}]
    assert len(steps) == 1
    assert phases == ["react"]


# ── 命令 ──────────────────────────────────────────────────────────────────


def test_run_command(monkeypatch):
    import findgo.agent.commands as commands

    agent = make_agent()
    monkeypatch.setattr(commands, "dispatch", lambda text, agent: "命令结果")

    outcome = TurnService(agent).run_command("/help")

    assert outcome.mode == "command"
    assert outcome.reply == "命令结果"
    assert outcome.tasks == []


def test_run_command_turns_errors_into_reply(monkeypatch):
    import findgo.agent.commands as commands

    agent = make_agent()

    def boom(text, agent):
        raise RuntimeError("炸了")

    monkeypatch.setattr(commands, "dispatch", boom)

    outcome = TurnService(agent).run_command("/anything")

    assert "命令执行失败" in outcome.reply


# ── 联网补足 ──────────────────────────────────────────────────────────────


def test_web_fill_in_hybrid_mode():
    agent = make_agent()
    tool = FakeWebTool()
    agent.tools["web_image_search"] = tool
    logs = []

    service = TurnService(agent, on_log=logs.append)
    service.web_fill([TaskSpec(id=1, query="猫", top_k=20)], [], search_mode="hybrid")

    assert tool.calls == [("猫", 20)]
    assert any("联网补充" in line for line in logs)


def test_web_fill_skipped_in_local_mode():
    agent = make_agent()
    tool = FakeWebTool()
    agent.tools["web_image_search"] = tool

    TurnService(agent).web_fill([TaskSpec(id=1, query="猫", top_k=20)], [], search_mode="local")

    assert tool.calls == []


def test_web_fill_skipped_when_stop_requested():
    agent = make_agent()
    tool = FakeWebTool()
    agent.tools["web_image_search"] = tool

    TurnService(agent).web_fill(
        [TaskSpec(id=1, query="猫", top_k=20)],
        [],
        search_mode="hybrid",
        should_stop=lambda: True,
    )

    assert tool.calls == []


# ── 以图搜图 ──────────────────────────────────────────────────────────────


def test_run_image_uses_tool_and_writes_memory():
    agent = make_agent()
    tool = FakeImageTool()
    agent.tools["image_search_by_image"] = tool

    outcome = TurnService(agent).run_image("D:/query.jpg", "找相似的")

    assert outcome.mode == "image"
    assert outcome.images == [{"path": "D:/similar.jpg"}]
    assert tool.calls == [{"image_path": "D:/query.jpg", "top_k": 12}]
    roles = [m["role"] for m in agent.memory.recent(n=10)]
    assert roles.count("user") == 1
    assert roles.count("assistant") == 1


def test_run_image_without_tool_reports_clearly():
    agent = make_agent()
    agent.tools.pop("image_search_by_image", None)

    outcome = TurnService(agent).run_image("D:/query.jpg")

    assert "没有注册以图搜图工具" in outcome.reply
