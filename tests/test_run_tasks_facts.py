"""批处理任务的事实更新开关测试。"""

from __future__ import annotations

from findgo.agent import FindgoAgent
from findgo.config import AgentConfig
from findgo.utils.schemas import TaskSpec


class _Result:
    def __init__(self):
        self.rank = 1
        self.path = "D:/cat.jpg"
        self.caption = "a cat"
        self.score = 0.9


class _Resp:
    def __init__(self):
        self.results = [_Result()]

    def status_line(self):
        return "1 张"


class _Searcher:
    def search_text(
        self, query, top_k=8, scope=None, all_results=False, should_stop=None, exclude_paths=None
    ):
        return _Resp()


class _Pipeline:
    def __init__(self):
        self.searcher = _Searcher()


class _Client:
    """第一次调用返回事实 patch，其余返回普通回复。"""

    def __init__(self, patch_response):
        self.patch_response = patch_response
        self.calls = 0

    def chat(self, messages, json_mode=False, max_tokens=512):
        content = messages[-1]["content"]
        if "结构化事实" in content or "patch" in content:
            self.calls += 1
            return self.patch_response
        return "找到 1 张图片。"


def make_agent(patch_response, tmp_path):
    cfg = AgentConfig()
    cfg.memory_persist = True
    cfg.memory_db_path = str(tmp_path / "facts.sqlite")
    client = _Client(patch_response)
    agent = FindgoAgent(config=cfg, client=client, pipeline=_Pipeline())
    return agent, client


def test_default_does_not_update_facts(tmp_path):
    agent, client = make_agent('{"action":"update","patch":{"count":9}}', tmp_path)
    agent.memory.patch_facts({"count": 2})

    task = TaskSpec(id=1, description="找猫", query="找猫", status="pending")
    agent.run_tasks([task], write_memory=False)

    assert agent.memory.get_facts()["count"] == 2
    assert client.calls == 0


def test_enabled_updates_facts(tmp_path):
    agent, client = make_agent('{"action":"update","patch":{"count":9}}', tmp_path)
    agent.memory.patch_facts({"count": 2})

    task = TaskSpec(id=1, description="找猫", query="找猫", status="pending")
    agent.run_tasks([task], write_memory=False, auto_update_facts=True)

    assert agent.memory.get_facts()["count"] == 9
    assert client.calls >= 1


def test_clear_action_wipes_facts(tmp_path):
    agent, _ = make_agent('{"action":"clear"}', tmp_path)
    agent.memory.patch_facts({"count": 2})

    task = TaskSpec(id=1, description="清空约束", query="清空约束", status="pending")
    agent.run_tasks([task], write_memory=False, auto_update_facts=True)

    assert agent.memory.get_facts() == {}
