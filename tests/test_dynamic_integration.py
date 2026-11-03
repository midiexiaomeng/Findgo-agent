"""动态工具与 Agent 集成测试。"""

from __future__ import annotations

from findgo.agent import FindgoAgent
from findgo.config import AgentConfig
from findgo.utils.schemas import TaskSpec


class _Pipeline:
    class _Searcher:
        def search_text(self, query, **kwargs):
            return type(
                "R",
                (),
                {
                    "results": [],
                    "status_line": staticmethod(lambda: "0 条"),
                },
            )()

    def __init__(self):
        self.searcher = self._Searcher()


def make_agent():
    cfg = AgentConfig()
    cfg.memory_persist = False
    return FindgoAgent(config=cfg, pipeline=_Pipeline())


SPEC = {
    "name": "count_width",
    "description": "统计图片宽度分布",
    "tags": ["read-only"],
    "timeout": 30,
    "args_schema_code": (
        "class Args(BaseModel):\n" '    folder: str = Field(..., description="目录")\n'
    ),
    "run_func_code": (
        "def run(self, args):\n"
        '    path = sandbox_check_path(args["folder"], "read")\n'
        '    return {"count": 0, "summary": "ok"}\n'
    ),
    "safety_note": "只读",
}


def test_register_dynamic_tool():
    agent = make_agent()
    tool = agent.register_dynamic_tool(SPEC)
    assert tool.name == "count_width"
    assert "count_width" in agent.tools
    assert tool.permission == "read"


def test_register_rejects_dangerous_source():
    agent = make_agent()
    bad = dict(SPEC)
    bad["run_func_code"] = "def run(self, args):\n    return eval('1')\n"
    try:
        agent.register_dynamic_tool(bad)
    except Exception as e:
        assert "禁止" in str(e)
    else:
        raise AssertionError("危险源码应该被拒绝")


def test_run_tasks_skips_unapproved_dynamic():
    agent = make_agent()
    task = TaskSpec(
        id=1,
        description="统计宽度",
        is_dynamic=True,
        dynamic_spec=SPEC,
        status="pending",
    )
    results = agent.run_tasks([task], write_memory=False)
    assert len(results) == 1
    assert "尚未批准" in results[0].reply


def test_task_spec_dynamic_fields():
    task = TaskSpec(is_dynamic=True, dynamic_spec=SPEC, dynamic_args={"folder": "D:/x"})
    assert task.is_dynamic is True
    assert task.dynamic_args["folder"] == "D:/x"


def test_run_tasks_executes_approved_dynamic_tool():
    """批准后的动态工具要真的跑起来：参数走 dynamic_args，结果进 reply。"""
    agent = make_agent()
    spec = dict(SPEC)
    spec["name"] = "count_width_ok"
    spec["run_func_code"] = (
        "def run(self, args):\n"
        '    folder = str(args.get("folder", ""))\n'
        '    return {"count": len(folder), "summary": f"统计完成：{folder}"}\n'
    )
    agent.register_dynamic_tool(spec)
    task = TaskSpec(
        id=1,
        description="统计宽度",
        is_dynamic=True,
        dynamic_spec=spec,
        dynamic_args={"folder": "abc"},
        status="pending",
    )

    results = agent.run_tasks([task], write_memory=False)

    assert len(results) == 1
    assert results[0].tool == "count_width_ok"
    assert results[0].reply == "统计完成：abc"


def test_run_tasks_reports_invalid_dynamic_args():
    """dynamic_args 不完整时应返回可读参数错误，而不是崩在编排层。"""
    agent = make_agent()
    agent.register_dynamic_tool(SPEC)
    task = TaskSpec(
        id=1,
        description="统计宽度",
        is_dynamic=True,
        dynamic_spec=SPEC,
        dynamic_args={},
        status="pending",
    )

    results = agent.run_tasks([task], write_memory=False)

    assert len(results) == 1
    assert "参数不合法" in results[0].reply
