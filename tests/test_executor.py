"""工具统一执行器测试。"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from imgsearch_agent.agent.executor import ToolExecutor
from imgsearch_agent.tools.base_tool import BaseTool


class _Args(BaseModel):
    query: str = Field(..., min_length=1)


class _SlowTool(BaseTool):
    name = "slow"
    description = "sleeps"
    args_schema = _Args

    def run(self, query: str):
        time.sleep(2)
        return {"ok": True}


class _BoomTool(BaseTool):
    name = "boom"
    description = "raises"
    args_schema = _Args

    def run(self, query: str):
        raise RuntimeError("炸了")


class _EchoTool(BaseTool):
    name = "echo"
    description = "echoes"
    args_schema = _Args

    def run(self, query: str):
        return {"query": query}


def test_timeout_returns_error():
    executor = ToolExecutor(default_timeout=0.2)
    result, _ = executor.execute(_SlowTool(), {"query": "x"}, timeout=0.2)
    assert result["error"] == "timeout"
    assert "超时" in result["summary"]


def test_exception_captured():
    executor = ToolExecutor()
    result, _ = executor.execute(_BoomTool(), {"query": "x"})
    assert result["error"] == "run_failed"
    assert "炸了" in result["summary"]


def test_invalid_args_rejected():
    executor = ToolExecutor()
    result, _ = executor.execute(_EchoTool(), {"query": ""})
    assert result["error"] == "invalid_args"


def test_normal_result():
    executor = ToolExecutor()
    result, _ = executor.execute(_EchoTool(), {"query": "猫"})
    assert result["query"] == "猫"


def test_truncate_long_string():
    executor = ToolExecutor(max_output_chars=10)
    out = executor._truncate("a" * 100)
    assert "已截断" in out
    assert len(out) < 100


def test_truncate_long_list():
    executor = ToolExecutor(max_output_chars=100)
    out = executor._truncate(list(range(500)))
    assert len(out) <= 201
