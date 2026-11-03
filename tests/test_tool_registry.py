"""工具注册一致性：tools.yaml 与构造注册表必须一一对应。

这里守住的是「配置里写了但实际没有」这类静默缺功能：
yaml 说启用 16 个工具，Agent 手里就该有这 16 个。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from findgo.config import AgentConfig, load_tools_config
from findgo.tools.registry import TOOL_BUILDERS, build_tools


class _Pipeline:
    """构造工具只需要一个引用，不会在初始化阶段碰向量库。"""

    searcher = None
    store = None


def test_yaml_and_registry_declare_same_tools():
    declared = {spec["name"] for spec in load_tools_config()}
    assert declared == set(TOOL_BUILDERS), "tools.yaml 与 TOOL_BUILDERS 不一致"


def test_every_declared_tool_has_a_description():
    for spec in load_tools_config():
        assert spec.get("description"), f"{spec.get('name')} 缺少给 Planner 的描述"


def test_build_tools_respects_selection():
    tools = build_tools(_Pipeline(), None, ["folder_scan", "image_retrieval"])

    assert list(tools) == ["folder_scan", "image_retrieval"]
    assert tools["folder_scan"].name == "folder_scan"


def test_build_tools_all_by_default():
    tools = build_tools(_Pipeline(), None)

    assert set(tools) == set(TOOL_BUILDERS)


def test_unknown_name_raises_instead_of_silently_skipping():
    with pytest.raises(ValueError, match="没有对应构造器"):
        build_tools(_Pipeline(), None, ["not_a_real_tool"])


def test_agent_registers_every_enabled_tool():
    cfg = AgentConfig()
    cfg.memory_persist = False

    from findgo.agent import FindgoAgent

    agent = FindgoAgent(config=cfg, pipeline=_Pipeline())

    enabled = {spec["name"] for spec in load_tools_config() if spec.get("enabled", True)}
    assert enabled <= set(agent.tools)
