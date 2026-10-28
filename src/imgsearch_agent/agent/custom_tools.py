"""全局自定义工具库：custom_tools.yaml 的读写与加载。

和内置 tools.yaml 完全隔离：
- 内置工具固定，不允许动态修改
- 自定义工具由用户确认后写入，可单独备份、清空

加载顺序：内置 → 自定义 → 会话临时
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from .dynamic import DynamicToolError, DynamicToolSpec, build_dynamic_tool

logger = logging.getLogger("imgsearch_agent.custom_tools")


def config_path() -> Path:
    from ..retriever.runtime import PROJECT_ROOT

    return Path(PROJECT_ROOT) / "configs" / "custom_tools.yaml"


def _read_raw() -> dict:
    path = config_path()
    if not path.exists():
        return {"tools": []}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.warning("读取 custom_tools.yaml 失败: %s", e)
        return {"tools": []}
    if not isinstance(data, dict):
        return {"tools": []}
    data.setdefault("tools", [])
    return data


def _write_raw(data: dict) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def list_custom_tools() -> list[dict]:
    """返回自定义工具的原始定义列表。"""
    items = _read_raw().get("tools") or []
    return [item for item in items if isinstance(item, dict)]


def save_custom_tool(spec: dict) -> None:
    """写入或覆盖一个自定义工具。"""
    parsed = DynamicToolSpec(**spec)
    payload = parsed.model_dump()

    data = _read_raw()
    tools = data.get("tools") or []
    replaced = False
    for index, item in enumerate(tools):
        if isinstance(item, dict) and item.get("name") == payload["name"]:
            tools[index] = payload
            replaced = True
            break
    if not replaced:
        tools.append(payload)

    data["tools"] = tools
    _write_raw(data)
    logger.info("自定义工具已保存: %s", payload["name"])


def delete_custom_tool(name: str) -> bool:
    """删除一个自定义工具，返回是否真的删掉了。"""
    data = _read_raw()
    tools = data.get("tools") or []
    remaining = [
        item for item in tools if not (isinstance(item, dict) and item.get("name") == name)
    ]
    if len(remaining) == len(tools):
        return False

    data["tools"] = remaining
    _write_raw(data)
    logger.info("自定义工具已删除: %s", name)
    return True


def load_custom_tools(sandbox=None) -> dict:
    """把自定义工具编译成可执行的 BaseTool 实例。

    单个工具出错不影响其他工具，也不阻止应用启动。
    """
    loaded: dict = {}
    for item in list_custom_tools():
        name = str(item.get("name") or "")
        if not name:
            continue
        try:
            spec = DynamicToolSpec(**item)
            loaded[name] = build_dynamic_tool(spec, sandbox)
            logger.info("已加载自定义工具: %s", name)
        except DynamicToolError as e:
            logger.warning("自定义工具 %s 加载被拒绝: %s", name, e)
        except Exception as e:
            logger.warning("自定义工具 %s 加载失败: %s", name, e)
    return loaded
