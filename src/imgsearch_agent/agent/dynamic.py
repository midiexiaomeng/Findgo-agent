"""动态工具：会话级临时能力的定义、加载与安全闸门。

三层工具来源：

1. 内置（tools.yaml）—— 固定，不允许动态修改
2. 全局自定义（custom_tools.yaml）—— 用户确认后保存
3. 会话临时（session.dynamic_tools）—— 会话销毁即丢失

动态工具的源码必须经过两道关：

- 静态扫描：命中 subprocess / eval / exec / os.system 等关键字直接拒绝
- 隔离执行：只用白名单库构建 namespace，路径校验函数注入进去
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("imgsearch_agent.dynamic")

#: 动态工具源码里出现这些片段直接拒绝
BLOCKED_TOKENS = (
    "subprocess",
    "os.system",
    "os.popen",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "shutil.rmtree",
    "socket",
    "__import__",
    "eval(",
    "exec(",
    "compile(",
    "open(",
    "pickle",
    "ctypes",
    "__subclasses__",
    "__globals__",
)

TAG_TO_PERMISSION = {
    "read-only": "read",
    "index-write": "index_write",
    "file-write": "file_write",
    "high-risk": "danger_delete",
}


class DynamicToolError(RuntimeError):
    """动态工具定义不合法，拒绝加载。"""


class DynamicToolSpec(BaseModel):
    """一份动态工具的完整定义。"""

    name: str = Field(..., min_length=2, max_length=64)
    description: str = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=lambda: ["read-only"])
    timeout: float = Field(60.0, ge=1.0, le=3600.0)
    args_schema_code: str = Field(..., min_length=1)
    run_func_code: str = Field(..., min_length=1)
    safety_note: str = ""


def scan_source(code: str) -> None:
    """静态扫描危险关键字，命中即抛错。"""
    lowered = (code or "").lower()
    for token in BLOCKED_TOKENS:
        if token.lower() in lowered:
            raise DynamicToolError(f"源码包含禁止的关键字：{token}")


def _build_namespace(sandbox):
    """只开放白名单库，其余一律不给。"""
    import pydantic
    from PIL import Image

    safe_builtins = {
        "__build_class__": __build_class__,
        "object": object,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
        "enumerate": enumerate,
        "range": range,
        "zip": zip,
        "any": any,
        "all": all,
        "abs": abs,
        "round": round,
        "isinstance": isinstance,
        "Exception": Exception,
        "ValueError": ValueError,
        "print": print,
    }

    return {
        "__builtins__": safe_builtins,
        "BaseModel": pydantic.BaseModel,
        "Field": pydantic.Field,
        "Path": Path,
        "Image": Image,
        "json": json,
        "re": re,
        "sandbox_check_path": sandbox.check if sandbox is not None else None,
    }


def build_dynamic_tool(spec: DynamicToolSpec, sandbox=None):
    """把 DynamicToolSpec 编译成一个可执行的 BaseTool 实例。"""
    from ..tools.base_tool import BaseTool

    scan_source(spec.args_schema_code)
    scan_source(spec.run_func_code)

    namespace = _build_namespace(sandbox)
    namespace["__name__"] = f"dynamic_{spec.name}"

    # ── 参数模型 ──
    try:
        code = compile(spec.args_schema_code, f"<dynamic:{spec.name}:args>", "exec")
        exec(code, namespace)  # noqa: S102 - 唯一入口，源码已过静态扫描
    except Exception as e:
        raise DynamicToolError(f"参数模型编译失败：{e}") from e

    schema_cls = namespace.get("Args")
    if schema_cls is None or not isinstance(schema_cls, type):
        raise DynamicToolError("args_schema_code 必须定义名为 Args 的模型")

    # ── run 函数 ──
    try:
        code = compile(spec.run_func_code, f"<dynamic:{spec.name}:run>", "exec")
        exec(code, namespace)  # noqa: S102
    except Exception as e:
        raise DynamicToolError(f"run 函数编译失败：{e}") from e

    run_fn = namespace.get("run")
    if not callable(run_fn):
        raise DynamicToolError("run_func_code 必须定义名为 run 的函数")

    tool_permission = "read"
    for tag in spec.tags:
        tool_permission = TAG_TO_PERMISSION.get(tag, tool_permission)

    class DynamicTool(BaseTool):
        name = spec.name
        description = spec.description
        args_schema = schema_cls
        permission = tool_permission
        timeout = spec.timeout

        def run(self, **kwargs):
            return run_fn(self, kwargs)

    tool = DynamicTool()
    tool.spec = spec  # 方便 UI 展示源码与安全提示
    return tool
