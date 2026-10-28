"""工具统一执行器。

把所有工具都要做的公共动作收敛到一处：

- 前置：pydantic 参数校验、Sandbox 路径校验（由工具内部完成）、should_stop 注入
- 执行：超时控制、异常捕获
- 后置：结果截断、耗时日志

工具本身只保留业务逻辑，不再各自写 try/except 和超时。
"""

from __future__ import annotations

import inspect
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any

from ..tools.base_tool import BaseTool
from ..utils.exceptions import ValidationError

logger = logging.getLogger("findgo.executor")


class ToolExecutor:
    def __init__(
        self,
        max_output_chars: int = 20000,
        default_timeout: float = 180.0,
        max_workers: int = 4,
    ):
        self.max_output_chars = max_output_chars
        self.default_timeout = default_timeout
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tool")

    def execute(
        self,
        tool: BaseTool,
        args: dict[str, Any],
        timeout: float | None = None,
        should_stop=None,
        allow_danger: bool = False,
        access_level: str = "auto",
    ) -> tuple[dict[str, Any], int]:
        started = time.time()

        try:
            permission = tool.permission_for(args)
        except Exception:
            permission = getattr(tool, "permission", "read")

        blocked = self._check_access(tool.name, permission, access_level, allow_danger)
        if blocked is not None:
            return blocked, 0

        try:
            payload = tool.validate(args)
        except ValidationError as e:
            logger.warning("参数被拒绝 %s: %s", tool.name, e)
            return {"error": "invalid_args", "summary": str(e)}, 0

        if should_stop is not None:
            try:
                if "should_stop" in inspect.signature(tool.run).parameters:
                    payload["should_stop"] = should_stop
            except (TypeError, ValueError):
                pass

        limit = float(timeout or getattr(tool, "timeout", self.default_timeout))
        future = self._pool.submit(tool.run, **payload)

        try:
            result = future.result(timeout=limit)
        except FuturesTimeout:
            future.cancel()
            logger.warning("工具超时 %s（%.0fs）", tool.name, limit)
            return {
                "error": "timeout",
                "summary": f"{tool.name} 执行超时（{limit:.0f} 秒）",
            }, 0
        except Exception as e:
            logger.warning("工具异常 %s: %s", tool.name, e)
            return {
                "error": "run_failed",
                "summary": f"{tool.name} 执行失败：{e}",
            }, 0

        if not isinstance(result, dict):
            result = {"value": result, "summary": str(result)}

        elapsed = time.time() - started
        logger.info("工具 %s 完成，耗时 %.2fs", tool.name, elapsed)
        return self._truncate(result), 0

    # ── 权限校验 ────────────────────────────────────────────────────────────

    @staticmethod
    def _check_access(
        tool_name: str,
        permission: str,
        access_level: str,
        allow_danger: bool,
    ) -> dict[str, Any] | None:
        """按当前权限等级判断能不能执行，返回拒绝原因或 None。

        四档：readonly 只放行读；ask 写操作转人工确认；
        auto 读写都放行；full 全放行。危险删除任何档位都要界面确认。
        """
        level = (access_level or "auto").lower()
        is_write = permission in ("index_write", "file_write", "danger_delete")

        if permission == "danger_delete" and not allow_danger:
            logger.warning("拒绝高危操作 %s（需要人工确认）", tool_name)
            return {
                "error": "need_confirm",
                "need_user_confirm": True,
                "summary": f"{tool_name} 属于高危删除，需要人工确认后执行",
            }

        if level == "readonly" and is_write:
            logger.warning("只读模式下拒绝写入 %s（%s）", tool_name, permission)
            return {
                "error": "readonly_blocked",
                "summary": (
                    f"只读模式：{tool_name} 需要 {permission} 权限，已阻止。"
                    "切换到自动审批或完整权限后重试。"
                ),
            }

        if level == "ask" and is_write:
            logger.info("操作前询问：%s（%s）", tool_name, permission)
            return {
                "error": "need_confirm",
                "need_user_confirm": True,
                "summary": f"操作前询问：{tool_name} 要执行 {permission} 操作，确认后才继续。",
            }

        return None

    # ── 输出截断 ────────────────────────────────────────────────────────────

    def _truncate(self, value: Any, limit: int | None = None) -> Any:
        budget = limit or self.max_output_chars

        if isinstance(value, str):
            if len(value) <= budget:
                return value
            return value[:budget] + "…（已截断）"

        if isinstance(value, list):
            out = [self._truncate(item, budget) for item in value[:200]]
            if len(value) > 200:
                out.append({"truncated": len(value) - 200})
            return out

        if isinstance(value, dict):
            return {k: self._truncate(v, budget) for k, v in value.items()}

        return value
