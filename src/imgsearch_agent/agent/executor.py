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

logger = logging.getLogger("imgsearch_agent.executor")


class ToolExecutor:
    def __init__(
        self,
        max_output_chars: int = 20000,
        default_timeout: float = 180.0,
        max_workers: int = 4,
    ):
        self.max_output_chars = max_output_chars
        self.default_timeout = default_timeout
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="tool"
        )

    def execute(
        self,
        tool: BaseTool,
        args: dict[str, Any],
        timeout: float | None = None,
        should_stop=None,
    ) -> tuple[dict[str, Any], int]:
        started = time.time()

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

        limit = float(timeout or self.default_timeout)
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
